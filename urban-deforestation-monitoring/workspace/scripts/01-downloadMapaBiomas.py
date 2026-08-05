import sys
import time
import shutil
import json
import os
import unicodedata
from dotenv import load_dotenv

try:
    import ee
    import geobr
except ImportError:
    pass

def baixar_mapbiomas():
    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        raise ValueError("A variável PROJECT_ROOT não foi encontrada.")

    if len(sys.argv) >= 3:
        CODE_MUNI = int(sys.argv[1])
        ANO = str(sys.argv[2])
    else:
        CODE_MUNI = int(os.environ.get('CODE_MUNI', 0))
        ANO = str(os.environ.get('ANO', ''))

    if not CODE_MUNI or not ANO:
        print("Erro: CODE_MUNI ou ANO não informados.")
        sys.exit(1)

    # Criação prévia de pastas por ano no Google Drive
    diretorio_data_input = os.path.join(project_root, "data", "input", "MapBiomas", ANO)
    diretorio_reports = os.path.join(project_root, "reports")

    os.makedirs(diretorio_data_input, exist_ok=True)
    os.makedirs(diretorio_reports, exist_ok=True)

    try:
        ee.Initialize(project='foresteyes-regioes-urbanas')
        print("Earth Engine inicializado com sucesso.")

        print(f"Buscando informações para o município: {CODE_MUNI} (Ano: {ANO})...")
        gdf_muni = geobr.read_municipality(code_muni=CODE_MUNI, year=2022)
        
        nome_cidade = gdf_muni['name_muni'].values[0]
        uf = gdf_muni['abbrev_state'].values[0]
        print(f"==================================================")
        print(f" CIDADE VALIDADA: {nome_cidade} - {uf} (IBGE: {CODE_MUNI} / ANO: {ANO})")
        print(f"==================================================")

        gdf_muni = gdf_muni.to_crs(epsg=4326)
        geojson_dict = json.loads(gdf_muni.to_json())
        coordenadas = geojson_dict['features'][0]['geometry']['coordinates']
        tipo_geometria = geojson_dict['features'][0]['geometry']['type']
        
        if tipo_geometria == 'MultiPolygon':
            limite_geopolitico = ee.Geometry.MultiPolygon(coordenadas)
        else:
            limite_geopolitico = ee.Geometry.Polygon(coordenadas)

        bounds = limite_geopolitico.bounds().getInfo()['coordinates'][0]
        lons = [p[0] for p in bounds]
        lats = [p[1] for p in bounds]
        oeste, leste = min(lons), max(lons)
        sul, norte = min(lats), max(lats)

        def limpar_para_ee(texto):
            nfkd = unicodedata.normalize('NFKD', texto)
            return "".join([c for c in nfkd if not unicodedata.combining(c)]).replace(" ", "_")

        cidade_limpa = limpar_para_ee(nome_cidade)
        nome_arquivo_mapbiomas = f'mapbiomas_lulc_10m_{cidade_limpa.lower()}_{ANO}.tif'
        caminho_mapbiomas_local = os.path.join(diretorio_data_input, nome_arquivo_mapbiomas)

        dados_json = {
            "code_muni": CODE_MUNI,
            "nome_cidade": nome_cidade,
            "uf": uf,
            "ano": int(ANO),
            "coordenadas_extremas": {"norte": norte, "sul": sul, "leste": leste, "oeste": oeste},
            "arquivo_mapbiomas": caminho_mapbiomas_local
        }
        
        caminho_json = os.path.join(diretorio_reports, f"{CODE_MUNI}_{ANO}.json")
        with open(caminho_json, 'w', encoding='utf-8') as f_json:
            json.dump(dados_json, f_json, indent=4, ensure_ascii=False)
        print(f"[RELATÓRIO] Arquivo JSON gerado em:\n-> {caminho_json}\n")

        BANDA_ANO = f'classification_{ANO}'
        asset_mapbiomas_10m = 'projects/mapbiomas-public/assets/brazil/lulc_10m/collection2/mapbiomas_10m_collection2_integration_v1'
        mapbiomas_10m = ee.Image(asset_mapbiomas_10m).select(BANDA_ANO)

        print("Recortando e mascarando o fundo externo para NoData...")
        imagem_recortada = mapbiomas_10m.clip(limite_geopolitico).unmask(0).short()

        PASTA_TEMPORARIA = 'MapBiomas_Temp'
        tarefa = ee.batch.Export.image.toDrive(
            image=imagem_recortada,
            description=f'Export_mapbiomas_{cidade_limpa.lower()}_{ANO}',
            folder=PASTA_TEMPORARIA,
            fileNamePrefix=f'mapbiomas_lulc_10m_{cidade_limpa.lower()}_{ANO}',
            region=limite_geopolitico.bounds(),
            scale=10,
            maxPixels=1e9,
            fileFormat='GeoTIFF',
            formatOptions={'noData': 0}
        )
        tarefa.start()
        print(f"Tarefa ID: {tarefa.id} iniciada. Aguardando processamento...")

        while True:
            status = ee.data.getTaskStatus(tarefa.id)[0]
            state = status.get('state')
            print(f"Status atual da tarefa: {state}...")
            if state == 'COMPLETED':
                break
            elif state in ['FAILED', 'CANCELLED']:
                error_msg = status.get('error_message', 'Erro desconhecido')
                raise Exception(f"Exportação falhou. Status: {state}. Erro: {error_msg}")
            time.sleep(20)

        raiz_drive = os.path.abspath(os.path.join(project_root, "../../../../.."))
        caminho_temp_local = os.path.join(raiz_drive, PASTA_TEMPORARIA)

        arquivo_origem = os.path.join(caminho_temp_local, nome_arquivo_mapbiomas)
        tentativas = 0
        while not os.path.exists(arquivo_origem) and tentativas < 15:
            time.sleep(5)
            tentativas += 1

        if os.path.exists(arquivo_origem):
            os.makedirs(diretorio_data_input, exist_ok=True)
            shutil.move(arquivo_origem, os.path.join(diretorio_data_input, nome_arquivo_mapbiomas))
            print(f"[SUCESSO] Arquivo movido para:\n-> {os.path.join(diretorio_data_input, nome_arquivo_mapbiomas)}")
            try:
                os.rmdir(caminho_temp_local)
            except OSError:
                pass
        else:
            print(f"[AVISO] Arquivo não encontrado em '{caminho_temp_local}'.")

    except Exception as e:
        print(f"\n[ERRO CRÍTICO] {e}")
        sys.exit(1)

if __name__ == "__main__":
    baixar_mapbiomas()