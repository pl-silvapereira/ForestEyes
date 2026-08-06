import sys
import time
import shutil
import ee
import geobr
import json
import os
import unicodedata
from dotenv import load_dotenv

if len(sys.argv) < 3:
    print("Erro: Parâmetros insuficientes. Uso: python 01-downloadMapaBiomas.py <code_muni> <ano>")
    sys.exit(1)

CODE_MUNI = int(sys.argv[1])
ANO = int(sys.argv[2])

load_dotenv()
project_root = os.getenv("PROJECT_ROOT")

diretorio_data_input = os.path.join(project_root, "data", "input", "MapBiomas", str(ANO))
diretorio_reports = os.path.join(project_root, "reports")

os.makedirs(diretorio_data_input, exist_ok=True)
os.makedirs(diretorio_reports, exist_ok=True)

try:
    print("Earth Engine inicializado com sucesso.")

    print(f"Buscando informações para o código de município: {CODE_MUNI}...")
    gdf_muni = geobr.read_municipality(code_muni=CODE_MUNI, year=2022)
    nome_cidade = gdf_muni['name_muni'].values[0]
    uf = gdf_muni['abbrev_state'].values[0]
    
    print(f"==================================================")
    print(f" CIDADE VALIDADA: {nome_cidade} - {uf} (IBGE: {CODE_MUNI} | ANO: {ANO})")
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
    oeste, leste, sul, norte = min(lons), max(lons), min(lats), max(lats)

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
        "ano": ANO,
        "coordenadas_extremas": {"norte": norte, "sul": sul, "leste": leste, "oeste": oeste},
        "arquivo_mapbiomas": caminho_mapbiomas_local
    }
    
    # ATENÇÃO: Salvando com o ano no nome do arquivo JSON
    caminho_json = os.path.join(diretorio_reports, f"{CODE_MUNI}_{ANO}.json")
    with open(caminho_json, 'w', encoding='utf-8') as f_json:
        json.dump(dados_json, f_json, indent=4, ensure_ascii=False)
    
    print(f"[RELATÓRIO] Arquivo JSON gerado em:\n-> {caminho_json}\n")

    BANDA_ANO = f'classification_{ANO}'
    asset_mapbiomas_10m = 'projects/mapbiomas-public/assets/brazil/lulc_10m/collection3/mapbiomas_10m_collection3_integration_v1'
    mapbiomas_10m = ee.Image(asset_mapbiomas_10m).select(BANDA_ANO)
    imagem_recortada = mapbiomas_10m.clip(limite_geopolitico).unmask(0).short()

    PASTA_TEMPORARIA = 'MapBiomas_Temp'
    print(f"Enviando tarefa para a pasta temporária '{PASTA_TEMPORARIA}' no Google Drive...")
    
    tarefa = ee.batch.Export.image.toDrive(
        image=imagem_recortada,
        description=f'Export_mapbiomas_lulc_10m_{cidade_limpa.lower()}_{ANO}',
        folder=PASTA_TEMPORARIA,
        fileNamePrefix=f'mapbiomas_lulc_10m_{cidade_limpa.lower()}_{ANO}',
        region=limite_geopolitico.bounds(),
        scale=10,
        maxPixels=1e9,
        fileFormat='GeoTIFF',
        formatOptions={'noData': 0}
    )

    tarefa.start()
    
    while True:
        status = ee.data.getTaskStatus(tarefa.id)[0]
        state = status.get('state')
        if state == 'COMPLETED':
            print("\n[SUCESSO] Processamento no Earth Engine concluído!")
            break
        elif state in ['FAILED', 'CANCELLED']:
            raise Exception(f"Exportação falhou. Status: {state}. Erro: {status.get('error_message')}")
        time.sleep(20)

    raiz_drive = os.path.abspath(os.path.join(project_root, "../../../../.."))
    caminho_temp_local = os.path.join(raiz_drive, PASTA_TEMPORARIA)
    arquivo_origem = os.path.join(caminho_temp_local, nome_arquivo_mapbiomas)
    
    tentativas = 0
    while not os.path.exists(arquivo_origem) and tentativas < 15:
        time.sleep(5)
        tentativas += 1

    if os.path.exists(arquivo_origem):
        shutil.move(arquivo_origem, caminho_mapbiomas_local)
        try: os.rmdir(caminho_temp_local)
        except OSError: pass

except Exception as e:
    print(f"\n[ERRO CRÍTICO] Ocorreu um erro durante o processamento: {e}")
    sys.exit(1)