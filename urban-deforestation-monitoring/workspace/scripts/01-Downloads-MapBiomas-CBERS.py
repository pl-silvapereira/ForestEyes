import sys
import time
import shutil
import ee
import geobr
import json
import os
import unicodedata
import google.auth
from datetime import date
from dotenv import load_dotenv
import requests
import numpy as np
import rasterio

try:
    from cbers4asat import Cbers4aAPI
    from cbers4asat.tools import rgbn_composite
except ImportError:
    pass

def inicializar_ee():
    print("Inicializando Google Earth Engine via credenciais da máquina...")
    credentials, _ = google.auth.default(
        scopes=['https://www.googleapis.com/auth/earthengine', 
                'https://www.googleapis.com/auth/cloud-platform']
    )
    ee.Initialize(credentials, project='foresteyes-regioes-urbanas')
    print("Earth Engine inicializado com sucesso.")

def baixar_mapbiomas(code_muni, ano, limite_geopolitico, nome_cidade, uf, projeto_root):
    def limpar_para_ee(texto):
        nfkd = unicodedata.normalize('NFKD', texto)
        return "".join([c for c in nfkd if not unicodedata.combining(c)]).replace(" ", "_")

    cidade_limpa = limpar_para_ee(nome_cidade)
    nome_arquivo_mapbiomas = f'mapbiomas_lulc_10m_{cidade_limpa.lower()}_{ano}.tif'
    
    diretorio_data_input = os.path.join(projeto_root, "data", "input", "MapBiomas", str(ano))
    os.makedirs(diretorio_data_input, exist_ok=True)
    caminho_mapbiomas_local = os.path.join(diretorio_data_input, nome_arquivo_mapbiomas)

    print(f"\n--- INICIANDO DOWNLOAD MAPBIOMAS ({ano}) ---")
    banda_ano = f'classification_{ano}'
    asset_mapbiomas_10m = 'projects/mapbiomas-public/assets/brazil/lulc_10m/collection3/mapbiomas_10m_collection3_integration_v1'
    mapbiomas_10m = ee.Image(asset_mapbiomas_10m).select(banda_ano)
    imagem_recortada = mapbiomas_10m.clip(limite_geopolitico).unmask(0).short()

    pasta_temporaria = f'MapBiomas_Temp_{ano}'
    print(f"Enviando tarefa para a pasta temporária exclusiva '{pasta_temporaria}' no Google Drive...")
    
    tarefa = ee.batch.Export.image.toDrive(
        image=imagem_recortada,
        description=f'Export_mapbiomas_lulc_10m_{cidade_limpa.lower()}_{ano}',
        folder=pasta_temporaria,
        fileNamePrefix=f'mapbiomas_lulc_10m_{cidade_limpa.lower()}_{ano}',
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
            print(f"[SUCESSO] MapBiomas {ano} concluído no Earth Engine!")
            break
        elif state in ['FAILED', 'CANCELLED']:
            raise Exception(f"Exportação MapBiomas {ano} falhou. Status: {state}. Erro: {status.get('error_message')}")
        time.sleep(20)

    raiz_drive = os.path.abspath(os.path.join(projeto_root, "../../../../.."))
    caminho_temp_local = os.path.join(raiz_drive, pasta_temporaria)
    arquivo_origem = os.path.join(caminho_temp_local, nome_arquivo_mapbiomas)
    
    tentativas = 0
    while not os.path.exists(arquivo_origem) and tentativas < 20:
        time.sleep(5)
        tentativas += 1

    if os.path.exists(arquivo_origem):
        shutil.move(arquivo_origem, caminho_mapbiomas_local)
        try: os.rmdir(caminho_temp_local)
        except OSError: pass
    else:
        raise Exception(f"[ERRO] O arquivo {nome_arquivo_mapbiomas} não foi encontrado no Drive após a exportação.")
    
    print(f"MapBiomas {ano} salvo com sucesso em: {caminho_mapbiomas_local}")
    return caminho_mapbiomas_local

def baixar_e_processar_cbers(code_muni, ano_fim, dados_json, projeto_root):
    print(f"\n--- INICIANDO DOWNLOAD E PROCESSAMENTO CBERS-4A ({ano_fim}) ---")
    coords = dados_json['coordenadas_extremas']
    nome_cidade = dados_json['nome_cidade']
    bbox_muni = [coords['oeste'], coords['sul'], coords['leste'], coords['norte']]

    pasta_entrada_cbers = os.path.join(projeto_root, "data", "input", "CBERS-4A-WPM", str(ano_fim))
    pasta_saida_pansharpening = os.path.join(projeto_root, "data", "output", "pansharpening", str(ano_fim))
    
    os.makedirs(pasta_entrada_cbers, exist_ok=True)
    os.makedirs(pasta_saida_pansharpening, exist_ok=True)

    email_inpe = 'pereira.pedro@unifesp.br'
    api = Cbers4aAPI(email_inpe)

    resultados = api.query(
        location=bbox_muni,
        initial_date=date(int(ano_fim), 1, 1),
        end_date=date(int(ano_fim), 12, 31),
        cloud=20, 
        limit=100,
        collections=['CBERS4A_WPM_L4_DN']
    )

    cenas_encontradas = resultados.get('features', [])
    if not cenas_encontradas:
        print(f"Nenhuma cena CBERS encontrada para o ano {ano_fim}.")
        sys.exit(1)

    cena_selecionada = cenas_encontradas[1] if len(cenas_encontradas) > 1 else cenas_encontradas[0]
    id_cena = cena_selecionada.get('id', 'CBERS_4A_WPM')
    
    ativos = cena_selecionada.get('assets', {})
    url_thumbnail = ativos['thumbnail']['href']
    
    url_base_tiff = url_thumbnail.replace("/datastore/BROWSE/", "/api/download/TIFF/").replace("/BROWSE/", "/api/download/TIFF/").replace(".jpg", "").replace(".png", "")
    if not url_base_tiff.endswith("_L4"):
        url_base_tiff += "_L4"
    
    bandas = ['BAND0', 'BAND1', 'BAND2', 'BAND3', 'BAND4']
    arquivos_baixados = {}
    
    for banda in bandas:
        url_download = f"{url_base_tiff}_{banda}.tif?email={email_inpe}"
        caminho_arquivo = os.path.join(pasta_entrada_cbers, f"{id_cena}_{banda}.tif")
        arquivos_baixados[banda] = caminho_arquivo

        if os.path.exists(caminho_arquivo) and os.path.getsize(caminho_arquivo) > 10000:
            continue

        sucesso = False
        for tentativa in range(1, 4):
            try:
                resposta = requests.get(url_download, stream=True, timeout=120)
                resposta.raise_for_status()
                with open(caminho_arquivo, 'wb') as f:
                    for chunk in resposta.iter_content(chunk_size=8192):
                        if chunk: f.write(chunk)
                sucesso = True
                break
            except requests.exceptions.RequestException:
                time.sleep(5)
        
        if not sucesso:
            sys.exit(1)

    nome_arquivo_stack = f"{code_muni}_{ano_fim}_CBERS_TRUE_COLOR_CLIPPED.tif"
    rgbn_composite(
        red=arquivos_baixados['BAND3'], green=arquivos_baixados['BAND2'], 
        blue=arquivos_baixados['BAND1'], nir=arquivos_baixados['BAND4'],   
        filename=nome_arquivo_stack, outdir=pasta_saida_pansharpening
    )
    
    caminho_stack = os.path.join(pasta_saida_pansharpening, nome_arquivo_stack)
    print(f"[SUCESSO] Stack CBERS gerado em: {caminho_stack}")
    
    # Gerar composições para Zooniverse com metadados NIR corretos
    gerar_composicoes_zooniverse(caminho_stack, pasta_saida_pansharpening, code_muni, ano_fim)

    return caminho_stack

def normalize_band(band_data):
    band_data = band_data.astype(np.float32)
    p2, p98 = np.percentile(band_data[band_data > 0], (2, 98)) if np.any(band_data > 0) else (0, 1)
    normalized = np.clip((band_data - p2) / (p98 - p2) * 255.0, 0, 255)
    return normalized.astype(np.uint8)

def gerar_composicoes_zooniverse(caminho_stack, pasta_saida, code_muni, ano):
    print("\n--- GERANDO COMPOSIÇÕES VISUAIS (ZOONIVERSE) COM METADADOS NIR ---")
    pasta_comp = os.path.join(pasta_saida, "composicoes")
    os.makedirs(pasta_comp, exist_ok=True)

    with rasterio.open(caminho_stack) as src:
        meta = src.meta.copy()
        img = src.read()

    if img.shape[0] >= 4:
        red_data = img[0]
        green_data = img[1]
        blue_data = img[2]
        nir_data = img[3]
    else:
        print("[AVISO] O stack não possui 4 bandas completas.")
        return

    print("Normalizando bandas para o padrão visual...")
    r_norm = normalize_band(red_data)
    g_norm = normalize_band(green_data)
    b_norm = normalize_band(blue_data)
    nir_norm = normalize_band(nir_data)

    meta_rgb = meta.copy()
    meta_rgb.update(count=3, dtype=rasterio.uint8)

    # 1. Cor Natural (R-G-B)
    path_rgb = os.path.join(pasta_comp, f"{code_muni}_{ano}_1_CorNatural_RGB.tif")
    with rasterio.open(path_rgb, "w", **meta_rgb) as dst:
        dst.write(r_norm, 1)
        dst.write(g_norm, 2)
        dst.write(b_norm, 3)
        dst.set_band_description(1, "Red")
        dst.set_band_description(2, "Green")
        dst.set_band_description(3, "Blue")
    print(f" -> Gerado: {path_rgb}")

    # 2. Falsa Cor (NIR-R-G)
    path_nir_rg = os.path.join(pasta_comp, f"{code_muni}_{ano}_2_FalsaCor_NIR-R-G.tif")
    with rasterio.open(path_nir_rg, "w", **meta_rgb) as dst:
        dst.write(nir_norm, 1)
        dst.write(r_norm, 2)
        dst.write(g_norm, 3)
        dst.set_band_description(1, "NIR")
        dst.set_band_description(2, "Red")
        dst.set_band_description(3, "Green")
    print(f" -> Gerado: {path_nir_rg}")

    # 3. Falsa Cor Alternativa (NIR-G-B)
    path_nir_gb = os.path.join(pasta_comp, f"{code_muni}_{ano}_3_FalsaCor_NIR-G-B.tif")
    with rasterio.open(path_nir_gb, "w", **meta_rgb) as dst:
        dst.write(nir_norm, 1)
        dst.write(g_norm, 2)
        dst.write(b_norm, 3)
        dst.set_band_description(1, "NIR")
        dst.set_band_description(2, "Green")
        dst.set_band_description(3, "Blue")
    print(f" -> Gerado: {path_nir_gb}")

    # 4. NDVI em tons de cinza
    path_ndvi = os.path.join(pasta_comp, f"{code_muni}_{ano}_4_NDVI_Cinza.tif")
    red_f = red_data.astype(np.float32)
    nir_f = nir_data.astype(np.float32)
    denominator = (nir_f + red_f)
    ndvi = np.zeros_like(red_f)
    valid_mask = denominator != 0
    ndvi[valid_mask] = (nir_f[valid_mask] - red_f[valid_mask]) / denominator[valid_mask]
    
    ndvi_scaled = np.clip((ndvi + 1) / 2 * 255, 0, 255).astype(np.uint8)
    meta_ndvi = meta.copy()
    meta_ndvi.update(count=1, dtype=rasterio.uint8)
    with rasterio.open(path_ndvi, "w", **meta_ndvi) as dst:
        dst.write(ndvi_scaled, 1)
        dst.set_band_description(1, "NDVI")
    print(f" -> Gerado: {path_ndvi}")

def main():
    if len(sys.argv) < 4:
        print("Erro: Parâmetros insuficientes.")
        print("Uso correto: python 01-Downloads-MapBiomas-CBERS.py <code_muni> <ano_inicio> <ano_fim>")
        print("Exemplo: python 01-Downloads-MapBiomas-CBERS.py 3549904 2023 2024")
        sys.exit(1)

    code_muni = int(sys.argv[1])
    ano_inicio = int(sys.argv[2])
    ano_fim = int(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        raise ValueError("A variável PROJECT_ROOT não foi encontrada no arquivo .env.")

    diretorio_reports = os.path.join(project_root, "reports")
    os.makedirs(diretorio_reports, exist_ok=True)

    print(f"Buscando informações municipais para o código: {code_muni}...")
    gdf_muni = geobr.read_municipality(code_muni=code_muni, year=2022)
    nome_cidade = gdf_muni['name_muni'].values[0]
    uf = gdf_muni['abbrev_state'].values[0]

    print("=" * 60)
    print(f" MUNICÍPIO: {nome_cidade} - {uf} (IBGE: {code_muni})")
    print(f" PERÍODO MAPBIOMAS: {ano_inicio} e {ano_fim} | CBERS: {ano_fim}")
    print("=" * 60)

    inicializar_ee()

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

    # 1. Download MapBiomas Ano Início
    caminho_mb_inicio = baixar_mapbiomas(code_muni, ano_inicio, limite_geopolitico, nome_cidade, uf, project_root)

    # 2. Download MapBiomas Ano Fim
    caminho_mb_fim = baixar_mapbiomas(code_muni, ano_fim, limite_geopolitico, nome_cidade, uf, project_root)

    dados_json = {
        "code_muni": code_muni,
        "nome_cidade": nome_cidade,
        "uf": uf,
        "ano_inicio": ano_inicio,
        "ano_fim": ano_fim,
        "coordenadas_extremas": {"norte": norte, "sul": sul, "leste": leste, "oeste": oeste},
        "arquivo_mapbiomas_inicio": caminho_mb_inicio,
        "arquivo_mapbiomas_fim": caminho_mb_fim
    }
    
    caminho_json = os.path.join(diretorio_reports, f"{code_muni}_{ano_inicio}_vs_{ano_fim}.json")
    with open(caminho_json, 'w', encoding='utf-8') as f_json:
        json.dump(dados_json, f_json, indent=4, ensure_ascii=False)
    print(f"\n[RELATÓRIO] Relatório consolidado gerado em:\n-> {caminho_json}\n")

    # 3. Download, Processamento CBERS e Geração das Composições Zooniverse
    caminho_cbers = baixar_e_processar_cbers(code_muni, ano_fim, dados_json, project_root)

    print(f"\n[SUCESSO] Processo unificado de downloads e composições finalizado com êxito!")

if __name__ == "__main__":
    main()