import sys
import ee
import geobr
import json
import os
import requests
from dotenv import load_dotenv

# 1. Capturar argumentos da linha de comando (.cmd)
# Uso esperado: python 01-downloadMapaBiomas.py <code_muni> <ano>
if len(sys.argv) < 3:
    print("Erro: Parâmetros insuficientes.")
    print("Uso correto: python 01-downloadMapaBiomas.py <code_muni> <ano>")
    sys.exit(1)

CODE_MUNI = int(sys.argv[1])
ANO = int(sys.argv[2])

# 2. Carregar variáveis de ambiente do arquivo .env
load_dotenv()
project_root = os.getenv("PROJECT_ROOT")

if not project_root:
    raise ValueError("A variável PROJECT_ROOT não foi encontrada no arquivo .env.")

# Diretório de destino local: workspace\data\input\MapBiomas
diretorio_destino = os.path.join(project_root, "data", "input", "MapBiomas")
os.makedirs(diretorio_destino, exist_ok=True)

try:
    # 3. Autenticar e inicializar a API do Earth Engine
    ee.Initialize(project='foresteyes-regioes-urbanas')
    print("Earth Engine inicializado com sucesso.")

    # 4. Carregar o limite geopolítico oficial usando o geobr e validar o nome da cidade
    print(f"Buscando informações para o código de município: {CODE_MUNI}...")
    gdf_muni = geobr.read_municipality(code_muni=CODE_MUNI, year=2022)
    
    # Validação e exibição do nome da cidade
    nome_cidade = gdf_muni['name_muni'].values[0]
    uf = gdf_muni['abbrev_state'].values[0]
    print(f"==================================================")
    print(f" CIDADE VALIDADA: {nome_cidade} - {uf} (IBGE: {CODE_MUNI})")
    print(f"==================================================")

    gdf_muni = gdf_muni.to_crs(epsg=4326)
    
    # Extração da geometria para o Earth Engine
    geojson_dict = json.loads(gdf_muni.to_json())
    coordenadas = geojson_dict['features'][0]['geometry']['coordinates']
    tipo_geometria = geojson_dict['features'][0]['geometry']['type']
    
    if tipo_geometria == 'MultiPolygon':
        limite_geopolitico = ee.Geometry.MultiPolygon(coordenadas)
    else:
        limite_geopolitico = ee.Geometry.Polygon(coordenadas)

    print("Limites oficiais carregados com sucesso no Earth Engine!")

    # 5. Calcular e exibir os 4 pontos (Norte, Sul, Leste e Oeste) para o INPE
    # Obtém os limites retangulares (bounds) da geometria do município
    bounds = limite_geopolitico.bounds().getInfo()['coordinates'][0]
    # bounds retorna uma lista de 4 pontos [[lon_min, lat_min], [lon_max, lat_min], [lon_max, lat_max], [lon_min, lat_max]]
    lons = [p[0] for p in bounds]
    lats = [p[1] for p in bounds]
    
    oeste = min(lons)
    leste = max(lons)
    sul = min(lats)
    norte = max(lats)

    print("\n--- COORDENADAS DOS 4 PONTOS (EXTREMAS) PARA O INPE ---")
    print(f"Norte (Latitude Máxima):  {norte}")
    print(f"Sul (Latitude Mínima):    {sul}")
    print(f"Leste (Longitude Máxima): {leste}")
    print(f"Oeste (Longitude Mínima): {oeste}")
    print("------------------------------------------------------\n")

    # 6. Carregar o Asset público do MapBiomas de 10 metros
    BANDA_ANO = f'classification_{ANO}'
    asset_mapbiomas_10m = 'projects/mapbiomas-public/assets/brazil/lulc_10m/collection2/mapbiomas_10m_collection2_integration_v1'
    mapbiomas_10m = ee.Image(asset_mapbiomas_10m).select(BANDA_ANO)

    print("Recortando e mascarando o fundo externo para NoData...")
    imagem_recortada = mapbiomas_10m.clip(limite_geopolitico).unmask(0).short()

    # 7. Iniciar o download direto para o diretório local
    nome_arquivo = f'mapbiomas_lulc_10m_{nome_cidade.lower().replace(" ", "_")}_{ANO}.tif'
    caminho_completo = os.path.join(diretorio_destino, nome_arquivo)
    
    print("Gerando URL de download no Earth Engine (isso pode levar alguns minutos)...")
    url_download = imagem_recortada.getDownloadURL({
        'region': limite_geopolitico,
        'scale': 10,
        'format': 'GEO_TIFF'
    })
    
    print(f"Baixando o arquivo para: {caminho_completo} ...")
    resposta = requests.get(url_download)
    
    if resposta.status_code == 200:
        with open(caminho_completo, 'wb') as f:
            f.write(resposta.content)
        print("\n[SUCESSO] Download concluído com fundo transparente!")
        print(f"Arquivo salvo em: {caminho_completo}")
    else:
        print(f"\n[ERRO] Falha no download. Código HTTP: {resposta.status_code}")

except Exception as e:
    print(f"\n[ERRO CRÍTICO] Ocorreu um erro durante o processamento: {e}")
    sys.exit(1)