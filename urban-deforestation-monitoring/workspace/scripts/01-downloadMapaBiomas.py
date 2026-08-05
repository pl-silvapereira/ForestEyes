import sys
import ee
import geobr
import json
import os
import unicodedata
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

# Diretório de destino local (caso queira manipular localmente no futuro)
diretorio_destino = os.path.join(project_root, "data", "input", "MapBiomas")
os.makedirs(diretorio_destino, exist_ok=True)

try:
    # 3. Autenticar e inicializar a API do Earth Engine
    ee.Authenticate(force=True)
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
    bounds = limite_geopolitico.bounds().getInfo()['coordinates'][0]
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

    # 7. Configurar a exportação via Tarefa Assíncrona para o Google Drive direcionando para a pasta específica
    def limpar_para_ee(texto):
        nfkd = unicodedata.normalize('NFKD', texto)
        return "".join([c for c in nfkd if not unicodedata.combining(c)]).replace(" ", "_")

    cidade_limpa = limpar_para_ee(nome_cidade)
    nome_arquivo = f'mapbiomas_lulc_10m_{cidade_limpa.lower()}_{ANO}'
    
    # Caminho exato dentro do Google Drive
    pasta_drive = 'Mestrado/04-Projeto ForestEyes/ForestEyes/urban-deforestation-monitoring/workspace/data/input/MapBiomas'
    
    print(f"Enviando tarefa de exportação para o Google Drive na pasta:\n-> {pasta_drive}")
    
    tarefa = ee.batch.Export.image.toDrive(
        image=imagem_recortada,
        description=f'Export_{nome_arquivo}',
        folder=pasta_drive,                  # Caminho completo estruturado no Drive
        fileNamePrefix=nome_arquivo,
        region=limite_geopolitico.bounds(),
        scale=10,                            # Resolução nativa de 10 metros mantida
        maxPixels=1e9,
        fileFormat='GeoTIFF',
        formatOptions={
            'noData': 0                      # Preserva a transparência nas bordas
        }
    )

    tarefa.start()
    
    print("\n[SUCESSO] Tarefa de exportação iniciada na nuvem do Google!")
    print(f"O arquivo '{nome_arquivo}.tif' será gerado automaticamente dentro da pasta especificada no seu Google Drive.")
    print("Acompanhe o andamento no painel Tasks do Earth Engine Code Editor.")

except Exception as e:
    print(f"\n[ERRO CRÍTICO] Ocorreu um erro durante o processamento: {e}")
    sys.exit(1)