import sys
import time
import shutil
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

# Definir e criar diretórios de trabalho locais necessários
diretorio_data_input = os.path.join(project_root, "data", "input", "MapBiomas")
diretorio_reports = os.path.join(project_root, "reports")

os.makedirs(diretorio_data_input, exist_ok=True)
os.makedirs(diretorio_reports, exist_ok=True)

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

    print("\n--- COORDENADAS DOS 4 PONTOS (EXTREMAS) PARAO INPE ---")
    print(f"Norte (Latitude Máxima):  {norte}")
    print(f"Sul (Latitude Mínima):    {sul}")
    print(f"Leste (Longitude Máxima): {leste}")
    print(f"Oeste (Longitude Mínima): {oeste}")
    print("------------------------------------------------------\n")

    # 6. Salvar o arquivo JSON com as coordenadas extremas e caminho do arquivo MapBiomas
    def limpar_para_ee(texto):
        nfkd = unicodedata.normalize('NFKD', texto)
        return "".join([c for c in nfkd if not unicodedata.combining(c)]).replace(" ", "_")

<<<<<<< HEAD
    try:
        #ee.Authenticate(force=True)
        ee.Initialize(project='foresteyes-regioes-urbanas')
        print("Earth Engine inicializado com sucesso.")
=======
    cidade_limpa = limpar_para_ee(nome_cidade)
    nome_arquivo_mapbiomas = f'mapbiomas_lulc_10m_{cidade_limpa.lower()}_{ANO}.tif'
    caminho_mapbiomas_local = os.path.join(diretorio_data_input, nome_arquivo_mapbiomas)
>>>>>>> parent of f902290 (inserido o contexto do ano nos arquivos e tambem adicionado um orchestrator para realizar a chamada do periodo de anos.)

    dados_json = {
        "code_muni": CODE_MUNI,
        "nome_cidade": nome_cidade,
        "uf": uf,
        "ano": ANO,
        "coordenadas_extremas": {
            "norte": norte,
            "sul": sul,
            "leste": leste,
            "oeste": oeste
        },
        "arquivo_mapbiomas": caminho_mapbiomas_local # <--- Nova linha armazenando o caminho
    }
    
    caminho_json = os.path.join(diretorio_reports, f"{CODE_MUNI}.json")
    with open(caminho_json, 'w', encoding='utf-8') as f_json:
        json.dump(dados_json, f_json, indent=4, ensure_ascii=False)
    
    print(f"[RELATÓRIO] Arquivo JSON gerado com sucesso em:\n-> {caminho_json}\n")

    # 7. Carregar o Asset público do MapBiomas de 10 metros
    BANDA_ANO = f'classification_{ANO}'
    asset_mapbiomas_10m = 'projects/mapbiomas-public/assets/brazil/lulc_10m/collection2/mapbiomas_10m_collection2_integration_v1'
    mapbiomas_10m = ee.Image(asset_mapbiomas_10m).select(BANDA_ANO)

    print("Recortando e mascarando o fundo externo para NoData...")
    imagem_recortada = mapbiomas_10m.clip(limite_geopolitico).unmask(0).short()

    # 8. Configurar a exportação via Tarefa Assíncrona utilizando pasta temporária
    def limpar_para_ee(texto):
        nfkd = unicodedata.normalize('NFKD', texto)
        return "".join([c for c in nfkd if not unicodedata.combining(c)]).replace(" ", "_")

    cidade_limpa = limpar_para_ee(nome_cidade)
    nome_arquivo = f'mapbiomas_lulc_10m_{cidade_limpa.lower()}_{ANO}.tif'
    
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
        formatOptions={
            'noData': 0
        }
    )

    tarefa.start()
    print(f"Tarefa ID: {tarefa.id} iniciada. Aguardando o processamento na nuvem...")

    # 9. Loop de monitoramento (Polling) até a conclusão da tarefa
    while True:
        status = ee.data.getTaskStatus(tarefa.id)[0]
        state = status.get('state')
        print(f"Status atual da tarefa: {state}...")
        
        if state == 'COMPLETED':
            print("\n[SUCESSO] Processamento no Earth Engine concluído!")
            break
        elif state in ['FAILED', 'CANCELLED']:
            error_msg = status.get('error_message', 'Erro desconhecido')
            raise Exception(f"A exportação falhou ou foi cancelada. Status: {state}. Erro: {error_msg}")
        
        time.sleep(20)

    # 10. Organização local via caminhos do projeto (`diretorio_data_input`)
    # Como o Google Drive Desktop sincroniza a nuvem, a pasta temporária aparecerá na raiz do Drive local.
    # Assumindo que a raiz do Drive está na mesma estrutura ou subindo níveis a partir de project_root:
    # project_root = ...\workspace -> Subindo 4 níveis chegamos na raiz do Drive (onde ficam as pastas do projeto e a pasta temp)
    raiz_drive = os.path.abspath(os.path.join(project_root, "../../../../.."))
    caminho_temp_local = os.path.join(raiz_drive, PASTA_TEMPORARIA)

    print("Aguardando a sincronização local do arquivo pelo Google Drive...")
    arquivo_origem = os.path.join(caminho_temp_local, nome_arquivo)
    
    # Loop de espera caso o Google Drive Desktop demore alguns segundos para sincronizar o .tif
    tentativas = 0
    while not os.path.exists(arquivo_origem) and tentativas < 15:
        time.sleep(5)
        tentativas += 1

    if os.path.exists(arquivo_origem):
        arquivo_destino = os.path.join(diretorio_data_input, nome_arquivo)
        
        # Move o arquivo para a pasta correta (diretorio_data_input)
        shutil.move(arquivo_origem, arquivo_destino)
        print(f"[SUCESSO] Arquivo movido para:\n-> {arquivo_destino}")
        
        # Remove a pasta temporária local vazia
        try:
            os.rmdir(caminho_temp_local)
            print("[LIMPEZA] Pasta temporária local 'MapBiomas_Temp' removida.")
        except OSError:
            pass
    else:
        print(f"[AVISO] O arquivo gerado não foi encontrado localmente em '{caminho_temp_local}'. Verifique se o Google Drive Desktop concluiu a sincronização.")

    print("\n[PROCESSO FINALIZADO COM SUCESSO]")

except Exception as e:
    print(f"\n[ERRO CRÍTICO] Ocorreu um erro durante o processamento: {e}")
    sys.exit(1)