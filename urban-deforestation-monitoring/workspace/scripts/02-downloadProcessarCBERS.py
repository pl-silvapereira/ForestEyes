import sys
import os
import json
from datetime import date
import requests
import time
from dotenv import load_dotenv
from cbers4asat import Cbers4aAPI
from cbers4asat.tools import rgbn_composite
# import rasterio as rio
# from rasterio.plot import show

def baixar_e_processar_cbers():
    # 1. Capturar argumentos da linha de comando (.cmd)
    # Uso esperado: python 02-downloadProcessarCBERS.py <code_muni>
    if len(sys.argv) < 2:
        print("Erro: Parâmetros insuficientes.")
        print("Uso correto: python 02-downloadProcessarCBERS.py <code_muni>")
        sys.exit(1)

    CODE_MUNI = int(sys.argv[1])

    # 2. Carregar variáveis de ambiente do arquivo .env
    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")

    if not project_root:
        raise ValueError("A variável PROJECT_ROOT não foi encontrada no arquivo .env.")

    # 3. Ler o arquivo JSON gerado pelo script 01 na pasta reports
    diretorio_reports = os.path.join(project_root, "reports")
    caminho_json = os.path.join(diretorio_reports, f"{CODE_MUNI}.json")
    
    if not os.path.exists(caminho_json):
        print(f"[ERRO CRÍTICO] O relatório JSON para o município {CODE_MUNI} não foi encontrado em:\n-> {caminho_json}")
        print("Execute primeiro o script 01 (01-downloadMapaBiomas.py) para gerar as coordenadas.")
        sys.exit(1)

    print(f"Lendo metadados e coordenadas do município do arquivo:\n-> {caminho_json}")
    with open(caminho_json, 'r', encoding='utf-8') as f_json:
        dados_json = json.load(f_json)

    coords = dados_json['coordenadas_extremas']
    ano_referencia = dados_json['ano']
    nome_cidade = dados_json['nome_cidade']

    # Monta a bbox dinamicamente com base no JSON: [oeste, sul, leste, norte]
    bbox_muni = [
        coords['oeste'],
        coords['sul'],
        coords['leste'],
        coords['norte']
    ]

    print(f"Município carregado: {nome_cidade} (Ano: {ano_referencia})")
    print(f"BBox utilizada: {bbox_muni}")

    # 4. Configurar diretórios baseados no PROJECT_ROOT
    pasta_entrada_cbers = os.path.join(project_root, "data", "input", "CBERS-4A-WPM")
    pasta_saida_pansharpening = os.path.join(project_root, "data", "output", "pansharpening")
    
    os.makedirs(pasta_entrada_cbers, exist_ok=True)
    os.makedirs(pasta_saida_pansharpening, exist_ok=True)
    os.makedirs(diretorio_reports, exist_ok=True)

    email_inpe = 'pereira.pedro@unifesp.br'
    print("Conectando ao catálogo LGI-CDSR do INPE...")
    api = Cbers4aAPI(email_inpe)

    print("Realizando a busca espacial e temporal...")
    resultados = api.query(
        location=bbox_muni,
        initial_date=date(ano_referencia, 1, 1),
        end_date=date(ano_referencia, 12, 31),
        cloud=15, 
        limit=100,
        collections=['CBERS4A_WPM_L4_DN']
    )

    cenas_encontradas = resultados.get('features', [])
    if not cenas_encontradas:
        print("Nenhuma cena encontrada para o período e região.")
        return

    # Seleciona a cena (ajuste o índice se necessário, mantendo o padrão original)
    cena_selecionada = cenas_encontradas[1] if len(cenas_encontradas) > 1 else cenas_encontradas[0]
    id_cena = cena_selecionada.get('id', 'CBERS_4A_WPM')
    print(f"\nSucesso! Cena selecionada: {id_cena}")
    
    ativos = cena_selecionada.get('assets', {})
    if 'thumbnail' not in ativos:
        print("Erro crítico: A API do INPE não retornou o thumbnail.")
        return
        
    url_thumbnail = ativos['thumbnail']['href']
    
    # -------------------------------------------------------------
    # Bypass: Engenharia reversa da URL da miniatura
    # -------------------------------------------------------------
    url_base_tiff = url_thumbnail.replace("/datastore/BROWSE/", "/api/download/TIFF/")
    url_base_tiff = url_base_tiff.replace("/BROWSE/", "/api/download/TIFF/")
    url_base_tiff = url_base_tiff.replace(".jpg", "").replace(".png", "")
    
    if not url_base_tiff.endswith("_L4"):
        url_base_tiff += "_L4"
    # -------------------------------------------------------------
    
    bandas = ['BAND0', 'BAND1', 'BAND2', 'BAND3', 'BAND4']
    arquivos_baixados = {}
    
    print("\nIniciando download das bandas com sistema de retentativas (Retry)...")
    
    for banda in bandas:
        url_download = f"{url_base_tiff}_{banda}.tif?email={email_inpe}"
        caminho_arquivo = os.path.join(pasta_entrada_cbers, f"{id_cena}_{banda}.tif")
        arquivos_baixados[banda] = caminho_arquivo

        if os.path.exists(caminho_arquivo) and os.path.getsize(caminho_arquivo) > 10000:
            print(f" -> ✓ {banda} já existe no disco. Pulando download.")
            continue

        print(f"Baixando {banda}...")
        sucesso = False
        
        for tentativa in range(1, 4):
            try:
                resposta = requests.get(url_download, stream=True, timeout=120)
                resposta.raise_for_status()
                
                with open(caminho_arquivo, 'wb') as f:
                    for chunk in resposta.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                print(f" -> ✓ Concluído (Tentativa {tentativa})")
                sucesso = True
                break
                
            except requests.exceptions.RequestException as e:
                print(f" -> ✗ Falha na tentativa {tentativa} para {banda}: {e}")
                if tentativa == 3:
                    print(f"    [DEBUG] A URL tentada foi: {url_download}")
                time.sleep(5)
        
        if not sucesso:
            print(f"Erro crítico: Não foi possível baixar a {banda} após 3 tentativas.")
            sys.exit(1)

    print("\nTodos os downloads concluídos! Iniciando composição RGBN (True Color)...")

    nome_arquivo_stack = f"{id_cena}_TRUE_COLOR.tif"
    
    # Executa a composição salvando diretamente na pasta de saída de pansharpening
    rgbn_composite(
        red=arquivos_baixados['BAND3'],   
        green=arquivos_baixados['BAND2'], 
        blue=arquivos_baixados['BAND1'],  
        nir=arquivos_baixados['BAND4'],   
        filename=nome_arquivo_stack,
        outdir=pasta_saida_pansharpening
    )
    
    caminho_stack = os.path.join(pasta_saida_pansharpening, nome_arquivo_stack)
    print(f"\n[SUCESSO] Composição salva com sucesso em:\n-> {caminho_stack}")

    # 5. Salvar o JSON de registros com os resultados desse script
    dados_resultado = {
        "code_muni": CODE_MUNI,
        "nome_cidade": nome_cidade,
        "ano_referencia": ano_referencia,
        "id_cena_selecionada": id_cena,
        "arquivos_bandas": arquivos_baixados,
        "composicao_true_color": caminho_stack
    }
    
    caminho_json_resultado = os.path.join(diretorio_reports, f"{CODE_MUNI}_cbers_results.json")
    with open(caminho_json_resultado, 'w', encoding='utf-8') as f_json_res:
        json.dump(dados_resultado, f_json_res, indent=4, ensure_ascii=False)
        
    print(f"[RELATÓRIO] Arquivo com o registro dos processamentos do CBERS gerado com sucesso em:\n-> {caminho_json_resultado}\n")

if __name__ == "__main__":
    baixar_e_processar_cbers()