import sys
import os
import json
from datetime import date
import requests
import time
from dotenv import load_dotenv

try:
    from cbers4asat import Cbers4aAPI
    from cbers4asat.tools import rgbn_composite
except ImportError:
    print("Aviso: Módulo 'cbers4asat' não encontrado. O script não executará corretamente.")

def baixar_e_processar_cbers():
    load_dotenv()
    
    if len(sys.argv) >= 3:
        CODE_MUNI = int(sys.argv[1])
        ANO = str(sys.argv[2])
    else:
        CODE_MUNI = int(os.environ.get('CODE_MUNI', 0))
        ANO = os.environ.get('ANO', '')

    if not CODE_MUNI or not ANO:
        print("Erro: CODE_MUNI ou ANO não informados. Configure os.environ ou passe como argumento.")
        sys.exit(1)

    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        raise ValueError("A variável PROJECT_ROOT não foi encontrada.")

    diretorio_reports = os.path.join(project_root, "reports")
    caminho_json = os.path.join(diretorio_reports, f"{CODE_MUNI}_{ANO}.json")
    
    if not os.path.exists(caminho_json):
        print(f"[ERRO CRÍTICO] O relatório JSON não foi encontrado:\n-> {caminho_json}")
        sys.exit(1)

    with open(caminho_json, 'r', encoding='utf-8') as f_json:
        dados_json = json.load(f_json)

    coords = dados_json['coordenadas_extremas']
    ano_referencia = int(dados_json['ano'])
    nome_cidade = dados_json['nome_cidade']

    bbox_muni = [coords['oeste'], coords['sul'], coords['leste'], coords['norte']]
    print(f"Município: {nome_cidade} (Ano: {ano_referencia}) | BBox: {bbox_muni}")

    pasta_entrada_cbers = os.path.join(project_root, "data", "input", "CBERS-4A-WPM", ANO)
    pasta_saida_pansharpening = os.path.join(project_root, "data", "output", "pansharpening", ANO)
    
    os.makedirs(pasta_entrada_cbers, exist_ok=True)
    os.makedirs(pasta_saida_pansharpening, exist_ok=True)

    email_inpe = 'pereira.pedro@unifesp.br'
    print("Conectando ao catálogo LGI-CDSR do INPE...")
    api = Cbers4aAPI(email_inpe)

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
        print(f"Nenhuma cena encontrada para o ano {ANO}.")
        return

    cena_selecionada = cenas_encontradas[1] if len(cenas_encontradas) > 1 else cenas_encontradas[0]
    id_cena = cena_selecionada.get('id', 'CBERS_4A_WPM')
    print(f"\nCena selecionada: {id_cena}")
    
    ativos = cena_selecionada.get('assets', {})
    url_thumbnail = ativos['thumbnail']['href']
    
    url_base_tiff = url_thumbnail.replace("/datastore/BROWSE/", "/api/download/TIFF/").replace("/BROWSE/", "/api/download/TIFF/").replace(".jpg", "").replace(".png", "")
    if not url_base_tiff.endswith("_L4"):
        url_base_tiff += "_L4"
    
    bandas = ['BAND0', 'BAND1', 'BAND2', 'BAND3', 'BAND4']
    arquivos_baixados = {}
    print("\nIniciando download das bandas (Retry ativo)...")
    
    for banda in bandas:
        url_download = f"{url_base_tiff}_{banda}.tif?email={email_inpe}"
        caminho_arquivo = os.path.join(pasta_entrada_cbers, f"{id_cena}_{banda}.tif")
        arquivos_baixados[banda] = caminho_arquivo

        if os.path.exists(caminho_arquivo) and os.path.getsize(caminho_arquivo) > 10000:
            print(f" -> ✓ {banda} já existe no disco.")
            continue

        sucesso = False
        for tentativa in range(1, 4):
            try:
                resposta = requests.get(url_download, stream=True, timeout=120)
                resposta.raise_for_status()
                with open(caminho_arquivo, 'wb') as f:
                    for chunk in resposta.iter_content(chunk_size=8192):
                        if chunk: f.write(chunk)
                print(f" -> ✓ {banda} baixada.")
                sucesso = True
                break
            except requests.exceptions.RequestException as e:
                time.sleep(5)
        
        if not sucesso:
            print(f"Erro crítico: Não foi possível baixar a {banda}.")
            sys.exit(1)

    print("\nIniciando composição RGBN (True Color)...")
    nome_arquivo_stack = f"{id_cena}_TRUE_COLOR_{ANO}.tif"
    
    rgbn_composite(
        red=arquivos_baixados['BAND3'], green=arquivos_baixados['BAND2'], 
        blue=arquivos_baixados['BAND1'], nir=arquivos_baixados['BAND4'],   
        filename=nome_arquivo_stack, outdir=pasta_saida_pansharpening
    )
    
    caminho_stack = os.path.join(pasta_saida_pansharpening, nome_arquivo_stack)
    print(f"[SUCESSO] Composição salva em: {caminho_stack}")

    dados_resultado = {
        "code_muni": CODE_MUNI, "nome_cidade": nome_cidade, "ano_referencia": ano_referencia,
        "id_cena_selecionada": id_cena, "arquivos_bandas": arquivos_baixados,
        "composicao_true_color": caminho_stack
    }
    
    caminho_json_resultado = os.path.join(diretorio_reports, f"{CODE_MUNI}_{ANO}_cbers_results.json")
    with open(caminho_json_resultado, 'w', encoding='utf-8') as f_json_res:
        json.dump(dados_resultado, f_json_res, indent=4, ensure_ascii=False)
        
    print(f"[RELATÓRIO] Arquivo CBERS gerado: {caminho_json_resultado}\n")

if __name__ == "__main__":
    baixar_e_processar_cbers()