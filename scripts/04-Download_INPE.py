import os
import glob
import requests
from tqdm import tqdm
import re
from dotenv import load_dotenv

# --- CONFIGURAÇÃO VIA .ENV ---
load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')

input_dir = os.path.join(ROOT, 'data', 'CBERS4A-WPM')
output_dir = os.path.join(ROOT, 'data', 'CBERS4A-WPM', 'Downloads')

# Localiza o arquivo TXT automaticamente
arquivos_txt = glob.glob(os.path.join(input_dir, "inpe_catalog_*.txt"))

if not arquivos_txt:
    print(f"❌ Nenhum catálogo encontrado em: {input_dir}")
    exit()

lista_links = arquivos_txt[0]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def limpar_nome_arquivo(url):
    nome_sujo = url.split('/')[-1].strip()
    nome_limpo = re.split(r'\.tif|\.xml', nome_sujo, flags=re.IGNORECASE)[0]
    extensao = ".tif" if ".tif" in nome_sujo.lower() else ".xml"
    return nome_limpo + extensao

def baixar_arquivo(url):
    nome_arquivo = limpar_nome_arquivo(url)
    caminho_destino = os.path.join(output_dir, nome_arquivo)
    
    if os.path.exists(caminho_destino):
        return

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(caminho_destino, 'wb') as f, tqdm(
            desc=nome_arquivo, total=total_size, unit='iB', unit_scale=True
        ) as bar:
            for data in response.iter_content(chunk_size=8192):
                bar.update(f.write(data))
    except Exception as e:
        print(f"❌ Erro em {nome_arquivo}: {e}")

if __name__ == "__main__":
    with open(lista_links, 'r') as f:
        links = [l.strip() for l in f.readlines() if l.startswith('http')]
    
    print(f"🚀 Baixando {len(links)} arquivos para {output_dir}...")
    for link in links:
        baixar_arquivo(link)