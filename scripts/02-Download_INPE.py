import os
import glob
import requests
from tqdm import tqdm
import re

# --- CONFIGURAÇÃO DE CAMINHOS ---
input_dir = '../data/CBERS4A-WPM/'
output_dir = '../data/CBERS4A-WPM/Downloads/'

# Localiza o arquivo TXT automaticamente
arquivos_txt = glob.glob(os.path.join(input_dir, "inpe_catalog_*.txt"))

if not arquivos_txt:
    print(f"❌ Nenhum arquivo de catálogo encontrado em: {input_dir}")
    exit()

lista_links = arquivos_txt[0]
print(f"📄 Lendo links de: {os.path.basename(lista_links)}")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def limpar_nome_arquivo(url):
    # Pega a última parte da URL
    nome_sujo = url.split('/')[-1].strip()
    # Remove tudo que vem depois de .tif ou .xml (incluindo o ?)
    nome_limpo = re.split(r'\.tif|\.xml', nome_sujo, flags=re.IGNORECASE)[0]
    # Readiciona a extensão correta
    extensao = ".tif" if ".tif" in nome_sujo.lower() else ".xml"
    return nome_limpo + extensao

def baixar_arquivo(url):
    nome_arquivo = limpar_nome_arquivo(url)
    caminho_destino = os.path.join(output_dir, nome_arquivo)
    
    if os.path.exists(caminho_destino):
        print(f"⏩ Arquivo já existe, pulando: {nome_arquivo}")
        return

    try:
        # Mantemos a URL original (com e-mail) para a requisição, 
        # mas salvamos com o nome limpo
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(caminho_destino, 'wb') as f, tqdm(
            desc=nome_arquivo,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=8192):
                size = f.write(data)
                bar.update(size)
                
    except Exception as e:
        print(f"❌ Erro ao baixar {nome_arquivo}: {e}")

def iniciar_processo():
    with open(lista_links, 'r') as f:
        links = [linha.strip() for linha in f.readlines() if linha.startswith('http')]
    
    total = len(links)
    print(f"🚀 Iniciando download de {total} arquivos...")
    
    for i, link in enumerate(links, 1):
        print(f"\n[{i}/{total}]")
        baixar_arquivo(link)

    print("\n✅ Processo finalizado!")

if __name__ == "__main__":
    iniciar_processo()