import os
import requests
from pathlib import Path

def download_imagens_cbers():
    # Caminhos configurados conforme seu ambiente
    caminho_catalogo = Path(r"C:\Users\PedroLuizdaSilvaPere\development\Projects\ForestEyes\Urban-SJC\CBERS-4A\inpe_catalog_2026_3_19_0_20_26.txt")
    diretorio_destino = Path(r"C:\Users\PedroLuizdaSilvaPere\development\Projects\ForestEyes\Urban-SJC\CBERS-4A\ObtencaoINPE")

    # Cria a pasta de destino se não existir
    diretorio_destino.mkdir(parents=True, exist_ok=True)

    if not caminho_catalogo.exists():
        print(f"Erro: Arquivo de catálogo não encontrado em: {caminho_catalogo}")
        return

    # Lê as URLs do arquivo de texto
    with open(caminho_catalogo, 'r') as f:
        urls = [linha.strip() for linha in f.readlines() if linha.strip().startswith('http')]

    print(f"Encontrados {len(urls)} links para download.")

    for i, url in enumerate(urls, 1):
        # Extrai o nome do arquivo da URL (antes dos parâmetros de email)
        nome_arquivo = url.split('/')[-1].split('?')[0]
        caminho_final = diretorio_destino / nome_arquivo

        if caminho_final.exists():
            print(f"[{i}/{len(urls)}] Pulando: {nome_arquivo} já existe.")
            continue

        print(f"[{i}/{len(urls)}] Baixando: {nome_arquivo}...")
        
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(caminho_final, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Concluído: {nome_arquivo}")
        except Exception as e:
            print(f"Falha ao baixar {nome_arquivo}: {e}")

    print("\n--- Processo de download finalizado ---")

if __name__ == "__main__":
    download_imagens_cbers()