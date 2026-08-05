import sys
import os
import json
import itertools
import numpy as np
import rasterio
from PIL import Image
from dotenv import load_dotenv

def aplicar_stretch_contraste(banda_matriz):
    """
    Aplica um stretch linear de contraste (2% - 98%) para visualização.
    Converte a matriz original (geralmente uint16) para uint8.
    """
    banda_mascarada = np.ma.masked_equal(banda_matriz, 0)
    p2, p98 = np.percentile(banda_mascarada.compressed(), (2, 98))
    banda_normalizada = np.clip((banda_matriz - p2) / (p98 - p2), 0, 1)
    return np.uint8(banda_normalizada * 255)

def gerar_composicoes(caminho_arquivo, diretorio_tif, diretorio_png, code_muni):
    """
    Lê o arquivo multiespectral único, gera permutações e exporta TIFs e PNGs.
    Retorna um dicionário com os metadados dos arquivos gerados.
    """
    os.makedirs(diretorio_tif, exist_ok=True)
    os.makedirs(diretorio_png, exist_ok=True)
    
    bandas_processadas = {}
    arquivos_gerados = {"tif": {}, "png": {}}
    
    # Nomes das bandas conforme a entrada
    nomes_bandas = ['B1_Azul', 'B2_Verde', 'B3_Vermelho', 'B4_NIR']
    
    print(f"Lendo arquivo base recortado:\n-> {caminho_arquivo}")
    
    with rasterio.open(caminho_arquivo) as src:
        # Prepara os metadados para salvar os arquivos TIF (3 bandas, uint8)
        metadados_tif = src.meta.copy()
        metadados_tif.update({
            'count': 3, 
            'dtype': 'uint8', 
            'driver': 'GTiff',
            'nodata': 0
        })
        
        # Lê cada uma das 4 bandas do arquivo único
        for i, nome in enumerate(nomes_bandas, start=1):
            matriz = src.read(i)
            bandas_processadas[nome] = aplicar_stretch_contraste(matriz)
            
    # Geração das permutações (24 combinações)
    todas_combinacoes = list(itertools.permutations(nomes_bandas, 3))
    print(f"\nGerando {len(todas_combinacoes)} composições em TIF e PNG...")
    
    for idx, composicao in enumerate(todas_combinacoes, start=1):
        r, g, b = composicao
        nome_base = f"{code_muni}_{r}_{g}_{b}"
        
        matriz_r = bandas_processadas[r]
        matriz_g = bandas_processadas[g]
        matriz_b = bandas_processadas[b]
        
        # ---------------------------------------------------------
        # 1. EXPORTAÇÃO EM PNG LEVE
        # ---------------------------------------------------------
        imagem_rgb = np.dstack((matriz_r, matriz_g, matriz_b))
        img = Image.fromarray(imagem_rgb)
        
        # Compressão/Redimensionamento proporcional (Max: 1920px)
        largura_maxima = 1920
        if img.width > largura_maxima:
            proporcao = largura_maxima / img.width
            nova_altura = int(img.height * proporcao)
            img = img.resize((largura_maxima, nova_altura), Image.Resampling.LANCZOS)
        
        nome_png = f"{nome_base}.png"
        caminho_png = os.path.join(diretorio_png, nome_png)
        img.save(caminho_png, optimize=True)
        arquivos_gerados["png"][nome_base] = caminho_png
        
        # ---------------------------------------------------------
        # 2. EXPORTAÇÃO EM TIF
        # ---------------------------------------------------------
        nome_tif = f"{nome_base}.tif"
        caminho_tif = os.path.join(diretorio_tif, nome_tif)
        
        with rasterio.open(caminho_tif, 'w', **metadados_tif) as dest:
            dest.write(matriz_r, 1) 
            dest.write(matriz_g, 2) 
            dest.write(matriz_b, 3)
        arquivos_gerados["tif"][nome_base] = caminho_tif
        
        print(f"[{idx}/24] Salvo: {r}/{g}/{b}")

    return arquivos_gerados

# ==========================================
# Automação de Leitura e Execução
# ==========================================
if __name__ == "__main__":
    # 1. Capturar argumento
    if len(sys.argv) < 2:
        print("Erro: Parâmetros insuficientes.")
        print("Uso correto: python 04-gerarMultibandas.py <code_muni>")
        sys.exit(1)

    CODE_MUNI = int(sys.argv[1])

    # 2. Carregar ambiente
    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        raise ValueError("A variável PROJECT_ROOT não foi encontrada no arquivo .env.")

    diretorio_reports = os.path.join(project_root, "reports")
    json_entrada = os.path.join(diretorio_reports, f"{CODE_MUNI}_clipped_results.json")

    # 3. Validar existência do relatório do script 03
    if not os.path.exists(json_entrada):
        print(f"[ERRO CRÍTICO] Arquivo JSON do recorte não encontrado:\n-> {json_entrada}")
        print("Certifique-se de executar o Script 03 primeiro.")
        sys.exit(1)

    # 4. Ler JSON para pegar o arquivo recortado
    with open(json_entrada, 'r', encoding='utf-8') as f:
        dados_json = json.load(f)
        
    arquivo_recortado = dados_json.get("arquivo_cbers_recortado")
    
    if not arquivo_recortado or not os.path.exists(arquivo_recortado):
        print("[ERRO] O caminho do arquivo recortado não foi encontrado ou não existe no disco.")
        sys.exit(1)

    # 5. Definir caminhos de saída
    pasta_base_saida = os.path.join(project_root, "data", "output", "pansharpening", "multispectral-RGBN-bands")
    pasta_tif = os.path.join(pasta_base_saida, "tif")
    pasta_png = os.path.join(pasta_base_saida, "png")

    print(f"==================================================")
    print(f" GERANDO 24 MULTIBANDAS: MUNICÍPIO {CODE_MUNI}")
    print(f"==================================================")

    # 6. Executar processamento
    arquivos_gerados = gerar_composicoes(arquivo_recortado, pasta_tif, pasta_png, CODE_MUNI)

    # 7. Salvar relatório JSON com os resultados
    dados_finais = {
        "code_muni": CODE_MUNI,
        "arquivo_origem_recortado": arquivo_recortado,
        "total_composicoes": 24,
        "arquivos_gerados": arquivos_gerados
    }
    
    json_final = os.path.join(diretorio_reports, f"{CODE_MUNI}_multibands_results.json")
    with open(json_final, 'w', encoding='utf-8') as f:
        json.dump(dados_finais, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Sucesso! Todas as imagens TIF e PNG foram geradas.")
    print(f"[RELATÓRIO] JSON com mapeamento dos 48 arquivos gerado em:\n-> {json_final}")