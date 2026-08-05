import os
import sys
import json
import itertools
import numpy as np
import rasterio
from PIL import Image
from dotenv import load_dotenv

def aplicar_stretch_contraste(banda_matriz):
    """
    Aplica um stretch linear de contraste (2% - 98%) para normalização e visualização.
    """
    banda_mascarada = np.ma.masked_equal(banda_matriz, 0)
    
    # Se a matriz for totalmente vazia (apenas NoData), retorna zeros
    if banda_mascarada.count() == 0:
        return np.zeros_like(banda_matriz, dtype=np.uint8)
        
    p2, p98 = np.percentile(banda_mascarada.compressed(), (2, 98))
    
    # Evita divisão por zero caso a banda seja homogênea
    if p98 == p2:
        return np.uint8(np.clip(banda_matriz, 0, 1) * 255)
        
    banda_normalizada = np.clip((banda_matriz - p2) / (p98 - p2), 0, 1)
    return np.uint8(banda_normalizada * 255)

def gerar_multibandas(caminho_arquivo, diretorio_saida_tif, diretorio_saida_png):
    """
    Lê o arquivo multiespectral recortado, gera as 24 permutações
    e exporta como PNGs leves e GeoTIFFs georreferenciados.
    """
    os.makedirs(diretorio_saida_tif, exist_ok=True)
    os.makedirs(diretorio_saida_png, exist_ok=True)
    
    bandas_processadas = {}
    nomes_bandas = ['B1_Azul', 'B2_Verde', 'B3_Vermelho', 'B4_NIR']
    
    print(f"Lendo arquivo base recortado:\n-> {caminho_arquivo}")
    
    arquivos_gerados = {"tif": [], "png": []}
    
    with rasterio.open(caminho_arquivo) as src:
        # 1. Copia os metadados espaciais (CRS, transform) para manter como GeoTIFF autêntico
        metadados = src.meta.copy()
        metadados.update({
            'count': 3,          # A saída terá 3 bandas (RGB)
            'dtype': 'uint8',    # Tipo de dado após o stretch
            'driver': 'GTiff',
            'nodata': 0
        })
        
        # 2. Lê e processa as 4 bandas da imagem fonte
        # Assume-se que a imagem original tem as bandas na ordem: 1, 2, 3, 4
        for i, nome in enumerate(nomes_bandas, start=1):
            matriz = src.read(i)
            bandas_processadas[nome] = aplicar_stretch_contraste(matriz)
            
    # 3. Geração das 24 permutações possíveis
    todas_combinacoes = list(itertools.permutations(nomes_bandas, 3))
    print(f"\nGerando {len(todas_combinacoes)} composições (GeoTIFF e PNG)...")
    
    for composicao in todas_combinacoes:
        r, g, b = composicao
        
        matriz_r = bandas_processadas[r]
        matriz_g = bandas_processadas[g]
        matriz_b = bandas_processadas[b]
        
        # -----------------------------------------------------
        # EXPORTAÇÃO EM PNG LEVE
        # -----------------------------------------------------
        imagem_rgb = np.dstack((matriz_r, matriz_g, matriz_b))
        img = Image.fromarray(imagem_rgb)
        
        largura_maxima = 1920
        if img.width > largura_maxima:
            proporcao = largura_maxima / img.width
            nova_altura = int(img.height * proporcao)
            img = img.resize((largura_maxima, nova_altura), Image.Resampling.LANCZOS)
        
        nome_png = f"preview_{r}_{g}_{b}.png"
        caminho_png = os.path.join(diretorio_saida_png, nome_png)
        img.save(caminho_png, optimize=True)
        arquivos_gerados["png"].append(caminho_png)
        
        # -----------------------------------------------------
        # EXPORTAÇÃO EM GEOTIFF
        # -----------------------------------------------------
        nome_tif = f"composicao_{r}_{g}_{b}.tif"
        caminho_tif = os.path.join(diretorio_saida_tif, nome_tif)
        
        with rasterio.open(caminho_tif, 'w', **metadados) as dest:
            dest.write(matriz_r, 1)
            dest.write(matriz_g, 2)
            dest.write(matriz_b, 3)
            
        arquivos_gerados["tif"].append(caminho_tif)
        
        print(f" -> ✓ {r} | {g} | {b} processada.")

    print(f"\n✅ {len(todas_combinacoes)} composições criadas com sucesso!")
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
    json_cbers_recortado = os.path.join(diretorio_reports, f"{CODE_MUNI}_clipped_results.json")

    # 3. Validar a existência do relatório da etapa anterior
    if not os.path.exists(json_cbers_recortado):
        print(f"[ERRO CRÍTICO] Arquivo JSON não encontrado: {json_cbers_recortado}")
        print("Certifique-se de executar o script 03 antes de gerar as multibandas.")
        sys.exit(1)

    # 4. Ler JSON para encontrar a imagem TIF recortada
    with open(json_cbers_recortado, 'r', encoding='utf-8') as f:
        dados_recorte = json.load(f)
        
    caminho_alvo = dados_recorte.get("arquivo_cbers_recortado")
    if not caminho_alvo or not os.path.exists(caminho_alvo):
        print("[ERRO] Arquivo CBERS recortado não encontrado no JSON ou no disco.")
        sys.exit(1)

    # 5. Definir os diretórios de saída exatos
    pasta_base_multibandas = os.path.join(project_root, "data", "output", "pansharpening", "multispectral-RGBN-bands")
    pasta_saida_tif = os.path.join(pasta_base_multibandas, "tif")
    pasta_saida_png = os.path.join(pasta_base_multibandas, "png")

    print(f"==================================================")
    print(f" INICIANDO GERAÇÃO MULTIESPECTRAL: MUNICÍPIO {CODE_MUNI}")
    print(f"==================================================")

    # 6. Executar o processamento
    arquivos_finais = gerar_multibandas(caminho_alvo, pasta_saida_tif, pasta_saida_png)

    # 7. Atualizar a pasta reports com o resultado deste script
    dados_finais = {
        "code_muni": CODE_MUNI,
        "total_composicoes": 24,
        "pasta_tif": pasta_saida_tif,
        "pasta_png": pasta_saida_png,
        "arquivos_tif_gerados": arquivos_finais["tif"],
        "arquivos_png_gerados": arquivos_finais["png"]
    }
    
    json_final = os.path.join(diretorio_reports, f"{CODE_MUNI}_multibands_results.json")
    
    with open(json_final, 'w', encoding='utf-8') as f:
        json.dump(dados_finais, f, indent=4, ensure_ascii=False)
        
    print(f"\n[RELATÓRIO] Processo concluído! Registro JSON atualizado em:\n-> {json_final}")