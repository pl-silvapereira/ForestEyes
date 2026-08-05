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
    Aplica um stretch linear de contraste (2% - 98%) EXCLUSIVAMENTE para a geração do PNG.
    """
    banda_mascarada = np.ma.masked_equal(banda_matriz, 0)
    
    if banda_mascarada.count() == 0:
        return np.zeros_like(banda_matriz, dtype=np.uint8)
        
    p2, p98 = np.percentile(banda_mascarada.compressed(), (2, 98))
    
    if p98 == p2:
        return np.uint8(np.clip(banda_matriz, 0, 1) * 255)
        
    banda_normalizada = np.clip((banda_matriz - p2) / (p98 - p2), 0, 1)
    return np.uint8(banda_normalizada * 255)

def gerar_multibandas(caminho_arquivo, diretorio_saida_tif, diretorio_saida_png):
    os.makedirs(diretorio_saida_tif, exist_ok=True)
    os.makedirs(diretorio_saida_png, exist_ok=True)
    
    bandas_brutas = {}
    bandas_png = {}
    nomes_bandas = ['B1_Azul', 'B2_Verde', 'B3_Vermelho', 'B4_NIR']
    
    print(f"Lendo arquivo base recortado:\n-> {caminho_arquivo}")
    
    arquivos_gerados = {"tif": [], "png": []}
    
    with rasterio.open(caminho_arquivo) as src:
        # Metadados para o GeoTIFF original. 
        # ATENÇÃO: NÃO forçamos 'uint8'. Mantemos o dtype original (ex: uint16) para preservar a radiometria.
        metadados = src.meta.copy()
        metadados.update({
            'count': 3,
            'nodata': 0
        })
        
        # Lê as bandas e separa a matriz bruta (para o TIF) da matriz tratada (para o PNG)
        for i, nome in enumerate(nomes_bandas, start=1):
            matriz = src.read(i)
            bandas_brutas[nome] = matriz
            bandas_png[nome] = aplicar_stretch_contraste(matriz)
            
    todas_combinacoes = list(itertools.permutations(nomes_bandas, 3))
    print(f"\nGerando {len(todas_combinacoes)} composições...")
    print("-> TIFs salvarão os dados brutos de reflectância (Visualização via QGIS).")
    print("-> PNGs salvarão versões tratadas para preview rápido (Visualização via Drive).\n")
    
    for composicao in todas_combinacoes:
        r, g, b = composicao
        
        # -----------------------------------------------------
        # 1. EXPORTAÇÃO EM GEOTIFF (Dados Brutos)
        # -----------------------------------------------------
        matriz_r_bruta = bandas_brutas[r]
        matriz_g_bruta = bandas_brutas[g]
        matriz_b_bruta = bandas_brutas[b]
        
        nome_tif = f"composicao_{r}_{g}_{b}.tif"
        caminho_tif = os.path.join(diretorio_saida_tif, nome_tif)
        
        with rasterio.open(caminho_tif, 'w', **metadados) as dest:
            dest.write(matriz_r_bruta, 1)
            dest.write(matriz_g_bruta, 2)
            dest.write(matriz_b_bruta, 3)
            
        arquivos_gerados["tif"].append(caminho_tif)

        # -----------------------------------------------------
        # 2. EXPORTAÇÃO EM PNG LEVE (Dados Esticados visualmente)
        # -----------------------------------------------------
        matriz_r_png = bandas_png[r]
        matriz_g_png = bandas_png[g]
        matriz_b_png = bandas_png[b]

        imagem_rgb = np.dstack((matriz_r_png, matriz_g_png, matriz_b_png))
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
        
        print(f" -> ✓ {r} | {g} | {b} processada (TIF bruto e PNG normalizado).")

    print(f"\n✅ {len(todas_combinacoes)} composições duplas criadas com sucesso!")
    return arquivos_gerados

# ==========================================
# Automação de Leitura e Execução
# ==========================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Erro: Parâmetros insuficientes.")
        print("Uso correto: python 04-gerarMultibandas.py <code_muni>")
        sys.exit(1)

    CODE_MUNI = int(sys.argv[1])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        raise ValueError("A variável PROJECT_ROOT não foi encontrada no arquivo .env.")

    diretorio_reports = os.path.join(project_root, "reports")
    json_cbers_recortado = os.path.join(diretorio_reports, f"{CODE_MUNI}_clipped_results.json")

    if not os.path.exists(json_cbers_recortado):
        print(f"[ERRO CRÍTICO] Arquivo JSON não encontrado: {json_cbers_recortado}")
        print("Certifique-se de executar o script 03 antes de gerar as multibandas.")
        sys.exit(1)

    with open(json_cbers_recortado, 'r', encoding='utf-8') as f:
        dados_recorte = json.load(f)
        
    caminho_alvo = dados_recorte.get("arquivo_cbers_recortado")
    if not caminho_alvo or not os.path.exists(caminho_alvo):
        print("[ERRO] Arquivo CBERS recortado não encontrado no JSON ou no disco.")
        sys.exit(1)

    pasta_base_multibandas = os.path.join(project_root, "data", "output", "pansharpening", "multispectral-RGBN-bands")
    pasta_saida_tif = os.path.join(pasta_base_multibandas, "tif")
    pasta_saida_png = os.path.join(pasta_base_multibandas, "png")

    print(f"==================================================")
    print(f" INICIANDO GERAÇÃO MULTIESPECTRAL: MUNICÍPIO {CODE_MUNI}")
    print(f"==================================================")

    arquivos_finais = gerar_multibandas(caminho_alvo, pasta_saida_tif, pasta_saida_png)

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