import os
import sys
import json
import itertools
import numpy as np
import rasterio
from PIL import Image
from dotenv import load_dotenv

def aplicar_stretch_contraste(banda_matriz):
    banda_mascarada = np.ma.masked_equal(banda_matriz, 0)
    
    if banda_mascarada.count() == 0:
        return np.zeros_like(banda_matriz, dtype=np.uint8)
        
    p2, p98 = np.percentile(banda_mascarada.compressed(), (2, 98))
    
    if p98 == p2:
        return np.uint8(np.clip(banda_matriz, 0, 1) * 255)
        
    banda_normalizada = np.clip((banda_matriz - p2) / (p98 - p2), 0, 1)
    return np.uint8(banda_normalizada * 255)

def gerar_multibandas(caminho_arquivo, diretorio_saida_tif, diretorio_saida_png, ano):
    os.makedirs(diretorio_saida_tif, exist_ok=True)
    os.makedirs(diretorio_saida_png, exist_ok=True)
    
    bandas_brutas = {}
    bandas_png = {}
    nomes_bandas = ['B1_Azul', 'B2_Verde', 'B3_Vermelho', 'B4_NIR']
    
    arquivos_gerados = {"tif": [], "png": []}
    
    with rasterio.open(caminho_arquivo) as src:
        metadados = src.meta.copy()
        metadados.update({'count': 3, 'nodata': 0})
        
        for i, nome in enumerate(nomes_bandas, start=1):
            matriz = src.read(i)
            bandas_brutas[nome] = matriz
            bandas_png[nome] = aplicar_stretch_contraste(matriz)
            
    todas_combinacoes = list(itertools.permutations(nomes_bandas, 3))
    print(f"Gerando {len(todas_combinacoes)} composições (Ano: {ano})...")
    
    for composicao in todas_combinacoes:
        r, g, b = composicao
        
        nome_tif = f"composicao_{r}_{g}_{b}_{ano}.tif"
        caminho_tif = os.path.join(diretorio_saida_tif, nome_tif)
        
        with rasterio.open(caminho_tif, 'w', **metadados) as dest:
            dest.write(bandas_brutas[r], 1)
            dest.write(bandas_brutas[g], 2)
            dest.write(bandas_brutas[b], 3)
        arquivos_gerados["tif"].append(caminho_tif)

        imagem_rgb = np.dstack((bandas_png[r], bandas_png[g], bandas_png[b]))
        img = Image.fromarray(imagem_rgb)
        
        largura_maxima = 1920
        if img.width > largura_maxima:
            proporcao = largura_maxima / img.width
            nova_altura = int(img.height * proporcao)
            img = img.resize((largura_maxima, nova_altura), Image.Resampling.LANCZOS)
        
        nome_png = f"preview_{r}_{g}_{b}_{ano}.png"
        caminho_png = os.path.join(diretorio_saida_png, nome_png)
        img.save(caminho_png, optimize=True)
        arquivos_gerados["png"].append(caminho_png)
        
    print(f"✅ {len(todas_combinacoes)} composições criadas com sucesso!")
    return arquivos_gerados

if __name__ == "__main__":
    load_dotenv()
    
    if len(sys.argv) >= 3:
        CODE_MUNI = int(sys.argv[1])
        ANO = str(sys.argv[2])
    else:
        CODE_MUNI = int(os.environ.get('CODE_MUNI', 0))
        ANO = os.environ.get('ANO', '')

    if not CODE_MUNI or not ANO: sys.exit(1)

    project_root = os.environ.get("PROJECT_ROOT")
    diretorio_reports = os.path.join(project_root, "reports")
    json_cbers_recortado = os.path.join(diretorio_reports, f"{CODE_MUNI}_{ANO}_clipped_results.json")

    if not os.path.exists(json_cbers_recortado):
        print(f"[ERRO] JSON do script 03 não encontrado: {json_cbers_recortado}")
        sys.exit(1)

    with open(json_cbers_recortado, 'r', encoding='utf-8') as f:
        caminho_alvo = json.load(f).get("arquivo_cbers_recortado")

    pasta_base_multibandas = os.path.join(project_root, "data", "output", "pansharpening", "multispectral-RGBN-bands", ANO)
    pasta_saida_tif = os.path.join(pasta_base_multibandas, "tif")
    pasta_saida_png = os.path.join(pasta_base_multibandas, "png")

    print(f"=== GERAÇÃO MULTIESPECTRAL: {CODE_MUNI} | ANO: {ANO} ===")
    arquivos_finais = gerar_multibandas(caminho_alvo, pasta_saida_tif, pasta_saida_png, ANO)

    dados_finais = {
        "code_muni": CODE_MUNI,
        "ano": ANO,
        "pasta_tif": pasta_saida_tif,
        "pasta_png": pasta_saida_png,
        "arquivos_tif_gerados": arquivos_finais["tif"],
        "arquivos_png_gerados": arquivos_finais["png"]
    }
    
    json_final = os.path.join(diretorio_reports, f"{CODE_MUNI}_{ANO}_multibands_results.json")
    with open(json_final, 'w', encoding='utf-8') as f:
        json.dump(dados_finais, f, indent=4, ensure_ascii=False)