import os
import sys
import json
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import shape
from dotenv import load_dotenv

def executar_recorte_por_vetorizacao(caminho_mapbiomas, caminho_cbers, caminho_saida):
    """
    Recorta a imagem de alta resolução extraindo o formato exato dos pixels válidos
    da imagem do MapBiomas através de vetorização on-the-fly e Geopandas.
    """
    print("1/3 - Extraindo o formato geopolítico exato do raster MapBiomas...")
    with rasterio.open(caminho_mapbiomas) as mb_src:
        mb_crs = mb_src.crs
        mb_data = mb_src.read(1)
        
        # Identifica o valor de "NoData" ou assume 0
        mb_nodata = mb_src.nodata if mb_src.nodata is not None else 0
        
        # Cria uma máscara restrita apenas aos pixels que contêm dados válidos do município
        mascara_cidade = (mb_data != mb_nodata) & (mb_data != 0)
        
        # Transforma os clusters de pixels em geometrias vetoriais (On-The-Fly)
        gerador_shapes = shapes(mascara_cidade.astype('uint8'), mask=mascara_cidade, transform=mb_src.transform)
        poligonos = [shape(geom) for geom, valor in gerador_shapes if valor == 1]
        
        if not poligonos:
            print("❌ Erro: Não foi possível identificar a área válida no MapBiomas.")
            sys.exit(1)
            
        # Agrupa os polígonos num único objeto geométrico (caso haja feições desconexas)
        gdf_mb = gpd.GeoDataFrame({'geometry': poligonos}, crs=mb_crs)
        poligono_cidade = gdf_mb.geometry.unary_union
        
        # Consolida o formato da cidade no sistema de coordenadas original
        gdf_base = gpd.GeoDataFrame({'geometry': [poligono_cidade]}, crs=mb_crs)

    print("2/3 - Alinhando projeções e recortando a imagem CBERS-4A...")
    with rasterio.open(caminho_cbers) as cbers_src:
        cbers_crs = cbers_src.crs
        
        # Reprojeta a geometria vetorial complexa para o CRS da imagem alvo
        gdf_base_alinhado = gdf_base.to_crs(cbers_crs)
        geometria_corte = [gdf_base_alinhado.geometry.iloc[0]]
        
        # Executa o recorte utilizando a geometria exata como máscara
        out_img, out_transform = mask(cbers_src, geometria_corte, crop=True, filled=True, nodata=0)
        
        # Atualiza os metadados espaciais
        out_meta = cbers_src.meta.copy()
        out_meta.update({
            "height": out_img.shape[1],
            "width": out_img.shape[2],
            "transform": out_transform,
            "nodata": 0
        })

        print("3/3 - Salvando o arquivo multiespectral final recortado...")
        
        # Garante que as subpastas existam antes de salvar
        os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
        
        with rasterio.open(caminho_saida, "w", **out_meta) as dest:
            dest.write(out_img)
            # Conserva a interpretação de cores original
            dest.colorinterp = cbers_src.colorinterp

    print(f"\n✅ Sucesso absoluto! A imagem perfeitamente alinhada foi salva em:\n-> {caminho_saida}")
    return caminho_saida

# ==========================================
# Automação de Leitura e Execução
# ==========================================
if __name__ == "__main__":
    # 1. Capturar argumento
    if len(sys.argv) < 2:
        print("Erro: Parâmetros insuficientes.")
        print("Uso correto: python 03-recortarCBERS.py <code_muni>")
        sys.exit(1)

    CODE_MUNI = int(sys.argv[1])

    # 2. Carregar ambiente
    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        raise ValueError("A variável PROJECT_ROOT não foi encontrada no arquivo .env.")

    diretorio_reports = os.path.join(project_root, "reports")
    json_mapbiomas = os.path.join(diretorio_reports, f"{CODE_MUNI}.json")
    json_cbers = os.path.join(diretorio_reports, f"{CODE_MUNI}_cbers_results.json")

    # 3. Validar existência dos relatórios gerados
    if not os.path.exists(json_mapbiomas) or not os.path.exists(json_cbers):
        print(f"[ERRO CRÍTICO] Arquivos JSON não encontrados na pasta reports para o município {CODE_MUNI}.")
        print("Certifique-se de executar os scripts 01 e 02 antes de recortar.")
        sys.exit(1)

    # 4. Ler JSON do MapBiomas para pegar o raster de máscara
    with open(json_mapbiomas, 'r', encoding='utf-8') as f:
        dados_mb = json.load(f)
        
    caminho_mapbiomas = dados_mb.get("arquivo_mapbiomas")
    if not caminho_mapbiomas or not os.path.exists(caminho_mapbiomas):
        print("[ERRO] Caminho do arquivo MapBiomas não encontrado no JSON ou o arquivo não existe no disco.")
        sys.exit(1)

    # 5. Ler JSON do CBERS para pegar o raster alvo da composição True Color
    with open(json_cbers, 'r', encoding='utf-8') as f:
        dados_cbers = json.load(f)
        
    caminho_cbers = dados_cbers.get("composicao_true_color")
    if not caminho_cbers or not os.path.exists(caminho_cbers):
        print("[ERRO] Caminho da composição CBERS não encontrado no JSON ou o arquivo não existe no disco.")
        sys.exit(1)

    # 6. Definir o caminho de saída final com a nova pasta "geopolitic-RGBN"
    nome_saida = f"{CODE_MUNI}_CBERS_TRUE_COLOR_CLIPPED.tif"
    pasta_saida = os.path.join(project_root, "data", "output", "pansharpening", "geopolitic-RGBN")
    caminho_saida = os.path.join(pasta_saida, nome_saida)

    print(f"==================================================")
    print(f" INICIANDO RECORTE: MUNICÍPIO {CODE_MUNI}")
    print(f"==================================================")
    print(f"MapBiomas (Máscara): {os.path.basename(caminho_mapbiomas)}")
    print(f"CBERS (Alvo): {os.path.basename(caminho_cbers)}\n")

    # 7. Executar a função
    arquivo_final = executar_recorte_por_vetorizacao(caminho_mapbiomas, caminho_cbers, caminho_saida)

    # 8. Atualizar a pasta reports com o resultado deste script
    dados_finais = {
        "code_muni": CODE_MUNI,
        "arquivo_cbers_recortado": arquivo_final
    }
    json_final = os.path.join(diretorio_reports, f"{CODE_MUNI}_clipped_results.json")
    
    with open(json_final, 'w', encoding='utf-8') as f:
        json.dump(dados_finais, f, indent=4, ensure_ascii=False)