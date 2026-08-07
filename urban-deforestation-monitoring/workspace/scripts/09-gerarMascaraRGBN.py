import os
import sys
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
import geobr
from dotenv import load_dotenv

def main():
    # Uso correto: python 09-gerarMascaraRGBN.py <code_muni> <ano_inicio> <ano_fim>
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 09-gerarMascaraRGBN.py <code_muni> <ano_inicio> <ano_fim>")
        print("Exemplo: python 09-gerarMascaraRGBN.py 3549904 2023 2024")
        sys.exit(1)

    code_muni = int(sys.argv[1])
    ano_inicio = str(sys.argv[2])
    ano_fim = str(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    # Nova subpasta solicitada: data/output/mask/rgbn/
    mask_rgbn_dir = os.path.join(project_root, "data", "output", "mask", "rgbn")
    os.makedirs(mask_rgbn_dir, exist_ok=True)

    print(f"Buscando informações para o código de município: {code_muni}...")
    try:
        gdf_info = geobr.read_municipality(code_muni=code_muni, year=2022)
        nome_cidade = gdf_info['name_muni'].values[0]
        uf = gdf_info['abbrev_state'].values[0]
    except Exception as e:
        nome_cidade = "Município"
        uf = "SP"

    # Caminho do shapefile delta gerado pelo Script 08
    delta_shp_path = os.path.join(project_root, "data", "output", "mask", f"{ano_inicio}_vs_{ano_fim}", f"{code_muni}_delta_vector_{ano_inicio}_vs_{ano_fim}.shp")
    
    # Caminho da imagem de satélite informada
    path_sat = os.path.join(project_root, "data", "output", "pansharpening", "geopolitic-RGBN", ano_fim, f"{code_muni}_{ano_fim}_CBERS_TRUE_COLOR_CLIPPED.tif")

    if not os.path.exists(delta_shp_path):
        print(f"[ERRO CRÍTICO] Shapefile delta do Script 08 não encontrado em:\n-> {delta_shp_path}")
        print("Execute o Script 08 primeiro para gerar o vetor de mudanças.")
        sys.exit(1)

    if not os.path.exists(path_sat):
        print(f"[ERRO CRÍTICO] Imagem de satélite não encontrada em:\n-> {path_sat}")
        sys.exit(1)

    print("=" * 115)
    print(f"🛰️ GERANDO MÁSCARA RGBN DE MUDANÇAS (FILTRADA POR NUVENS E DELTA)")
    print(f"📍 MUNICÍPIO: {nome_cidade} - {uf} | PERÍODO: {ano_inicio} vs {ano_fim}")
    print("=" * 115)

    print("Carregando imagem de satélite e shapefile delta...")
    with rasterio.open(path_sat) as src_sat:
        meta_sat = src_sat.meta.copy()
        sat_img = src_sat.read() # Lê todas as bandas (ex: R, G, B, ...)
        transform = src_sat.transform
        crs = src_sat.crs
        height, width = src_sat.height, src_sat.width

    # Carregar vetor delta e reprojetar para o CRS da imagem de satélite, se necessário
    gdf_delta = gpd.read_file(delta_shp_path)
    if gdf_delta.crs != crs:
        gdf_delta = gdf_delta.to_crs(crs)

    print("Rasterizando geometrias de mudança do Script 08...")
    shapes_delta = [(geom, 1) for geom in gdf_delta.geometry]
    mask_delta_raster = rasterize(
        shapes_delta, 
        out_shape=(height, width), 
        transform=transform, 
        fill=0, 
        dtype=np.uint8
    )

    print("Detectando e mascarando nuvens na imagem de satélite...")
    # Normalizar valores de acordo com o tipo de dado (geralmente uint8 [0-255] ou uint16 [0-65535])
    # Nuvens possuem reflectância muito alta (brancas) em todas as bandas RGB principais (bandas 1, 2, 3)
    if sat_img.shape[0] >= 3:
        r, g, b = sat_img[0], sat_img[1], sat_img[2]
        
        # Limiar adaptativo para considerar nuvem branca (ex: > 85% do valor máximo da banda)
        max_val = np.max(sat_img)
        limiar_nuvem = max_val * 0.82 if max_val > 255 else 215 

        # Máscara de nuvens: True onde for nuvem muito branca em RGB simultaneamente
        is_cloud = (r >= limiar_nuvem) & (g >= limiar_nuvem) & (b >= limiar_nuvem)
    else:
        is_cloud = np.zeros((height, width), dtype=bool)

    # Construir a máscara final combinada:
    # 1. Deve ser um pixel de mudança do Delta (mask_delta_raster == 1)
    # 2. NÃO pode ser nuvem (~is_cloud)
    mascara_final = (mask_delta_raster == 1) & (~is_cloud)

    print("Aplicando máscara preta nas áreas estáveis e cobertas por nuvens...")
    masked_sat = np.zeros_like(sat_img)
    for i in range(sat_img.shape[0]):
        # Onde a máscara for verdadeira, mantém o satélite; senão, aplica 0 (preto)
        masked_sat[i] = np.where(mascara_final, sat_img[i], 0)

    # Salvar a nova imagem TIF na subpasta mask/rgbn/
    out_tif_name = f"{code_muni}_masked_rgbn_delta_{ano_inicio}_vs_{ano_fim}.tif"
    out_tif_path = os.path.join(mask_rgbn_dir, out_tif_name)

    meta_sat.update(nodata=0)
    with rasterio.open(out_tif_path, 'w', **meta_sat) as dst:
        dst.write(masked_sat)

    print(f"\n[SUCESSO] Imagem de máscara RGBN gerada com sucesso em:\n-> {out_tif_path}")

if __name__ == "__main__":
    main()