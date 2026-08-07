import os
import sys
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import pandas as pd
import numpy as np
import geobr
from dotenv import load_dotenv

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 08-gerarImagemDelta.py <code_muni> <ano_inicio> <ano_fim>")
        print("Exemplo: python 08-gerarImagemDelta.py 3549904 2023 2024")
        sys.exit(1)

    code_muni = int(sys.argv[1])
    ano_inicio = str(sys.argv[2])
    ano_fim = str(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    mask_dir = os.path.join(project_root, "data", "output", "mask", f"{ano_inicio}_vs_{ano_fim}")
    os.makedirs(mask_dir, exist_ok=True)

    print(f"Buscando informações para o código de município: {code_muni}...")
    try:
        gdf_info = geobr.read_municipality(code_muni=code_muni, year=2022)
        nome_cidade = gdf_info['name_muni'].values[0]
        uf = gdf_info['abbrev_state'].values[0]
    except Exception as e:
        nome_cidade = "Município"
        uf = "SP"
        print(f"⚠️ Aviso ao buscar nome da cidade: {e}")

    path_shp_inicio = os.path.join(project_root, "data", "output", "classification", ano_inicio, f"{code_muni}_Classificado_ForestEyes_{ano_inicio}.shp")
    path_shp_fim = os.path.join(project_root, "data", "output", "classification", ano_fim, f"{code_muni}_Classificado_ForestEyes_{ano_fim}.shp")
    path_sat = os.path.join(project_root, "data", "output", "pansharpening", "multispectral-RGBN-bands", ano_fim, f"{code_muni}_multispectral_RGBN_{ano_fim}.tif")

    if not os.path.exists(path_shp_inicio) or not os.path.exists(path_shp_fim):
        print(f"[ERRO CRÍTICO] Shapefiles de classificação não encontrados para {ano_inicio} e/ou {ano_fim}.")
        print(f" -> [{ano_inicio}]: {path_shp_inicio}")
        print(f" -> [{ano_fim}]: {path_shp_fim}")
        sys.exit(1)

    print("=" * 115)
    print(f"🖼️ GERANDO IMAGENS DELTA A PARTIR DOS SHAPEFILES")
    print(f"📍 MUNICÍPIO: {nome_cidade} - {uf} | PERÍODO: {ano_inicio} vs {ano_fim}")
    print("=" * 115)

    print("Carregando shapefiles...")
    gdf1 = gpd.read_file(path_shp_inicio)
    gdf2 = gpd.read_file(path_shp_fim)

    utm_crs = gdf1.estimate_utm_crs()
    gdf1 = gdf1.to_crs(utm_crs)
    gdf2 = gdf2.to_crs(utm_crs)

    # Identificar automaticamente a coluna de código/classe no GeoDataFrame
    def obter_coluna_classe(gdf):
        candidatos = ['class_code', 'gridcode', 'id', 'value', 'class_id', 'DN']
        for c in candidatos:
            if c in gdf.columns:
                return c
        # Se não achar nenhuma conhecida, pega a primeira coluna numérica que não seja geometria
        for col in gdf.columns:
            if col != 'geometry' and pd.api.types.is_numeric_dtype(gdf[col]):
                return col
        return None

    col1 = obter_coluna_classe(gdf1)
    col2 = obter_coluna_classe(gdf2)

    # Definir resolução espacial de 10 metros
    resolution = 10.0
    xmin, ymin, xmax, ymax = gdf2.total_bounds
    width = int(np.ceil((xmax - xmin) / resolution))
    height = int(np.ceil((ymax - ymin) / resolution))

    transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)

    meta = {
        'driver': 'GTiff',
        'dtype': 'int32',
        'count': 1,
        'width': width,
        'height': height,
        'transform': transform,
        'crs': utm_crs,
        'nodata': 0
    }

    print(f"Rasterizando base do ano inicial (usando coluna: {col1})...")
    if col1:
        shapes1 = [(geom, int(val) if pd.notnull(val) else 1) for geom, val in zip(gdf1.geometry, gdf1[col1])]
    else:
        shapes1 = [(geom, 1) for geom in gdf1.geometry]
    arr1 = rasterize(shapes1, out_shape=(height, width), transform=transform, fill=0, dtype=rasterio.int32)

    print(f"Rasterizando base do ano final (usando coluna: {col2})...")
    if col2:
        shapes2 = [(geom, int(val) if pd.notnull(val) else 1) for geom, val in zip(gdf2.geometry, gdf2[col2])]
    else:
        shapes2 = [(geom, 1) for geom in gdf2.geometry]
    arr2 = rasterize(shapes2, out_shape=(height, width), transform=transform, fill=0, dtype=rasterio.int32)

    # Gerar imagem delta: onde arr1 != arr2 mantemos arr2, senão máscara preta (0)
    delta_class = np.where(arr1 != arr2, arr2, 0).astype(np.int32)

    out_class_name = f"{code_muni}_delta_classification_{ano_inicio}_vs_{ano_fim}.tif"
    out_class_path = os.path.join(mask_dir, out_class_name)

    with rasterio.open(out_class_path, 'w', **meta) as dst:
        dst.write(delta_class, 1)

    print(f"✓ Imagem delta da classificação salva em:\n-> {out_class_path}")

    # Gerar imagem de satélite com máscara aplicada (se houver o raster de satélite)
    if os.path.exists(path_sat):
        print("Aplicando máscara delta sobre o raster de satélite...")
        with rasterio.open(path_sat) as src_sat:
            meta_sat = src_sat.meta.copy()
            sat_img = src_sat.read()

            mascara_mudanca = (delta_class != 0)
            delta_sat = np.zeros_like(sat_img)
            for i in range(sat_img.shape[0]):
                delta_sat[i] = np.where(mascara_mudanca, sat_img[i], 0)

            out_sat_name = f"{code_muni}_delta_satellite_{ano_inicio}_vs_{ano_fim}.tif"
            out_sat_path = os.path.join(mask_dir, out_sat_name)

            meta_sat.update(nodata=0)
            with rasterio.open(out_sat_path, 'w', **meta_sat) as dst:
                dst.write(delta_sat)

        print(f"✓ Imagem delta de satélite salva em:\n-> {out_sat_path}")
    else:
        print(f"⚠️ [AVISO] Raster de satélite correspondente não encontrado. Apenas a imagem delta rasterizada foi gerada.")

    print("\n[SUCESSO] Processo de geração de imagens delta concluído!")

if __name__ == "__main__":
    main()