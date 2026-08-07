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

    path_shp_inicio = os.path.join(project_root, "data", "output", "classification", ano_inicio, f"{code_muni}_Classificado_ForestEyes_{ano_inicio}.shp")
    path_shp_fim = os.path.join(project_root, "data", "output", "classification", ano_fim, f"{code_muni}_Classificado_ForestEyes_{ano_fim}.shp")

    if not os.path.exists(path_shp_inicio) or not os.path.exists(path_shp_fim):
        print(f"[ERRO CRÍTICO] Shapefiles não encontrados para {ano_inicio} e/ou {ano_fim}.")
        sys.exit(1)

    print("=" * 115)
    print(f"🖼️ GERANDO IMAGENS DELTA COM PALETA DE CORES PERSONALIZADA")
    print(f"📍 MUNICÍPIO: {nome_cidade} - {uf} | PERÍODO: {ano_inicio} vs {ano_fim}")
    print("=" * 115)

    gdf1 = gpd.read_file(path_shp_inicio)
    gdf2 = gpd.read_file(path_shp_fim)

    utm_crs = gdf1.estimate_utm_crs()
    gdf1 = gdf1.to_crs(utm_crs)
    gdf2 = gdf2.to_crs(utm_crs)

    def obter_coluna_classe(gdf):
        for c in ['class_code', 'gridcode', 'id', 'value', 'class_id', 'DN']:
            if c in gdf.columns:
                return c
        for col in gdf.columns:
            if col != 'geometry' and pd.api.types.is_numeric_dtype(gdf[col]):
                return col
        return None

    col1 = obter_coluna_classe(gdf1)
    col2 = obter_coluna_classe(gdf2)

    resolution = 10.0
    xmin, ymin, xmax, ymax = gdf2.total_bounds
    width = int(np.ceil((xmax - xmin) / resolution))
    height = int(np.ceil((ymax - ymin) / resolution))
    transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)

    shapes1 = [(geom, int(val) if pd.notnull(val) else 1) for geom, val in zip(gdf1.geometry, gdf1[col1])] if col1 else [(geom, 1) for geom in gdf1.geometry]
    arr1 = rasterize(shapes1, out_shape=(height, width), transform=transform, fill=0, dtype=rasterio.int32)

    shapes2 = [(geom, int(val) if pd.notnull(val) else 1) for geom, val in zip(gdf2.geometry, gdf2[col2])] if col2 else [(geom, 1) for geom in gdf2.geometry]
    arr2 = rasterize(shapes2, out_shape=(height, width), transform=transform, fill=0, dtype=rasterio.int32)

    # Onde não houve mudança (arr1 == arr2), o valor vira 0 (máscara preta)
    delta_class = np.where(arr1 != arr2, arr2, 0).astype(np.int32)

    # Função auxiliar para converter cor HEX para tupla RGB (0-255)
    def hex_to_rgb(hex_str):
        hex_str = hex_str.lstrip('#')
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

    # Mapeamento oficial baseado nas suas categorias e códigos do MapBiomas
    # Construindo o dicionário de cores completo para o Rasterio (RGB)
    colormap = {0: (0, 0, 0)} # 0 = Máscara preta para áreas sem alteração

    mapeamento_hex = {
        # Floresta
        3: '#006400',
        # Floresta Antrópica
        9: '#93c47d',
        # Vegetacao Herbacea e Arbustiva
        11: '#a8c04d', 12: '#a8c04d', 36: '#a8c04d',
        # Agropecuaria (Campos, Lavouras)
        15: '#edde8e', 19: '#edde8e', 20: '#edde8e', 21: '#edde8e', 
        39: '#edde8e', 41: '#edde8e', 46: '#edde8e', 48: '#edde8e',
        # Infraestrutura Urbana
        24: '#d4271e', 25: '#d4271e',
        # Nao Observado (Agua, Rocha)
        29: '#0000ff', 31: '#0000ff', 33: '#0000ff',
        # Descartadas
        4: '#A9A9A9', 5: '#A9A9A9', 6: '#A9A9A9', 23: '#A9A9A9', 
        27: '#A9A9A9', 30: '#A9A9A9', 32: '#A9A9A9', 35: '#A9A9A9', 
        40: '#A9A9A9', 47: '#A9A9A9', 49: '#A9A9A9', 50: '#A9A9A9', 
        62: '#A9A9A9', 75: '#A9A9A9'
    }

    for code, hex_val in mapeamento_hex.items():
        colormap[code] = hex_to_rgb(hex_val)

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

    out_class_path = os.path.join(mask_dir, f"{code_muni}_delta_classification_{ano_inicio}_vs_{ano_fim}.tif")
    
    with rasterio.open(out_class_path, 'w', **meta) as dst:
        dst.write(delta_class, 1)
        dst.write_colormap(1, colormap)

    print(f"✓ Imagem delta colorida com máscara preta salva em:\n-> {out_class_path}")

if __name__ == "__main__":
    main()