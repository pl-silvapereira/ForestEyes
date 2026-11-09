#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Etapa 08.1: Preparação de Dados (Shapefile 3 Classes -> Matriz Numpy e PNGs)
==============================================================================
Entradas:
 - Shapefile 3 Classes: data/output/segmentation/{ano_fim}/{code_muni}_Segmentacao_3Classes_{ano_fim}.shp
 - CBERS-4A 2m: data/output/pansharpening/{ano_fim}/{code_muni}_{ano_fim}_CBERS_TRUE_COLOR_2M.tif

Saídas em data/processing/:
 - slics/matriz_classes_{code_muni}_{ano_fim}.npy  (1=Floresta, 2=Não-Floresta, 0=Não Segmentar)
 - rgb/{code_muni}_rgb_{ano_fim}.png
 - ndvi/{code_muni}_ndvi_{ano_fim}.png
 - nir/{code_muni}_nir_{ano_fim}.png
==============================================================================
"""

import sys
import os
import gc
import warnings
import duckdb

if not hasattr(duckdb.DuckDBPyRelation, "fetchone"):
    duckdb.DuckDBPyRelation.fetchone = lambda self: self.fetchall()[0]

import cv2
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)

def normalizar_8bits(array):
    """Normaliza matriz radiométrica para o intervalo 0-255 (uint8) com stretch 2-98%."""
    valid_mask = array > 0
    if not np.any(valid_mask):
        return np.zeros(array.shape, dtype=np.uint8)
    p2, p98 = np.percentile(array[valid_mask], (2, 98))
    if p98 == p2:
        p98 = p2 + 1.0
    norm = np.clip((array - p2) / (p98 - p2) * 255.0, 0, 255)
    return norm.astype(np.uint8)

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 08.1-preparacaoDados.py <code_muni> <ano_inicio> <ano_fim>")
        sys.exit(1)

    code_muni = str(sys.argv[1])
    ano_inicio = str(sys.argv[2])
    ano_fim = str(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    path_shp_3classes = os.path.join(project_root, "data", "output", "segmentation", ano_fim, f"{code_muni}_Segmentacao_3Classes_{ano_fim}.shp")
    path_cbers = os.path.join(project_root, "data", "output", "pansharpening", ano_fim, f"{code_muni}_{ano_fim}_CBERS_TRUE_COLOR_2M.tif")

    dir_processing = os.path.join(project_root, "data", "processing")
    dir_slics = os.path.join(dir_processing, "slics")
    dir_rgb = os.path.join(dir_processing, "rgb")
    dir_ndvi = os.path.join(dir_processing, "ndvi")
    dir_nir = os.path.join(dir_processing, "nir")

    for d in [dir_slics, dir_rgb, dir_ndvi, dir_nir]:
        os.makedirs(d, exist_ok=True)

    path_matriz_out = os.path.join(dir_slics, f"matriz_classes_{code_muni}_{ano_fim}.npy")
    path_png_rgb = os.path.join(dir_rgb, f"{code_muni}_rgb_{ano_fim}.png")
    path_png_ndvi = os.path.join(dir_ndvi, f"{code_muni}_ndvi_{ano_fim}.png")
    path_png_nir = os.path.join(dir_nir, f"{code_muni}_nir_{ano_fim}.png")

    if not os.path.exists(path_shp_3classes) or not os.path.exists(path_cbers):
        print(f"❌ Erro Crítico: Arquivos de entrada ausentes:")
        print(f" - Shapefile: {path_shp_3classes}")
        print(f" - Raster CBERS: {path_cbers}")
        sys.exit(1)

    print("=" * 80)
    print(f"🚀 PREPARAÇÃO DE DADOS (3 CLASSES -> SLIC BASE) - IBGE {code_muni} ({ano_fim})")
    print("=" * 80)

    # 1. Leitura de metadados do CBERS
    print("1. Lendo cena raster CBERS-4A (2m)...")
    with rasterio.open(path_cbers) as src:
        cbers_shape = (src.height, src.width)
        cbers_crs = src.crs
        cbers_transform = src.transform
        num_bands = src.count

        band_r = src.read(1).astype(np.float32)
        band_g = src.read(2).astype(np.float32)
        band_b = src.read(3).astype(np.float32)
        band_nir = src.read(4).astype(np.float32) if num_bands >= 4 else None

    # 2. Leitura e rasterização do Shapefile 3 Classes
    print("2. Rasterizando Shapefile 3 Classes (Floresta=1, Não-Floresta=2, Não Segmentar=0)...")
    gdf_3classes = gpd.read_file(path_shp_3classes).to_crs(cbers_crs)

    col_classe = "class_name" if "class_name" in gdf_3classes.columns else ("classe" if "classe" in gdf_3classes.columns else None)
    if not col_classe:
        col_classe = gdf_3classes.columns[0]

    gdf_floresta = gdf_3classes[gdf_3classes[col_classe].astype(str).str.contains("Floresta", case=False, na=False) & 
                                ~gdf_3classes[col_classe].astype(str).str.contains("Nao", case=False, na=False)]
    
    gdf_nao_floresta = gdf_3classes[gdf_3classes[col_classe].astype(str).str.contains("Nao_Floresta|Não_Floresta|Nao Floresta", case=False, na=False)]

    matriz_classes = np.zeros(cbers_shape, dtype=np.uint8)

    if not gdf_floresta.empty:
        shapes_flor = [(geom, 1) for geom in gdf_floresta.geometry if geom.is_valid]
        mask_flor = rasterize(shapes_flor, out_shape=cbers_shape, transform=cbers_transform, fill=0, dtype=np.uint8)
        matriz_classes[mask_flor == 1] = 1

    if not gdf_nao_floresta.empty:
        shapes_nflor = [(geom, 2) for geom in gdf_nao_floresta.geometry if geom.is_valid]
        mask_nflor = rasterize(shapes_nflor, out_shape=cbers_shape, transform=cbers_transform, fill=0, dtype=np.uint8)
        matriz_classes[mask_nflor == 2] = 2

    np.save(path_matriz_out, matriz_classes)
    print(f" ✅ Matriz de classes salva em: {path_matriz_out}")

    # 3. Geração das composições 8-bit PNG
    print("3. Exportando composições visuais RGB, NDVI e NIR em formato PNG...")
    r_8 = normalizar_8bits(band_r)
    g_8 = normalizar_8bits(band_g)
    b_8 = normalizar_8bits(band_b)
    img_rgb = cv2.merge([b_8, g_8, r_8])
    cv2.imwrite(path_png_rgb, img_rgb)

    if band_nir is not None:
        nir_8 = normalizar_8bits(band_nir)
        img_nir = cv2.merge([g_8, r_8, nir_8])  # Falsa Cor NIR
        cv2.imwrite(path_png_nir, img_nir)

        denom = band_nir + band_r
        denom[denom == 0] = 1e-6
        ndvi = (band_nir - band_r) / denom
        ndvi_scaled = np.clip((ndvi + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
        img_ndvi = cv2.merge([ndvi_scaled, ndvi_scaled, ndvi_scaled])
        cv2.imwrite(path_png_ndvi, img_ndvi)

    print("🎉 Etapa 08.1 concluída com sucesso!")

if __name__ == "__main__":
    main()