import sys
import os
import gc
import duckdb

if not hasattr(duckdb.DuckDBPyRelation, "fetchone"):
    duckdb.DuckDBPyRelation.fetchone = lambda self: self.fetchall()[0]

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import geopandas as gpd
import cv2
from dotenv import load_dotenv

def calcular_percentis_banda(src, band_idx, decimation=10):
    """Calcula p2 e p98 da banda via amostragem rápida por decimação."""
    out_shape = (1, max(1, src.height // decimation), max(1, src.width // decimation))
    amostra = src.read(band_idx, out_shape=out_shape).astype(np.float32)
    validos = amostra[amostra > 0]
    if len(validos) > 0:
        p2, p98 = np.percentile(validos, (2, 98))
    else:
        p2, p98 = 0, 1
    if p98 == p2:
        p98 = p2 + 1
    del amostra, validos
    gc.collect()
    return float(p2), float(p98)

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

    print("=" * 80, flush=True)
    print(f"🚀 SCRIPT 08.1 - PREPARAÇÃO DE DADOS & RASTERIZAÇÃO 3 CLASSES ({ano_fim})", flush=True)
    print(f"📌 PROJECT_ROOT resolvido: {project_root}", flush=True)
    print("=" * 80, flush=True)

    dir_processing = os.path.join(project_root, "data", "processing")
    dir_proc_rgb = os.path.join(dir_processing, "rgb")
    dir_proc_ndvi = os.path.join(dir_processing, "ndvi")
    dir_proc_nir = os.path.join(dir_processing, "nir")
    dir_proc_masks_png = os.path.join(dir_processing, "masks", "png_truth_masks")
    dir_proc_masks_npy = os.path.join(dir_processing, "masks", "npy_truth_masks")

    for folder in [dir_proc_rgb, dir_proc_ndvi, dir_proc_nir, dir_proc_masks_png, dir_proc_masks_npy]:
        os.makedirs(folder, exist_ok=True)

    path_cbers_2m = os.path.join(project_root, "data", "output", "pansharpening", ano_fim, f"{code_muni}_{ano_fim}_CBERS_TRUE_COLOR_2M.tif")
    path_shp_3classes = os.path.join(project_root, "data", "output", "segmentation", ano_fim, f"{code_muni}_Segmentacao_3Classes_{ano_fim}.shp")

    if not os.path.exists(path_cbers_2m):
        print(f"❌ [ERRO CRÍTICO 08.1] Imagem CBERS-4A 2m não encontrada em:\n    {path_cbers_2m}", flush=True)
        sys.exit(1)
    if not os.path.exists(path_shp_3classes):
        print(f"❌ [ERRO CRÍTICO 08.1] Shapefile 3Classes não encontrado em:\n    {path_shp_3classes}", flush=True)
        sys.exit(1)

    with rasterio.open(path_cbers_2m) as src:
        height, width = src.height, src.width
        num_bands = src.count
        crs_img = src.crs
        transform_img = src.transform

        print("1. Calculando percentis radiométricos...", flush=True)
        p2_b, p98_b = calcular_percentis_banda(src, 1)
        p2_g, p98_g = calcular_percentis_banda(src, 2)
        p2_r, p98_r = calcular_percentis_banda(src, 3)
        p2_nir, p98_nir = calcular_percentis_banda(src, 4) if num_bands >= 4 else (0, 1)

        tile_size = 2048

        print("2. Gerando RGB Natural...", flush=True)
        rgb_composite = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                h_win, w_win = min(tile_size, height - y), min(tile_size, width - x)
                win = Window(x, y, w_win, h_win)

                r_t = src.read(3, window=win).astype(np.float32)
                g_t = src.read(2, window=win).astype(np.float32)
                b_t = src.read(1, window=win).astype(np.float32)

                rgb_composite[y:y+h_win, x:x+w_win, 0] = np.clip((r_t - p2_r) / (p98_r - p2_r) * 255.0, 0, 255).astype(np.uint8)
                rgb_composite[y:y+h_win, x:x+w_win, 1] = np.clip((g_t - p2_g) / (p98_g - p2_g) * 255.0, 0, 255).astype(np.uint8)
                rgb_composite[y:y+h_win, x:x+w_win, 2] = np.clip((b_t - p2_b) / (p98_b - p2_b) * 255.0, 0, 255).astype(np.uint8)

        output_rgb_path = os.path.join(dir_proc_rgb, f"{code_muni}_rgb_{ano_fim}.png")
        cv2.imwrite(output_rgb_path, cv2.cvtColor(rgb_composite, cv2.COLOR_RGB2BGR))
        del rgb_composite
        gc.collect()

        if num_bands >= 4:
            print("3. Gerando Falsa Cor NIR e NDVI...", flush=True)
            nir_composite = np.zeros((height, width, 3), dtype=np.uint8)
            ndvi_scaled = np.zeros((height, width), dtype=np.uint8)
            for y in range(0, height, tile_size):
                for x in range(0, width, tile_size):
                    h_win, w_win = min(tile_size, height - y), min(tile_size, width - x)
                    win = Window(x, y, w_win, h_win)

                    nir_t = src.read(4, window=win).astype(np.float32)
                    r_t = src.read(3, window=win).astype(np.float32)
                    g_t = src.read(2, window=win).astype(np.float32)

                    nir_composite[y:y+h_win, x:x+w_win, 0] = np.clip((nir_t - p2_nir) / (p98_nir - p2_nir) * 255.0, 0, 255).astype(np.uint8)
                    nir_composite[y:y+h_win, x:x+w_win, 1] = np.clip((r_t - p2_r) / (p98_r - p2_r) * 255.0, 0, 255).astype(np.uint8)
                    nir_composite[y:y+h_win, x:x+w_win, 2] = np.clip((g_t - p2_g) / (p98_g - p2_g) * 255.0, 0, 255).astype(np.uint8)

                    denom = nir_t + r_t
                    denom[denom == 0] = 1e-6
                    ndvi_tile = (nir_t - r_t) / denom
                    ndvi_scaled[y:y+h_win, x:x+w_win] = np.interp(ndvi_tile, (-1.0, 1.0), (0, 255)).astype(np.uint8)

            cv2.imwrite(os.path.join(dir_proc_nir, f"{code_muni}_nir_{ano_fim}.png"), cv2.cvtColor(nir_composite, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(dir_proc_ndvi, f"{code_muni}_ndvi_{ano_fim}.png"), ndvi_scaled)
            del nir_composite, ndvi_scaled
            gc.collect()

        print("4. Rasterizando Shapefile 3Classes...", flush=True)
        gdf_3c = gpd.read_file(path_shp_3classes)
        if gdf_3c.crs != crs_img:
            gdf_3c = gdf_3c.to_crs(crs_img)

        # Identificação dinâmica do nome da coluna da classe
        col_class = None
        for c in ['class_name', 'classe', 'class', 'target']:
            if c in gdf_3c.columns:
                col_class = c
                break
        if not col_class:
            col_class = gdf_3c.columns[0]

        def class_to_code(cname):
            s = str(cname).lower().strip()
            if "nao_floresta" in s or "nao floresta" in s:
                return 2  # Vermelho (Nao_Floresta)
            elif "floresta" in s:
                return 1  # Verde (Floresta)
            return 0      # Preto (Nao_Segmentar)

        shapes = [(geom, class_to_code(cname)) for geom, cname in zip(gdf_3c.geometry, gdf_3c[col_class]) if not geom.is_empty]

        gt_3classes = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform_img,
            fill=0,
            dtype=np.uint8
        )

        out_npy_3c = os.path.join(dir_proc_masks_npy, f"{code_muni}_GT_3Classes_{ano_fim}.npy")
        out_npy_gt_u = os.path.join(dir_proc_masks_npy, f"{code_muni}_GT_U_{ano_fim}.npy")
        out_png_3c = os.path.join(dir_proc_masks_png, f"{code_muni}_GT_3Classes_{ano_fim}.png")

        np.save(out_npy_3c, gt_3classes)
        np.save(out_npy_gt_u, gt_3classes)
        cv2.imwrite(out_png_3c, gt_3classes)

        print(f" ✅ Matriz NPY gerada: {out_npy_3c} ({os.path.getsize(out_npy_3c):,} bytes)", flush=True)
        print(f" ✅ Matriz NPY gerada: {out_npy_gt_u} ({os.path.getsize(out_npy_gt_u):,} bytes)", flush=True)
        print("✅ SCRIPT 08.1 CONCLUÍDO COM SUCESSO!", flush=True)

if __name__ == "__main__":
    main()