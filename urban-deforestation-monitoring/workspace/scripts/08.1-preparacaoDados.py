import sys
import os
import gc
import duckdb

# Correção de compatibilidade entre DuckDB e geobr no Python 3.13
if not hasattr(duckdb.DuckDBPyRelation, "fetchone"):
    duckdb.DuckDBPyRelation.fetchone = lambda self: self.fetchall()[0]

import numpy as np
import rasterio
from rasterio.windows import Window
import cv2
from PIL import Image
from dotenv import load_dotenv

def calcular_percentis_banda(src, band_idx, decimation=10):
    """Calcula p2 e p98 da banda em milissegundos via amostragem rápida por decimação."""
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

    # 1. Estrutura de pastas em data/processing
    dir_processing = os.path.join(project_root, "data", "processing")
    dir_proc_rgb = os.path.join(dir_processing, "rgb")
    dir_proc_ndvi = os.path.join(dir_processing, "ndvi")
    dir_proc_masks_png = os.path.join(dir_processing, "masks", "png_truth_masks")
    dir_proc_masks_npy = os.path.join(dir_processing, "masks", "npy_truth_masks")

    for folder in [dir_proc_rgb, dir_proc_ndvi, dir_proc_masks_png, dir_proc_masks_npy]:
        os.makedirs(folder, exist_ok=True)

    dir_pan = os.path.join(project_root, "data", "output", "pansharpening", ano_fim)
    dir_gt = os.path.join(project_root, "data", "output", "ground_truth", ano_fim)

    path_cbers_2m = os.path.join(dir_pan, f"{code_muni}_{ano_fim}_CBERS_TRUE_COLOR_2M.tif")
    path_gt_mb = os.path.join(dir_gt, f"{code_muni}_GT_MapBiomas_{ano_fim}.tif")
    path_gt_u = os.path.join(dir_gt, f"{code_muni}_GT_U_{ano_fim}.tif")
    path_gt_m = os.path.join(dir_gt, f"{code_muni}_GT_M_{ano_fim}.tif")

    if not os.path.exists(path_cbers_2m):
        print(f"[ERRO CRÍTICO] Imagem CBERS-4A 2m não encontrada: {path_cbers_2m}")
        sys.exit(1)

    print("=" * 80, flush=True)
    print(f"🚀 SCRIPT 08.1 - PREPARAÇÃO OTIMIZADA EM TILES ({ano_fim})", flush=True)
    print("=" * 80, flush=True)

    with rasterio.open(path_cbers_2m) as src:
        height, width = src.height, src.width
        num_bands = src.count

        print("1. Calculando estatísticas radiométricas (p2/p98)...", flush=True)
        p2_r, p98_r = calcular_percentis_banda(src, 1)
        p2_g, p98_g = calcular_percentis_banda(src, 2)
        p2_b, p98_b = calcular_percentis_banda(src, 3)

        # 2. Gerar Composição RGB em Tiles (uint8 direto)
        print("2. Processando composição RGB em blocos...", flush=True)
        rgb_composite = np.zeros((height, width, 3), dtype=np.uint8)
        
        tile_size = 2048
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                h_win = min(tile_size, height - y)
                w_win = min(tile_size, width - x)
                win = Window(x, y, w_win, h_win)

                r_tile = src.read(1, window=win).astype(np.float32)
                g_tile = src.read(2, window=win).astype(np.float32)
                b_tile = src.read(3, window=win).astype(np.float32)

                rgb_composite[y:y+h_win, x:x+w_win, 0] = np.clip((r_tile - p2_r) / (p98_r - p2_r) * 255.0, 0, 255).astype(np.uint8)
                rgb_composite[y:y+h_win, x:x+w_win, 1] = np.clip((g_tile - p2_g) / (p98_g - p2_g) * 255.0, 0, 255).astype(np.uint8)
                rgb_composite[y:y+h_win, x:x+w_win, 2] = np.clip((b_tile - p2_b) / (p98_b - p2_b) * 255.0, 0, 255).astype(np.uint8)

        output_rgb_path = os.path.join(dir_proc_rgb, f"{code_muni}_rgb_{ano_fim}.png")
        cv2.imwrite(output_rgb_path, cv2.cvtColor(rgb_composite, cv2.COLOR_RGB2BGR))
        print(f"   -> Composição RGB salva em: {output_rgb_path}", flush=True)
        del rgb_composite
        gc.collect()

        # 3. Gerar NDVI em Tiles
        if num_bands >= 4:
            print("3. Calculando NDVI em blocos...", flush=True)
            ndvi_scaled = np.zeros((height, width), dtype=np.uint8)

            for y in range(0, height, tile_size):
                for x in range(0, width, tile_size):
                    h_win = min(tile_size, height - y)
                    w_win = min(tile_size, width - x)
                    win = Window(x, y, w_win, h_win)

                    r_tile = src.read(1, window=win).astype(np.float32)
                    nir_tile = src.read(4, window=win).astype(np.float32)

                    denom = nir_tile + r_tile
                    denom[denom == 0] = 1e-6
                    ndvi_tile = (nir_tile - r_tile) / denom

                    ndvi_scaled[y:y+h_win, x:x+w_win] = np.interp(ndvi_tile, (-1.0, 1.0), (0, 255)).astype(np.uint8)

            output_ndvi_path = os.path.join(dir_proc_ndvi, f"{code_muni}_ndvi_{ano_fim}.png")
            cv2.imwrite(output_ndvi_path, ndvi_scaled)
            print(f"   -> NDVI salvo em: {output_ndvi_path}", flush=True)
            del ndvi_scaled
            gc.collect()

    # 4. Exportação das Máscaras (PNG e NPY)
    print("4. Salvando conjuntos verdade em .png e .npy em data/processing/...", flush=True)
    gt_dict = {
        'GT_MapBiomas': path_gt_mb,
        'GT_U': path_gt_u,
        'GT_M': path_gt_m
    }

    for name_gt, path_gt in gt_dict.items():
        if os.path.exists(path_gt):
            with rasterio.open(path_gt) as src_gt:
                mask_data = src_gt.read(1).astype(np.uint8)
                
                out_npy = os.path.join(dir_proc_masks_npy, f"{code_muni}_{name_gt}_{ano_fim}.npy")
                np.save(out_npy, mask_data)
                
                out_png = os.path.join(dir_proc_masks_png, f"{code_muni}_{name_gt}_{ano_fim}.png")
                cv2.imwrite(out_png, mask_data)
                
                print(f"   -> {name_gt}: Salvo NPY ({out_npy}) e PNG ({out_png})", flush=True)
                del mask_data
                gc.collect()

    print("\n✅ SUCESSO! Script 08.1 concluído sem estourar RAM.")

if __name__ == "__main__":
    main()