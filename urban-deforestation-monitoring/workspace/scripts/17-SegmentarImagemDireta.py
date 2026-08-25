import os
import sys
import rasterio
from rasterio.windows import Window
from skimage.segmentation import slic, find_boundaries
import numpy as np
from dotenv import load_dotenv

def main():
    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(diretorio_scripts))

    # Diretórios de entrada e saída
    segmentation_dir = os.path.join(project_root, "data", "output", "mask", "segmentation")
    os.makedirs(segmentation_dir, exist_ok=True)

    # Caminho exato da imagem de entrada fornecida
    path_sat = os.path.join(
        project_root, "data", "output", "pansharpening", "geopolitic-RGBN", 
        "2024", "CBERS4A_WPM20214220240930_TRUE_COLOR_2024.tif"
    )

    if not os.path.exists(path_sat):
        # Tenta buscar diretamente pelo caminho absoluto caso o .env não esteja configurado
        path_sat = "/content/drive/MyDrive/Mestrado/04-Projeto ForestEyes/ForestEyes/urban-deforestation-monitoring/workspace/data/output/pansharpening/geopolitic-RGBN/2024/3549904_2024_CBERS_TRUE_COLOR_CLIPPED.tif"
        if not os.path.exists(path_sat):
            print(f"[ERRO CRÍTICO] Imagem não encontrada em:\n-> {path_sat}")
            sys.exit(1)

    print("=" * 115)
    print(f"🧩 GERANDO SEGMENTAÇÃO SLIC DIRETA (IMAGEM: CBERS4A_WPM20214220240930_TRUE_COLOR_2024.tif)")
    print(f"📁 DESTINO: {segmentation_dir}")
    print("=" * 115)

    with rasterio.open(path_sat) as src_sat:
        meta_sat = src_sat.meta.copy()
        height = src_sat.height
        width = src_sat.width
        sat_data_full = src_sat.read() # Lê todas as bandas

    # Pré-normalização global de contraste para preservar as cores originais
    rgb_full_normalized = np.zeros((3, height, width), dtype=np.uint8)
    for b_idx in range(min(3, sat_data_full.shape[0])):
        band_data = sat_data_full[b_idx].astype(np.float32)
        p2, p98 = np.percentile(band_data[band_data > 0], (2, 98)) if np.any(band_data > 0) else (0, 1)
        if p98 > p2:
            normalized = np.clip((band_data - p2) / (p98 - p2), 0, 1) * 255.0
        else:
            normalized = np.clip(band_data, 0, 255)
        rgb_full_normalized[b_idx] = normalized.astype(np.uint8)

    TILE_SIZE = 1500
    global_id_offset = 0

    meta_labels = meta_sat.copy()
    meta_labels.update({"dtype": rasterio.int32, "count": 1, "nodata": 0})

    meta_vis = meta_sat.copy()
    meta_vis.update({"dtype": rasterio.uint8, "count": 3, "nodata": 0, "photometric": "RGB"})

    saida_labels = os.path.join(segmentation_dir, "CBERS4A_segmentation_labels_2024.tif")
    saida_visual = os.path.join(segmentation_dir, "CBERS4A_segmentation_visual_2024.tif")

    print(f"Iniciando processamento em blocos (Tiling) de {TILE_SIZE}px...")

    with rasterio.open(saida_labels, "w", **meta_labels) as dst_labels:
        with rasterio.open(saida_visual, "w", **meta_vis) as dst_vis:
            
            n_rows = int(np.ceil(height / TILE_SIZE))
            n_cols = int(np.ceil(width / TILE_SIZE))
            total_blocos = n_rows * n_cols
            bloco_atual = 0

            for row in range(0, height, TILE_SIZE):
                for col in range(0, width, TILE_SIZE):
                    bloco_atual += 1
                    win_h = min(TILE_SIZE, height - row)
                    win_w = min(TILE_SIZE, width - col)
                    window = Window(col, row, win_w, win_h)
                    
                    rgb_tile = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                    for b in range(3):
                        rgb_tile[:, :, b] = rgb_full_normalized[b, row:row+win_h, col:col+win_w]

                    sat_img_tile = np.moveaxis(sat_data_full[:, row:row+win_h, col:col+win_w], 0, -1).astype(np.float32)
                    
                    print(f"[{bloco_atual:03d}/{total_blocos}] Segmentando trecho ({win_w}x{win_h}px)...")
                    
                    n_seg_tile = max(30, int(1500 * ((win_h * win_w) / (TILE_SIZE * TILE_SIZE))))

                    segments = slic(
                        sat_img_tile, 
                        n_segments=n_seg_tile, 
                        compactness=10.0, 
                        convert2lab=False, 
                        max_num_iter=5,
                        start_label=1
                    )
                    
                    max_id_local = segments.max()
                    if max_id_local > 0:
                        segments[segments > 0] += global_id_offset
                        global_id_offset += max_id_local
                    
                    # Desenha contornos em amarelo vibrante ([255, 255, 0]) para realce visual
                    borders = find_boundaries(segments, mode='outer')
                    rgb_tile[borders] = [255, 255, 0]
                    
                    dst_labels.write(segments.astype(np.int32), 1, window=window)
                    for b in range(3):
                        dst_vis.write(rgb_tile[:, :, b], b+1, window=window)

    print(f"\n[SUCESSO] Segmentação concluída e salva com sucesso!")
    print(f"📊 Raster de Rótulos: {saida_labels}")
    print(f"🖼️ Raster Visual com Contornos: {saida_visual}")

if __name__ == "__main__":
    main()