import os
import sys
import rasterio
from rasterio.windows import Window
from skimage.segmentation import slic, find_boundaries
import numpy as np
from dotenv import load_dotenv

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 11-segmentarDelta.py <code_muni> <ano_inicio> <ano_fim>")
        sys.exit(1)

    code_muni = int(sys.argv[1])
    ano_inicio = str(sys.argv[2])
    ano_fim = str(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    segmentation_dir = os.path.join(project_root, "data", "output", "mask", "segmentation")
    os.makedirs(segmentation_dir, exist_ok=True)

    path_sat_original = os.path.join(
        project_root, "data", "output", "pansharpening", "geopolitic-RGBN", 
        ano_fim, f"{code_muni}_{ano_fim}_CBERS_TRUE_COLOR_CLIPPED.tif"
    )

    path_mask_black = os.path.join(
        project_root, "data", "output", "mask", "black", 
        f"{code_muni}_masked_black_transparent_{ano_inicio}_vs_{ano_fim}.tif"
    )

    if not os.path.exists(path_sat_original) or not os.path.exists(path_mask_black):
        print(f"[ERRO CRÍTICO] Arquivos de entrada não encontrados.")
        sys.exit(1)

    print("=" * 115)
    print(f"🧩 GERANDO SEGMENTAÇÃO PRESERVANDO CORES ORIGINAIS DO CBERS (ANO: {ano_fim})")
    print(f"📍 MUNICÍPIO: {code_muni} | PERÍODO: {ano_inicio} vs {ano_fim}")
    print("=" * 115)

    with rasterio.open(path_sat_original) as src_sat:
        meta_sat = src_sat.meta.copy()
        height = src_sat.height
        width = src_sat.width
        sat_data_full = src_sat.read() # Lê todas as bandas na íntegra

    with rasterio.open(path_mask_black) as src_mask:
        mask_bands = src_mask.read()
        if mask_bands.shape[0] >= 4:
            alpha_channel = mask_bands[3]
            mask_mudanca = (alpha_channel == 0)
        else:
            r_m, g_m, b_m = mask_bands[0], mask_bands[1], mask_bands[2]
            mask_mudanca = (r_m > 0) | (g_m > 0) | (b_m > 0)

    saida_labels = os.path.join(segmentation_dir, f"{code_muni}_segmentation_labels_{ano_inicio}_vs_{ano_fim}.tif")
    saida_visual = os.path.join(segmentation_dir, f"{code_muni}_segmentation_visual_{ano_inicio}_vs_{ano_fim}.tif")

    # Pré-calcula globalmente os limites de contraste para o RGB original do CBERS não perder a cor
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

    print(f"Iniciando segmentação SLIC com blocos de {TILE_SIZE}px...")

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
                    
                    mask_tile = mask_mudanca[row:row+win_h, col:col+win_w]
                    
                    # Extrai o tile já normalizado globalmente (preservando cor original idêntica ao clipped)
                    rgb_tile = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                    for b in range(3):
                        rgb_tile[:, :, b] = rgb_full_normalized[b, row:row+win_h, col:col+win_w]

                    # Extrai o tile float para o SLIC
                    sat_img_tile = np.moveaxis(sat_data_full[:, row:row+win_h, col:col+win_w], 0, -1).astype(np.float32)

                    if not np.any(mask_tile):
                        dst_labels.write(np.zeros((win_h, win_w), dtype=np.int32), 1, window=window)
                        for b in range(3):
                            dst_vis.write(rgb_tile[:, :, b], b+1, window=window)
                        continue
                    
                    print(f"[{bloco_atual:03d}/{total_blocos}] Segmentando trechos de mudança...")
                    
                    n_seg_tile = max(30, int(1500 * ((win_h * win_w) / (TILE_SIZE * TILE_SIZE))))

                    segments = slic(
                        sat_img_tile, 
                        n_segments=n_seg_tile, 
                        compactness=10.0, 
                        mask=mask_tile, 
                        convert2lab=False, 
                        max_num_iter=5,
                        start_label=1
                    )
                    
                    segments[~mask_tile] = 0
                    
                    max_id_local = segments.max()
                    if max_id_local > 0:
                        segments[segments > 0] += global_id_offset
                        global_id_offset += max_id_local
                    
                    # Desenha contornos em branco (ou amarelo) estritamente onde houve mudança
                    borders = find_boundaries(segments, mode='outer')
                    # Definido como Branco (255, 255, 255) para destacar bem sobre a floresta/satélite
                    rgb_tile[borders] = [255, 255, 255]
                    
                    dst_labels.write(segments.astype(np.int32), 1, window=window)
                    for b in range(3):
                        dst_vis.write(rgb_tile[:, :, b], b+1, window=window)

    print(f"\n[SUCESSO] Segmentação concluída com cores preservadas!")
    print(f"📊 Labels: {saida_labels}")
    print(f"🖼️ Visual (Clipped original + Segmentação): {saida_visual}")

if __name__ == "__main__":
    main()