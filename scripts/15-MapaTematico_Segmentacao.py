import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from skimage.segmentation import slic, find_boundaries
from dotenv import load_dotenv
import gc
import time

def executar_mapa_tematico_blocos():
    load_dotenv()
    ROOT = os.getenv('PROJECT_ROOT')
    if not ROOT:
        print("❌ Erro: Variável 'PROJECT_ROOT' não encontrada no .env.")
        return

    dir_mapbiomas = os.path.join(ROOT, 'data', 'MapBiomas')
    dir_output = os.path.join(ROOT, 'data', 'Output')

    imagem_sat_path = os.path.join(dir_output, "02_SJC_Recortado_MapBiomas.tif")
    saida_visual = os.path.join(dir_output, "15_SJC_Mapa_Verde_Vermelho_Segmentado.tif")

    if not os.path.exists(imagem_sat_path):
        print(f"❌ Erro: Arquivo {imagem_sat_path} não encontrado.")
        return

    busca = glob.glob(os.path.join(dir_mapbiomas, "*2021*coverage*10m*.tif"))
    if not busca:
        busca = glob.glob(os.path.join(dir_mapbiomas, "*coverage_10m*.tif"))
    mapbiomas_path = busca[0]

    # -------------------------------------------------------------
    # 1. PREPARAÇÃO DA BASE
    # -------------------------------------------------------------
    print("1/3 - Preparando metadados e alinhando classes...")
    with rasterio.open(imagem_sat_path) as sat_src:
        meta_sat = sat_src.meta.copy()
        height, width = sat_src.height, sat_src.width
        sat_transform, sat_crs = sat_src.transform, sat_src.crs

    with rasterio.open(mapbiomas_path) as mb_src:
        mb_aligned = np.zeros((height, width), dtype=np.uint8)
        reproject(
            source=rasterio.band(mb_src, 1),
            destination=mb_aligned,
            src_transform=mb_src.transform,
            src_crs=mb_src.crs,
            dst_transform=sat_transform,
            dst_crs=sat_crs,
            resampling=Resampling.nearest
        )

    # -------------------------------------------------------------
    # 2. DEFINIÇÃO DAS CORES E ALVOS
    # -------------------------------------------------------------
    ids_floresta = [1, 3, 4, 5, 6, 49]
    ids_nao_floresta = [9, 10, 11, 12, 13, 29, 32, 50] 

    mask_floresta = np.isin(mb_aligned, ids_floresta)
    mask_nao_floresta = np.isin(mb_aligned, ids_nao_floresta)
    
    # --- CÁLCULO AUTOMÁTICO DE ALVO ---
    area_total_mascara = np.sum(mask_floresta | mask_nao_floresta)
    TAMANHO_DESEJADO_PX = 1000 
    ALVO_SUPERPIXELS = max(1, int(area_total_mascara / TAMANHO_DESEJADO_PX))
    
    print(f"-> Meta ajustada automaticamente: {ALVO_SUPERPIXELS} superpíxeis para atingir ~{TAMANHO_DESEJADO_PX}px cada.")
    
    del mb_aligned
    gc.collect()

    # -------------------------------------------------------------
    # 3. PROCESSAMENTO EM BLOCOS (À PROVA DE TRAVAMENTO)
    # -------------------------------------------------------------
    TILE_SIZE = 2000
    meta_vis = meta_sat.copy()
    meta_vis.update({"dtype": rasterio.uint8, "count": 3, "nodata": None, "photometric": "RGB"})

    print(f"2/3 - Iniciando segmentação (Processamento em Blocos)...")
    
    n_rows = int(np.ceil(height / TILE_SIZE))
    n_cols = int(np.ceil(width / TILE_SIZE))
    total_blocos = n_rows * n_cols
    bloco_atual = 0
    start_time = time.time()

    with rasterio.open(imagem_sat_path) as sat_src:
        with rasterio.open(saida_visual, "w", **meta_vis) as dst_vis:
            
            for row in range(0, height, TILE_SIZE):
                for col in range(0, width, TILE_SIZE):
                    bloco_atual += 1
                    
                    win_h = min(TILE_SIZE, height - row)
                    win_w = min(TILE_SIZE, width - col)
                    window = Window(col, row, win_w, win_h)
                    
                    tile_f = mask_floresta[row:row+win_h, col:col+win_w]
                    tile_nf = mask_nao_floresta[row:row+win_h, col:col+win_w]
                    tile_mask = tile_f | tile_nf
                    
                    # Prepara a tela base (Fundo Verde/Vermelho e Máscara Preta)
                    rgb_tile = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                    rgb_tile[tile_f] = [0, 255, 0]      
                    rgb_tile[tile_nf] = [255, 0, 0]     
                    
                    area_tile = np.sum(tile_mask)

                    if area_tile > 0:
                        # Lê apenas o bloco da imagem
                        patch_sat = sat_src.read((1,2,3), window=window)
                        patch_sat = np.moveaxis(patch_sat, 0, -1).astype(np.float32)
                        
                        for b in range(patch_sat.shape[2]):
                            p99 = np.percentile(patch_sat[:,:,b], 99)
                            if p99 > 0:
                                patch_sat[:,:,b] = np.clip((patch_sat[:,:,b] / p99) * 255.0, 0, 255.0)
                        patch_sat = patch_sat.astype(np.uint8)

                        # Distribui os superpíxeis proporcionalmente ao tamanho do bloco
                        n_seg_patch = max(1, int(ALVO_SUPERPIXELS * (area_tile / area_total_mascara)))

                        seg_patch = slic(
                            patch_sat, n_segments=n_seg_patch, compactness=10.0, 
                            mask=tile_mask, convert2lab=False, enforce_connectivity=False, 
                            max_num_iter=5, start_label=1
                        )
                        seg_patch[~tile_mask] = 0

                        # Desenha as bordas amarelas
                        borders = find_boundaries(seg_patch, mode='inner', background=0)
                        rgb_tile[borders] = [255, 255, 0]

                    # Grava direto no disco, limpando a memória RAM
                    for b in range(3):
                        dst_vis.write(rgb_tile[:, :, b], b+1, window=window)
                        
                    print(f"   Bloco [{bloco_atual:03d}/{total_blocos}] processado e salvo...")

    tempo_total = round(time.time() - start_time, 1)
    
    print(f"\n3/3 - 🎉 Processo finalizado com perfeição em {tempo_total} segundos!")
    print(f"Arquivo salvo em:\n{saida_visual}")

if __name__ == "__main__":
    executar_mapa_tematico_blocos()