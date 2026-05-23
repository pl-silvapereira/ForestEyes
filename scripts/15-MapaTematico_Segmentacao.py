import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from skimage.segmentation import slic, find_boundaries
from dotenv import load_dotenv
import gc

def executar_mapa_e_segmentacao():
    load_dotenv()
    ROOT = os.getenv('PROJECT_ROOT')
    if not ROOT:
        print("❌ Erro: Variável 'PROJECT_ROOT' não encontrada no .env.")
        return

    dir_mapbiomas = os.path.join(ROOT, 'data', 'MapBiomas')
    dir_output = os.path.join(ROOT, 'data', 'Output')

    # Entradas
    imagem_sat_path = os.path.join(dir_output, "02_SJC_Recortado_MapBiomas.tif")
    
    # Saídas
    saida_mapa_cores = os.path.join(dir_output, "15_SJC_Mapa_Cores.tif")
    saida_labels = os.path.join(dir_output, "15_SJC_Superpixels_Labels.tif")
    saida_visual = os.path.join(dir_output, "15_SJC_Vis_Superpixels.tif")

    if not os.path.exists(imagem_sat_path):
        print(f"❌ Erro: Arquivo {imagem_sat_path} não encontrado.")
        return

    mapbiomas_files = glob.glob(os.path.join(dir_mapbiomas, "*2021*coverage*10m*.tif"))
    if not mapbiomas_files:
        mapbiomas_files = glob.glob(os.path.join(dir_mapbiomas, "*coverage_10m*.tif"))
    mapbiomas_path = mapbiomas_files[0]

    # -------------------------------------------------------------
    # 1. ALINHAMENTO DO MAPBIOMAS
    # -------------------------------------------------------------
    print("1/4 - Lendo metadados e alinhando máscara do MapBiomas...")
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
    # 2. DEFINIÇÃO DE CLASSES E CORES
    # -------------------------------------------------------------
    print("2/4 - Gerando Mapa Temático (Verde, Vermelho, Preto)...")
    ids_floresta = [1, 3, 4, 5, 6, 49]
    ids_nao_floresta = [10, 11, 12, 32, 50, 13]
    # O restante será considerado Máscara (Preto)

    mask_floresta = np.isin(mb_aligned, ids_floresta)
    mask_nao_floresta = np.isin(mb_aligned, ids_nao_floresta)
    
    # A área que será segmentada (Verde + Vermelho)
    mask_segmentacao = mask_floresta | mask_nao_floresta

    # Criação do TIF RGB temático
    mapa_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    mapa_rgb[mask_floresta] = [0, 255, 0]       # Verde
    mapa_rgb[mask_nao_floresta] = [255, 0, 0]   # Vermelho
    # As áreas que não são nem floresta nem não-floresta permanecem [0,0,0] (Preto)

    meta_mapa = meta_sat.copy()
    meta_mapa.update({"dtype": rasterio.uint8, "count": 3, "nodata": None, "photometric": "RGB"})
    
    with rasterio.open(saida_mapa_cores, "w", **meta_mapa) as dst_mapa:
        for b in range(3):
            dst_mapa.write(mapa_rgb[:, :, b], b+1)

    del mapa_rgb
    gc.collect()

    # -------------------------------------------------------------
    # 3. SEGMENTAÇÃO MASKSLIC (PROCESSAMENTO EM BLOCOS)
    # -------------------------------------------------------------
    TILE_SIZE = 1500  
    global_id_offset = 0

    meta_labels = meta_sat.copy()
    meta_labels.update({"dtype": rasterio.int32, "count": 1, "nodata": 0})

    meta_vis = meta_sat.copy()
    meta_vis.update({"dtype": rasterio.uint8, "count": 3, "nodata": 0, "photometric": "RGB"})

    print("3/4 - Iniciando segmentação nas áreas Verdes e Vermelhas...")
    
    with rasterio.open(imagem_sat_path) as sat_src:
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
                        
                        # Recorte da máscara combinada (Verde + Vermelho)
                        mask_tile = mask_segmentacao[row:row+win_h, col:col+win_w]
                        
                        sat_data = sat_src.read(window=window)
                        sat_img_tile = np.moveaxis(sat_data, 0, -1).astype(np.float32)
                        
                        rgb_tile = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                        num_bandas = min(3, sat_img_tile.shape[2])
                        
                        for i in range(num_bandas):
                            band_max = np.percentile(sat_img_tile[:,:,i], 99)
                            if band_max > 0:
                                rgb_tile[:,:,i] = np.clip((sat_img_tile[:,:,i] / band_max) * 255.0, 0, 255).astype(np.uint8)
                        
                        # Se não houver nada para segmentar no bloco (tudo preto)
                        if not np.any(mask_tile):
                            dst_labels.write(np.zeros((win_h, win_w), dtype=np.int32), 1, window=window)
                            for b in range(3):
                                dst_vis.write(rgb_tile[:, :, b], b+1, window=window)
                            continue
                            
                        print(f"[{bloco_atual:03d}/{total_blocos}] Segmentando Floresta/Não-Floresta...")
                        
                        n_seg_tile = max(50, int(2000 * ((win_h * win_w) / (TILE_SIZE * TILE_SIZE))))

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
                            
                        borders = find_boundaries(segments, mode='outer')
                        rgb_tile[borders] = [255, 255, 0] # Contorno Amarelo
                            
                        dst_labels.write(segments.astype(np.int32), 1, window=window)
                        for b in range(3):
                            dst_vis.write(rgb_tile[:, :, b], b+1, window=window)

    print(f"\n4/4 - 🎉 Processo finalizado com sucesso!")
    print(f"🎨 Mapa de Cores (Verde/Vermelho/Preto): {saida_mapa_cores}")
    print(f"📊 Arquivo de Labels Segmentados: {saida_labels}")
    print(f"🖼️ Imagem CBERS com Contornos: {saida_visual}")

if __name__ == "__main__":
    executar_mapa_e_segmentacao()