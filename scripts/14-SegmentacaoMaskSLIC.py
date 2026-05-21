import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from skimage.segmentation import slic, find_boundaries
from dotenv import load_dotenv
import gc

def executar_maskslic_corrigido():
    load_dotenv()
    ROOT = os.getenv('PROJECT_ROOT')
    if not ROOT:
        print("❌ Erro: Variável 'PROJECT_ROOT' não encontrada no .env.")
        return

    dir_mapbiomas = os.path.join(ROOT, 'data', 'MapBiomas')
    dir_output = os.path.join(ROOT, 'data', 'Output')

    imagem_sat_path = os.path.join(dir_output, "02_SJC_Recortado_MapBiomas.tif")
    
    # Arquivo 1: Os dados puros (IDs dos superpixels para o Machine Learning)
    saida_labels = os.path.join(dir_output, "03_SJC_Superpixels_Labels.tif")
    # Arquivo 2: A imagem visual (Satélite + Contorno Amarelo para você visualizar no QGIS)
    saida_visual = os.path.join(dir_output, "04_SJC_Vis_Superpixels.tif")

    if not os.path.exists(imagem_sat_path):
        print(f"❌ Erro: Arquivo {imagem_sat_path} não encontrado.")
        return

    mapbiomas_files = glob.glob(os.path.join(dir_mapbiomas, "*2021*coverage*10m*.tif"))
    if not mapbiomas_files:
        mapbiomas_files = glob.glob(os.path.join(dir_mapbiomas, "*coverage_10m*.tif"))
    mapbiomas_path = mapbiomas_files[0]

    # -------------------------------------------------------------
    # 1. METADADOS E REPROJEÇÃO DA MÁSCARA
    # -------------------------------------------------------------
    print("1/3 - Lendo metadados e alinhando máscara do MapBiomas...")
    with rasterio.open(imagem_sat_path) as sat_src:
        meta_sat = sat_src.meta.copy()
        height = sat_src.height
        width = sat_src.width
        sat_transform = sat_src.transform
        sat_crs = sat_src.crs

    with rasterio.open(mapbiomas_path) as mb_src:
        mb_resampled = np.zeros((height, width), dtype=np.uint8)
        reproject(
            source=rasterio.band(mb_src, 1),
            destination=mb_resampled,
            src_transform=mb_src.transform,
            src_crs=mb_src.crs,
            dst_transform=sat_transform,
            dst_crs=sat_crs,
            resampling=Resampling.nearest
        )

    # Classes que queremos DESCONSIDERAR (Máscara de exclusão)
    agropecuaria = [14, 15, 18, 19, 39, 20, 40, 62, 41, 36, 46, 47, 35, 48, 9, 21]
    infra_urbana = [24]
    agua_rocha = [26, 33, 31, 29] 
    ruido_descartadas = [27]
    ids_mascara_ignorar = agropecuaria + infra_urbana + agua_rocha + ruido_descartadas

    # INVERSÃO DA LÓGICA: Queremos segmentar tudo que NÃO faz parte da máscara (ou seja, Floresta)
    mask_floresta = ~np.isin(mb_resampled, ids_mascara_ignorar)
    
    del mb_resampled 
    gc.collect()

    # -------------------------------------------------------------
    # 2. PREPARAÇÃO DOS ARQUIVOS DE SAÍDA
    # -------------------------------------------------------------
    TILE_SIZE = 1500  
    global_id_offset = 0

    # Metadados para o raster de labels (matriz matemática de IDs)
    meta_labels = meta_sat.copy()
    meta_labels.update({"dtype": rasterio.int32, "count": 1, "nodata": 0})

    # Metadados para o raster visual (imagem colorida 8-bits RGB)
    meta_vis = meta_sat.copy()
    meta_vis.update({"dtype": rasterio.uint8, "count": 3, "nodata": 0, "photometric": "RGB"})

    print(f"2/3 - Iniciando segmentação SLIC dividida em blocos de {TILE_SIZE}px...")
    
    with rasterio.open(imagem_sat_path) as sat_src:
        with rasterio.open(saida_labels, "w", **meta_labels) as dst_labels:
            with rasterio.open(saida_visual, "w", **meta_vis) as dst_vis:
                
                n_rows = int(np.ceil(height / TILE_SIZE))
                n_cols = int(np.ceil(width / TILE_SIZE))
                total_blocos = n_rows * n_cols
                bloco_atual = 0

                # -------------------------------------------------------------
                # 3. LOOP DE PROCESSAMENTO (TILING)
                # -------------------------------------------------------------
                for row in range(0, height, TILE_SIZE):
                    for col in range(0, width, TILE_SIZE):
                        bloco_atual += 1
                        
                        win_h = min(TILE_SIZE, height - row)
                        win_w = min(TILE_SIZE, width - col)
                        window = Window(col, row, win_w, win_h)
                        
                        mask_tile = mask_floresta[row:row+win_h, col:col+win_w]
                        
                        # Lê o bloco da imagem do satélite
                        sat_data = sat_src.read(window=window)
                        sat_img_tile = np.moveaxis(sat_data, 0, -1).astype(np.float32)
                        
                        # Extrai as bandas RGB (3 primeiras bandas) e normaliza para 8-bits (0-255)
                        rgb_tile = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                        num_bandas = min(3, sat_img_tile.shape[2])
                        
                        for i in range(num_bandas):
                            band_max = np.percentile(sat_img_tile[:,:,i], 99)
                            if band_max > 0:
                                rgb_tile[:,:,i] = np.clip((sat_img_tile[:,:,i] / band_max) * 255.0, 0, 255).astype(np.uint8)
                        
                        # Se não houver floresta no bloco, salva a imagem original sem bordas e pula o SLIC
                        if not np.any(mask_tile):
                            print(f"[{bloco_atual:03d}/{total_blocos}] Pulo: Área 100% Infraestrutura/Agropecuária.")
                            dst_labels.write(np.zeros((win_h, win_w), dtype=np.int32), 1, window=window)
                            for b in range(3):
                                dst_vis.write(rgb_tile[:, :, b], b+1, window=window)
                            continue
                            
                        print(f"[{bloco_atual:03d}/{total_blocos}] Segmentando Floresta e Desenhando Contornos...")
                        
                        # Ajusta a quantidade de segmentos proporcionalmente ao bloco
                        n_seg_tile = max(50, int(2000 * ((win_h * win_w) / (TILE_SIZE * TILE_SIZE))))

                        # Roda o MaskSLIC
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
                            
                        # DESENHA O CONTORNO AMARELO NA IMAGEM RGB
                        # find_boundaries localiza as bordas dos superpixels gerados
                        borders = find_boundaries(segments, mode='outer')
                        
                        # Onde houver borda, pinta de amarelo (R=255, G=255, B=0)
                        rgb_tile[borders] = [255, 255, 0]
                            
                        # Grava as matrizes no disco
                        dst_labels.write(segments.astype(np.int32), 1, window=window)
                        for b in range(3):
                            dst_vis.write(rgb_tile[:, :, b], b+1, window=window)

    print(f"\n3/3 - 🎉 Sucesso! Processo finalizado.")
    print(f"📊 Arquivo de Dados (Labels): {saida_labels}")
    print(f"🖼️ Arquivo Visual (Contornos): {saida_visual}")

if __name__ == "__main__":
    executar_maskslic_corrigido()