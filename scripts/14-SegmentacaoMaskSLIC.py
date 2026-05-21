import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from skimage.segmentation import slic
from dotenv import load_dotenv

def executar_maskslic_blocos():
    load_dotenv()
    ROOT = os.getenv('PROJECT_ROOT')
    if not ROOT:
        print("❌ Erro: Variável 'PROJECT_ROOT' não encontrada no .env.")
        return

    dir_mapbiomas = os.path.join(ROOT, 'data', 'MapBiomas')
    dir_output = os.path.join(ROOT, 'data', 'Output')

    imagem_sat_path = os.path.join(dir_output, "02_SJC_Recortado_MapBiomas.tif")
    saida_segments = os.path.join(dir_output, "03_SJC_Superpixels_MaskSLIC.tif")

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
    print("1/3 - Lendo metadados e alinhando máscara...")
    with rasterio.open(imagem_sat_path) as sat_src:
        meta_saida = sat_src.meta.copy()
        height = sat_src.height
        width = sat_src.width
        sat_transform = sat_src.transform
        sat_crs = sat_src.crs

    # Lê o MapBiomas e reprojeta para encaixar no tamanho do Satélite
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

    agropecuaria = [14, 15, 18, 19, 39, 20, 40, 62, 41, 36, 46, 47, 35, 48, 9, 21]
    infra_urbana = [24]
    agua_rocha = [26, 33, 31, 29] 
    ruido_descartadas = [27]
    ids_mascara = agropecuaria + infra_urbana + agua_rocha + ruido_descartadas

    mask_boolean = np.isin(mb_resampled, ids_mascara)
    
    # Limpa a matriz gigante da memória, manteremos só a booleana
    del mb_resampled 

    # -------------------------------------------------------------
    # 2. PREPARAÇÃO PARA O PROCESSAMENTO EM BLOCOS
    # -------------------------------------------------------------
    TILE_SIZE = 1500  # Processa em blocos de 1500x1500 pixels
    global_id_offset = 0

    meta_saida.update({
        "dtype": rasterio.int32,  
        "count": 1,
        "nodata": 0
    })

    print(f"2/3 - Iniciando segmentação SLIC dividida em blocos de {TILE_SIZE}px...")
    
    with rasterio.open(imagem_sat_path) as sat_src:
        with rasterio.open(saida_segments, "w", **meta_saida) as dst:
            
            # Calcula o número total de blocos apenas para controle visual
            n_rows = int(np.ceil(height / TILE_SIZE))
            n_cols = int(np.ceil(width / TILE_SIZE))
            total_blocos = n_rows * n_cols
            bloco_atual = 0

            # -------------------------------------------------------------
            # 3. LOOP DE CORTE E COSTURA (TILING)
            # -------------------------------------------------------------
            for row in range(0, height, TILE_SIZE):
                for col in range(0, width, TILE_SIZE):
                    bloco_atual += 1
                    
                    win_h = min(TILE_SIZE, height - row)
                    win_w = min(TILE_SIZE, width - col)
                    window = Window(col, row, win_w, win_h)
                    
                    # Puxa o recorte equivalente da máscara global
                    mask_tile = mask_boolean[row:row+win_h, col:col+win_w]
                    
                    # Se não houver nenhum pixel de interesse na máscara, pula direto
                    if not np.any(mask_tile):
                        print(f"[{bloco_atual:03d}/{total_blocos}] Pulo: Área de floresta pura")
                        continue
                        
                    print(f"[{bloco_atual:03d}/{total_blocos}] Processando superpixels...")
                    
                    # Lê APENAS o pequeno bloco da imagem do disco, sem sobrecarregar a RAM
                    sat_data = sat_src.read(window=window)
                    sat_img_tile = np.moveaxis(sat_data, 0, -1).astype(np.float32)
                    
                    # Normaliza brilho do bloco para o SLIC funcionar
                    for i in range(sat_img_tile.shape[2]):
                        band_max = np.percentile(sat_img_tile[:,:,i], 99)
                        if band_max > 0:
                            sat_img_tile[:,:,i] = np.clip((sat_img_tile[:,:,i] / band_max) * 255.0, 0, 255)
                    sat_img_tile = sat_img_tile.astype(np.uint8)
                    
                    # Ajusta a quantidade de segmentos proporcionalmente ao bloco para manter o tamanho do superpixel constante
                    n_seg_tile = max(50, int(2000 * ((win_h * win_w) / (TILE_SIZE * TILE_SIZE))))

                    # Roda o algoritmo instantaneamente no bloco pequeno
                    segments = slic(
                        sat_img_tile, 
                        n_segments=n_seg_tile, 
                        compactness=10.0, 
                        mask=mask_tile, 
                        convert2lab=False, 
                        max_num_iter=5,
                        start_label=1
                    )
                    
                    # Garante que as bordas da máscara ficaram devidamente com ID=0
                    segments[~mask_tile] = 0
                    
                    # Se superpixels foram criados, adicionamos um deslocamento contínuo aos IDs 
                    # para que o ID 1 do bloco 2 não se misture com o ID 1 do bloco 1.
                    max_id_local = segments.max()
                    if max_id_local > 0:
                        segments[segments > 0] += global_id_offset
                        global_id_offset += max_id_local
                        
                    # Grava o resultado do bloco diretamente no arquivo final
                    dst.write(segments.astype(np.int32), 1, window=window)

    print(f"\n3/3 - 🎉 Sucesso! Total de {global_id_offset} superpixels únicos gerados e salvos em:")
    print(saida_segments)

if __name__ == "__main__":
    executar_maskslic_blocos()