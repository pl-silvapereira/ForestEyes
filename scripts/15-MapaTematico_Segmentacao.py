import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from skimage.segmentation import slic, find_boundaries
from dotenv import load_dotenv
import gc

def executar_mapa_segmentado_tematico():
    load_dotenv()
    ROOT = os.getenv('PROJECT_ROOT')
    if not ROOT:
        print("❌ Erro: Variável 'PROJECT_ROOT' não encontrada no .env.")
        return

    dir_mapbiomas = os.path.join(ROOT, 'data', 'MapBiomas')
    dir_output = os.path.join(ROOT, 'data', 'Output')

    # Arquivo de entrada (usado para calcular os superpixels)
    imagem_sat_path = os.path.join(dir_output, "02_SJC_Recortado_MapBiomas.tif")
    
    # Arquivo único de saída (Mapa Temático Segmentado)
    saida_visual = os.path.join(dir_output, "15_SJC_Mapa_Segmentado_Tematico.tif")

    if not os.path.exists(imagem_sat_path):
        print(f"❌ Erro: Arquivo {imagem_sat_path} não encontrado.")
        return

    # Busca dinâmica pelo arquivo base do MapBiomas
    mapbiomas_files = glob.glob(os.path.join(dir_mapbiomas, "*2021*coverage*10m*.tif"))
    if not mapbiomas_files:
        mapbiomas_files = glob.glob(os.path.join(dir_mapbiomas, "*coverage_10m*.tif"))
    if not mapbiomas_files:
        print("❌ Erro: Arquivo do MapBiomas não encontrado.")
        return
        
    mapbiomas_path = mapbiomas_files[0]

    # -------------------------------------------------------------
    # 1. METADADOS E ALINHAMENTO DO MAPBIOMAS
    # -------------------------------------------------------------
    print("1/3 - Lendo metadados e alinhando máscara do MapBiomas...")
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

    # Definição dos IDs das classes
    ids_floresta = [1, 3, 4, 5, 6, 49]
    ids_nao_floresta = [10, 11, 12, 32, 50, 13]
    # (O restante dos IDs será automaticamente tratado como a máscara preta)

    # -------------------------------------------------------------
    # 2. CONFIGURAÇÃO DE PROCESSAMENTO EM BLOCOS (TILING)
    # -------------------------------------------------------------
    TILE_SIZE = 1500  

    # O arquivo final será uma imagem RGB 8-bits
    meta_vis = meta_sat.copy()
    meta_vis.update({
        "dtype": rasterio.uint8, 
        "count": 3, 
        "nodata": None, 
        "photometric": "RGB"
    })

    print("2/3 - Iniciando cálculo de superpixels e geração do mapa temático...")
    
    with rasterio.open(imagem_sat_path) as sat_src:
        with rasterio.open(saida_visual, "w", **meta_vis) as dst_vis:
            
            n_rows = int(np.ceil(height / TILE_SIZE))
            n_cols = int(np.ceil(width / TILE_SIZE))
            total_blocos = n_rows * n_cols
            bloco_atual = 0

            # -------------------------------------------------------------
            # 3. LOOP DE SEGMENTAÇÃO E PINTURA
            # -------------------------------------------------------------
            for row in range(0, height, TILE_SIZE):
                for col in range(0, width, TILE_SIZE):
                    bloco_atual += 1
                    
                    win_h = min(TILE_SIZE, height - row)
                    win_w = min(TILE_SIZE, width - col)
                    window = Window(col, row, win_w, win_h)
                    
                    # Extrai o bloco correspondente da grade do MapBiomas
                    mb_tile = mb_aligned[row:row+win_h, col:col+win_w]
                    
                    # Cria as máscaras binárias locais
                    tile_floresta = np.isin(mb_tile, ids_floresta)
                    tile_nao_floresta = np.isin(mb_tile, ids_nao_floresta)
                    mask_segmentacao = tile_floresta | tile_nao_floresta
                    
                    # Cria a tela RGB base para a visualização (inicia toda preta)
                    rgb_out_tile = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                    
                    # Pinta o fundo com as cores temáticas (A imagem CBERS é descartada visualmente)
                    rgb_out_tile[tile_floresta] = [0, 255, 0]      # Verde
                    rgb_out_tile[tile_nao_floresta] = [255, 0, 0]  # Vermelho
                    # O resto da matriz continua [0,0,0] (Preto absoluto para a máscara)
                    
                    # Se houver área para segmentar, processamos o algoritmo MaskSLIC
                    if np.any(mask_segmentacao):
                        # Lê os dados reais do CBERS APENAS para alimentar a matemática do algoritmo
                        sat_data = sat_src.read(window=window)
                        sat_img_tile = np.moveaxis(sat_data, 0, -1).astype(np.float32)
                        
                        # Normalização rápida na RAM para o SLIC funcionar
                        num_bandas = min(3, sat_img_tile.shape[2])
                        for i in range(num_bandas):
                            band_max = np.percentile(sat_img_tile[:,:,i], 99)
                            if band_max > 0:
                                sat_img_tile[:,:,i] = np.clip((sat_img_tile[:,:,i] / band_max) * 255.0, 0, 255)
                        sat_img_tile = sat_img_tile.astype(np.uint8)
                        
                        n_seg_tile = max(50, int(2000 * ((win_h * win_w) / (TILE_SIZE * TILE_SIZE))))

                        # SLIC guiado pela imagem real, mas restrito à área Verde/Vermelha
                        segments = slic(
                            sat_img_tile, 
                            n_segments=n_seg_tile, 
                            compactness=10.0, 
                            mask=mask_segmentacao, 
                            convert2lab=False, 
                            max_num_iter=5,
                            start_label=1
                        )
                        
                        segments[~mask_segmentacao] = 0
                        
                        # Extrai os contornos e desenha em Amarelo por cima da base sólida
                        borders = find_boundaries(segments, mode='inner', background=0)
                        rgb_out_tile[borders] = [255, 255, 0]
                        
                        # Limpa os dados do CBERS da RAM após o cálculo
                        del sat_data, sat_img_tile
                    
                    # Escreve o bloco diretamente no arquivo .tif
                    for b in range(3):
                        dst_vis.write(rgb_out_tile[:, :, b], b+1, window=window)

    print(f"\n3/3 - 🎉 Processo finalizado com sucesso!")
    print(f"🖼️ Arquivo Temático Segmentado salvo em:\n{saida_visual}")

if __name__ == "__main__":
    executar_mapa_segmentado_tematico()