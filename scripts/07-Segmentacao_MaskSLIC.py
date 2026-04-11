import os
import gc
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from skimage.segmentation import slic
from skimage.util import img_as_float
from dotenv import load_dotenv

load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')
dir_out = os.path.join(ROOT, 'data', 'Output')

path_img_cbers = os.path.join(dir_out, "02_SJC_Recortado_MapBiomas.tif")
path_mudancas = os.path.join(dir_out, "04_SJC_Mapa_Mudancas_21_23.tif")
saida_segmentos = os.path.join(dir_out, "03_SJC_Segmentacao_SLIC_Mudancas.tif")

def executar_slic_extremo_baixo_consumo():
    print("🚀 Iniciando Segmentação Ultra-Leve (Modo Janelas 512px)...")
    
    with rasterio.open(path_img_cbers) as src_rgb, rasterio.open(path_mudancas) as src_mud:
        with WarpedVRT(src_mud, crs=src_rgb.crs, transform=src_rgb.transform, 
                       width=src_rgb.width, height=src_rgb.height, 
                       resampling=Resampling.nearest) as vrt_mud:
            
            meta = src_rgb.meta.copy()
            meta.update(dtype=rasterio.int32, count=1, nodata=0)
            
            with rasterio.open(saida_segmentos, 'w', **meta) as dst:
                id_global_offset = 0
                tile_size = 512 # Blocos pequenos para não estourar a RAM
                
                for j in range(0, src_rgb.height, tile_size):
                    for i in range(0, src_rgb.width, tile_size):
                        window = Window(i, j, min(tile_size, src_rgb.width - i), min(tile_size, src_rgb.height - j))
                        
                        mask_tile = vrt_mud.read(1, window=window) > 0
                        if not np.any(mask_tile):
                            continue
                        
                        # Lê e normaliza o bloco
                        img_tile = src_rgb.read([1, 2, 3], window=window)
                        img_tile = np.nan_to_num(img_tile)
                        
                        # Normalização simplificada para poupar CPU/RAM
                        img_float = np.transpose(img_tile, (1, 2, 0)).astype(np.float32)
                        for b in range(3):
                            max_val = np.max(img_float[:,:,b])
                            if max_val > 0: img_float[:,:,b] /= max_val
                        
                        # SLIC otimizado para o Colab
                        # compactness=10 mantém a pureza da classe seguindo as bordas
                        segmentos_bloco = slic(
                            img_float, 
                            n_segments=150, 
                            compactness=10, 
                            mask=mask_tile, 
                            start_label=1,
                            enforce_connectivity=True
                        )

                        if np.any(segmentos_bloco > 0):
                            mask_valid = (segmentos_bloco > 0)
                            segmentos_bloco[mask_valid] += id_global_offset
                            id_global_offset = np.max(segmentos_bloco)
                            dst.write(segmentos_bloco.astype(np.int32), 1, window=window)
                        
                        # Limpeza profunda de memória
                        del img_tile, img_float, segmentos_bloco
                        gc.collect()

    print(f"✅ Sucesso! {id_global_offset} superpixels gerados apenas nas áreas de mudança.")

if __name__ == "__main__":
    executar_slic_extremo_baixo_consumo()