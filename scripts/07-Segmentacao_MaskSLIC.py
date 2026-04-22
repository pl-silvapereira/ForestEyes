import os
import gc
import warnings
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from skimage.segmentation import slic
from scipy.ndimage import binary_opening
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')
dir_out = os.path.join(ROOT, 'data', 'Output')

path_rgb = os.path.join(dir_out, "02_SJC_Recortado_MapBiomas.tif")
path_mudancas = os.path.join(dir_out, "04_SJC_Mapa_Mudancas_21_23.tif")
path_sombras = os.path.join(dir_out, "05_SJC_Mascara_Sombras.tif")
saida_segmentos = os.path.join(dir_out, "03_SJC_Segmentacao_SLIC_Mudancas.tif")

def executar_slic_sem_sombras():
    print("🚀 Segmentação Adaptativa: Focando em mudanças e ignorando sombras...")
    
    with rasterio.open(path_rgb) as src_rgb, \
         rasterio.open(path_mudancas) as src_mud, \
         rasterio.open(path_sombras) as src_sombras:
        
        with WarpedVRT(src_mud, crs=src_rgb.crs, transform=src_rgb.transform, width=src_rgb.width, height=src_rgb.height, resampling=Resampling.nearest) as vrt_mud, \
             WarpedVRT(src_sombras, crs=src_rgb.crs, transform=src_rgb.transform, width=src_rgb.width, height=src_rgb.height, resampling=Resampling.nearest) as vrt_somb:
            
            meta = src_rgb.meta.copy()
            meta.update(dtype=rasterio.int32, count=1, nodata=0)
            
            with rasterio.open(saida_segmentos, 'w', **meta) as dst:
                id_global_offset = 0
                tile_size = 1024 
                
                for j in range(0, src_rgb.height, tile_size):
                    for i in range(0, src_rgb.width, tile_size):
                        window = Window(i, j, min(tile_size, src_rgb.width - i), min(tile_size, src_rgb.height - j))
                        
                        mudanca_tile = vrt_mud.read(1, window=window)
                        sombra_tile = vrt_somb.read(1, window=window)
                        
                        # A MÁGICA AQUI: Tem que ter mudado (>0) E não pode ser sombra (==0)
                        mask_tile = (mudanca_tile > 0) & (sombra_tile == 0)
                        mask_tile = binary_opening(mask_tile, structure=np.ones((3,3))) # Limpa ruído
                        
                        if np.sum(mask_tile) < 100: continue 
                        
                        img_tile = src_rgb.read([1, 2, 3], window=window).astype(np.float32)
                        for b in range(3):
                            p2, p98 = np.percentile(img_tile[b], (2, 98))
                            img_tile[b] = np.clip((img_tile[b] - p2) / (p98 - p2 + 1e-5), 0, 1)
                        
                        img_slic = np.transpose(img_tile, (1, 2, 0))
                        
                        segmentos_bloco = slic(img_slic, n_segments=50, compactness=0.5, 
                                               mask=mask_tile, start_label=1, enforce_connectivity=True)

                        if np.any(segmentos_bloco > 0):
                            segmentos_bloco[segmentos_bloco > 0] += id_global_offset
                            id_global_offset = np.max(segmentos_bloco)
                            dst.write(segmentos_bloco.astype(np.int32), 1, window=window)
                        
                        del img_tile, img_slic, segmentos_bloco
                        gc.collect()

    print(f"✅ Segmentação Limpa concluída. Total de objetos: {id_global_offset}")

if __name__ == "__main__":
    executar_slic_sem_sombras()