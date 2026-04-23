import os
import gc
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from skimage.segmentation import slic
from scipy.ndimage import binary_opening
from dotenv import load_dotenv

load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')
dir_out = os.path.join(ROOT, 'data', 'Output')

path_rgb = os.path.join(dir_out, "02_SJC_Recortado_MapBiomas.tif")
path_mud = os.path.join(dir_out, "04_SJC_Mapa_Mudancas_21_23.tif")
path_som = os.path.join(dir_out, "05_SJC_Mascara_Sombras.tif")
saida_seg = os.path.join(dir_out, "03_SJC_Segmentacao_SLIC_Mudancas.tif")

def executar_slic_blocos_rigidos():
    print("🚀 Gerando Blocos Quadrados Rígidos (Compactness=50)...")
    with rasterio.open(path_rgb) as src_rgb, \
         rasterio.open(path_mud) as src_mud, \
         rasterio.open(path_som) as src_som:
        
        with WarpedVRT(src_mud, crs=src_rgb.crs, transform=src_rgb.transform, width=src_rgb.width, height=src_rgb.height, resampling=Resampling.nearest) as v_mud, \
             WarpedVRT(src_som, crs=src_rgb.crs, transform=src_rgb.transform, width=src_rgb.width, height=src_rgb.height, resampling=Resampling.nearest) as v_som:
            
            meta = src_rgb.meta.copy()
            meta.update(dtype=rasterio.int32, count=1, nodata=0)
            
            with rasterio.open(saida_seg, 'w', **meta) as dst:
                id_offset = 0
                for j in range(0, src_rgb.height, 1024):
                    for i in range(0, src_rgb.width, 1024):
                        win = Window(i, j, min(1024, src_rgb.width - i), min(1024, src_rgb.height - j))
                        mask = (v_mud.read(1, window=win) > 0) & (v_som.read(1, window=win) == 0)
                        mask = binary_opening(mask, structure=np.ones((3,3)))
                        
                        if np.sum(mask) < 50: continue
                        
                        img = src_rgb.read([1, 2, 3], window=win).astype(np.float32)
                        for b in range(3):
                            p2, p98 = np.percentile(img[b], (2, 98))
                            img[b] = np.clip((img[b]-p2)/(p98-p2+1e-5), 0, 1)
                        
                        # Compactness=50 e n_segments menor para criar blocos maiores e quadrados
                        seg = slic(np.transpose(img, (1, 2, 0)), n_segments=100, 
                                   compactness=50.0, mask=mask, start_label=1, 
                                   enforce_connectivity=True)

                        if np.any(seg > 0):
                            seg[seg > 0] += id_offset
                            id_offset = np.max(seg)
                            dst.write(seg.astype(np.int32), 1, window=win)
                        gc.collect()
    print(f"✅ Segmentação em blocos concluída: {id_offset} objetos.")

if __name__ == "__main__":
    executar_slic_blocos_rigidos()