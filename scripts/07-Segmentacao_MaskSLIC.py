import os
import gc
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from skimage.segmentation import slic
from scipy.ndimage import binary_opening # Limpeza de ruído
from dotenv import load_dotenv

load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')
dir_out = os.path.join(ROOT, 'data', 'Output')

path_img_cbers = os.path.join(dir_out, "02_SJC_Recortado_MapBiomas.tif")
path_mudancas = os.path.join(dir_out, "04_SJC_Mapa_Mudancas_21_23.tif")
saida_segmentos = os.path.join(dir_out, "03_SJC_Segmentacao_SLIC_Mudancas.tif")

# Categorias foco (Vegetação)
ids_vegetacao = [3, 9, 11, 12, 36]

def executar_slic_vegetacao_relevante():
    print("🚀 Segmentação Adaptativa: Removendo ruído e focando em manchas de vegetação...")
    
    with rasterio.open(path_img_cbers) as src_rgb, rasterio.open(path_mudancas) as src_mud:
        with WarpedVRT(src_mud, crs=src_rgb.crs, transform=src_rgb.transform, 
                       width=src_rgb.width, height=src_rgb.height, 
                       resampling=Resampling.nearest) as vrt_mud:
            
            meta = src_rgb.meta.copy()
            meta.update(dtype=rasterio.int32, count=1, nodata=0)
            
            with rasterio.open(saida_segmentos, 'w', **meta) as dst:
                id_global_offset = 0
                tile_size = 1024 
                
                for j in range(0, src_rgb.height, tile_size):
                    for i in range(0, src_rgb.width, tile_size):
                        window = Window(i, j, min(tile_size, src_rgb.width - i), min(tile_size, src_rgb.height - j))
                        
                        mudanca_tile = vrt_mud.read(1, window=window)
                        mask_tile = np.isin(mudanca_tile, ids_vegetacao)
                        
                        # --- ESTRATÉGIA DE LIMPEZA ---
                        # Remove pequenos grupos de pixels (ruído) antes de segmentar
                        mask_tile = binary_opening(mask_tile, structure=np.ones((3,3)))
                        
                        if np.sum(mask_tile) < 100: continue # Mínimo de 100 pixels por bloco
                        
                        img_tile = src_rgb.read([1, 2, 3], window=window).astype(np.float32)
                        for b in range(3):
                            p2, p98 = np.percentile(img_tile[b], (2, 98))
                            img_tile[b] = np.clip((img_tile[b] - p2) / (p98 - p2 + 1e-5), 0, 1)
                        
                        img_slic = np.transpose(img_tile, (1, 2, 0))
                        
                        # Ajuste: compactness=1 para equilibrar cor e forma
                        segmentos_bloco = slic(img_slic, n_segments=50, compactness=1, 
                                               mask=mask_tile, start_label=1, enforce_connectivity=True)

                        if np.any(segmentos_bloco > 0):
                            segmentos_bloco[segmentos_bloco > 0] += id_global_offset
                            id_global_offset = np.max(segmentos_bloco)
                            dst.write(segmentos_bloco.astype(np.int32), 1, window=window)
                        
                        del img_tile, img_slic, segmentos_bloco
                        gc.collect()

    print(f"✅ Segmentação concluída. Total de objetos de vegetação: {id_global_offset}")

if __name__ == "__main__":
    executar_slic_vegetacao_relevante()