import os
import gc
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from skimage.segmentation import slic
from dotenv import load_dotenv

load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')
dir_out = os.path.join(ROOT, 'data', 'Output')

path_img_cbers = os.path.join(dir_out, "02_SJC_Recortado_MapBiomas.tif")
path_mudancas = os.path.join(dir_out, "04_SJC_Mapa_Mudancas_21_23.tif")
saida_segmentos = os.path.join(dir_out, "03_SJC_Segmentacao_SLIC_Mudancas.tif")

def executar_slic_orientado_a_objeto():
    print("🚀 Iniciando Segmentação Adaptativa (Superpixels por Classe)...")
    
    with rasterio.open(path_img_cbers) as src_rgb, rasterio.open(path_mudancas) as src_mud:
        with WarpedVRT(src_mud, crs=src_rgb.crs, transform=src_rgb.transform, 
                       width=src_rgb.width, height=src_rgb.height, 
                       resampling=Resampling.nearest) as vrt_mud:
            
            meta = src_rgb.meta.copy()
            meta.update(dtype=rasterio.int32, count=1, nodata=0)
            
            with rasterio.open(saida_segmentos, 'w', **meta) as dst:
                id_global_offset = 0
                tile_size = 800 
                
                for j in range(0, src_rgb.height, tile_size):
                    for i in range(0, src_rgb.width, tile_size):
                        window = Window(i, j, min(tile_size, src_rgb.width - i), min(tile_size, src_rgb.height - j))
                        mask_tile = vrt_mud.read(1, window=window) > 0
                        
                        if np.sum(mask_tile) < 20: continue
                        
                        img_tile = src_rgb.read([1, 2, 3], window=window).astype(np.float32)
                        for b in range(3):
                            p2, p98 = np.percentile(img_tile[b], (2, 98))
                            img_tile[b] = np.clip((img_tile[b] - p2) / (p98 - p2 + 1e-5), 0, 1)
                        
                        img_slic = np.transpose(img_tile, (1, 2, 0))
                        
                        # --- O SEGREDO ESTÁ AQUI ---
                        # Compactness em 0.1 força o superpixel a seguir APENAS a cor/classe
                        # n_segments vira apenas uma sugestão inicial
                        segmentos_bloco = slic(
                            img_slic, 
                            n_segments=60, 
                            compactness=0.1, 
                            mask=mask_tile, 
                            start_label=1,
                            enforce_connectivity=True,
                            min_size_factor=0.5 # Ajuda a fundir pequenos pedaços da mesma classe
                        )

                        if np.any(segmentos_bloco > 0):
                            mask_valid = (segmentos_bloco > 0)
                            segmentos_bloco[mask_valid] += id_global_offset
                            id_global_offset = np.max(segmentos_bloco)
                            dst.write(segmentos_bloco.astype(np.int32), 1, window=window)
                        
                        del img_tile, img_slic, segmentos_bloco
                        gc.collect()

    print(f"✅ Segmentação Orientada a Objetos concluída. Total de objetos: {id_global_offset}")

if __name__ == "__main__":
    executar_slic_orientado_a_objeto()