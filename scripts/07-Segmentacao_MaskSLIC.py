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

def executar_slic_alta_performance():
    print(f"🚀 Modo Alta Performance: Otimizando para {os.sys.platform} com 12.7GB RAM")
    
    with rasterio.open(path_img_cbers) as src_rgb, rasterio.open(path_mudancas) as src_mud:
        with WarpedVRT(src_mud, crs=src_rgb.crs, transform=src_rgb.transform, 
                       width=src_rgb.width, height=src_rgb.height, 
                       resampling=Resampling.nearest) as vrt_mud:
            
            meta = src_rgb.meta.copy()
            meta.update(dtype=rasterio.int32, count=1, nodata=0)
            
            with rasterio.open(saida_segmentos, 'w', **meta) as dst:
                id_global_offset = 0
                # Aumentamos o tile para 800px para processar mais área por vez, 
                # mas mantendo segurança para os 12GB.
                tile_size = 800 
                
                for j in range(0, src_rgb.height, tile_size):
                    for i in range(0, src_rgb.width, tile_size):
                        window = Window(i, j, min(tile_size, src_rgb.width - i), min(tile_size, src_rgb.height - j))
                        
                        mask_tile = vrt_mud.read(1, window=window) > 0
                        if not np.any(mask_tile):
                            continue
                        
                        # Lendo as bandas já convertendo para float32 (economiza uma conversão posterior)
                        img_tile = src_rgb.read([1, 2, 3], window=window).astype(np.float32)
                        
                        # Normalização In-Place: sobrescreve a matriz atual em vez de criar uma nova
                        for b in range(3):
                            p2, p98 = np.percentile(img_tile[b], (2, 98))
                            img_tile[b] = np.clip((img_tile[b] - p2) / (p98 - p2 + 1e-5), 0, 1)
                        
                        # Transposição e Segmentação
                        img_slic = np.transpose(img_tile, (1, 2, 0))
                        
                        # Reduzimos n_segments para 100 por bloco para ser mais rápido
                        segmentos_bloco = slic(
                            img_slic, 
                            n_segments=100, 
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
                        
                        # LIBERAÇÃO DE MEMÓRIA CRÍTICA
                        del img_tile, img_slic, segmentos_bloco
                        gc.collect()

    print(f"✅ Concluído! {id_global_offset} superpixels gerados com sucesso.")

if __name__ == "__main__":
    executar_slic_alta_performance()