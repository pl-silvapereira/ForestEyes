import os
import glob
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import numpy as np

# --- CONFIGURAÇÃO ---
download_dir = '../data/CBERS4A-WPM/Downloads/'
mapbiomas_dir = '../data/MapBiomas/'
output_dir = '../data/Output/'
output_name = 'SJC_CBERS4A_WPM_Final_05m.tif'

def realizar_pansharpening_definitivo():
    bandas = {i: glob.glob(os.path.join(download_dir, f"*BAND{i}.tif"))[0] for i in range(4)}
    path_mask = glob.glob(os.path.join(mapbiomas_dir, "*coverage*.tif"))[0]
    
    print("🚀 Iniciando Fusão (Pan-sharpening)...")

    with rasterio.open(bandas[0]) as pan_src:
        profile = pan_src.profile.copy()
        pan_crs = pan_src.crs
        pan_transform = pan_src.transform
        total_rows = pan_src.height

        profile.update({
            'count': 3, 'dtype': rasterio.uint8, 'nodata': 0,
            'compress': 'lzw', 'tiled': True, 'blockxsize': 512, # Aumentado para 512 para ser mais rápido
            'blockysize': 512, 'interleave': 'pixel'
        })

        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        with rasterio.open(path_mask) as mask_src:
            mask_crs = mask_src.crs
            
            with rasterio.open(os.path.join(output_dir, output_name), 'w', **profile) as dst:
                windows = list(dst.block_windows())
                total_blocks = len(windows)
                
                for i, (_, window) in enumerate(windows):
                    pan_block = pan_src.read(1, window=window)
                    
                    # Alinhamento Geográfico
                    bounds_utm = rasterio.windows.bounds(window, pan_transform)
                    bounds_mask = transform_bounds(pan_crs, mask_crs, *bounds_utm)
                    mask_window = from_bounds(*bounds_mask, transform=mask_src.transform)
                    
                    mask_block = mask_src.read(
                        1, window=mask_window, out_shape=pan_block.shape, 
                        resampling=Resampling.nearest, boundless=True, fill_value=0
                    )

                    # Bandas RGB
                    rgb_stack = []
                    for b_key in [3, 2, 1]: 
                        with rasterio.open(bandas[b_key]) as src_banda:
                            data_block = src_banda.read(
                                1, window=window, out_shape=pan_block.shape, 
                                resampling=Resampling.bilinear, boundless=True, fill_value=0
                            )
                            
                            pan_f = pan_block.astype(np.float32)
                            mean_pan = pan_f.mean() if pan_f.mean() > 1 else 1.0
                            fusaun = (data_block.astype(np.float32) * pan_f) / mean_pan
                            
                            # Contraste local para visibilidade
                            p_min, p_max = np.percentile(fusaun, (2, 98))
                            fusaun = np.clip((fusaun - p_min) * (255.0 / (p_max - p_min + 1e-6)), 0, 255)
                            
                            # Máscara Geopolítica
                            fusaun[mask_block == 0] = 0
                            rgb_stack.append(fusaun.astype(np.uint8))
                    
                    # Escreve as 3 bandas de uma vez no bloco
                    for b_idx, data in enumerate(rgb_stack, 1):
                        dst.write(data, b_idx, window=window)
                    
                    # Feedback de progresso a cada 100 blocos
                    if i % 100 == 0:
                        percentual = (i / total_blocks) * 100
                        print(f"🛰️ Progresso: {percentual:.2f}% | Bloco {i}/{total_blocks} | Linha Atual: {window.row_off}", end='\r')

    print(f"\n\n✅ Concluído! Imagem de SJC gerada: {output_name}")

if __name__ == "__main__":
    realizar_pansharpening_definitivo()