import os
import sys
import numpy as np
import rasterio
from rasterio.enums import Resampling, ColorInterp
from rasterio.vrt import WarpedVRT

# ==========================================
# 1. CONFIGURAÇÃO DE CAMINHOS AUTOMÁTICA
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

dir_downloads = os.path.join(ROOT, 'data', 'CBERS4A-WPM', 'Downloads')
dir_output = os.path.join(ROOT, 'data', 'Output')

banda1_blue = os.path.join(dir_downloads, "CBERS_4A_WPM_20230724_202_142_L4_BAND1.tif")
banda2_green = os.path.join(dir_downloads, "CBERS_4A_WPM_20230724_202_142_L4_BAND2.tif")
banda3_red = os.path.join(dir_downloads, "CBERS_4A_WPM_20230724_202_142_L4_BAND3.tif")
banda0_pan = os.path.join(dir_downloads, "CBERS_4A_WPM_20230724_202_142_L4_BAND0.tif")

saida_final = os.path.join(dir_output, "01_SJC_Pansharpened_Exato.tif")

def executar_pansharpening():
    print(f"Diretório Raiz: {ROOT}")
    print("🚀 Iniciando Pansharpening com Alinhamento Geográfico (VRT)...")
    
    with rasterio.open(banda0_pan) as pan_src:
        profile = pan_src.profile.copy()
        
        # Propriedades extraídas do QGIS, mas COM compressão inteligente
        profile.update({
            'count': 3,
            'dtype': rasterio.int16,
            'nodata': 0,
            'crs': 'EPSG:32723',
            'photometric': 'RGB',
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512,
            'compress': 'lzw',
            'bigtiff': 'yes'
        })
        
        # O PULO DO GATO: Configuração do Virtual Raster (VRT)
        # Isso força as bandas de 8m (RGB) a se esticarem para a grade de 2m (PAN) fisicamente
        vrt_options = {
            'resampling': Resampling.cubic,
            'crs': pan_src.crs,
            'transform': pan_src.transform,
            'height': pan_src.height,
            'width': pan_src.width,
        }
        
        # Abre as bandas RGB originais
        with rasterio.open(banda3_red) as b_red, \
             rasterio.open(banda2_green) as b_green, \
             rasterio.open(banda1_blue) as b_blue:
             
             # Aplica a lupa virtual (WarpedVRT) em cima de cada banda colorida
             with WarpedVRT(b_red, **vrt_options) as vrt_red, \
                  WarpedVRT(b_green, **vrt_options) as vrt_green, \
                  WarpedVRT(b_blue, **vrt_options) as vrt_blue, \
                  rasterio.open(saida_final, 'w', **profile) as dst:
                  
                  dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]
                  
                  windows = [w for _, w in dst.block_windows()]
                  total = len(windows)
                  
                  for i, window in enumerate(windows):
                      # Agora sim, ler a 'window' vai ler EXATAMENTE a mesma latitude/longitude em todas as imagens
                      pan = pan_src.read(1, window=window).astype(np.float32)
                      
                      multi = np.array([
                          vrt_red.read(1, window=window).astype(np.float32),
                          vrt_green.read(1, window=window).astype(np.float32),
                          vrt_blue.read(1, window=window).astype(np.float32)
                      ])
                      
                      soma = np.sum(multi, axis=0)
                      soma[soma == 0] = 1.0  
                      
                      fused = np.zeros_like(multi, dtype=np.int16)
                      for idx in range(3):
                          calc = (multi[idx] / soma) * pan
                          fused[idx] = np.clip(calc, -32768, 32767).astype(np.int16)
                          fused[idx][pan == 0] = 0
                          
                      dst.write(fused, window=window)
                      
                      if i % 50 == 0:
                          sys.stdout.write(f"\r  📊 Progresso da Fusão: {(i/total)*100:.1f}% ")
                          sys.stdout.flush()
                          
    print("\r  📊 Progresso da Fusão: 100.0% ")
    print(f"\n✅ Sucesso! O arquivo final foi salvo em: {saida_final}")

if __name__ == '__main__':
    executar_pansharpening()