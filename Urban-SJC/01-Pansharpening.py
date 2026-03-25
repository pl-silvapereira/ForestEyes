import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
import numpy as np
from pathlib import Path

def executar_pansharpening_estavel():
    # Caminhos
    base_path = Path(r"C:\Users\PedroLuizdaSilvaPere\development\Projects\ForestEyes\Urban-SJC\CBERS-4A")
    origem = base_path / "ObtencaoINPE"
    saida = base_path / "SJC_Master_Cientifico_4Bandas_2m.tif"
    
    # Cena: 04/09/2024 - Órbita 201/142
    prefixo = "CBERS_4A_WPM_20240904_201_142_L4_BAND"
    f_pan = origem / f"{prefixo}0.tif"
    f_red = origem / f"{prefixo}3.tif"
    f_gre = origem / f"{prefixo}2.tif"
    f_blu = origem / f"{prefixo}1.tif"
    f_nir = origem / f"{prefixo}4.tif"

    crs_final = "EPSG:31983"

    print(f"Iniciando Pansharpening BigTIFF para: {prefixo}...")

    with rasterio.open(str(f_pan)) as src_pan:
        # Perfil simplificado e robusto para arquivos gigantes
        perfil = {
            'driver': 'GTiff',
            'height': src_pan.height,
            'width': src_pan.width,
            'count': 4,
            'dtype': 'uint16',
            'crs': crs_final,
            'transform': src_pan.transform,
            'nodata': 0,
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'BIGTIFF': 'YES'  # ESSENCIAL: Permite arquivos maiores que 4GB
        }

        with rasterio.open(str(f_red)) as src_r, \
             rasterio.open(str(f_gre)) as src_g, \
             rasterio.open(str(f_blu)) as src_b, \
             rasterio.open(str(f_nir)) as src_n:
            
            vrt_opts = {'resampling': Resampling.bilinear, 'crs': crs_final, 
                        'transform': src_pan.transform, 'height': src_pan.height, 'width': src_pan.width}
            
            # Usamos context managers para garantir que os arquivos fechem corretamente se houver erro
            with WarpedVRT(src_r, **vrt_opts) as vr, \
                 WarpedVRT(src_g, **vrt_opts) as vg, \
                 WarpedVRT(src_b, **vrt_opts) as vb, \
                 WarpedVRT(src_n, **vrt_opts) as vn:
                
                with rasterio.open(str(saida), 'w', **perfil) as dst:
                    print("Processando e gravando blocos (isso pode demorar alguns minutos)...")
                    for ij, window in dst.block_windows():
                        pan = src_pan.read(1, window=window).astype('float32')
                        r = vr.read(1, window=window).astype('float32')
                        g = vg.read(1, window=window).astype('float32')
                        b = vb.read(1, window=window).astype('float32')
                        n = vn.read(1, window=window).astype('float32')

                        # Algoritmo Brovey
                        denominador = (r + g + b) + 1.0
                        ratio = pan / denominador
                        
                        dst.write((r * ratio).astype('uint16'), window=window, indexes=1)
                        dst.write((g * ratio).astype('uint16'), window=window, indexes=2)
                        dst.write((b * ratio).astype('uint16'), window=window, indexes=3)
                        dst.write((n * ratio).astype('uint16'), window=window, indexes=4)

    print(f"\nSUCESSO! Imagem master gerada com suporte BigTIFF: {saida.name}")

if __name__ == "__main__":
    executar_pansharpening_estavel()