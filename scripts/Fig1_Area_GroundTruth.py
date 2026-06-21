import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
import matplotlib.pyplot as plt
from dotenv import load_dotenv

def gerar_figura_1():
    load_dotenv()
    ROOT = os.getenv('PROJECT_ROOT')
    dir_output = os.path.join(ROOT, 'data', 'Output')
    dir_mapbiomas = os.path.join(ROOT, 'data', 'MapBiomas')
    
    sat_path = os.path.join(dir_output, "02_SJC_Recortado_MapBiomas.tif")
    mapbiomas_path = glob.glob(os.path.join(dir_mapbiomas, "*coverage_10m*.tif"))[0]
    
    saida_figura = os.path.join(dir_output, "Fig1_CBERS_MapBiomas.png")

    with rasterio.open(sat_path) as sat_src:
        # Lê uma janela representativa de 2000x2000 para a figura não ficar pesada
        window = Window(1000, 1000, 2000, 2000)
        sat_crop = sat_src.read((1,2,3), window=window).astype(np.float32)
        sat_transform = sat_src.window_transform(window)
        sat_crs = sat_src.crs

    # Normaliza RGB
    sat_rgb = np.zeros((2000, 2000, 3), dtype=np.uint8)
    for b in range(3):
        p2, p98 = np.percentile(sat_crop[b], (2, 98))
        sat_rgb[:,:,b] = np.clip((sat_crop[b] - p2) / (p98 - p2) * 255.0, 0, 255.0)

    # Alinha MapBiomas
    with rasterio.open(mapbiomas_path) as mb_src:
        mb_crop = np.zeros((2000, 2000), dtype=np.uint8)
        reproject(
            source=rasterio.band(mb_src, 1), destination=mb_crop,
            src_transform=mb_src.transform, src_crs=mb_src.crs,
            dst_transform=sat_transform, dst_crs=sat_crs, resampling=Resampling.nearest
        )

    # Cria máscara binária para o plot (Floresta = Verde, Não Floresta = Vermelho)
    ids_floresta = [1, 3, 4, 5, 6, 49]
    ids_nao_floresta = [9, 10, 11, 12, 13, 29, 32, 50]
    
    mb_colorido = np.zeros((2000, 2000, 3), dtype=np.uint8)
    mb_colorido[np.isin(mb_crop, ids_floresta)] = [0, 255, 0]
    mb_colorido[np.isin(mb_crop, ids_nao_floresta)] = [255, 0, 0]

    # Plotagem
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(sat_rgb)
    axes[0].set_title("(a) CBERS-4A Study Area")
    axes[0].axis('off')
    
    axes[1].imshow(mb_colorido)
    axes[1].set_title("(b) MapBiomas Ground Truth")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(saida_figura, dpi=300)
    print(f"Figura 1 salva em: {saida_figura}")

if __name__ == "__main__":
    gerar_figura_1()