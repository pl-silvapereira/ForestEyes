import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import center_of_mass
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
from dotenv import load_dotenv

def gerar_figuras_4_5():
    load_dotenv()
    ROOT = os.getenv('PROJECT_ROOT')
    dir_output = os.path.join(ROOT, 'data', 'Output')
    
    sat_path = os.path.join(dir_output, "02_SJC_Recortado_MapBiomas.tif")
    csv_path = os.path.join(dir_output, "19_SJC_Alvos_Campanha.csv")
    npy_path = os.path.join(dir_output, "17_SJC_Matriz_Segmentacao.npy")
    saida_figura = os.path.join(dir_output, "Fig4_5_Compositions.png")

    df = pd.read_csv(csv_path, sep=';')
    matriz_seg = np.load(npy_path)
    
    # Pega um segmento perfeito de floresta como exemplo
    sp_id = int(df.iloc[0]['ID_Segmento'])
    
    with rasterio.open(sat_path) as sat_src:
        cy, cx = center_of_mass(matriz_seg == sp_id)
        cy, cx = int(cy), int(cx)
        
        window = Window(cx - 128, cy - 128, 256, 256)
        sat_crop = sat_src.read(window=window).astype(np.float32)
        seg_crop = matriz_seg[cy-128:cy+128, cx-128:cx+128]
        borda = find_boundaries(seg_crop == sp_id, mode='thick')

        def norm(b):
            p2, p98 = np.percentile(b, (2, 98))
            return np.clip((b - p2) / (p98 - p2) * 255.0, 0, 255.0).astype(np.uint8)

        r, g, b_band = norm(sat_crop[0]), norm(sat_crop[1]), norm(sat_crop[2])
        rgb = np.dstack((r, g, b_band))
        
        # Assume banda 4 como NIR se existir para Falsa Cor e NDVI
        if sat_src.count >= 4:
            nir = sat_crop[3]
            falsa_cor = np.dstack((norm(nir), r, g))
            num, den = (nir - sat_crop[0]), (nir + sat_crop[0])
            ndvi = np.divide(num, den, out=np.zeros_like(num), where=den!=0)
            cmap = plt.get_cmap('RdYlGn')
            ndvi_img = (cmap((ndvi + 1) / 2.0)[:, :, :3] * 255).astype(np.uint8)
        else:
            falsa_cor, ndvi_img = rgb.copy(), rgb.copy()

        # Aplica bordas
        for img in [rgb, falsa_cor, ndvi_img]:
            img[borda] = [0, 255, 0]

        # Composições simulando o Flip Mode
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(rgb); axes[0].set_title("RGB (True Color)"); axes[0].axis('off')
        axes[1].imshow(falsa_cor); axes[1].set_title("False Color (NIR-R-G)"); axes[1].axis('off')
        axes[2].imshow(ndvi_img); axes[2].set_title("NDVI Representation"); axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(saida_figura, dpi=300)
        print(f"Figuras 4/5 salvas em: {saida_figura}")

if __name__ == "__main__":
    gerar_figuras_4_5()