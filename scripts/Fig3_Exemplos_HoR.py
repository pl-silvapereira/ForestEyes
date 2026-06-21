import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import center_of_mass
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
from dotenv import load_dotenv

def gerar_figura_3():
    load_dotenv()
    ROOT = os.getenv('PROJECT_ROOT')
    dir_output = os.path.join(ROOT, 'data', 'Output')
    
    sat_path = os.path.join(dir_output, "02_SJC_Recortado_MapBiomas.tif")
    csv_path = os.path.join(dir_output, "17_SJC_Estatisticas_Superpixels.csv")
    npy_path = os.path.join(dir_output, "17_SJC_Matriz_Segmentacao.npy")
    saida_figura = os.path.join(dir_output, "Fig3_HoR_Examples.png")

    df = pd.read_csv(csv_path, sep=';')
    matriz_seg = np.load(npy_path)
    
    # Seleciona os 4 casos da Figura 3
    alvos = {
        "(a) Forest, HoR = 1": df[(df['Classe_Majoritaria'] == 'Floresta') & (df['Taxa_HoR'] == 100)].iloc[0]['ID_Segmento'],
        "(b) Forest, HoR ~ 0.7": df[(df['Classe_Majoritaria'] == 'Floresta') & (df['Taxa_HoR'] >= 70) & (df['Taxa_HoR'] <= 75)].iloc[0]['ID_Segmento'],
        "(c) Non-Forest, HoR = 1": df[(df['Classe_Majoritaria'] == 'Nao_Floresta') & (df['Taxa_HoR'] == 100)].iloc[0]['ID_Segmento'],
        "(d) Non-Forest, HoR ~ 0.7": df[(df['Classe_Majoritaria'] == 'Nao_Floresta') & (df['Taxa_HoR'] >= 70) & (df['Taxa_HoR'] <= 75)].iloc[0]['ID_Segmento']
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    
    with rasterio.open(sat_path) as sat_src:
        for i, (titulo, sp_id) in enumerate(alvos.items()):
            sp_id = int(sp_id)
            cy, cx = center_of_mass(matriz_seg == sp_id)
            cy, cx = int(cy), int(cx)
            
            window = Window(cx - 128, cy - 128, 256, 256)
            sat_crop = sat_src.read((1,2,3), window=window).astype(np.float32)
            seg_crop = matriz_seg[cy-128:cy+128, cx-128:cx+128]
            
            rgb = np.zeros((256, 256, 3), dtype=np.uint8)
            for b in range(3):
                p2, p98 = np.percentile(sat_crop[b], (2, 98))
                rgb[:,:,b] = np.clip((sat_crop[b] - p2) / (p98 - p2) * 255.0, 0, 255.0)
            
            borda = find_boundaries(seg_crop == sp_id, mode='thick')
            rgb[borda] = [0, 255, 0] # Marca o segmento
            
            axes[i].imshow(rgb)
            axes[i].set_title(titulo)
            axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(saida_figura, dpi=300)
    print(f"Figura 3 salva em: {saida_figura}")

if __name__ == "__main__":
    gerar_figura_3()