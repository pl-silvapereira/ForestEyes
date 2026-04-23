import os
import pandas as pd
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from PIL import Image
from skimage.segmentation import find_boundaries
from dotenv import load_dotenv

load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')
dir_out = os.path.join(ROOT, 'data', 'Output')
dir_zoo = os.path.join(ROOT, 'data', 'Zooniverse_Amostra_30_70')
os.makedirs(dir_zoo, exist_ok=True)

def gerar_divisao_30_70():
    print("🧠 Filtrando Pureza e Dividindo Dataset (30/70)...")
    path_rgb = os.path.join(dir_out, "02_SJC_Recortado_MapBiomas.tif")
    path_seg = os.path.join(dir_out, "03_SJC_Segmentacao_SLIC_Mudancas.tif")
    path_mud = os.path.join(dir_out, "04_SJC_Mapa_Mudancas_21_23.tif")

    with rasterio.open(path_rgb) as s_rgb, rasterio.open(path_seg) as s_seg, rasterio.open(path_mud) as s_mud:
        with WarpedVRT(s_mud, crs=s_rgb.crs, transform=s_rgb.transform, width=s_rgb.width, height=s_rgb.height, resampling=Resampling.nearest) as v_mud:
            img = s_rgb.read([1, 2, 3])
            seg_full = s_seg.read(1)
            mud_full = v_mud.read(1)

    ids = np.unique(seg_full[seg_full > 0])
    dados = []
    for sid in ids:
        mask = (seg_full == sid)
        # TESTE DE PUREZA ABSOLUTA: Apenas uma classe de mudança no bloco
        classes = np.unique(mud_full[mask][mud_full[mask] > 0])
        if len(classes) == 1:
            dados.append({'id': sid, 'classe': int(classes[0]), 'area': np.sum(mask)})

    df = pd.DataFrame(dados)
    zoo_list, ml_list = [], []
    # Divisão 30/70 por classe para manter o equilíbrio
    for cl, group in df.groupby('classe'):
        # Ordenamos por área para pegar os 'melhores' (maiores) para o Zooniverse
        sorted_g = group.sort_values(by='area', ascending=False)
        split = int(len(sorted_g) * 0.3)
        zoo_list.append(sorted_g.iloc[:split])
        ml_list.append(sorted_g.iloc[split:])

    df_zoo = pd.concat(zoo_list)
    df_ml = pd.concat(ml_list)
    
    print(f"📊 Zooniverse: {len(df_zoo)} | Machine Learning: {len(df_ml)}")

    # Exportação das imagens (30%)
    manifesto = []
    for _, row in df_zoo.iterrows():
        m = (seg_full == row['id'])
        c = np.argwhere(m)
        y1, x1 = max(0, c[:,0].min()-30), max(0, c[:,1].min()-30)
        y2, x2 = min(img.shape[1], c[:,0].max()+30), min(img.shape[2], c[:,1].max()+30)
        
        chip = img[:, y1:y2, x1:x2]
        chip_norm = np.zeros((chip.shape[1], chip.shape[2], 3), dtype=np.uint8)
        for b in range(3):
            p2, p98 = np.percentile(chip[b], (2, 98))
            chip_norm[:,:,b] = np.uint8(np.clip((chip[b]-p2)/(p98-p2+1e-5)*255, 0, 255))
        
        chip_norm[find_boundaries(m[y1:y2, x1:x2], mode='thick')] = [255, 255, 0]
        name = f"ZOO_CL_{row['classe']}_ID_{row['id']}.png"
        Image.fromarray(chip_norm).save(os.path.join(dir_zoo, name))
        manifesto.append({"image_name": name, "id_segmento": row['id'], "classe": row['classe']})

    pd.DataFrame(manifesto).to_csv(os.path.join(dir_zoo, "manifest_zooniverse.csv"), index=False)
    df_ml.to_csv(os.path.join(dir_zoo, "dataset_ML_70.csv"), index=False)
    print("✅ Processo 30/70 finalizado.")

if __name__ == "__main__":
    gerar_divisao_30_70()