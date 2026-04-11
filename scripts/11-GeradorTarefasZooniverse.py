import os
import pandas as pd
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')
dir_out = os.path.join(ROOT, 'data', 'Output')
dir_zoo = os.path.join(ROOT, 'data', 'Zooniverse_Hotspots')

if not os.path.exists(dir_zoo): os.makedirs(dir_zoo)

def exportar_hotspots_zooniverse():
    path_rgb = os.path.join(dir_out, "02_SJC_Recortado_MapBiomas.tif")
    path_seg = os.path.join(dir_out, "03_SJC_Segmentacao_SLIC_Total.tif")
    path_mud = os.path.join(dir_out, "04_SJC_Mapa_Mudancas_21_23.tif")

    print("🎯 Filtrando apenas mudanças para a campanha...")

    with rasterio.open(path_rgb) as src_rgb, \
         rasterio.open(path_seg) as src_seg, \
         rasterio.open(path_mud) as src_mud:
        
        img_data = src_rgb.read([1, 2, 3])
        # Normalização básica para PNG
        img_rgb = np.array([np.clip((b - np.percentile(b, 2)) / (np.percentile(b, 98) - np.percentile(b, 2)) * 255, 0, 255) for b in img_data]).astype(np.uint8)
        
        segmentos = src_seg.read(1)
        
        with WarpedVRT(src_mud, crs=src_rgb.crs, transform=src_rgb.transform, 
                       width=src_rgb.width, height=src_rgb.height, resampling=Resampling.nearest) as vrt:
            mudancas = vrt.read(1)

        # IDs de segmentos que tocam a mudança
        ids_hotspots = np.unique(segmentos[mudancas > 0])
        ids_hotspots = ids_hotspots[ids_hotspots > 0]
        
        print(f"🔥 Reduzindo de milhares para apenas {len(ids_hotspots)} imagens de mudança.")

        manifesto = []
        for seg_id in ids_hotspots:
            mask = (segmentos == seg_id)
            coords = np.argwhere(mask)
            if coords.size == 0: continue
            
            y_min, x_min = coords.min(axis=0) - 25
            y_max, x_max = coords.max(axis=0) + 25
            
            y_min, x_min = max(0, y_min), max(0, x_min)
            y_max, x_max = min(img_rgb.shape[1], y_max), min(img_rgb.shape[2], x_max)

            chip = img_rgb[:, y_min:y_max, x_min:x_max]
            img_name = f"SJC_Change_ID_{seg_id}.png"
            Image.fromarray(np.transpose(chip, (1, 2, 0))).save(os.path.join(dir_zoo, img_name))
            
            manifesto.append({"image_name": img_name, "segment_id": seg_id})

        # Salva o manifesto para o upload no Zooniverse
        pd.DataFrame(manifesto).to_csv(os.path.join(dir_zoo, "manifest.csv"), index=False)
        print(f"✅ Campanha pronta em {dir_zoo} com {len(manifesto)} imagens.")

if __name__ == "__main__":
    exportar_hotspots_zooniverse()