import os
import gc
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
dir_zoo = os.path.join(ROOT, 'data', 'Zooniverse_Foco_Vegetacao')
os.makedirs(dir_zoo, exist_ok=True)

def gerar_campanha_alinhada():
    print("🛡️ Iniciando Script 11 (Alinhamento de Resolução 2m vs 10m)...")
    
    path_rgb = os.path.join(dir_out, "02_SJC_Recortado_MapBiomas.tif")
    path_seg = os.path.join(dir_out, "03_SJC_Segmentacao_SLIC_Mudancas.tif")
    path_mud = os.path.join(dir_out, "04_SJC_Mapa_Mudancas_21_23.tif")

    with rasterio.open(path_rgb) as src_rgb, \
         rasterio.open(path_seg) as src_seg, \
         rasterio.open(path_mud) as src_mud:
        
        # 1. ALINHAMENTO CRÍTICO: Reprojeta o mapa de mudanças para a resolução do CBERS (2m)
        # Isso resolve o IndexError garantindo que 'mask' e 'mapa_mudanca' tenham o mesmo shape
        with WarpedVRT(src_mud, crs=src_rgb.crs, transform=src_rgb.transform, 
                       width=src_rgb.width, height=src_rgb.height, 
                       resampling=Resampling.nearest) as vrt_mud:
            
            print("⏳ Lendo dados alinhados para a RAM...")
            img_full = src_rgb.read([1, 2, 3])
            segmentos_full = src_seg.read(1)
            mapa_mud_full = vrt_mud.read(1) # Agora ele tem o mesmo tamanho que 'segmentos_full'
            
        ids_brutos = np.unique(segmentos_full[segmentos_full > 0])
        print(f"📊 Total para processar: {len(ids_brutos)} superpixels.")
        
        area_min_pixels = 600 
        manifesto = []

        for idx, seg_id in enumerate(ids_brutos):
            mask = (segmentos_full == seg_id)
            
            # Agora a máscara funciona perfeitamente!
            if np.sum(mask) < area_min_pixels:
                continue

            classes_no_seg = mapa_mud_full[mask]
            classes_puras = np.unique(classes_no_seg[classes_no_seg > 0])
            
            if len(classes_puras) != 1:
                continue
            
            # Recorte
            coords = np.argwhere(mask)
            y_min, x_min = max(0, coords[:,0].min()-25), max(0, coords[:,1].min()-25)
            y_max, x_max = min(img_full.shape[1], coords[:,0].max()+25), min(img_full.shape[2], coords[:,1].max()+25)

            chip_data = img_full[:, y_min:y_max, x_min:x_max]
            
            # Normalização e Borda
            chip_norm = np.zeros((chip_data.shape[1], chip_data.shape[2], 3), dtype=np.uint8)
            for b in range(3):
                p2, p98 = np.percentile(chip_data[b], (2, 98))
                chip_norm[:,:,b] = np.uint8(np.clip((chip_data[b] - p2) / (p98 - p2 + 1e-5) * 255, 0, 255))

            mask_local = mask[y_min:y_max, x_min:x_max]
            bordas = find_boundaries(mask_local, mode='thick')
            chip_norm[bordas] = [255, 255, 0]
            
            img_name = f"SJC_VEG_{seg_id}.png"
            Image.fromarray(chip_norm).save(os.path.join(dir_zoo, img_name))
            manifesto.append({"image_name": img_name, "id": seg_id})

            if idx % 200 == 0:
                print(f"✅ {idx} processados...")
                gc.collect()

        pd.DataFrame(manifesto).to_csv(os.path.join(dir_zoo, "manifest.csv"), index=False)
        print(f"🏁 Finalizado! {len(manifesto)} imagens prontas para o Zooniverse.")

if __name__ == "__main__":
    gerar_campanha_alinhada()