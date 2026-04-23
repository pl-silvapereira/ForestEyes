import os
import gc  # <--- CORREÇÃO AQUI
import pandas as pd
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from PIL import Image
from skimage.segmentation import find_boundaries
from dotenv import load_dotenv

# --- CONFIGURAÇÃO ---
load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')
dir_out = os.path.join(ROOT, 'data', 'Output')
# Nova pasta para a campanha refinada
dir_zoo = os.path.join(ROOT, 'data', 'Zooniverse_Campanha_Zoom_Foco')
os.makedirs(dir_zoo, exist_ok=True)

def gerar_campanha_zoom_foco_unico():
    print("🧠 Gerando imagens 2048x2048 com Zoom Máximo e Foco em Ponto Único...")
    
    path_rgb = os.path.join(dir_out, "02_SJC_Recortado_MapBiomas.tif")
    path_seg = os.path.join(dir_out, "03_SJC_Segmentacao_SLIC_Mudancas.tif")
    path_mud = os.path.join(dir_out, "04_SJC_Mapa_Mudancas_21_23.tif")

    with rasterio.open(path_rgb) as src_rgb, \
         rasterio.open(path_seg) as src_seg, \
         rasterio.open(path_mud) as src_mud:
        
        with WarpedVRT(src_mud, crs=src_rgb.crs, transform=src_rgb.transform, 
                       width=src_rgb.width, height=src_rgb.height, 
                       resampling=Resampling.nearest) as vrt_mud:
            
            # Carrega dados alinhados para a RAM
            img_full = src_rgb.read([1, 2, 3])
            segmentos_full = src_seg.read(1)
            mapa_mud_full = vrt_mud.read(1)
            
        ids_brutos = np.unique(segmentos_full[segmentos_full > 0])
        total_ids = len(ids_brutos)
        print(f"📦 Analisando {total_ids} superpixels candidatos...")

        manifesto = []
        res_final = 2048 

        for idx, seg_id in enumerate(ids_brutos):
            mask_full = (segmentos_full == seg_id)
            
            # --- GARANTIA DE CLASSE ÚNICA ---
            classes_no_segmento = mapa_mud_full[mask_full]
            classes_unicas = np.unique(classes_no_segmento[classes_no_segmento > 0])
            
            if len(classes_unicas) != 1:
                continue 
            
            classe_id = int(classes_unicas[0])

            # --- ESTRATÉGIA DE ZOOM MÁXIMO E SIMETRIA QUADRADA ---
            coords = np.argwhere(mask_full)
            cy, cx = coords.mean(axis=0).astype(int) 
            
            y_min, x_min = coords[:,0].min(), coords[:,1].min()
            y_max, x_max = coords[:,0].max(), coords[:,1].max()
            
            obj_h = y_max - y_min
            obj_w = x_max - x_min
            
            # Margem para contexto e simetria
            tam_recorte = max(obj_h, obj_w) + 20 
            
            metade = tam_recorte // 2
            y1, y2 = max(0, cy - metade), min(img_full.shape[1], cy + metade)
            x1, x2 = max(0, cx - metade), min(img_full.shape[2], cx + metade)
            
            dif_y = tam_recorte - (y2 - y1)
            dif_x = tam_recorte - (x2 - x1)
            
            if dif_y > 0:
                y1 = max(0, y1 - dif_y // 2)
                y2 = min(img_full.shape[1], y2 + (dif_y - (cy - metade - y1)))
            if dif_x > 0:
                x1 = max(0, x1 - dif_x // 2)
                x2 = min(img_full.shape[2], x2 + (dif_x - (cx - metade - x1)))

            # --- PROCESSAMENTO DO RECORTE ---
            chip = img_full[:, y1:y2, x1:x2]
            
            # Ignora se, mesmo após compensação, a imagem estiver cortada na borda extrema da cidade
            if chip.shape[1] == 0 or chip.shape[2] == 0:
                continue

            chip_norm = np.zeros((chip.shape[1], chip.shape[2], 3), dtype=np.uint8)
            for b in range(3):
                p2, p98 = np.percentile(chip[b], (2, 98))
                chip_norm[:,:,b] = np.uint8(np.clip((chip[b] - p2) / (p98 - p2 + 1e-5) * 255, 0, 255))

            # --- MARCAÇÃO ÚNICA ---
            mask_local = mask_full[y1:y2, x1:x2]
            bordas = find_boundaries(mask_local, mode='thick')
            chip_norm[bordas] = [255, 255, 0] 
            
            # --- SALVAMENTO E REDIMENSIONAMENTO ---
            img_png = Image.fromarray(chip_norm)
            final_img = img_png.resize((res_final, res_final), Image.Resampling.LANCZOS)
            
            img_name = f"SJC_Task_ID_{seg_id}.png"
            final_img.save(os.path.join(dir_zoo, img_name))
            
            manifesto.append({
                "image_name": img_name,
                "id_segmento": seg_id,
                "classe_mapbiomas": classe_id
            })

            # Feedback e limpeza de memória
            if idx % 100 == 0:
                print(f"✅ Processado: {idx}/{total_ids}...")
                gc.collect() 

        pd.DataFrame(manifesto).to_csv(os.path.join(dir_zoo, "manifest.csv"), index=False)
        print(f"✅ Campanha refinada finalizada com {len(manifesto)} tarefas quadradas de foco único.")

if __name__ == "__main__":
    gerar_campanha_zoom_foco_unico()