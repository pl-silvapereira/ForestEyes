import os
import gc
import pandas as pd
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from PIL import Image
from skimage.segmentation import find_boundaries
from scipy.ndimage import find_objects # <--- A MÁGICA DA VELOCIDADE
from dotenv import load_dotenv

# --- CONFIGURAÇÃO DA CAMPANHA ---
load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')
dir_out = os.path.join(ROOT, 'data', 'Output')
dir_zoo = os.path.join(ROOT, 'data', 'Zooniverse_Campanha_Nitida')
os.makedirs(dir_zoo, exist_ok=True)

MAX_IMAGENS_ZOONIVERSE = 150

def gerar_campanha_foco_nitidez():
    print(f"🧠 Iniciando Geração (Versão Turbo): As {MAX_IMAGENS_ZOONIVERSE} maiores áreas...")
    
    path_rgb = os.path.join(dir_out, "02_SJC_Recortado_MapBiomas.tif")
    path_seg = os.path.join(dir_out, "03_SJC_Segmentacao_SLIC_Mudancas.tif")
    path_mud = os.path.join(dir_out, "04_SJC_Mapa_Mudancas_21_23.tif")

    with rasterio.open(path_rgb) as src_rgb, \
         rasterio.open(path_seg) as src_seg, \
         rasterio.open(path_mud) as src_mud:
        
        with WarpedVRT(src_mud, crs=src_rgb.crs, transform=src_rgb.transform, 
                       width=src_rgb.width, height=src_rgb.height, 
                       resampling=Resampling.nearest) as vrt_mud:
            
            img_full = src_rgb.read([1, 2, 3])
            segmentos_full = src_seg.read(1)
            mapa_mud_full = vrt_mud.read(1)
            
        print("📊 1/3: Mapeando regiões na memória (Isso leva poucos segundos agora)...")
        
        # --- A GRANDE OTIMIZAÇÃO: FIND_OBJECTS ---
        # Isso cria uma "caixa" (bounding box) para todos os IDs de uma vez só em C++ por baixo dos panos!
        caixas = find_objects(segmentos_full)
        
        dados_superpixels = []
        
        for seg_id, caixa in enumerate(caixas, start=1):
            if caixa is None: continue # Pula se o ID não existir
            
            # Olhamos APENAS para dentro da caixa, e não para a imagem inteira! (Super rápido)
            mask_local = (segmentos_full[caixa] == seg_id)
            area = np.sum(mask_local)
            
            if area < 100: continue 
            
            classes_no_segmento = mapa_mud_full[caixa][mask_local]
            classes_unicas = np.unique(classes_no_segmento[classes_no_segmento > 0])
            
            if len(classes_unicas) == 1:
                dados_superpixels.append({
                    'id': seg_id, 
                    'classe': int(classes_unicas[0]), 
                    'area': area,
                    'caixa': caixa # Guardamos a caixa para usar no Passo 3 e poupar mais tempo!
                })

        df_todos = pd.DataFrame(dados_superpixels)
        
        if df_todos.empty:
            print("❌ Nenhum superpixel atendeu aos critérios. Verifique as imagens base.")
            return

        num_classes = df_todos['classe'].nunique()
        limite_por_classe = MAX_IMAGENS_ZOONIVERSE // num_classes
        
        print(f"⚖️ 2/3: Selecionando as {limite_por_classe} MAIORES imagens de cada classe...")
        
        df_zooniverse = pd.DataFrame()
        df_machine_learning = pd.DataFrame()
        
        for classe, group in df_todos.groupby('classe'):
            group = group.sort_values(by='area', ascending=False)
            df_zooniverse = pd.concat([df_zooniverse, group.head(limite_por_classe)])
            df_machine_learning = pd.concat([df_machine_learning, group.iloc[limite_por_classe:]])
            
        print(f"🎯 ALVO: {len(df_zooniverse)} imagens Premium para humanos.")
        
        # Remove a coluna da 'caixa' do CSV do Machine Learning para não dar erro de formatação
        df_machine_learning_csv = df_machine_learning.drop(columns=['caixa'])
        df_machine_learning_csv.to_csv(os.path.join(dir_zoo, "dataset_Machine_Learning.csv"), index=False)

        print("🖼️ 3/3: Recortando pixels originais (Zero Desfoque)...")
        manifesto = []

        for idx, row in enumerate(df_zooniverse.itertuples()):
            seg_id = row.id
            classe_id = row.classe
            caixa = row.caixa # Usa a caixa calculada no Passo 1!
            
            # As coordenadas da caixa já nos dão o y_min, y_max, x_min, x_max muito mais rápido
            y_min, y_max = caixa[0].start, caixa[0].stop
            x_min, x_max = caixa[1].start, caixa[1].stop
            
            cy = (y_min + y_max) // 2
            cx = (x_min + x_max) // 2
            
            obj_h = y_max - y_min
            obj_w = x_max - x_min
            
            tam_recorte = max(obj_h, obj_w) + 150 
            tam_recorte = max(tam_recorte, 600) 
            
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

            chip = img_full[:, y1:y2, x1:x2]
            if chip.shape[1] == 0 or chip.shape[2] == 0: continue

            chip_norm = np.zeros((chip.shape[1], chip.shape[2], 3), dtype=np.uint8)
            for b in range(3):
                p2, p98 = np.percentile(chip[b], (2, 98))
                chip_norm[:,:,b] = np.uint8(np.clip((chip[b] - p2) / (p98 - p2 + 1e-5) * 255, 0, 255))

            # A máscara local agora é criada direto pelo recorte
            mask_local = (segmentos_full[y1:y2, x1:x2] == seg_id)
            bordas = find_boundaries(mask_local, mode='thick')
            chip_norm[bordas] = [255, 255, 0] 
            
            img_name = f"SJC_Task_ID_{seg_id}.png"
            Image.fromarray(chip_norm).save(os.path.join(dir_zoo, img_name))
            
            manifesto.append({
                "image_name": img_name,
                "id_segmento": seg_id,
                "classe_mapbiomas": classe_id,
                "area_pixels": row.area
            })
            
            print(f"✅ Recortada Imagem {idx + 1}/{len(df_zooniverse)}...")

        pd.DataFrame(manifesto).to_csv(os.path.join(dir_zoo, "manifest.csv"), index=False)
        print(f"✅ SUCESSO! Campanha perfeita gerada em: {dir_zoo}")

if __name__ == "__main__":
    gerar_campanha_foco_nitidez()