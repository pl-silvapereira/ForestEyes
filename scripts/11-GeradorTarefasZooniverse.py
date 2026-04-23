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

# --- CONFIGURAÇÃO DA CAMPANHA ---
load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')
dir_out = os.path.join(ROOT, 'data', 'Output')
dir_zoo = os.path.join(ROOT, 'data', 'Zooniverse_Campanha_Nitida')
os.makedirs(dir_zoo, exist_ok=True)

# Define o tamanho ideal e enxuto para os voluntários não cansarem
MAX_IMAGENS_ZOONIVERSE = 150

def gerar_campanha_foco_nitidez():
    print(f"🧠 Iniciando Geração: As {MAX_IMAGENS_ZOONIVERSE} maiores áreas de mudança (Nitidez 100%)...")
    
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
            
        ids_brutos = np.unique(segmentos_full[segmentos_full > 0])
        print("📊 1/3: Analisando candidatos (mudanças 2021-2023 sem sombras)...")

        dados_superpixels = []
        
        for seg_id in ids_brutos:
            mask = (segmentos_full == seg_id)
            area = np.sum(mask)
            
            # Pula coisas muito pequenas
            if area < 100: continue 
            
            classes_no_segmento = mapa_mud_full[mask]
            classes_unicas = np.unique(classes_no_segmento[classes_no_segmento > 0])
            
            # Exige Pureza Absoluta
            if len(classes_unicas) == 1:
                dados_superpixels.append({
                    'id': seg_id, 
                    'classe': int(classes_unicas[0]), 
                    'area': area
                })

        df_todos = pd.DataFrame(dados_superpixels)
        num_classes = df_todos['classe'].nunique()
        limite_por_classe = MAX_IMAGENS_ZOONIVERSE // num_classes
        
        print(f"⚖️ 2/3: Selecionando as {limite_por_classe} MAIORES imagens de cada classe...")
        
        df_zooniverse = pd.DataFrame()
        df_machine_learning = pd.DataFrame()
        
        for classe, group in df_todos.groupby('classe'):
            # Ordenar por ÁREA (Os gigantes vão para os humanos para garantir alta resolução visual)
            group = group.sort_values(by='area', ascending=False)
            
            df_zooniverse = pd.concat([df_zooniverse, group.head(limite_por_classe)])
            df_machine_learning = pd.concat([df_machine_learning, group.iloc[limite_por_classe:]])
            
        print(f"🎯 ALVO: {len(df_zooniverse)} imagens Premium para humanos.")
        print(f"🤖 SALVO: {len(df_machine_learning)} imagens cadastradas para o Machine Learning.")

        # PASSO 3: O Segredo da Nitidez (Sem Resize!)
        print("🖼️ 3/3: Recortando pixels originais do satélite (Zero Desfoque)...")
        manifesto = []

        for idx, row in enumerate(df_zooniverse.itertuples()):
            seg_id = row.id
            classe_id = row.classe
            mask_full = (segmentos_full == seg_id)
            
            coords = np.argwhere(mask_full)
            cy, cx = coords.mean(axis=0).astype(int) 
            
            y_min, x_min = coords[:,0].min(), coords[:,1].min()
            y_max, x_max = coords[:,0].max(), coords[:,1].max()
            
            obj_h = y_max - y_min
            obj_w = x_max - x_min
            
            # Garantimos um crop nativo simétrico (quadrado).
            # Pegamos o tamanho do objeto e adicionamos 150 pixels de margem para o voluntário entender o "contexto".
            # Forçamos um mínimo de 600px para a imagem não ficar pequena na tela do Zooniverse, MAS USANDO PIXELS REAIS.
            tam_recorte = max(obj_h, obj_w) + 150 
            tam_recorte = max(tam_recorte, 600) 
            
            metade = tam_recorte // 2
            y1, y2 = max(0, cy - metade), min(img_full.shape[1], cy + metade)
            x1, x2 = max(0, cx - metade), min(img_full.shape[2], cx + metade)
            
            # Centraliza perfeitamente
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

            # Melhoramento de brilho/contraste para o olho humano
            chip_norm = np.zeros((chip.shape[1], chip.shape[2], 3), dtype=np.uint8)
            for b in range(3):
                p2, p98 = np.percentile(chip[b], (2, 98))
                chip_norm[:,:,b] = np.uint8(np.clip((chip[b] - p2) / (p98 - p2 + 1e-5) * 255, 0, 255))

            # PONTO ÚNICO: Borda amarela espessa APENAS no superpixel principal
            mask_local = mask_full[y1:y2, x1:x2]
            bordas = find_boundaries(mask_local, mode='thick')
            chip_norm[bordas] = [255, 255, 0] 
            
            # SALVA DIRETO SEM ESTICAR (Garante a Nitidez do CBERS)
            img_name = f"SJC_Task_ID_{seg_id}.png"
            Image.fromarray(chip_norm).save(os.path.join(dir_zoo, img_name))
            
            manifesto.append({
                "image_name": img_name,
                "id_segmento": seg_id,
                "classe_mapbiomas": classe_id,
                "area_pixels": row.area
            })
            
            print(f"✅ Recortada Imagem {idx + 1}/{len(df_zooniverse)}...")
            gc.collect()

        pd.DataFrame(manifesto).to_csv(os.path.join(dir_zoo, "manifest.csv"), index=False)
        df_machine_learning.to_csv(os.path.join(dir_zoo, "dataset_Machine_Learning.csv"), index=False)
        
        print(f"✅ SUCESSO! Campanha perfeita gerada em: {dir_zoo}")

if __name__ == "__main__":
    gerar_campanha_foco_nitidez()