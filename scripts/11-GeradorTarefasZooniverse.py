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
dir_zoo = os.path.join(ROOT, 'data', 'Zooniverse_Campanha_30_70')
os.makedirs(dir_zoo, exist_ok=True)

def gerar_divisao_30_70():
    print("🧠 Iniciando Amostragem Estratificada (30% Zooniverse / 70% Machine Learning)...")
    
    path_rgb = os.path.join(dir_out, "02_SJC_Recortado_MapBiomas.tif")
    path_seg = os.path.join(dir_out, "03_SJC_Segmentacao_SLIC_Mudancas.tif")
    path_mud = os.path.join(dir_out, "04_SJC_Mapa_Mudancas_21_23.tif")

    with rasterio.open(path_rgb) as src_rgb, rasterio.open(path_seg) as src_seg, rasterio.open(path_mud) as src_mud:
        with WarpedVRT(src_mud, crs=src_rgb.crs, transform=src_rgb.transform, width=src_rgb.width, height=src_rgb.height, resampling=Resampling.nearest) as vrt_mud:
            
            img_full = src_rgb.read([1, 2, 3])
            segmentos_full = src_seg.read(1)
            mapa_mud_full = vrt_mud.read(1)
            
        ids_brutos = np.unique(segmentos_full[segmentos_full > 0])
        
        # ETAPA 1: Levantar os dados de todos os superpixels
        print("📊 Analisando qualidade e tamanho dos superpixels...")
        dados_superpixels = []
        
        for seg_id in ids_brutos:
            mask = (segmentos_full == seg_id)
            area = np.sum(mask)
            if area < 100: continue # Ignora micro-ruídos
            
            classes = np.unique(mapa_mud_full[mask][mapa_mud_full[mask] > 0])
            if len(classes) != 1: continue # Garante pureza
            
            dados_superpixels.append({
                'id_segmento': seg_id,
                'classe': int(classes[0]),
                'area_pixels': area
            })
            
        df_todos = pd.DataFrame(dados_superpixels)
        
        # ETAPA 2: O Split 30/70 por Categoria
        df_zooniverse = pd.DataFrame()
        df_machine_learning = pd.DataFrame()
        
        for classe, group in df_todos.groupby('classe'):
            # Ordena pelos maiores (mais fáceis para o humano ver = 'melhores')
            group = group.sort_values(by='area_pixels', ascending=False)
            
            # Calcula o corte de 30%
            corte = int(len(group) * 0.30)
            
            df_zooniverse = pd.concat([df_zooniverse, group.iloc[:corte]])
            df_machine_learning = pd.concat([df_machine_learning, group.iloc[corte:]])
            
        print(f"🎯 Separação concluída:\n - {len(df_zooniverse)} imagens irão para o Zooniverse (Humanos)\n - {len(df_machine_learning)} imagens irão para o modelo (ML)")
        
        # ETAPA 3: Exportar imagens apenas para os 30% do Zooniverse
        print("🖼️ Exportando chips de imagem para a campanha...")
        manifesto = []
        
        for idx, row in df_zooniverse.iterrows():
            seg_id = row['id_segmento']
            mask = (segmentos_full == seg_id)
            
            coords = np.argwhere(mask)
            y_min, x_min = max(0, coords[:,0].min()-25), max(0, coords[:,1].min()-25)
            y_max, x_max = min(img_full.shape[1], coords[:,0].max()+25), min(img_full.shape[2], coords[:,1].max()+25)

            chip_data = img_full[:, y_min:y_max, x_min:x_max]
            chip_norm = np.zeros((chip_data.shape[1], chip_data.shape[2], 3), dtype=np.uint8)
            for b in range(3):
                p2, p98 = np.percentile(chip_data[b], (2, 98))
                chip_norm[:,:,b] = np.uint8(np.clip((chip_data[b] - p2) / (p98 - p2 + 1e-5) * 255, 0, 255))

            mask_local = mask[y_min:y_max, x_min:x_max]
            bordas = find_boundaries(mask_local, mode='thick')
            chip_norm[bordas] = [255, 255, 0]
            
            img_name = f"SJC_Class_{row['classe']}_ID_{seg_id}.png"
            Image.fromarray(chip_norm).save(os.path.join(dir_zoo, img_name))
            
            manifesto.append({
                "image_name": img_name,
                "id_segmento": seg_id,
                "classe_mapbiomas": row['classe'],
                "destino": "Zooniverse_Treino"
            })

        # Salva o Manifesto do Zooniverse
        pd.DataFrame(manifesto).to_csv(os.path.join(dir_zoo, "manifest.csv"), index=False)
        
        # Salva o Dataset do Machine Learning (para você usar no futuro)
        df_machine_learning.to_csv(os.path.join(dir_zoo, "dataset_Machine_Learning_70_pct.csv"), index=False)
        print("✅ Pipeline Finalizado! Dados prontos para humanos e máquinas.")

if __name__ == "__main__":
    gerar_divisao_30_70()