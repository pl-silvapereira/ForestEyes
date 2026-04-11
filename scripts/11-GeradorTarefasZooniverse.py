import os
import pandas as pd
import numpy as np
import rasterio
from PIL import Image
from skimage.segmentation import find_boundaries
from dotenv import load_dotenv

# --- CONFIGURAÇÃO ---
load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')
dir_out = os.path.join(ROOT, 'data', 'Output')
dir_zoo = os.path.join(ROOT, 'data', 'Zooniverse_Campaign_SJC')

if not os.path.exists(dir_zoo):
    os.makedirs(dir_zoo)

# Arquivos de entrada (Gerados no Script 06 e no Script 07-Otimizado)
path_rgb = os.path.join(dir_out, "02_SJC_Recortado_MapBiomas.tif")
path_seg = os.path.join(dir_out, "03_SJC_Segmentacao_SLIC_Mudancas.tif")

def gerar_campanha_hotspots():
    print("🎯 Iniciando extração de superpixels das áreas de mudança...")
    
    with rasterio.open(path_rgb) as src_rgb, rasterio.open(path_seg) as src_seg:
        # 1. Leitura e Normalização para Visualização Humana (Stretch 2-98%)
        img_data = src_rgb.read([1, 2, 3])
        img_display = np.zeros(img_data.shape, dtype=np.uint8)
        
        for b in range(3):
            # Normaliza para o olho humano ver as cores reais do satélite
            p2, p98 = np.percentile(img_data[b], (2, 98))
            img_display[b] = np.clip((img_data[b] - p2) / (p98 - p2 + 1e-5) * 255, 0, 255).astype(np.uint8)
        
        img_display = np.transpose(img_display, (1, 2, 0))
        segmentos = src_seg.read(1)
        
        # 2. Identificação de IDs únicos (Apenas onde houve mudança)
        ids_validos = np.unique(segmentos[segmentos > 0])
        print(f"📦 Total de tarefas detectadas: {len(ids_validos)}")

        manifesto = []
        
        # 3. Exportação dos Recortes (Chips)
        for seg_id in ids_validos:
            mask = (segmentos == seg_id)
            coords = np.argwhere(mask)
            
            # Bounding Box com margem de 20 pixels para dar contexto ao voluntário
            y_min, x_min = coords.min(axis=0) - 20
            y_max, x_max = coords.max(axis=0) + 20
            
            y_min, x_min = max(0, y_min), max(0, x_min)
            y_max, x_max = min(img_display.shape[0], y_max), min(img_display.shape[1], x_max)

            # Recorte do chip
            chip = img_display[y_min:y_max, x_min:x_max].copy()
            
            # Opcional: Desenhar borda amarela para destacar o objeto da mudança
            mask_local = mask[y_min:y_max, x_min:x_max]
            bordas = find_boundaries(mask_local, mode='thick')
            chip[bordas] = [255, 255, 0] 
            
            # Salvar PNG
            img_name = f"SJC_Change_Task_{seg_id}.png"
            Image.fromarray(chip).save(os.path.join(dir_zoo, img_name))
            
            manifesto.append({
                "image_name": img_name,
                "subject_id": seg_id,
                "city": "São José dos Campos",
                "source": "CBERS-04A WPM"
            })

        # 4. Criar o Manifesto CSV para o Zooniverse
        df_manifesto = pd.DataFrame(manifesto)
        df_manifesto.to_csv(os.path.join(dir_zoo, "manifest.csv"), index=False)
        
    print(f"✅ Campanha finalizada! {len(manifesto)} imagens salvas.")
    print(f"📂 Pasta pronta para upload: {dir_zoo}")

if __name__ == "__main__":
    gerar_campanha_hotspots()