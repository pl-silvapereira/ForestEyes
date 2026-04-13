import os
import pandas as pd
import numpy as np
import rasterio
from PIL import Image
from skimage.segmentation import find_boundaries
from dotenv import load_dotenv

load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')
dir_out = os.path.join(ROOT, 'data', 'Output')
dir_zoo = os.path.join(ROOT, 'data', 'Zooniverse_Foco_Impacto')

if not os.path.exists(dir_zoo): os.makedirs(dir_zoo)

# Metadados ISO refinados
iso_metadata = {
    3: "ISO 37120: Floresta Nativa - Preservacao e Qualidade do Ar",
    9: "ISO 37120: Floresta Antropica - Recuperacao Verde",
    11: "ISO 37122: Vegetacao Herbacea - Permeabilidade e Risco de Fogo",
    12: "ISO 37122: Formacao Arbustiva - Biodiversidade Local",
    36: "ISO 37122: Campo Alagado - Protecao de Recursos Hidricos"
}

def gerar_campanha_alta_relevancia():
    print("🚀 Aplicando Funil de Alta Relevância (Foco: Qualidade > Quantidade)...")
    
    path_rgb = os.path.join(dir_out, "02_SJC_Recortado_MapBiomas.tif")
    path_seg = os.path.join(dir_out, "03_SJC_Segmentacao_SLIC_Mudancas.tif")
    path_mud = os.path.join(dir_out, "04_SJC_Mapa_Mudancas_21_23.tif")

    with rasterio.open(path_rgb) as src_rgb, \
         rasterio.open(path_seg) as src_seg, \
         rasterio.open(path_mud) as src_mud:
        
        img_data = src_rgb.read([1, 2, 3])
        # Normalização com brilho extra para facilitar a visão do voluntário
        img_display = np.zeros(img_data.shape, dtype=np.uint8)
        for b in range(3):
            p2, p98 = np.percentile(img_data[b], (2, 98))
            img_display[b] = np.clip((img_data[b] - p2) / (p98 - p2 + 1e-5) * 255, 0, 255).astype(np.uint8)
        
        img_display = np.transpose(img_display, (1, 2, 0))
        segmentos = src_seg.read(1)
        mapa_mudanca = src_mud.read(1)
        
        ids_brutos = np.unique(segmentos[segmentos > 0])
        manifesto = []

        # --- FILTROS AGRESSIVOS PARA A TESE ---
        # 1. Área Mínima: 500 pixels (2.000m²) - Foca em desmatamentos/mudanças reais
        area_min_pixels = 500 

        for seg_id in ids_brutos:
            mask = (segmentos == seg_id)
            
            # Filtro 1: Tamanho do objeto
            if np.sum(mask) < area_min_pixels:
                continue

            # Filtro 2: Pureza de Classe (Apenas 1 classe por superpixel)
            classes_dentro = mapa_mudanca[mask]
            classes_unicas = np.unique(classes_dentro[classes_dentro > 0])
            
            if len(classes_unicas) != 1:
                continue # Descarta se houver mistura de classes
            
            class_id = int(classes_unicas[0])

            # Filtro 3: Priorização (Opcional - se quiser reduzir ainda mais)
            # if class_id not in [3, 9]: continue # Exemplo: focar só em floresta

            # --- PROCESSAMENTO DO RECORTE ---
            coords = np.argwhere(mask)
            y_min, x_min = max(0, coords.min(axis=0)[0]-30), max(0, coords.min(axis=0)[1]-30)
            y_max, x_max = min(img_display.shape[0], coords.max(axis=0)[0]+30), min(img_display.shape[1], coords.max(axis=0)[1]+30)

            chip = img_display[y_min:y_max, x_min:x_max].copy()
            
            # Borda Amarela (Pureza Visual)
            mask_local = mask[y_min:y_max, x_min:x_max]
            bordas = find_boundaries(mask_local, mode='thick')
            chip[bordas] = [255, 255, 0]
            
            img_name = f"SJC_PURE_VEG_{seg_id}.png"
            Image.fromarray(chip).save(os.path.join(dir_zoo, img_name))
            
            manifesto.append({
                "image_name": img_name,
                "id_segmento": seg_id,
                "classe_mapbiomas": class_id,
                "area_m2": np.sum(mask) * 4,
                "indicador_iso": iso_metadata.get(class_id, "Monitoramento Ambiental")
            })

        # Salva Manifesto
        pd.DataFrame(manifesto).to_csv(os.path.join(dir_zoo, "manifest.csv"), index=False)
        print(f"✅ Filtro concluído! Reduzimos para {len(manifesto)} tarefas de alta qualidade.")

if __name__ == "__main__":
    gerar_campanha_alta_relevancia()