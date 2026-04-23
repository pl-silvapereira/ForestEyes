import os
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
            
            # Carrega dados alinhados para a RAM (certifique-se de ter RAM disponível)
            img_full = src_rgb.read([1, 2, 3])
            segmentos_full = src_seg.read(1)
            mapa_mud_full = vrt_mud.read(1)
            
        ids_brutos = np.unique(segmentos_full[segmentos_full > 0])
        total_ids = len(ids_brutos)
        print(f"📦 Analisando {total_ids} superpixels candidatos...")

        manifesto = []
        
        # Resolução Final Solicitada
        res_final = 2048 

        for idx, seg_id in enumerate(ids_brutos):
            # Máscara binária para o ID atual em toda a imagem
            mask_full = (segmentos_full == seg_id)
            
            # --- GARANTIA DE CLASSE ÚNICA ---
            classes_no_segmento = mapa_mud_full[mask_full]
            classes_unicas = np.unique(classes_no_segmento[classes_no_segmento > 0])
            
            if len(classes_unicas) != 1:
                continue # Descarta se houver mistura de classes
            
            classe_id = int(classes_unicas[0])

            # --- ESTRATÉGIA DE ZOOM MÁXIMO E SIMETRIA QUADRADA ---
            # 1. Calcular Centroide e BBox do Superpixel Alvo
            coords = np.argwhere(mask_full)
            cy, cx = coords.mean(axis=0).astype(int) # Centroide
            
            # BBox do objeto
            y_min, x_min = coords[:,0].min(), coords[:,1].min()
            y_max, x_max = coords[:,0].max(), coords[:,1].max()
            
            # Dimensões do objeto
            obj_h = y_max - y_min
            obj_w = x_max - x_min
            
            # Define o tamanho do recorte quadrado baseado na MAIOR dimensão do objeto, 
            # adicionando uma margem mínima (ex: 20 pixels) para contexto.
            # Isso garante que o objeto preencha a maior parte do quadro (alto zoom).
            tam_recorte = max(obj_h, obj_w) + 20 
            
            # 2. Definir Janela Quadrada Centrada no Objeto
            metade = tam_recorte // 2
            y1, y2 = max(0, cy - metade), min(img_full.shape[1], cy + metade)
            x1, x2 = max(0, cx - metade), min(img_full.shape[2], cx + metade)
            
            # Ajuste final para garantir que seja sempre um quadrado perfeito, mesmo nas bordas da imagem original
            dif_y = tam_recorte - (y2 - y1)
            dif_x = tam_recorte - (x2 - x1)
            
            # Expande para compensar perda nas bordas (se necessário e possível)
            if dif_y > 0:
                y1 = max(0, y1 - dif_y // 2)
                y2 = min(img_full.shape[1], y2 + (dif_y - (cy - metade - y1)))
            if dif_x > 0:
                x1 = max(0, x1 - dif_x // 2)
                x2 = min(img_full.shape[2], x2 + (dif_x - (cx - metade - x1)))

            # --- PROCESSAMENTO DO RECORTE ---
            # Extração do chip RGB
            chip = img_full[:, y1:y2, x1:x2]
            
            # Normalização Linear (Linear Stretch 2-98%) para visualização humana
            chip_norm = np.zeros((chip.shape[1], chip.shape[2], 3), dtype=np.uint8)
            for b in range(3):
                p2, p98 = np.percentile(chip[b], (2, 98))
                chip_norm[:,:,b] = np.uint8(np.clip((chip[b] - p2) / (p98 - p2 + 1e-5) * 255, 0, 255))

            # --- MARCAÇÃO ÚNICA (Foco em Ponto Único) ---
            # Criamos uma máscara local APENAS para o ID alvo dentro do recorte.
            # Isso impede que bordas de superpixels vizinhos apareçam.
            mask_local = mask_full[y1:y2, x1:x2]
            
            # Desenha a borda amarela (espesa) apenas para o objeto centralizado
            bordas = find_boundaries(mask_local, mode='thick')
            chip_norm[bordas] = [255, 255, 0] # Amarelo ForestEyes
            
            # --- SALVAMENTO E REDIMENSIONAMENTO ---
            # Converte para imagem PIL e redimensiona para 2048x2048 px usando Lanczos
            img_png = Image.fromarray(chip_norm)
            final_img = img_png.resize((res_final, res_final), Image.Resampling.LANCZOS)
            
            # Salva PNG com nome único vinculado ao ID
            img_name = f"SJC_Task_ID_{seg_id}.png"
            final_img.save(os.path.join(dir_zoo, img_name))
            
            manifesto.append({
                "image_name": img_name,
                "id_segmento": seg_id,
                "classe_mapbiomas": classe_id
            })

            # Feedback de progresso
            if idx % 100 == 0:
                print(f"✅ Processado: {idx}/{total_ids}...")
                gc.collect() # Limpeza de memória agressiva

        # Gera o manifesto CSV para upload no Zooniverse
        pd.DataFrame(manifesto).to_csv(os.path.join(dir_zoo, "manifest.csv"), index=False)
        print(f"✅ Campanha refinada finalizada com {len(manifesto)} tarefas quadradas de foco único.")

if __name__ == "__main__":
    gerar_campanha_zoom_foco_unico()