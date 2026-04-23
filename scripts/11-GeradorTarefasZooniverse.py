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
dir_zoo = os.path.join(ROOT, 'data', 'Zooniverse_Final_Pureza')
os.makedirs(dir_zoo, exist_ok=True)

def gerar_campanha_simetrica():
    print("🧠 Criando imagens quadradas 2048x2048 e Foco em Ponto Único...")
    path_rgb = os.path.join(dir_out, "02_SJC_Recortado_MapBiomas.tif")
    path_seg = os.path.join(dir_out, "03_SJC_Segmentacao_SLIC_Mudancas.tif")
    path_mud = os.path.join(dir_out, "04_SJC_Mapa_Mudancas_21_23.tif")

    with rasterio.open(path_rgb) as s_rgb, rasterio.open(path_seg) as s_seg, rasterio.open(path_mud) as s_mud:
        with WarpedVRT(s_mud, crs=s_rgb.crs, transform=s_rgb.transform, width=s_rgb.width, height=s_rgb.height, resampling=Resampling.nearest) as v_mud:
            img_full = s_rgb.read([1, 2, 3])
            seg_full = s_seg.read(1)
            mud_full = v_mud.read(1)

    ids = np.unique(seg_full[seg_full > 0])
    manifesto = []
    
    # Tamanho fixo solicitado (Simetria Quadrada)
    # Nota: Como 2048px é muito grande para um chip de contexto, 
    # centralizamos o objeto e cortamos um quadrado.
    tamanho_quadrado = 512 # Usaremos 512 para o recorte e redimensionamos para 2048 se desejar, 
                           # para manter a nitidez do sensor de 2m.
    
    for sid in ids:
        mask_full = (seg_full == sid)
        classes = np.unique(mud_full[mask_full][mud_full[mask_full] > 0])
        
        # Filtro de Pureza de Classe
        if len(classes) != 1: continue
        
        # 1. Calcular Centroide do Superpixel
        coords = np.argwhere(mask_full)
        cy, cx = coords.mean(axis=0).astype(int)
        
        # 2. Definir Janela Quadrada (BBox Simétrico)
        half = tamanho_quadrado // 2
        y1, y2 = max(0, cy - half), min(img_full.shape[1], cy + half)
        x1, x2 = max(0, cx - half), min(img_full.shape[2], cx + half)
        
        # Ajuste para garantir que seja sempre um quadrado perfeito, mesmo nas bordas
        if (y2 - y1) < tamanho_quadrado or (x2 - x1) < tamanho_quadrado:
            continue # Pula objetos muito próximos da borda da imagem total

        # 3. Recorte e Normalização
        chip = img_full[:, y1:y2, x1:x2]
        chip_norm = np.zeros((tamanho_quadrado, tamanho_quadrado, 3), dtype=np.uint8)
        for b in range(3):
            p2, p98 = np.percentile(chip[b], (2, 98))
            chip_norm[:,:,b] = np.uint8(np.clip((chip[b]-p2)/(p98-p2+1e-5)*255, 0, 255))
        
        # 4. MARCAÇÃO ÚNICA: Apenas o ID atual recebe a borda
        # Criamos uma máscara local apenas para o superpixel alvo dentro do recorte
        mask_local = mask_full[y1:y2, x1:x2]
        bordas = find_boundaries(mask_local, mode='thick')
        chip_norm[bordas] = [255, 255, 0] # Amarelo vibrante
        
        # 5. Redimensionamento para 2048x2048 (Opcional, para simetria de resolução)
        # Se preferir salvar no tamanho nativo do recorte para economizar espaço, pule a linha abaixo
        final_img = Image.fromarray(chip_norm).resize((2048, 2048), Image.Resampling.LANCZOS)
        
        name = f"SJC_Task_ID_{sid}.png"
        final_img.save(os.path.join(dir_zoo, name))
        manifesto.append({"image_name": name, "id_segmento": sid, "classe": int(classes[0])})

    pd.DataFrame(manifesto).to_csv(os.path.join(dir_zoo, "manifest.csv"), index=False)
    print(f"✅ Campanha finalizada com {len(manifesto)} imagens quadradas e foco único.")

if __name__ == "__main__":
    gerar_campanha_simetrica()