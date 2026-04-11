import os
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from skimage.segmentation import slic
from dotenv import load_dotenv

# --- CONFIGURAÇÃO ---
load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')
dir_out = os.path.join(ROOT, 'data', 'Output')

path_img_cbers = os.path.join(dir_out, "02_SJC_Recortado_MapBiomas.tif")
path_mudancas = os.path.join(dir_out, "04_SJC_Mapa_Mudancas_21_23.tif")
saida_segmentos = os.path.join(dir_out, "03_SJC_Segmentacao_SLIC_Mudancas.tif")

def executar_slic_direcionado():
    print("🚀 Iniciando Segmentação SLIC focada em áreas de mudança...")
    
    with rasterio.open(path_img_cbers) as src_rgb, rasterio.open(path_mudancas) as src_mud:
        # 1. Alinha a máscara de mudança (10m) com a imagem CBERS (2m)
        with WarpedVRT(src_mud, crs=src_rgb.crs, transform=src_rgb.transform, 
                       width=src_rgb.width, height=src_rgb.height, 
                       resampling=Resampling.nearest) as vrt_mud:
            
            mask_mudanca = vrt_mud.read(1) > 0
            img_rgb = src_rgb.read([1, 2, 3]) # Lendo as bandas
            
            # 2. Normalização para o SLIC (8-bit) - Corrigido 'img_data' para 'img_rgb'
            img_rgb_norm = np.array([
                np.clip((b - np.percentile(b, 2)) / (np.percentile(b, 98) - np.percentile(b, 2)) * 255, 0, 255) 
                for b in img_rgb
            ]).astype(np.uint8)
            
            # Transpor para o formato (H, W, C) exigido pelo scikit-image
            img_slic = np.transpose(img_rgb_norm, (1, 2, 0))

            # 3. Gerar Superpixels Puros
            # Compactness=10 faz o segmento "grudar" na borda da mudança, garantindo classe única
            print("⏳ Processando superpixels sobre a máscara de diferença...")
            segmentos = slic(
                img_slic, 
                n_segments=8000,   # Aumentei para garantir segmentos menores e mais puros
                compactness=10, 
                mask=mask_mudanca, 
                start_label=1
            )

            # 4. Salvar o resultado
            meta = src_rgb.meta.copy()
            meta.update(dtype=rasterio.int32, count=1, nodata=0)
            with rasterio.open(saida_segmentos, 'w', **meta) as dst:
                dst.write(segmentos.astype(np.int32), 1)

    print(f"✅ Arquivo de segmentação gerado: {os.path.basename(saida_segmentos)}")

if __name__ == "__main__":
    executar_slic_direcionado()