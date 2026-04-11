import os
import sys
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from skimage.segmentation import slic
from dotenv import load_dotenv

load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')
dir_out = os.path.join(ROOT, 'data', 'Output')

path_img_cbers = os.path.join(dir_out, "02_SJC_Recortado_MapBiomas.tif")
path_mudancas = os.path.join(dir_out, "04_SJC_Mapa_Mudancas_21_23.tif")
saida_segmentos = os.path.join(dir_out, "03_SJC_Segmentacao_SLIC_Mudancas.tif")

def executar_slic_por_blocos():
    print("🚀 Iniciando Segmentação Otimizada (Processamento por Blocos)...")
    
    with rasterio.open(path_img_cbers) as src_rgb, rasterio.open(path_mudancas) as src_mud:
        # 1. Alinhamento da Máscara via VRT (Não consome RAM)
        with WarpedVRT(src_mud, crs=src_rgb.crs, transform=src_rgb.transform, 
                       width=src_rgb.width, height=src_rgb.height, 
                       resampling=Resampling.nearest) as vrt_mud:
            
            meta = src_rgb.meta.copy()
            meta.update(dtype=rasterio.int32, count=1, nodata=0)
            
            with rasterio.open(saida_segmentos, 'w', **meta) as dst:
                id_global_offset = 0
                # Dividimos a imagem em blocos de 1024x1024 pixels
                tile_size = 1024 
                
                for j in range(0, src_rgb.height, tile_size):
                    for i in range(0, src_rgb.width, tile_size):
                        # Define a janela de leitura
                        window = Window(i, j, min(tile_size, src_rgb.width - i), min(tile_size, src_rgb.height - j))
                        
                        # Lê apenas o pedaço da máscara de mudança
                        mask_tile = vrt_mud.read(1, window=window) > 0
                        
                        # Se não houver mudança neste bloco, pula para o próximo
                        if not np.any(mask_tile):
                            continue
                        
                        # Lê apenas o pedaço da imagem RGB
                        img_tile = src_rgb.read([1, 2, 3], window=window)
                        
                        # Normalização local (8-bit)
                        img_norm = np.array([
                            np.clip((b - np.percentile(b, 2)) / (np.percentile(b, 98) - np.percentile(b, 2) + 1e-5) * 255, 0, 255) 
                            for b in img_tile
                        ]).astype(np.uint8)
                        
                        img_slic = np.transpose(img_norm, (1, 2, 0))

                        # Executa o SLIC apenas no bloco atual
                        segmentos_bloco = slic(
                            img_slic, 
                            n_segments=200, # Menos segmentos por bloco para ser rápido
                            compactness=10, 
                            mask=mask_tile, 
                            start_label=1
                        )

                        # Ajusta os IDs para serem únicos globalmente
                        segmentos_validos = (segmentos_bloco > 0)
                        if np.any(segmentos_validos):
                            segmentos_bloco[segmentos_validos] += id_global_offset
                            id_global_offset = np.max(segmentos_bloco)
                            
                            # Escreve o bloco no arquivo de saída
                            dst.write(segmentos_bloco.astype(np.int32), 1, window=window)
                
                print(f"📊 Progresso: Blocos processados com sucesso.")

    print(f"✅ Segmentação salva: {os.path.basename(saida_segmentos)}")

if __name__ == "__main__":
    executar_slic_por_blocos()