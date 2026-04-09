import os
import sys
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from skimage.segmentation import slic
from dotenv import load_dotenv

# --- CONFIGURAÇÃO VIA .ENV ---
load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')

dir_output = os.path.join(ROOT, 'data', 'Output')
dir_mapbiomas = os.path.join(ROOT, 'data', 'MapBiomas')

img_cbers_path = os.path.join(dir_output, "02_SJC_Recortado_MapBiomas.tif")
mapbiomas_path = os.path.join(dir_mapbiomas, "2023_coverage_coverage_10m_1-95-2928_faf8d377-614d-4e94-a539-5201bc76c6ef.tif")
saida_segmentos = os.path.join(dir_output, "03_SJC_Segmentacao_SLIC_Total.tif")

def executar_slic_por_blocos():
    print("1/3 - Preparando imagens e alinhamento (Modo Baixo Consumo de RAM)...")
    
    with rasterio.open(img_cbers_path) as cbers_src, rasterio.open(mapbiomas_path) as mb_src:
        
        # Configuração do VRT (Alinhamento MapBiomas -> CBERS)
        vrt_options = {
            'resampling': Resampling.nearest,
            'crs': cbers_src.crs,
            'transform': cbers_src.transform,
            'height': cbers_src.height,
            'width': cbers_src.width,
        }
        
        # Configura o arquivo de saída para aceitar escrita em blocos
        profile = cbers_src.profile.copy()
        profile.update({
            'dtype': rasterio.int32, 
            'count': 1, 
            'compress': 'lzw', 
            'nodata': 0,
            'tiled': True,
            'blockxsize': 1024, # Tamanho do bloco (ajuda a limitar a RAM)
            'blockysize': 1024
        })

        print("2/3 - Iniciando processamento em blocos (Isso evita travamentos)...")
        
        with WarpedVRT(mb_src, **vrt_options) as vrt_mb, \
             rasterio.open(saida_segmentos, 'w', **profile) as dst:
            
            # Pega todas as "janelas" (blocos) que formam a imagem
            windows = [w for _, w in dst.block_windows()]
            total_blocos = len(windows)
            
            # Variável para garantir que nenhum superpixel tenha o mesmo ID de outro bloco
            id_global_offset = 0 
            blocos_processados = 0
            
            for i, window in enumerate(windows):
                # 1. Carrega APENAS aquele pedacinho da imagem de alta resolução
                img_rgb = cbers_src.read([1, 2, 3], window=window)
                
                # Se o bloco for apenas fundo preto, pula para economizar tempo
                if np.max(img_rgb) == 0:
                    continue
                    
                img_rgb = np.moveaxis(img_rgb, 0, -1)
                
                # Normalização de brilho local
                p98 = np.percentile(img_rgb[img_rgb > 0], 98) if np.any(img_rgb > 0) else 1
                img_rgb_8b = np.clip(img_rgb / p98 * 255, 0, 255).astype(np.uint8)

                # 2. Carrega APENAS aquele pedacinho do MapBiomas
                mb_data = vrt_mb.read(1, window=window)
                mascara_cidade = (mb_data != 0)
                
                # Se não houver cidade neste bloco, pula
                if not np.any(mascara_cidade):
                    continue

                # 3. Aplica o SLIC apenas neste bloco (Gasta muito pouca RAM)
                # Pede cerca de 100 segmentos por bloco de 1024x1024
                segmentos = slic(img_rgb_8b, n_segments=100, compactness=15, mask=mascara_cidade, start_label=1)
                
                # 4. Ajusta os IDs para não repetirem com os blocos anteriores
                # Exemplo: Bloco 1 vai do ID 1 ao 100. Bloco 2 vai do 101 ao 200.
                segmentos_validos = (segmentos > 0)
                if np.any(segmentos_validos):
                    segmentos[segmentos_validos] += id_global_offset
                    id_global_offset = np.max(segmentos)
                
                # 5. Salva o bloco no disco rígido imediatamente e limpa a memória
                dst.write(segmentos.astype(np.int32), 1, window=window)
                blocos_processados += 1
                
                if i % 10 == 0:
                    sys.stdout.write(f"\r  📊 Progresso: {(i/total_blocos)*100:.1f}% ({blocos_processados} blocos válidos) ")
                    sys.stdout.flush()

        print(f"\r  📊 Progresso: 100.0% ({blocos_processados} blocos processados no total) ")
            
    print(f"\n✅ Concluído! O arquivo foi gerado consumindo pouquíssima memória RAM.")
    print(f"Total de superpixels gerados na cidade: {id_global_offset}")

if __name__ == '__main__':
    executar_slic_por_blocos()