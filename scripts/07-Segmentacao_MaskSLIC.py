import os
import gc
import warnings
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from skimage.segmentation import slic
from dotenv import load_dotenv

# Ignorar avisos de clusters vazios
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')
dir_out = os.path.join(ROOT, 'data', 'Output')

path_img_cbers = os.path.join(dir_out, "02_SJC_Recortado_MapBiomas.tif")
path_mudancas = os.path.join(dir_out, "04_SJC_Mapa_Mudancas_21_23.tif")
saida_segmentos = os.path.join(dir_out, "03_SJC_Segmentacao_SLIC_Mudancas.tif")

# --- DICIONÁRIO DE CATEGORIZAÇÃO (Foco em Vegetação conforme solicitado) ---
class_info = {}

def registar_categoria(ids, name, color, iso120, iso122, iso123):
    for i in ids:
        class_info[i] = {'name': name, 'color': color}

# 1. FLORESTAS
registar_categoria([3], 'Floresta', '#006400', 
    'Area verde (ha) por 100.000 hab. e indices de qualidade do ar.', 
    'Monitorizacao IoT de desmatamento e saude vegetal.', 
    'Mitigacao de Ilhas de Calor Urbanas (UHI).')

registar_categoria([9], 'Floresta Antrópica', '#93c47d', 
    'Area verde (ha) por 100.000 hab. e indices de qualidade do ar.', 
    'Monitorizacao IoT de desmatamento e saude vegetal.', 
    'Mitigacao de Ilhas de Calor Urbanas (UHI).')

# 2. VEGETAÇÃO HERBÁCEA E ARBUSTIVA
registar_categoria([11, 12, 36], 'Vegetacao Herbacea e Arbustiva', '#a8c04d', 
    'Manutencao da biodiversidade local e % de areas nao pavimentadas.', 
    'Monitorizacao preditiva de risco de queimadas.', 
    'Areas de amortecimento (Buffer Zones) e permeabilidade do solo.')

# OBS: Demais categorias foram removidas para focar na detecção de mudanças vegetais

def executar_slic_vegetacao_pura():
    print("🚀 Iniciando Segmentação Adaptativa focada em Florestas e Vegetação...")
    
    with rasterio.open(path_img_cbers) as src_rgb, rasterio.open(path_mudancas) as src_mud:
        with WarpedVRT(src_mud, crs=src_rgb.crs, transform=src_rgb.transform, 
                       width=src_rgb.width, height=src_rgb.height, 
                       resampling=Resampling.nearest) as vrt_mud:
            
            meta = src_rgb.meta.copy()
            meta.update(dtype=rasterio.int32, count=1, nodata=0)
            
            with rasterio.open(saida_segmentos, 'w', **meta) as dst:
                id_global_offset = 0
                tile_size = 800 
                ids_alvos = list(class_info.keys()) # Apenas os IDs de vegetação definidos acima
                
                for j in range(0, src_rgb.height, tile_size):
                    for i in range(0, src_rgb.width, tile_size):
                        window = Window(i, j, min(tile_size, src_rgb.width - i), min(tile_size, src_rgb.height - j))
                        
                        mudanca_tile = vrt_mud.read(1, window=window)
                        # Máscara: Pixels que estão no mapa de mudança E pertencem às classes de vegetação
                        mask_tile = np.isin(mudanca_tile, ids_alvos)
                        
                        if np.sum(mask_tile) < 20: continue
                        
                        img_tile = src_rgb.read([1, 2, 3], window=window).astype(np.float32)
                        for b in range(3):
                            p2, p98 = np.percentile(img_tile[b], (2, 98))
                            img_tile[b] = np.clip((img_tile[b] - p2) / (p98 - p2 + 1e-5), 0, 1)
                        
                        img_slic = np.transpose(img_tile, (1, 2, 0))
                        
                        # Compactness baixa (0.1) para que o superpixel se molde à forma da vegetação
                        segmentos_bloco = slic(
                            img_slic, 
                            n_segments=60, 
                            compactness=0.1, 
                            mask=mask_tile, 
                            start_label=1,
                            enforce_connectivity=True
                        )

                        if np.any(segmentos_bloco > 0):
                            mask_valid = (segmentos_bloco > 0)
                            segmentos_bloco[mask_valid] += id_global_offset
                            id_global_offset = np.max(segmentos_bloco)
                            dst.write(segmentos_bloco.astype(np.int32), 1, window=window)
                        
                        del img_tile, img_slic, segmentos_bloco
                        gc.collect()

    print(f"✅ Segmentação Vegetal concluída. Total de objetos: {id_global_offset}")

if __name__ == "__main__":
    executar_slic_vegetacao_pura()