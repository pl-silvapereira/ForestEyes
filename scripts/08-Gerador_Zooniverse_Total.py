import os
import csv
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from skimage.segmentation import find_boundaries
import scipy.ndimage as ndi
from PIL import Image
from dotenv import load_dotenv

# --- CONFIGURAÇÃO VIA .ENV ---
load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')

dir_output = os.path.join(ROOT, 'data', 'Output')
dir_mapbiomas = os.path.join(ROOT, 'data', 'MapBiomas')
dir_zooniverse = os.path.join(ROOT, 'data', 'Zooniverse_Campaign')

img_cbers_path = os.path.join(dir_output, "02_SJC_Recortado_MapBiomas.tif")
saida_segmentos = os.path.join(dir_output, "03_SJC_Segmentacao_SLIC_Total.tif")
mapbiomas_path = os.path.join(dir_mapbiomas, "2023_coverage_coverage_10m_1-95-2928_faf8d377-614d-4e94-a539-5201bc76c6ef.tif")

# ==========================================
# DICIONÁRIO COMPLETO MAPBIOMAS + ISO
# ==========================================
class_metadata = {}
def registar_categoria(ids, name, color, iso120, iso122, iso123):
    for i in ids:
        class_metadata[i] = {'name': name, 'color': color, 'iso120': iso120, 'iso122': iso122, 'iso123': iso123}

registar_categoria([3], 'Floresta', '#006400', 'Area verde (ha) por 100.000 hab. e indices de qualidade do ar.', 'Monitorizacao IoT de desmatamento e saude vegetal.', 'Mitigacao de Ilhas de Calor Urbanas (UHI).')
registar_categoria([9], 'Floresta Antrópica', '#93c47d', 'Area verde (ha) por 100.000 hab. e indices de qualidade do ar.', 'Monitorizacao IoT de desmatamento e saude vegetal.', 'Mitigacao de Ilhas de Calor Urbanas (UHI).')
registar_categoria([11, 12, 36], 'Vegetacao Herbacea e Arbustiva', '#a8c04d', 'Manutencao da biodiversidade local e % de areas nao pavimentadas.', 'Monitorizacao preditiva de risco de queimadas.', 'Areas de amortecimento (Buffer Zones) e permeabilidade do solo.')
registar_categoria([15, 19, 20, 21, 39, 41, 46, 48], 'Agropecuaria (Campos, Lavouras)', '#edde8e', 'Protecao de terras araveis contra o espraiamento urbano (Urban Sprawl).', 'Agrotech, agricultura de precisao e rastreabilidade local.', 'Garantia de seguranca alimentar e autoabastecimento.')
registar_categoria([24, 25], 'Infraestrutura Urbana', '#d4271e', 'Densidade populacional, acesso a moradia e crescimento da mancha urbana.', 'Infraestrutura conectada (Smart Grids) e mobilidade inteligente.', 'Vulnerabilidade da infraestrutura critica frente a desastres.')
registar_categoria([29, 31, 33], 'Nao Observado (Agua, Rocha)', '#0000ff', 'Disponibilidade, acesso e qualidade das reservas hidricas superficiais.', 'Telemetria e sensores para controlo de qualidade e nivel.', 'Prevencao de inundacoes e respeito as Areas de Preservacao Permanente (APP).')
registar_categoria([4, 5, 6, 23, 27, 30, 32, 35, 40, 47, 49, 50, 62, 75], 'Descartadas', '#A9A9A9', 'N/A - Filtragem de dados para precisao dos calculos.', 'N/A - Otimizacao de processamento analitico.', 'N/A - Remocao de anomalias da modelagem de risco.')

def gerar_imagens_zooniverse_total():
    if not os.path.exists(dir_zooniverse): os.makedirs(dir_zooniverse)

    with rasterio.open(img_cbers_path) as src_rgb, \
         rasterio.open(saida_segmentos) as src_seg, \
         rasterio.open(mapbiomas_path) as src_mb:
        
        # 1. Carrega APENAS a matriz de segmentos para a memória (Muito leve)
        segmentos = src_seg.read(1)
        ids_segmentos = np.unique(segmentos)[1:] # Ignora ID 0

        vrt_options = {'resampling': Resampling.nearest, 'crs': src_rgb.crs, 'transform': src_rgb.transform, 'height': src_rgb.height, 'width': src_rgb.width}
        vrt_mb = WarpedVRT(src_mb, **vrt_options)

        manifesto = []
        print(f"Gerando tarefas para {len(ids_segmentos)} segmentos...")
        
        # 2. Localiza as coordenadas geográficas de TODOS os segmentos em menos de 1 segundo!
        # Isso substitui os milhares de 'np.where()' que estavam travando a CPU.
        print("Mapeando coordenadas (Bounding Boxes)...")
        slices = ndi.find_objects(segmentos)
        
        print("Processando imagens direto do disco (Baixo consumo de RAM)...")
        for seg_id in ids_segmentos:
            # Encontra as coordenadas deste segmento exato
            slc = slices[seg_id - 1] if seg_id - 1 < len(slices) else None
            if slc is None: continue
            
            y_slice, x_slice = slc
            y_min, y_max = y_slice.start, y_slice.stop
            x_min, x_max = x_slice.start, x_slice.stop
            
            # Margem de 100 pixels (Contexto para o voluntário)
            m = 100
            y_min_m = max(0, y_min - m)
            y_max_m = min(src_rgb.height, y_max + m)
            x_min_m = max(0, x_min - m)
            x_max_m = min(src_rgb.width, x_max + m)
            
            # 3. O PULO DO GATO: Cria uma "janela" virtual
            janela = Window(col_off=x_min_m, row_off=y_min_m, width=(x_max_m - x_min_m), height=(y_max_m - y_min_m))
            
            # Lê APENAS o quadradinho pequeno direto do arquivo pesado no disco!
            img_crop = src_rgb.read([1, 2, 3], window=janela)
            img_crop = np.moveaxis(img_crop, 0, -1)
            
            # Ajuste de brilho dinâmico e local (Apenas para este pequeno quadradinho)
            valid_pixels = img_crop[img_crop > 0]
            if valid_pixels.size > 0:
                p1, p99 = np.percentile(valid_pixels, (1, 99))
                if p99 > p1:
                    img_crop_8b = np.clip((img_crop - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
                else:
                    img_crop_8b = img_crop.astype(np.uint8)
            else:
                img_crop_8b = img_crop.astype(np.uint8)
            
            # Recorta os segmentos e o mapbiomas no mesmo quadradinho
            seg_crop = segmentos[y_min_m:y_max_m, x_min_m:x_max_m]
            mb_crop = vrt_mb.read(1, window=janela)
            
            mascara_alvo = (seg_crop == seg_id)
            classes_no_segmento = mb_crop[mascara_alvo]
            
            classes_validas = classes_no_segmento[classes_no_segmento != 0]
            if len(classes_validas) == 0: continue
            
            classe_dominante_id = np.bincount(classes_validas).argmax()
            
            info_classe = class_metadata.get(classe_dominante_id, {
                'name': f'Desconhecida (ID {classe_dominante_id})', 
                'iso120': 'N/A', 'iso122': 'N/A', 'iso123': 'N/A'
            })
            
            # Pinta a borda amarela
            bordas = find_boundaries(mascara_alvo, mode='thick')
            img_crop_8b[bordas] = [255, 255, 0]
            
            nome_arquivo = f"SJC_Task_{seg_id:05d}.png"
            Image.fromarray(img_crop_8b).save(os.path.join(dir_zooniverse, nome_arquivo))
            
            manifesto.append({
                "image_name": nome_arquivo,
                "segment_id": seg_id,
                "mapbiomas_class": info_classe['name'],
                "iso_37120_impact": info_classe['iso120'],
                "iso_37122_impact": info_classe['iso122'],
                "iso_37123_impact": info_classe['iso123']
            })
            
            if len(manifesto) % 500 == 0: 
                print(f"   -> {len(manifesto)} tarefas exportadas e salvas em disco...")

        print("Gerando arquivo manifest.csv enriquecido...")
        with open(os.path.join(dir_zooniverse, "manifest.csv"), mode='w', newline='', encoding='utf-8') as csv_file:
            colunas = ["image_name", "segment_id", "mapbiomas_class", "iso_37120_impact", "iso_37122_impact", "iso_37123_impact"]
            writer = csv.DictWriter(csv_file, fieldnames=colunas)
            writer.writeheader()
            writer.writerows(manifesto)

        vrt_mb.close()
        print(f"✅ Concluído! O uso de RAM foi mantido no mínimo. Imagens na pasta: {dir_zooniverse}")

if __name__ == '__main__':
    gerar_imagens_zooniverse_total()