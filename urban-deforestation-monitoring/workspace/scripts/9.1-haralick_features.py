import os
import numpy as np
import pandas as pd
import mahotas as mh
from tqdm import tqdm
import rasterio
from scipy.ndimage import find_objects
from joblib import Parallel, delayed

# ==============================================================================
# 1. CONFIGURAÇÃO DO DIRETÓRIO RAIZ DO PROJETO
# ==============================================================================
PROJECT_ROOT = '/content/drive/MyDrive/Mestrado/04-Projeto ForestEyes/ForestEyes/urban-deforestation-monitoring'


# ==============================================================================
# 2. FUNÇÃO AUXILIAR PARA UM ÚNICO SEGMENTO (RECORTADO POR BOUNDING BOX)
# ==============================================================================
def _process_single_segment(segment_id, sub_gray, sub_mask, max_val, is_float_0_1):
    segment_region = sub_gray * sub_mask

    # Normaliza para [0, 255] uint8
    if is_float_0_1:
        normalized_segment = (segment_region * 255).astype(np.uint8)
    else:
        # Evita overflow em imagens uint16 (CBERS-4A pan-sharpened)
        if max_val > 0:
            normalized_segment = ((segment_region / max_val) * 255).astype(np.uint8)
        else:
            return None

    if np.sum(normalized_segment) == 0:
        return None

    # Calcula os descritores de Haralick ignorando os pixels zero (fundo do recorte)
    try:
        haralick_features = mh.features.haralick(normalized_segment, ignore_zeros=True).mean(axis=0)
        return [int(segment_id)] + haralick_features.tolist()
    except ValueError:
        # Ocorre apenas se o segmento tiver 1 único pixel isolado sem vizinhos
        return None


# ==============================================================================
# 3. FUNÇÃO PRINCIPAL DE CÁLCULO DOS DESCRITORES DE HARALICK (OTIMIZADA)
# ==============================================================================
def haralick(gray_image, segments, n_jobs=-1):
    """
    Calcula os 13 descritores de Haralick por segmento utilizando recorte por
    bounding box e processamento paralelo real em múltiplos processos (loky).
    """
    haralick_feature_names = [
        "ASM", "Contrast", "Correlation", "Variance", "IDM",
        "Sum_Average", "Sum_Variance", "Sum_Entropy", "Entropy",
        "Diff_Variance", "Diff_Entropy",
        "IMC1", "IMC2"
    ]

    # Encontra instantaneamente o bounding box de todos os IDs de segmentos
    slices = find_objects(segments)

    # Identifica os IDs válidos (ignorando 0, que representa o fundo filtrado)
    segment_ids = np.unique(segments)
    segment_ids = segment_ids[segment_ids > 0]

    max_val = float(gray_image.max())
    is_float_0_1 = np.issubdtype(gray_image.dtype, np.floating) and (max_val <= 1.0)

    # Gera apenas os pequenos recortes (bounding boxes) para enviar aos processos
    def task_generator():
        for seg_id in segment_ids:
            sl = slices[seg_id - 1]
            if sl is None:
                continue
            sub_mask = (segments[sl] == seg_id)
            if not np.any(sub_mask):
                continue
            yield (seg_id, gray_image[sl], sub_mask, max_val, is_float_0_1)

    # Executa em paralelo distribuindo entre todos os núcleos de CPU disponíveis
    results = Parallel(n_jobs=n_jobs, backend="loky", batch_size=500)(
        delayed(_process_single_segment)(*args)
        for args in tqdm(task_generator(), total=len(segment_ids), desc="Extraindo Haralick")
    )

    # Filtra eventuais retornos nulos
    features_list = [res for res in results if res is not None]

    columns = ['Segment_id'] + haralick_feature_names
    features_df = pd.DataFrame(features_list, columns=columns)

    return features_df


# ==============================================================================
# 4. EXECUÇÃO PRINCIPAL
# ==============================================================================
def main():
    methods = ['SLIC']
    study_areas = ['3549904_2024']

    # Caminhos estruturados a partir do PROJECT_ROOT
    pansharpening_tif_dir = os.path.join(PROJECT_ROOT, 'workspace/data/output/pansharpening/2024')
    slic_dir = os.path.join(PROJECT_ROOT, 'workspace/data/processing/slics')
    output_ml_dir = os.path.join(PROJECT_ROOT, 'workspace/data/ML/csv_haralick_features/bands_1234')

    for method in methods:
        print(f'Método: {method}')

        for study_area in study_areas:
            pansharpened_2m_path = os.path.join(
                pansharpening_tif_dir,
                f'{study_area}_CBERS_TRUE_COLOR_2M.tif'
            )

            # Mapeamento das 4 bandas do stack pan-sharpened gerado no script 01:
            # Índice 1: Red (BAND3), Índice 2: Green (BAND2), Índice 3: Blue (BAND1), Índice 4: NIR (BAND4)
            band_sources = [
                ('1', pansharpened_2m_path, 3),  # BAND1 - Blue (2m fusionada)
                ('2', pansharpened_2m_path, 2),  # BAND2 - Green (2m fusionada)
                ('3', pansharpened_2m_path, 1),  # BAND3 - Red (2m fusionada)
                ('4', pansharpened_2m_path, 4),  # BAND4 - NIR (2m fusionada)
            ]

            # Carrega a segmentação filtrada
            segmentation_path = os.path.join(slic_dir, f'mask_slic_filtered_{study_area}.npy')
            current_segmentation = np.load(segmentation_path)

            segment_ids = np.unique(current_segmentation)
            print(f'Área de Estudo: {study_area}')
            print(f'Shape da máscara: {current_segmentation.shape} | Segmentos únicos (c/ fundo): {len(segment_ids)}')

            # Cria a subpasta de saída dentro de workspace/data/ML caso ainda não exista
            output_area_dir = os.path.join(output_ml_dir, f'AE-{study_area}')
            os.makedirs(output_area_dir, exist_ok=True)

            for band_name, tif_path, band_idx in band_sources:
                output_csv_path = os.path.join(
                    output_area_dir,
                    f'haralick_features_sa_{study_area}_B{band_name}.csv'
                )

                # Se a banda já foi processada anteriormente, pula automaticamente
                if os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 1000:
                    print(f"[OK] Banda B{band_name} já processada em '{output_csv_path}'. Pulando...\n")
                    continue

                if not os.path.exists(tif_path):
                    print(f"[ERRO] Arquivo não encontrado: {tif_path}. Pulando B{band_name}...\n")
                    continue

                with rasterio.open(tif_path) as current_band_data:
                    if band_idx > current_band_data.count:
                        print(
                            f"[AVISO] O arquivo {os.path.basename(tif_path)} possui apenas "
                            f"{current_band_data.count} bandas. Pulando B{band_name}...\n"
                        )
                        continue
                    current_band = current_band_data.read(band_idx)

                if current_band.shape != current_segmentation.shape:
                    print(
                        f"[AVISO] Shape da banda B{band_name} {current_band.shape} "
                        f"difere da máscara {current_segmentation.shape}. Pulando banda...\n"
                    )
                    continue

                print(
                    f'Computando features de Haralick para B{band_name} '
                    f'(índice {band_idx} de {os.path.basename(tif_path)})...'
                )
                features_df = haralick(current_band, current_segmentation, n_jobs=-1)

                features_df.to_csv(output_csv_path, index=False)
                print(f"Features salvas em '{output_csv_path}'.\n\n")


if __name__ == '__main__':
    main()