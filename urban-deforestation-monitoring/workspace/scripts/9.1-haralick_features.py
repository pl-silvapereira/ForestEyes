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
def _process_single_segment(segment_id, slice_obj, gray_image, segments, is_float_0_1):
    if slice_obj is None:
        return None

    # Recorta apenas a caixa delimitadora (bounding box) do segmento
    sub_mask = (segments[slice_obj] == segment_id)
    if not np.any(sub_mask):
        return None

    sub_gray = gray_image[slice_obj]
    segment_region = sub_gray * sub_mask

    # Normaliza para [0, 255] uint8
    if is_float_0_1:
        normalized_segment = (segment_region * 255).astype(np.uint8)
    else:
        # Caso a imagem original venha em uint16 (ex: 0-1023 ou 0-10000), evita overflow
        max_val = gray_image.max()
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
    bounding box e processamento paralelo em CPU.
    """
    haralick_feature_names = [
        "ASM", "Contrast", "Correlation", "Variance", "IDM",
        "Sum_Average", "Sum_Variance", "Sum_Entropy", "Entropy",
        "Diff_Variance", "Diff_Entropy",
        "IMC1", "IMC2"
    ]

    # Encontra instantaneamente o bounding box de todos os IDs de segmentos (1 única passada na matriz)
    slices = find_objects(segments)

    # Identifica os IDs válidos (ignorando 0, que representa o fundo filtrado)
    segment_ids = np.unique(segments)
    segment_ids = segment_ids[segment_ids > 0]

    # Verifica se a imagem já está normalizada entre 0 e 1
    is_float_0_1 = np.issubdtype(gray_image.dtype, np.floating) and (gray_image.max() <= 1.0)

    # Executa em paralelo usando todos os núcleos de CPU disponíveis (n_jobs=-1)
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_process_single_segment)(
            seg_id,
            slices[seg_id - 1],
            gray_image,
            segments,
            is_float_0_1
        )
        for seg_id in tqdm(segment_ids, desc="Extraindo Haralick")
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

    # Área de estudo única: São José dos Campos (IBGE 3549904) - Ano 2024
    study_areas = ['3549904_2024']
    band_names = ['0', '1', '2', '3', '4']

    # Caminhos estruturados a partir do PROJECT_ROOT
    input_tif_dir = os.path.join(PROJECT_ROOT, 'workspace/data/input/CBERS-4A-WPM/2024')
    slic_dir = os.path.join(PROJECT_ROOT, 'workspace/data/processing/slics')
    output_ml_dir = os.path.join(PROJECT_ROOT, 'workspace/data/ML/csv_haralick_features/bands_01234')

    for method in methods:
        print(f'Método: {method}')

        for study_area in study_areas:
            # Atenção: ajuste o sufixo (_recortado.tif ou .tif) conforme os arquivos salvos na sua pasta
            bands_path = [
                os.path.join(input_tif_dir, 'CBERS4A_WPM20214220240930_BAND0_recortado.tif'),
                os.path.join(input_tif_dir, 'CBERS4A_WPM20214220240930_BAND1_recortado.tif'),
                os.path.join(input_tif_dir, 'CBERS4A_WPM20214220240930_BAND2_recortado.tif'),
                os.path.join(input_tif_dir, 'CBERS4A_WPM20214220240930_BAND3_recortado.tif'),
                os.path.join(input_tif_dir, 'CBERS4A_WPM20214220240930_BAND4_recortado.tif')
            ]

            # Carrega a segmentação filtrada
            segmentation_path = os.path.join(slic_dir, f'mask_slic_filtered_{study_area}.npy')
            current_segmentation = np.load(segmentation_path)

            segment_ids = np.unique(current_segmentation)
            print(f'Area de Estudo: {study_area}')
            print(f'Shape da máscara: {current_segmentation.shape} | Segmentos únicos (c/ fundo): {len(segment_ids)}')

            output_area_dir = os.path.join(output_ml_dir, f'AE-{study_area}')
            os.makedirs(output_area_dir, exist_ok=True)

            for idx, tif_band in enumerate(bands_path):
                if not os.path.exists(tif_band):
                    print(f"Arquivo não encontrado: {tif_band}. Pulando...\n")
                    continue

                with rasterio.open(tif_band) as current_band_data:
                    current_band = current_band_data.read(1)

                if current_band.shape != current_segmentation.shape:
                    print(
                        f"AVISO: Shape da banda B{band_names[idx]} {current_band.shape} "
                        f"difere da máscara {current_segmentation.shape}. Pulando banda...\n"
                    )
                    continue

                print(f'Computando as features de Haralick para a banda: {tif_band}')
                features_df = haralick(current_band, current_segmentation, n_jobs=-1)

                output_csv_path = os.path.join(
                    output_area_dir,
                    f'haralick_features_sa_{study_area}_B{band_names[idx]}.csv'
                )
                features_df.to_csv(output_csv_path, index=False)
                print(f"Features salvas em '{output_csv_path}'.\n\n")


if __name__ == '__main__':
    main()