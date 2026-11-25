import os
import numpy as np
import pandas as pd
import mahotas as mh
from tqdm import tqdm
import rasterio

# ==============================================================================
# 1. CONFIGURAÇÃO DO DIRETÓRIO RAIZ DO PROJETO
# ==============================================================================
PROJECT_ROOT = '/content/drive/MyDrive/Mestrado/04-Projeto ForestEyes/ForestEyes/urban-deforestation-monitoring'


# ==============================================================================
# 2. FUNÇÃO DE CÁLCULO DOS DESCRITORES DE HARALICK
# ==============================================================================
def haralick(gray_image, segments):
    """
    Retorna um DataFrame com as features de Haralick e os IDs dos segmentos.
    """
    # Obtém IDs únicos dos segmentos
    segment_ids = np.unique(segments)

    # Lista para armazenar os resultados
    features_list = []

    # Nomes dos descritores de Haralick
    haralick_feature_names = [
        "ASM", "Contrast", "Correlation", "Variance", "IDM",
        "Sum_Average", "Sum_Variance", "Sum_Entropy", "Entropy",
        "Diff_Variance", "Diff_Entropy",
        "IMC1", "IMC2"
    ]

    # Loop sobre cada segmento
    for segment_id in tqdm(segment_ids):

        # Cria uma máscara booleana para o segmento atual
        mask = (segments == segment_id)

        # Verifica se o segmento não está vazio
        if np.sum(mask) == 0:
            continue

        # Extrai a região do segmento (mantém os valores originais dos pixels cinzas.
        # O restante vira 0, ao serem multiplicados por False)
        segment_region = gray_image * mask

        # Normaliza a imagem para a faixa [0, 255]
        normalized_segment = (segment_region * 255).astype(np.uint8)

        # Verifica se o segmento não é completamente vazio
        if np.sum(normalized_segment) == 0:
            continue

        # Calcula os descritores de Haralick
        haralick_features = mh.features.haralick(normalized_segment).mean(axis=0)

        # Adiciona os resultados à lista
        features_list.append([segment_id] + haralick_features.tolist())

    # Cria um DataFrame com os resultados
    columns = ['Segment_id'] + haralick_feature_names
    features_df = pd.DataFrame(features_list, columns=columns)

    return features_df


# ==============================================================================
# 3. EXECUÇÃO PRINCIPAL
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

        for id_area, study_area in enumerate(study_areas):
            # Lista com os caminhos das bandas do CBERS-4A
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
            print(f'Shape da máscara: {current_segmentation.shape} | Segmentos únicos: {len(segment_ids)}')

            # Cria a subpasta de saída dentro de workspace/data/ML caso ainda não exista
            output_area_dir = os.path.join(output_ml_dir, f'AE-{study_area}')
            os.makedirs(output_area_dir, exist_ok=True)

            for id, tif_band in enumerate(bands_path):
                with rasterio.open(tif_band) as current_band_data:
                    current_band = current_band_data.read(1)

                # Verifica compatibilidade de dimensões (H, W) entre a banda e a máscara SLIC
                if current_band.shape != current_segmentation.shape:
                    print(
                        f"AVISO: Shape da banda B{band_names[id]} {current_band.shape} "
                        f"difere da máscara {current_segmentation.shape}. Pulando banda...\n"
                    )
                    continue

                print(f'Computando as features de Haralick para a banda: {tif_band}')
                features_df = haralick(current_band, current_segmentation)

                # Salva o DataFrame em um arquivo CSV
                output_csv_path = os.path.join(
                    output_area_dir,
                    f'haralick_features_sa_{study_area}_B{band_names[id]}.csv'
                )
                features_df.to_csv(output_csv_path, index=False)
                print(f"Features salvas em '{output_csv_path}'.\n\n")


if __name__ == '__main__':
    main()