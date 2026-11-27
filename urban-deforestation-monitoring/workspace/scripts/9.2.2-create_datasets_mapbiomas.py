import os
import pandas as pd
import numpy as np

# ==============================================================================
# 1. CONFIGURAÇÃO DO DIRETÓRIO RAIZ E PARÂMETROS DO EXPERIMENTO (MAPBIOMAS)
# ==============================================================================
PROJECT_ROOT = '/content/drive/MyDrive/Mestrado/04-Projeto ForestEyes/ForestEyes/urban-deforestation-monitoring'

STUDY_AREA = '3549904_2024'
CODE_MUNI = '3549904'
BANDS = [4, 3, 2, 1]  # Bandas CBERS-4A fusionadas (NIR, Red, Green, Blue)

# Estratificação solicitada: 150 segmentos no total (100 Forest + 50 Non-forest)
# - Forest:     50 HoR == 1.0 + 50 HoR mais próximo possível de 0.7
# - Non-forest: 25 HoR == 1.0 + 25 HoR mais próximo possível de 0.7
SAMPLES_CONFIG = {
    'Forest': {'hor_1': 50, 'hor_07': 50},
    'Non-forest': {'hor_1': 25, 'hor_07': 25},
}
RANDOM_SEED = 42


# ==============================================================================
# 2. FUNÇÕES AUXILIARES DE CARREGAMENTO E AMOSTRAGEM ESTRATIFICADA
# ==============================================================================
def get_available_workflow_tasks(project_root, code_muni):
    """
    Lê os IDs das tarefas existentes nas pastas forest e non-forest do workflow Zooniverse.
    """
    base_wf = os.path.join(
        project_root,
        f'workspace/data/out_workflow/cbers_workflow/AE-X{code_muni}'
    )
    valid_ids = set()
    for cls_folder in ['forest', 'non-forest']:
        folder_path = os.path.join(base_wf, cls_folder)
        if os.path.exists(folder_path):
            for entry in os.listdir(folder_path):
                if entry.startswith('task_'):
                    try:
                        valid_ids.add(int(entry.replace('task_', '')))
                    except ValueError:
                        pass
    return valid_ids


def load_all_haralick_bands(project_root, study_area, bands):
    """
    Carrega os 4 CSVs de Haralick e empilha em formato longo (Segment_id, band, 13 features).
    """
    haralick_dir = os.path.join(
        project_root,
        f'workspace/data/ML/csv_haralick_features/bands_1234/AE-{study_area}'
    )
    dfs_bands = []
    for band in bands:
        file_path = os.path.join(haralick_dir, f'haralick_features_sa_{study_area}_B{band}.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Arquivo de Haralick não encontrado: {file_path}")

        df_b = pd.read_csv(file_path)
        df_b.rename(columns={'Segment ID': 'Segment_id'}, inplace=True)
        df_b['Segment_id'] = df_b['Segment_id'].astype(int)
        df_b.insert(1, 'band', int(band))
        dfs_bands.append(df_b)

    return pd.concat(dfs_bands, ignore_index=True)


def load_mapbiomas_segments(project_root, study_area):
    """
    Carrega os segmentos aprovados na etapa 08.4 (verdade de referência MapBiomas).
    """
    csv_dir = os.path.join(project_root, 'workspace/data/processing/csv')
    df_forest = pd.read_csv(os.path.join(csv_dir, 'forest', f'df_forest_sort_{study_area}.csv'))
    df_forest['MapBiomas_class'] = 'Forest'

    df_non_forest = pd.read_csv(os.path.join(csv_dir, 'non-forest', f'df_non-forest_sort_{study_area}.csv'))
    df_non_forest['MapBiomas_class'] = 'Non-forest'

    df_all = pd.concat([df_forest, df_non_forest], ignore_index=True)
    df_all.rename(columns={'Segment_label': 'Segment_id'}, inplace=True)
    df_all['Segment_id'] = df_all['Segment_id'].astype(int)
    df_all['Study_Area'] = study_area
    return df_all[['Segment_id', 'Study_Area', 'MapBiomas_class', 'HoR', 'Size']]


def sample_stratified_by_hor(df_class, n_hor_1=50, n_hor_07=50, seed=42):
    """
    Seleciona n_hor_1 segmentos com HoR == 1.0 e n_hor_07 segmentos
    com HoR mais próximo de 0.7 (HoR >= 0.7).
    """
    # Subgrupo 1: HoR == 1.0
    df_hor_100 = df_class[np.isclose(df_class['HoR'], 1.0, atol=1e-6)]
    if len(df_hor_100) < n_hor_1:
        raise ValueError(f"Apenas {len(df_hor_100)} segmentos com HoR=1.0 disponíveis (necessário: {n_hor_1}).")
    sample_100 = df_hor_100.sample(n=n_hor_1, random_state=seed).copy()
    sample_100['HoR_group'] = 'HoR_1.0'

    # Subgrupo 2: HoR == 0.7 ou o mais próximo disso (excluindo os já selecionados no subgrupo 1)
    remaining = df_class[~df_class['Segment_id'].isin(sample_100['Segment_id'])].copy()
    remaining = remaining[remaining['HoR'] >= 0.7].copy()
    if len(remaining) < n_hor_07:
        raise ValueError(f"Apenas {len(remaining)} segmentos restantes para HoR≈0.7 (necessário: {n_hor_07}).")

    remaining['_dist_07'] = (remaining['HoR'] - 0.7).abs()
    # Ordena pela menor distância de 0.7 (com desempate aleatório reprodutível)
    remaining = remaining.sample(frac=1.0, random_state=seed).sort_values(by='_dist_07', kind='mergesort')
    sample_070 = remaining.head(n_hor_07).drop(columns=['_dist_07']).copy()
    sample_070['HoR_group'] = 'HoR_0.7'

    return pd.concat([sample_100, sample_070], ignore_index=True)


# ==============================================================================
# 3. EXECUÇÃO PRINCIPAL (9.2.2 - MAPBIOMAS)
# ==============================================================================
def main():
    print("=" * 80)
    print(f"ETAPA 9.2.2 - DATASETS DE TREINO E TESTE: VERDADE MAPBIOMAS ({STUDY_AREA})")
    print("=" * 80)

    out_base = os.path.join(PROJECT_ROOT, 'workspace/data/ML/datasets/mapbiomas')
    train_dir = os.path.join(out_base, 'train')
    test_dir = os.path.join(out_base, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print("1. Verificando tarefas disponíveis nas pastas do workflow (forest / non-forest)...")
    valid_task_ids = get_available_workflow_tasks(PROJECT_ROOT, CODE_MUNI)
    print(f"   -> Tarefas encontradas nas pastas: {len(valid_task_ids):,}")

    print("2. Carregando segmentos aprovados com referência MapBiomas...")
    df_meta = load_mapbiomas_segments(PROJECT_ROOT, STUDY_AREA)
    if valid_task_ids:
        df_meta = df_meta[df_meta['Segment_id'].isin(valid_task_ids)].copy()

    print(f"   -> Total de segmentos elegíveis: {len(df_meta):,}")

    print("3. Aplicando amostragem estratificada (100 Forest [50+50] + 50 Non-forest [25+25])...")
    train_forest = sample_stratified_by_hor(
        df_meta[df_meta['MapBiomas_class'] == 'Forest'],
        n_hor_1=SAMPLES_CONFIG['Forest']['hor_1'],
        n_hor_07=SAMPLES_CONFIG['Forest']['hor_07'],
        seed=RANDOM_SEED
    )
    train_non_forest = sample_stratified_by_hor(
        df_meta[df_meta['MapBiomas_class'] == 'Non-forest'],
        n_hor_1=SAMPLES_CONFIG['Non-forest']['hor_1'],
        n_hor_07=SAMPLES_CONFIG['Non-forest']['hor_07'],
        seed=RANDOM_SEED
    )

    df_train_meta = pd.concat([train_forest, train_non_forest], ignore_index=True)
    train_ids = set(df_train_meta['Segment_id'])

    # Todos os demais segmentos aprovados (fora dos 150 de treino) vão para Teste
    df_test_meta = df_meta[~df_meta['Segment_id'].isin(train_ids)].copy()

    total_train_segs = len(df_train_meta)
    print(f"\nResumo da seleção de TREINO ({total_train_segs} segmentos MapBiomas):")
    print(df_train_meta.groupby(['MapBiomas_class', 'HoR_group'])['HoR'].agg(['count', 'min', 'max']))

    print("\n4. Cruzando com as 13 features de Haralick nas bandas [4, 3, 2, 1]...")
    df_haralick = load_all_haralick_bands(PROJECT_ROOT, STUDY_AREA, BANDS)
    band_order_df = pd.DataFrame({'band': BANDS, '_band_order': range(len(BANDS))})

    df_train_final = (
        df_train_meta
        .merge(df_haralick, on='Segment_id', how='inner')
        .merge(band_order_df, on='band', how='left')
        .sort_values(by=['Segment_id', '_band_order'])
        .drop(columns=['_band_order'])
    )

    df_test_final = (
        df_test_meta
        .merge(df_haralick, on='Segment_id', how='inner')
        .merge(band_order_df, on='band', how='left')
        .sort_values(by=['Segment_id', '_band_order'])
        .drop(columns=['_band_order'])
    )

    train_csv_path = os.path.join(train_dir, f'sjc_{STUDY_AREA}_mapbiomas_train.csv')
    test_csv_path = os.path.join(test_dir, f'sjc_{STUDY_AREA}_mapbiomas_test.csv')

    df_train_final.to_csv(train_csv_path, index=False)
    df_test_final.to_csv(test_csv_path, index=False)

    print("\n" + "=" * 80)
    print("SUCESSO (9.2.2 - MAPBIOMAS)!")
    print(f" -> TREINO ({len(df_train_final)//len(BANDS)} segmentos | {len(df_train_final)} linhas): {train_csv_path}")
    print(f" -> TESTE  ({len(df_test_final)//len(BANDS):,} segmentos | {len(df_test_final):,} linhas): {test_csv_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()