import os
import sys

# Deve vir ANTES de qualquer import que carregue TensorFlow/absl (ex: umap)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

# ==============================================================================
# 1. IMPORTAÇÃO DE BIBLIOTECAS
# ==============================================================================
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    ConfusionMatrixDisplay
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


# ==============================================================================
# 2. DEFINIÇÃO DE PARÂMETROS E DIRETÓRIOS (ESTRUTURA data/ML/datasets/mapbiomas)
# ==============================================================================
t_start = time.time()

# Bandas CBERS-4A WPM fusionado (2m): [4, 3, 2, 1] (NIR, Red, Green, Blue)
bands_list = [4, 3, 2, 1]  # Pode testar subconjuntos, ex: [4, 3, 2]

# Áreas de estudo para descartar (ex: ['3549904_2024']), vazio = usa tudo
sa_to_discard_train = []
sa_to_discard_test = []

# Coluna alvo de verdade terrestre
target_col = 'MapBiomas_class'

# Lista explícita das 13 features de textura de Haralick (evita erro de índice)
HARALICK_COLS = [
    'ASM', 'Contrast', 'Correlation', 'Variance', 'IDM',
    'Sum_Average', 'Sum_Variance', 'Sum_Entropy', 'Entropy',
    'Diff_Variance', 'Diff_Entropy', 'IMC1', 'IMC2'
]

# Resolve o diretório workspace automaticamente a partir de workspace/scripts/9.3-svm.py
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(script_dir)
data_dir = os.path.join(workspace_dir, 'data')
ml_dir = os.path.join(data_dir, 'ML')
mapbiomas_dataset_dir = os.path.join(ml_dir, 'datasets', 'mapbiomas')

train_filename = 'sjc_3549904_2024_mapbiomas_train.csv'
test_filename = 'sjc_3549904_2024_mapbiomas_test.csv'

train_file_path = os.path.join(mapbiomas_dataset_dir, 'train', train_filename)
test_file_path = os.path.join(mapbiomas_dataset_dir, 'test', test_filename)

# Fallback rápido apenas dentro de data/ML caso mude de subpasta
if not os.path.isfile(train_file_path) or not os.path.isfile(test_file_path):
    for root, _, files in os.walk(ml_dir):
        if train_filename in files:
            train_file_path = os.path.join(root, train_filename)
        if test_filename in files:
            test_file_path = os.path.join(root, test_filename)

if not os.path.isfile(train_file_path) or not os.path.isfile(test_file_path):
    raise FileNotFoundError(
        f"Arquivos não encontrados.\n"
        f"Esperado treino: {train_file_path}\n"
        f"Esperado teste:  {test_file_path}"
    )

# Diretório de saída dentro de workspace/data/ML/results_mapbiomas_only
output_dir = os.path.join(ml_dir, 'results_mapbiomas_only')
os.makedirs(output_dir, exist_ok=True)

log_lines = []


def log_print(msg=""):
    """Imprime no console/célula com flush imediato e acumula para salvar em .txt."""
    print(msg, flush=True)
    log_lines.append(str(msg))


log_print(f"Train CSV localizado: {train_file_path}")
log_print(f"Test CSV localizado:  {test_file_path}")
log_print(f"Resultados serão salvos em: {output_dir}\n")


# ==============================================================================
# 3. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
# ==============================================================================
def remove_samples_per_areas(df, sa_to_discard):
    if not sa_to_discard:
        return df.copy()
    return df[~df['Study_Area'].isin(sa_to_discard)].reset_index(drop=True)


df_train_raw = pd.read_csv(train_file_path, sep=',')
df_test_raw = pd.read_csv(test_file_path, sep=',')

X_train_df = remove_samples_per_areas(df_train_raw, sa_to_discard_train)
X_test_df = remove_samples_per_areas(df_test_raw, sa_to_discard_test)

log_print(f"Linhas carregadas -> Treino: {len(X_train_df)} | Teste: {len(X_test_df)}")
log_print(f"Bandas disponíveis no treino: {sorted(X_train_df['band'].unique())}")


# ==============================================================================
# 4. UNIÃO VETORIZADA DAS FEATURES POR SEGMENTO (1 LINHA = 13 x N_BANDAS)
# ==============================================================================
def assign_hor_group(hor_series):
    """Deriva HoR_group caso a coluna não esteja presente no CSV (ex: conjunto de teste)."""
    conds = [
        hor_series >= 0.9999,
        (hor_series >= 0.7) & (hor_series < 0.9999),
    ]
    choices = ['HoR_1.0', 'HoR_0.7']
    return np.select(conds, choices, default='HoR_<0.7')


def build_segment_dataset(dataframe, selected_bands, feature_cols, target_column):
    """
    Versão vetorizada de alta performance: pivota as bandas por Segment_id
    sem loops Python, reduzindo drasticamente o consumo de RAM e o tempo.
    """
    df_filtered = dataframe[dataframe['band'].isin(selected_bands)].copy()

    # Filtra apenas segmentos completos (com todas as bandas selecionadas)
    band_counts = df_filtered.groupby('Segment_id')['band'].nunique()
    valid_segments = band_counts[band_counts == len(selected_bands)].index
    df_valid = df_filtered[df_filtered['Segment_id'].isin(valid_segments)]

    # Metadados únicos por segmento (seleciona apenas colunas presentes no CSV)
    desired_meta_cols = ['Segment_id', 'Study_Area', target_column, 'HoR', 'Size', 'HoR_group']
    available_meta_cols = [c for c in desired_meta_cols if c in df_valid.columns]

    meta_segments = (
        df_valid[available_meta_cols]
        .drop_duplicates(subset=['Segment_id'])
        .sort_values('Segment_id')
        .rename(columns={target_column: 'MapBiomas_class'})
        .reset_index(drop=True)
    )

    # Garante a coluna HoR_group mesmo se não vier pronta no CSV de teste
    if 'HoR_group' not in meta_segments.columns and 'HoR' in meta_segments.columns:
        meta_segments['HoR_group'] = assign_hor_group(meta_segments['HoR'])

    # Pivota as 13 features de Haralick por (Segment_id x band)
    pivoted = df_valid.pivot(index='Segment_id', columns='band', values=feature_cols)
    pivoted = pivoted.sort_index()

    # Reordena colunas na ordem exata de `selected_bands` (ex: B4_ASM, ..., B1_IMC2)
    ordered_cols = [(feat, b) for b in selected_bands for feat in feature_cols]
    pivoted = pivoted[ordered_cols]
    pivoted.columns = [f"B{b}_{feat}" for (feat, b) in ordered_cols]

    X_segments = pivoted.reset_index(drop=True)
    return X_segments, meta_segments


new_X_train, meta_train = build_segment_dataset(X_train_df, bands_list, HARALICK_COLS, target_col)
new_X_test, meta_test = build_segment_dataset(X_test_df, bands_list, HARALICK_COLS, target_col)

# Libera dataframes brutos da memória
del df_train_raw, df_test_raw, X_train_df, X_test_df

y_train_mapbiomas = meta_train['MapBiomas_class'].values
y_test_mapbiomas = meta_test['MapBiomas_class'].values

log_print("\n=== Resumo dos Segmentos Processados ===")
log_print(f"Shape X_train: {new_X_train.shape} (Segmentos x Features)")
log_print(f"Shape X_test:  {new_X_test.shape} (Segmentos x Features)")

log_print("\nDistribuição de Classes - TREINO (MapBiomas):")
log_print(meta_train['MapBiomas_class'].value_counts().to_string())

log_print("\nDistribuição de Classes - TESTE (MapBiomas):")
log_print(meta_test['MapBiomas_class'].value_counts().to_string())


# ==============================================================================
# 5. NORMALIZAÇÃO DOS DADOS (MAPBIOMAS-BASED)
# ==============================================================================
mapbiomas_scaler = StandardScaler()
X_train_mapbiomas_scaled = mapbiomas_scaler.fit_transform(new_X_train)
X_test_mapbiomas_scaled = mapbiomas_scaler.transform(new_X_test)


# ==============================================================================
# 6. TREINAMENTO COM 5-FOLD STRATIFIED CROSS-VALIDATION E AVALIAÇÃO NO TESTE
# ==============================================================================
def train_5_fold_cross_validation(model, cv, X_train, y_train):
    scores_cv = []
    X_arr = np.array(X_train)
    y_arr = np.array(y_train)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_arr, y_arr), start=1):
        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

        model.fit(X_tr, y_tr)
        y_val_pred = model.predict(X_val)
        scores_cv.append(balanced_accuracy_score(y_val, y_val_pred))

    log_print(f"Mean 5-Fold CV Balanced Accuracy: {np.mean(scores_cv):.4f} (+/- {np.std(scores_cv):.4f})")

    # Reajusta o modelo final com 100% dos dados de treino normalizados
    model.fit(X_arr, y_arr)
    return model


# class_weight=None respeita a proporção 2:1 (100 Forest : 50 Non-forest)
svm_mapbiomas_based = SVC(C=1.0, kernel='linear', class_weight=None, random_state=42, verbose=False)
cv_mapbiomas_based = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

log_print("\n=== Treinamento SVM (MapBiomas-based) ===")
svm_mapbiomas_based = train_5_fold_cross_validation(
    svm_mapbiomas_based,
    cv_mapbiomas_based,
    X_train_mapbiomas_scaled,
    y_train_mapbiomas
)

# Predição no conjunto de teste
y_test_pred = svm_mapbiomas_based.predict(X_test_mapbiomas_scaled)
bal_acc_test = balanced_accuracy_score(y_test_mapbiomas, y_test_pred)
acc_test = accuracy_score(y_test_mapbiomas, y_test_pred)
f1_macro_test = f1_score(y_test_mapbiomas, y_test_pred, average='macro')

log_print("\n=== Resultados no Conjunto de Teste (SJC - MapBiomas-based) ===")
log_print(f"Balanced Accuracy: {bal_acc_test:.4f}")
log_print(f"Accuracy:          {acc_test:.4f}")
log_print(f"F1-Score (Macro):  {f1_macro_test:.4f}\n")
log_print(classification_report(y_test_mapbiomas, y_test_pred, digits=4))

# Matriz de Confusão
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test_mapbiomas,
    y_test_pred,
    cmap='Greens',
    ax=ax
)
plt.title(f"SVM MapBiomas-based (Bal. Acc: {bal_acc_test:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix_mapbiomas.png'), dpi=300)
plt.show()
plt.close()


# ==============================================================================
# 7. ANÁLISE POR GRUPO DE HOMOGENEIDADE (HoR_group) E MARGEM SVM
# ==============================================================================
if 'HoR_group' in meta_test.columns:
    log_print("\n=== Desempenho por HoR_group no Teste ===")
    for hor_grp in sorted(meta_test['HoR_group'].unique()):
        idx = (meta_test['HoR_group'] == hor_grp).values
        if len(np.unique(y_test_mapbiomas[idx])) > 1:
            b_acc = balanced_accuracy_score(y_test_mapbiomas[idx], y_test_pred[idx])
            log_print(f"Grupo = {hor_grp} | Segmentos = {np.sum(idx)} | Bal. Acc. = {b_acc:.4f}")
        else:
            acc_grp = accuracy_score(y_test_mapbiomas[idx], y_test_pred[idx])
            log_print(f"Grupo = {hor_grp} | Segmentos = {np.sum(idx)} | Acc. (classe única) = {acc_grp:.4f}")

# Gráfico de Margem (Decision Function) do SVM no treino por HoR_group e Classe
scores_train = svm_mapbiomas_based.decision_function(X_train_mapbiomas_scaled)
hor_groups = sorted(meta_train['HoR_group'].unique())

plt.figure(figsize=(8, 5))
for pos, grp in enumerate(hor_groups):
    mask_f = (meta_train['HoR_group'] == grp) & (meta_train['MapBiomas_class'] == 'Forest')
    mask_nf = (meta_train['HoR_group'] == grp) & (meta_train['MapBiomas_class'] == 'Non-forest')

    plt.scatter(np.repeat(pos - 0.05, mask_f.sum()), scores_train[mask_f], color='green', alpha=0.7, label='Forest' if pos == 0 else "")
    plt.scatter(np.repeat(pos + 0.05, mask_nf.sum()), scores_train[mask_nf], color='orange', alpha=0.7, label='Non-forest' if pos == 0 else "")

plt.axhline(0, color='black', linestyle='--')
plt.xticks(range(len(hor_groups)), hor_groups)
plt.xlabel("Grupo de Homogeneidade (HoR_group)")
plt.ylabel("Decision Function (Margem SVM)")
plt.title("Margem do SVM por HoR_group e Classe (Treino)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'svm_margin_by_hor.png'), dpi=300)
plt.show()
plt.close()


# ==============================================================================
# 8. REDUÇÃO DE DIMENSIONALIDADE: PCA, UMAP E t-SNE (2D E 3D)
# ==============================================================================
# 8.1 PCA 2D
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_train_mapbiomas_scaled)

plt.figure(figsize=(8, 6))
for classe, cor in [('Forest', 'green'), ('Non-forest', 'orange')]:
    idx = (y_train_mapbiomas == classe)
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=classe, color=cor, alpha=0.8)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("PCA 2D - Treino SJC (MapBiomas)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pca_2d_mapbiomas.png'), dpi=300)
plt.show()
plt.close()

# 8.2 UMAP 2D
n_neighbors_umap = min(15, max(2, len(X_train_mapbiomas_scaled) - 1))
reducer = umap.UMAP(n_neighbors=n_neighbors_umap, random_state=42)
X_umap = reducer.fit_transform(X_train_mapbiomas_scaled)

plt.figure(figsize=(8, 6))
for classe, cor in [('Forest', 'green'), ('Non-forest', 'orange')]:
    idx = (y_train_mapbiomas == classe)
    plt.scatter(X_umap[idx, 0], X_umap[idx, 1], label=classe, color=cor, alpha=0.8)
plt.title("UMAP 2D - Treino SJC (MapBiomas)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'umap_2d_mapbiomas.png'), dpi=300)
plt.show()
plt.close()

# 8.3 t-SNE 2D (com PCA intermediário dinâmico)
n_pca_inter = min(20, X_train_mapbiomas_scaled.shape[1], X_train_mapbiomas_scaled.shape[0] - 1)
X_pca_inter = PCA(n_components=n_pca_inter, random_state=42).fit_transform(X_train_mapbiomas_scaled)

perp = min(30, max(5, len(X_train_mapbiomas_scaled) // 4))
tsne_2d = TSNE(n_components=2, perplexity=perp, random_state=42)
X_tsne_2 = tsne_2d.fit_transform(X_pca_inter)

plt.figure(figsize=(8, 6))
for classe, cor in [('Forest', 'green'), ('Non-forest', 'orange')]:
    idx = (y_train_mapbiomas == classe)
    plt.scatter(X_tsne_2[idx, 0], X_tsne_2[idx, 1], label=classe, color=cor, alpha=0.8)
plt.title("t-SNE 2D por Classe - Treino SJC (MapBiomas)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'tsne_2d_mapbiomas.png'), dpi=300)
plt.show()
plt.close()

# 8.4 t-SNE 3D Interativo (Plotly -> exibe na célula e salva em HTML)
tsne_3d = TSNE(n_components=3, perplexity=perp, random_state=42)
X_tsne_3 = tsne_3d.fit_transform(X_pca_inter)

df_for_plot = pd.DataFrame({
    'TSNE1': X_tsne_3[:, 0],
    'TSNE2': X_tsne_3[:, 1],
    'TSNE3': X_tsne_3[:, 2],
    'Classe': y_train_mapbiomas,
    'HoR_group': meta_train['HoR_group'].values,
    'Segment_id': meta_train['Segment_id'].values
})

fig = px.scatter_3d(
    df_for_plot,
    x='TSNE1',
    y='TSNE2',
    z='TSNE3',
    color='Classe',
    symbol='HoR_group',
    hover_data=['Segment_id', 'HoR_group'],
    color_discrete_map={'Forest': 'green', 'Non-forest': 'orange'},
    title='t-SNE 3D - SJC (MapBiomas)'
)
fig.write_html(os.path.join(output_dir, 'tsne_3d_mapbiomas.html'))
fig.show()


# ==============================================================================
# 9. CÁLCULO DE ÁREA EM HECTARES (CBERS-4A 2m x 2m = 4 m² por pixel)
# ==============================================================================
PIXEL_AREA_M2 = 4.0  # Resolução espacial CBERS-4A fusionado (2m)
M2_PER_HECTARE = 10000.0

results_test_df = meta_test.copy()
results_test_df['Predicted_class'] = y_test_pred
results_test_df['Area_ha'] = (results_test_df['Size'] * PIXEL_AREA_M2) / M2_PER_HECTARE

area_real = results_test_df.groupby('MapBiomas_class')['Area_ha'].sum()
area_pred = results_test_df.groupby('Predicted_class')['Area_ha'].sum()

area_summary = pd.DataFrame({
    'Area_Real_MapBiomas_ha': area_real,
    'Area_Predita_SVM_ha': area_pred
}).fillna(0)
area_summary['Diferenca_ha'] = area_summary['Area_Predita_SVM_ha'] - area_summary['Area_Real_MapBiomas_ha']

log_print("\n=== Estimativa de Área (Hectares) no Conjunto de Teste ===")
log_print(area_summary.round(4).to_string())

# Matriz de Confusão em Hectares (Verdade MapBiomas vs. Predição SVM CBERS-4A)
area_cm = pd.crosstab(
    results_test_df['MapBiomas_class'],
    results_test_df['Predicted_class'],
    values=results_test_df['Area_ha'],
    aggfunc='sum',
    margins=True,
    margins_name='Total_ha'
).fillna(0)

log_print("\n=== Matriz de Confusão em HECTARES (MapBiomas 2020-2024 vs. SVM CBERS-4A 2024) ===")
log_print(area_cm.round(4).to_string())

# Salva as tabelas finais em CSV
results_test_df.to_csv(os.path.join(output_dir, 'predictions_test_sjc_mapbiomas.csv'), index=False)
area_summary.to_csv(os.path.join(output_dir, 'area_summary_sjc_mapbiomas.csv'))
area_cm.to_csv(os.path.join(output_dir, 'confusion_matrix_hectares_sjc.csv'))

elapsed = time.time() - t_start
log_print(f"\nScript 9.3-svm.py finalizado com sucesso em {elapsed:.2f}s!")

# Salva o log completo em TXT na pasta de saída
with open(os.path.join(output_dir, 'execution_log_mapbiomas.txt'), 'w', encoding='utf-8') as f:
    f.write("\n".join(log_lines) + "\n")