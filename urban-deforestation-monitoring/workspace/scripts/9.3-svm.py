# ==============================================================================
# 1. IMPORTAÇÃO DE BIBLIOTECAS
# ==============================================================================
import os
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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suprime logs INFO/WARNING do TensorFlow/absl

# ==============================================================================
# 2. DEFINIÇÃO DE PARÂMETROS E DIRETÓRIOS (BUSCA AUTOMÁTICA EM workspace/data)
# ==============================================================================
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

def find_csv_file(search_root, target_filename):
    for root, _, files in os.walk(search_root):
        if target_filename in files:
            return os.path.join(root, target_filename)
    return None

train_filename = 'sjc_3549904_2024_mapbiomas_train.csv'
test_filename = 'sjc_3549904_2024_mapbiomas_test.csv'

train_file_path = find_csv_file(data_dir, train_filename)
test_file_path = find_csv_file(data_dir, test_filename)

# Diagnóstico caso algum dos arquivos tenha nome ligeiramente diferente
if train_file_path is None or test_file_path is None:
    available_csvs = [
        os.path.relpath(os.path.join(r, f), data_dir)
        for r, _, fs in os.walk(ml_dir if os.path.exists(ml_dir) else data_dir)
        for f in fs if f.endswith('.csv')
    ]
    raise FileNotFoundError(
        f"Arquivo não encontrado em '{data_dir}'.\n"
        f"Procurados: '{train_filename}' e '{test_filename}'.\n"
        f"CSVs encontrados em data/ML: {available_csvs}"
    )

print(f"Train CSV localizado: {train_file_path}")
print(f"Test CSV localizado:  {test_file_path}")

# Diretório de saída para figuras, HTML e tabelas CSV
output_dir = os.path.join(ml_dir if os.path.exists(ml_dir) else data_dir, 'results_mapbiomas_only')
os.makedirs(output_dir, exist_ok=True)
print(f"Resultados serão salvos em: {output_dir}\n")


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

print(f"Linhas carregadas -> Treino: {len(X_train_df)} | Teste: {len(X_test_df)}")
print(f"Bandas disponíveis no treino: {sorted(X_train_df['band'].unique())}")


# ==============================================================================
# 4. UNIÃO DAS FEATURES POR SEGMENTO (1 LINHA POR SEGMENTO = 13 x N_BANDAS)
# ==============================================================================
def build_segment_dataset(dataframe, selected_bands, feature_cols, target_column):
    """
    Agrupa as linhas por Segment_id, concatena as features de Haralick na ordem
    definida em `selected_bands` e extrai os metadados/rótulos únicos por segmento.
    """
    band_order = {b: i for i, b in enumerate(selected_bands)}
    df_filtered = dataframe[dataframe['band'].isin(selected_bands)].copy()
    df_filtered['band_rank'] = df_filtered['band'].map(band_order)
    df_filtered = df_filtered.sort_values(by=['Segment_id', 'band_rank'], kind='stable')

    X_rows = []
    meta_rows = []

    for seg_id, group in df_filtered.groupby('Segment_id', sort=False):
        if len(group) != len(selected_bands):
            print(f"[AVISO] Segment_id {seg_id} possui {len(group)} bandas (esperado: {len(selected_bands)}). Ignorado.")
            continue

        seg_features = group[feature_cols].values.flatten()
        X_rows.append(seg_features)

        first_row = group.iloc[0]
        meta_rows.append({
            'Segment_id': seg_id,
            'Study_Area': first_row['Study_Area'],
            'MapBiomas_class': first_row[target_column],
            'HoR': first_row['HoR'],
            'Size': first_row['Size'],
            'HoR_group': first_row['HoR_group']
        })

    col_names = [f"B{b}_{feat}" for b in selected_bands for feat in feature_cols]
    X_segments = pd.DataFrame(X_rows, columns=col_names)
    meta_segments = pd.DataFrame(meta_rows)

    return X_segments, meta_segments

new_X_train, meta_train = build_segment_dataset(X_train_df, bands_list, HARALICK_COLS, target_col)
new_X_test, meta_test = build_segment_dataset(X_test_df, bands_list, HARALICK_COLS, target_col)

y_train_mapbiomas = meta_train['MapBiomas_class'].values
y_test_mapbiomas = meta_test['MapBiomas_class'].values

print("\n=== Resumo dos Segmentos Processados ===")
print(f"Shape X_train: {new_X_train.shape} (Segmentos x Features)")
print(f"Shape X_test:  {new_X_test.shape} (Segmentos x Features)")

print("\nDistribuição de Classes - TREINO (MapBiomas):")
print(meta_train['MapBiomas_class'].value_counts())

print("\nDistribuição de Classes - TESTE (MapBiomas):")
print(meta_test['MapBiomas_class'].value_counts())


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

    print(f"Mean 5-Fold CV Balanced Accuracy: {np.mean(scores_cv):.4f} (+/- {np.std(scores_cv):.4f})")

    # Reajusta o modelo final com 100% dos dados de treino normalizados
    model.fit(X_arr, y_arr)
    return model

svm_mapbiomas_based = SVC(C=1.0, kernel='linear', random_state=42, verbose=False)
cv_mapbiomas_based = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n=== Treinamento SVM (MapBiomas-based) ===")
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

print("\n=== Resultados no Conjunto de Teste (SJC - MapBiomas-based) ===")
print(f"Balanced Accuracy: {bal_acc_test:.4f}")
print(f"Accuracy:          {acc_test:.4f}")
print(f"F1-Score (Macro):  {f1_macro_test:.4f}\n")
print(classification_report(y_test_mapbiomas, y_test_pred, digits=4))

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
plt.close()


# ==============================================================================
# 7. ANÁLISE POR GRUPO DE HOMOGENEIDADE (HoR_group) E MARGEM SVM
# ==============================================================================
print("\n=== Desempenho por HoR_group no Teste ===")
for hor_grp in sorted(meta_test['HoR_group'].unique()):
    idx = (meta_test['HoR_group'] == hor_grp).values
    if len(np.unique(y_test_mapbiomas[idx])) > 1:
        b_acc = balanced_accuracy_score(y_test_mapbiomas[idx], y_test_pred[idx])
        print(f"Grupo = {hor_grp} | Segmentos = {np.sum(idx)} | Bal. Acc. = {b_acc:.4f}")
    else:
        acc_grp = accuracy_score(y_test_mapbiomas[idx], y_test_pred[idx])
        print(f"Grupo = {hor_grp} | Segmentos = {np.sum(idx)} | Acc. (classe única) = {acc_grp:.4f}")

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
plt.close()

# 8.4 t-SNE 3D Interativo (Plotly -> salvo em HTML)
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

print("\n=== Estimativa de Área (Hectares) no Conjunto de Teste ===")
print(area_summary.round(4))

# Salva as tabelas finais em CSV
results_test_df.to_csv(os.path.join(output_dir, 'predictions_test_sjc_mapbiomas.csv'), index=False)
area_summary.to_csv(os.path.join(output_dir, 'area_summary_sjc_mapbiomas.csv'))

print("\nScript 9.3-svm.py finalizado com sucesso!")