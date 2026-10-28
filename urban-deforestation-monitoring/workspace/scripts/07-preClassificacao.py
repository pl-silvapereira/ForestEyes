import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importações Geoespaciais
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from matplotlib.colors import ListedColormap

# Importações do Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    balanced_accuracy_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)

# =====================================================================
# 1. CAMINHOS E DIRETÓRIOS (GOOGLE DRIVE)
# =====================================================================
BASE_DIR = '/content/drive/MyDrive/Mestrado/04-Projeto ForestEyes/ForestEyes/urban-deforestation-monitoring/workspace'

# Entradas CSV (Características)
PATH_FOREST = f'{BASE_DIR}/reports/dataframes_csv/forest/df_forest_sort_3549904_2024.csv'
PATH_NON_FOREST = f'{BASE_DIR}/reports/dataframes_csv/non-forest/df_non-forest_sort_3549904_2024.csv'

# Entradas Espaciais (Base para gerar os mapas)
SHP_SEGMENTOS = f'{BASE_DIR}/data/output/segmentation/2024/3549904_Segmentar_vs_NaoSegmentar_2024.shp'
TIF_ORIGINAL = f'{BASE_DIR}/data/output/pansharpening/2024/3549904_2024_CBERS_TRUE_COLOR_2M.tif'

# Saídas (Onde os arquivos gerados serão salvos)
OUT_DIR = f'{BASE_DIR}/data/reports/2024_Classificacao_Desmatamento'

# Criar a pasta de saída se ela não existir
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Diretório de saída garantido em: {OUT_DIR}")

# =====================================================================
# 2. FUNÇÃO DE REESTRUTURAÇÃO (ACHATAMENTO DE BANDAS)
# =====================================================================
def preparar_dataframe(df_path, label_alvo):
    df = pd.read_csv(df_path)
    col_segmento = 'Segment_ID' if 'Segment_ID' in df.columns else 'segment_id' if 'segment_id' in df.columns else None
    col_banda = 'Band' if 'Band' in df.columns else 'band' if 'band' in df.columns else None

    if col_segmento and col_banda:
        cols_features = [c for c in df.columns if c not in [col_segmento, col_banda, 'geometry']]
        df_flat = df.pivot(index=col_segmento, columns=col_banda, values=cols_features)
        df_flat.columns = [f"{feat}_B{band}" for feat, band in df_flat.columns]
        df_flat = df_flat.reset_index()
        df = df_flat
    
    if 'geometry' in df.columns:
        df = df.drop(columns=['geometry'])
        
    df['Target'] = label_alvo
    return df

# =====================================================================
# 3. CARREGAMENTO E PREPARAÇÃO DA MATRIZ
# =====================================================================
print("\nCarregando dados estruturados...")
df_forest = preparar_dataframe(PATH_FOREST, label_alvo=1)
df_non_forest = preparar_dataframe(PATH_NON_FOREST, label_alvo=0)
df_completo = pd.concat([df_forest, df_non_forest], ignore_index=True)

# Guardar os IDs para cruzar com o Shapefile depois
col_id = 'Segment_ID' if 'Segment_ID' in df_completo.columns else 'segment_id'
ids_completo = df_completo[col_id].values
y = df_completo['Target'].values

# Filtrar colunas puramente numéricas para o treinamento
cols_to_drop = ['Target', 'Segment_ID', 'segment_id', 'Class', 'classe', 'Label']
df_features = df_completo.drop(columns=[c for c in cols_to_drop if c in df_completo.columns], errors='ignore')
df_features_numericas = df_features.select_dtypes(include=[np.number])
X = df_features_numericas.values

print(f"Amostras: {len(X)} | Features: {X.shape[1]}")

# =====================================================================
# 4. TREINAMENTO DO MODELO SVM
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTreinando classificador SVM (isso pode levar alguns instantes)...")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# =====================================================================
# 5. AVALIAÇÃO E EXPORTAÇÃO DE RELATÓRIOS TEXTUAIS E GRÁFICOS
# =====================================================================
y_pred = svm_model.predict(X_test_scaled)
b_acc = balanced_accuracy_score(y_test, y_pred)
class_names = ['Não-Floresta (0)', 'Floresta (1)']
relatorio_txt = classification_report(y_test, y_pred, target_names=class_names)

# Salvar Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
disp.plot(cmap=plt.cm.Greens, ax=ax_cm, values_format='d')
plt.title('Matriz de Confusão: Superpixels MaskSLIC')
plt.savefig(f'{OUT_DIR}/matriz_confusao.png', dpi=300, bbox_inches='tight')
plt.close()

# =====================================================================
# 6. GERAÇÃO DE RESULTADOS ESPACIAIS (SHAPEFILE E CÁLCULO DE ÁREA)
# =====================================================================
print("\nProcessando cruzamento espacial e gerando mapas...")

# Prever a classe para TODO o dataset (para podermos mapear toda a imagem)
X_all_scaled = scaler.transform(X)
y_pred_all = svm_model.predict(X_all_scaled)

# Criar DataFrame com o ID do segmento e o resultado previsto
df_resultados = pd.DataFrame({col_id: ids_completo, 'Predicao': y_pred_all})

# Carregar o Shapefile original dos segmentos
gdf_seg = gpd.read_file(SHP_SEGMENTOS)
col_shp_id = 'Segment_ID' if 'Segment_ID' in gdf_seg.columns else 'segment_id'

# Fazer a junção (Merge) usando o ID
gdf_map = gdf_seg.merge(df_resultados, left_on=col_shp_id, right_on=col_id, how='left')

# Segmentos sem previsão (não estavam nos CSVs) recebem -1 (Não Analisada)
gdf_map['Predicao'] = gdf_map['Predicao'].fillna(-1).astype(int)

# Criar coluna em texto para facilitar visualização no QGIS
condicoes = [gdf_map['Predicao'] == -1, gdf_map['Predicao'] == 0, gdf_map['Predicao'] == 1]
escolhas = ['Não Analisada', 'Não-Floresta', 'Floresta']
gdf_map['Classe'] = np.select(condicoes, escolhas, default='Erro')

# Calcular a Área (Projetando para UTM fuso 23S - EPSG:31983 - São José dos Campos)
# Isso garante que o cálculo .area seja estritamente em metros quadrados reais.
gdf_calc = gdf_map.to_crs(epsg=31983)
gdf_desmatamento = gdf_calc[gdf_calc['Predicao'] == 0]

area_m2 = gdf_desmatamento.area.sum()
# Resolução do CBERS Pan-sharpened é 2m. Logo, 1 pixel = 2x2 = 4m².
area_px = int(area_m2 / 4) 
houve_desmatamento = area_m2 > 0

# Salvar Relatório Final em Texto
texto_relatorio = f"""RELATÓRIO DE MONITORAMENTO - FORESTEYES
=======================================
Acurácia Balanceada (Teste): {b_acc:.4f}

Relatório de Classificação (Conjunto de Teste):
{relatorio_txt}

DADOS DE DESMATAMENTO (Área Total Analisada):
Houve desmatamento detectado? {'SIM' if houve_desmatamento else 'NÃO'}
Área total de Não-Floresta (Desmatamento): {area_m2:,.2f} m²
Quantidade equivalente em Pixels (CBERS 2m): {area_px:,} px
=======================================
"""
with open(f'{OUT_DIR}/relatorio_resultados.txt', 'w', encoding='utf-8') as f:
    f.write(texto_relatorio)
print(texto_relatorio)

# Salvar o Shapefile
shp_out_path = f'{OUT_DIR}/resultado_classificacao_3549904.shp'
gdf_map.to_file(shp_out_path)

# Gerar Visualização em PNG do Mapa (Cinza, Vermelho, Verde)
fig_map, ax_map = plt.subplots(figsize=(10, 10))
cmap_map = ListedColormap(['gray', 'red', 'green']) # Cores rigorosamente alinhadas: -1, 0, 1
gdf_map.plot(column='Predicao', cmap=cmap_map, legend=False, ax=ax_map)
plt.title('Mapa de Classificação: Cinza(Não Analisada), Verde(Floresta), Vermelho(Não-Floresta)')
plt.axis('off')
plt.savefig(f'{OUT_DIR}/mapa_classificacao_cores.png', dpi=300, bbox_inches='tight')
plt.close()

# =====================================================================
# 7. GERAÇÃO DA IMAGEM TIF (MÁSCARA DE DESMATAMENTO)
# =====================================================================
if houve_desmatamento:
    print("Gerando imagem TIF (True Color) focada nas áreas de desmatamento...")
    
    # Abrir a imagem TIF original
    with rasterio.open(TIF_ORIGINAL) as src:
        # Garantir que a geometria de recorte está no mesmo CRS da imagem original
        geometrias_desmatamento = gdf_map[gdf_map['Predicao'] == 0].to_crs(src.crs).geometry.tolist()
        
        # Realizar a máscara (tudo que não for desmatamento ficará preto / nodata=0)
        out_image, out_transform = mask(src, geometrias_desmatamento, crop=False, filled=True, nodata=0)
        out_meta = src.meta.copy()

    # Atualizar metadados para salvar o novo arquivo
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    tif_out_path = f'{OUT_DIR}/desmatamento_mascara_CBERS.tif'
    with rasterio.open(tif_out_path, "w", **out_meta) as dest:
        dest.write(out_image)
        
    print(f"TIF de desmatamento gerado com sucesso em: {tif_out_path}")
else:
    print("Nenhum desmatamento detectado pelo modelo. Geração do TIF de máscara ignorada.")

print(f"\nPipeline concluído com sucesso! Todos os arquivos estão disponíveis na pasta:\n{OUT_DIR}")