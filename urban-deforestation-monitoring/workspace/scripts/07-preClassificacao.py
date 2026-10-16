import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
# 1. CAMINHOS DOS ARQUIVOS (GOOGLE DRIVE)
# =====================================================================
BASE_DIR = '/content/drive/MyDrive/Mestrado/04-Projeto ForestEyes/ForestEyes/urban-deforestation-monitoring/workspace'

# Dataframes gerados pelo Script 06 (Atributos de textura GLCM, espectrais, etc.)
PATH_FOREST = f'{BASE_DIR}/reports/dataframes_csv/forest/df_forest_sort_3549904_2024.csv'
PATH_NON_FOREST = f'{BASE_DIR}/reports/dataframes_csv/non-forest/df_non-forest_sort_3549904_2024.csv'

# =====================================================================
# 2. FUNÇÃO DE REESTRUTURAÇÃO (ACHATAMENTO DE BANDAS)
# =====================================================================
def preparar_dataframe(df_path, label_alvo):
    """
    Carrega o CSV, verifica se as bandas estão em linhas separadas para o mesmo 
    segmento e, se sim, 'achata' (pivot) para que 1 Segmento = 1 Linha com todas as features.
    Adiciona a coluna Target (1 = Floresta, 0 = Não-Floresta).
    """
    df = pd.read_csv(df_path)
    
    # Padronizando nomes das colunas de identificação caso existam (ajuste conforme seu CSV)
    col_segmento = 'Segment_ID' if 'Segment_ID' in df.columns else 'segment_id' if 'segment_id' in df.columns else None
    col_banda = 'Band' if 'Band' in df.columns else 'band' if 'band' in df.columns else None

    # Se os dados estiverem em formato longo (1 linha por banda do segmento)
    if col_segmento and col_banda:
        # Pega as colunas que são atributos (exclui ID, Banda e geometria se houver)
        cols_features = [c for c in df.columns if c not in [col_segmento, col_banda, 'geometry']]
        
        # Faz o Pivot: Transforma as bandas em novas colunas de atributos
        df_flat = df.pivot(index=col_segmento, columns=col_banda, values=cols_features)
        
        # Achata o MultiIndex das colunas (Ex: 'Contrast', 1 -> 'Contrast_B1')
        df_flat.columns = [f"{feat}_B{band}" for feat, band in df_flat.columns]
        df_flat = df_flat.reset_index()
        df = df_flat
    
    # Remove coluna de geometria do shapefile, caso tenha sido exportada no CSV
    if 'geometry' in df.columns:
        df = df.drop(columns=['geometry'])
        
    # Adiciona a classe (Ground Truth)
    df['Target'] = label_alvo
    return df

# =====================================================================
# 3. CARREGAMENTO E CONSOLIDAÇÃO DOS DADOS
# =====================================================================
print("Carregando e reestruturando relatórios do Script 06...")

# Carrega e rotula os dados: Floresta = 1, Não-Floresta = 0
df_forest_final = preparar_dataframe(PATH_FOREST, label_alvo=1)
df_non_forest_final = preparar_dataframe(PATH_NON_FOREST, label_alvo=0)

# Une os dois dataframes em um conjunto de dados completo
df_completo = pd.concat([df_forest_final, df_non_forest_final], ignore_index=True)

# 1. Isolamos o vetor 'y' (Target numérico: 0 e 1)
y = df_completo['Target'].values

# 2. Criamos uma lista com as colunas que NÃO são atributos (features)
cols_to_drop = ['Target']
if 'Segment_ID' in df_completo.columns: cols_to_drop.append('Segment_ID')
if 'segment_id' in df_completo.columns: cols_to_drop.append('segment_id')
if 'Class' in df_completo.columns: cols_to_drop.append('Class')
if 'classe' in df_completo.columns: cols_to_drop.append('classe')
if 'Label' in df_completo.columns: cols_to_drop.append('Label')

# 3. Removemos as colunas indesejadas (ignorando erros se elas não existirem)
df_features = df_completo.drop(columns=cols_to_drop, errors='ignore')

# 4. FILTRO DE SEGURANÇA: Selecionar estritamente colunas com tipos numéricos (int, float)
# Isso ignora automaticamente qualquer coluna de texto ('floresta') que tenha sobrado
df_features_numericas = df_features.select_dtypes(include=[np.number])

# 5. Transformamos no array X
X = df_features_numericas.values

print(f"Total de Amostras: {len(X)}")
print(f"Total de Features Estritamente Numéricas por Superpixel: {X.shape[1]}")
print(f"Distribuição Real - Floresta: {sum(y==1)} | Não-Floresta: {sum(y==0)}")

# =====================================================================
# 4. DIVISÃO EM TREINO E TESTE
# =====================================================================
# Dividindo dados (Ex: 70% Treino, 30% Teste). 
# stratify=y garante que a proporção das classes seja mantida nos conjuntos.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

print(f"\nAmostras de Treino: {len(X_train)} | Amostras de Teste: {len(X_test)}")

# =====================================================================
# 5. PADRONIZAÇÃO (STANDARD SCALER)
# =====================================================================
print("Aplicando StandardScaler nas features dos superpixels...")
scaler = StandardScaler()

# O scaler mapeia a distribuição a partir do treino para evitar vazamento de dados (data leakage)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================================
# 6. TREINAMENTO DO MODELO SVM
# =====================================================================
print("Iniciando treinamento do classificador SVM (Kernel RBF)...")

# class_weight='balanced' ajusta pesos internamente devido ao desbalanceamento das classes
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
svm_model.fit(X_train_scaled, y_train)

print("Treinamento concluído!")

# =====================================================================
# 7. AVALIAÇÃO COM O GROUND TRUTH
# =====================================================================
print("\n--- AVALIAÇÃO DE DESEMPENHO NO CONJUNTO DE TESTE ---")
y_pred = svm_model.predict(X_test_scaled)

b_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Acurácia Balanceada (Balanced Accuracy): {b_acc:.4f}\n")

class_names = ['Não-Floresta (0)', 'Floresta (1)']
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=class_names))

# =====================================================================
# 8. MATRIZ DE CONFUSÃO
# =====================================================================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap=plt.cm.Greens, ax=ax, values_format='d')
plt.title('Matriz de Confusão: Superpixels MaskSLIC (CBERS-4A)')
plt.show()