# ==============================================================================
# Etapa 8.3: Extração de Características e Rotulagem dos Superpíxeis
# ==============================================================================
import numpy as np
import pandas as pd
from scipy import ndimage

# 1. Obter lista de rótulos únicos de superpíxeis
unique_labels = np.unique(img_slic)
num_superpixels = len(unique_labels)
print(f"Processando {num_superpixels} superpíxeis...")

# Dicionário para armazenar o dataset tabular
dataset = {'sp_id': unique_labels}

# 2. Extração de atributos espectrais (Média e Desvio Padrão por canal)
# Trata imagens monocromáticas ou multicanal (ex: RGB / 4 bandas)
num_channels = img.shape[2] if img.ndim == 3 else 1

for c in range(num_channels):
    channel_data = img[:, :, c] if num_channels > 1 else img
    dataset[f'mean_ch{c+1}'] = ndimage.mean(channel_data, labels=img_slic, index=unique_labels)
    dataset[f'std_ch{c+1}'] = ndimage.standard_deviation(channel_data, labels=img_slic, index=unique_labels)

# 3. Determinação da classe predominante (Ground Truth) em cada superpíxel
if mask.ndim == 3:
    # Codificação única das cores RGB para contagem vetorial rápida
    mask_encoded = (
        mask[:, :, 0].astype(np.uint64) << 32
    ) | (
        mask[:, :, 1].astype(np.uint64) << 16
    ) | mask[:, :, 2].astype(np.uint64)
    
    majority_codes = ndimage.labeled_comprehension(
        mask_encoded,
        img_slic,
        unique_labels,
        lambda x: np.bincount(x).argmax(),
        np.uint64,
        0
    )
    dataset['target_class_code'] = majority_codes
else:
    majority_classes = ndimage.labeled_comprehension(
        mask,
        img_slic,
        unique_labels,
        lambda x: np.bincount(x).argmax(),
        int,
        -1
    )
    dataset['target_class'] = majority_classes

# 4. Criação e salvamento do DataFrame
df_features = pd.DataFrame(dataset)
output_csv_path = 'csv_superpixel_features/features_x01.csv'
df_features.to_csv(output_csv_path, index=False)

print(f"Extração concluída com sucesso! Dataset salvo em: {output_csv_path}")
print(f"Dimensão da tabela de atributos: {df_features.shape}")
df_features.head()