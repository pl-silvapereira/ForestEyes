#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Etapa 08.3: Extração de Características e Rotulagem de Superpíxeis
==============================================================================
Descrição:
  - Carrega a imagem original, o mapa de superpíxeis (SLIC) e o ground truth.
  - Extrai métricas estatísticas espectrais (média e desvio padrão por canal).
  - Determina a classe majoritária de cada superpíxel com base na máscara.
  - Exporta o conjunto de dados tabular resultante no formato CSV.
==============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import ndimage
from PIL import Image

# ------------------------------------------------------------------------------
# 1. Configuração de Caminhos (Ajuste conforme a estrutura do projeto)
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

PATH_IMG = os.path.join(DATA_DIR, "processed", "imagem_entrada.npy")       # Imagem original (RGB / Multiespectral)
PATH_SLIC = os.path.join(DATA_DIR, "processed", "superpixels_slic.npy")    # Matriz img_slic gerada no 08.2
PATH_MASK = os.path.join(DATA_DIR, "processed", "mascara_groundtruth.npy") # Máscara de referência/rótulos
PATH_OUTPUT_CSV = os.path.join(DATA_DIR, "output", "features_superpixels.csv")


def carregar_dados():
    """Carrega a imagem, a matriz de superpíxeis e a máscara de ground truth."""
    if not os.path.exists(PATH_SLIC):
        raise FileNotFoundError(f"Arquivo de superpíxeis não encontrado: {PATH_SLIC}")
    if not os.path.exists(PATH_IMG):
        raise FileNotFoundError(f"Arquivo de imagem não encontrado: {PATH_IMG}")
    if not os.path.exists(PATH_MASK):
        raise FileNotFoundError(f"Arquivo de máscara não encontrado: {PATH_MASK}")

    # Suporta arquivos .npy (NumPy) ou imagens raster (.png, .tif)
    img_slic = np.load(PATH_SLIC) if PATH_SLIC.endswith(".npy") else np.array(Image.open(PATH_SLIC))
    img = np.load(PATH_IMG) if PATH_IMG.endswith(".npy") else np.array(Image.open(PATH_IMG))
    mask = np.load(PATH_MASK) if PATH_MASK.endswith(".npy") else np.array(Image.open(PATH_MASK))

    return img_slic, img, mask


def extracao_caracteristicas(img_slic, img, mask):
    """Calcula estatísticas por superpíxel e atribui o rótulo majoritário."""
    # Obter rótulos únicos e contagem
    unique_labels = np.unique(img_slic)
    num_superpixels = len(unique_labels)
    print(f"[INFO] Processando {num_superpixels} superpíxeis...")

    # Dicionário base para a tabela de atributos
    dataset = {'sp_id': unique_labels}

    # 1. Extração de atributos espectrais (Média e Desvio Padrão por canal)
    num_channels = img.shape[2] if img.ndim == 3 else 1

    for c in range(num_channels):
        channel_data = img[:, :, c] if num_channels > 1 else img
        dataset[f'mean_ch{c+1}'] = ndimage.mean(channel_data, labels=img_slic, index=unique_labels)
        dataset[f'std_ch{c+1}'] = ndimage.standard_deviation(channel_data, labels=img_slic, index=unique_labels)

    # 2. Votação majoritária para definição da classe (Ground Truth)
    if mask.ndim == 3:
        # Se a máscara for RGB, codifica a cor em um único número inteiro de 64-bit
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
        # Máscara 2D com classes indexadas por inteiros (ex: 0=Fundo, 1=Vegetação, 2=Desmatamento)
        majority_classes = ndimage.labeled_comprehension(
            mask,
            img_slic,
            unique_labels,
            lambda x: np.bincount(x).argmax(),
            int,
            -1
        )
        dataset['target_class'] = majority_classes

    return pd.DataFrame(dataset)


def main():
    print("[INFO] Iniciando etapa 08.3: Extração de Características...")
    
    # Carregamento dos dados
    img_slic, img, mask = carregar_dados()

    # Processamento
    df_features = extracao_caracteristicas(img_slic, img, mask)

    # Salvamento do CSV
    os.makedirs(os.path.dirname(PATH_OUTPUT_CSV), exist_ok=True)
    df_features.to_csv(PATH_OUTPUT_CSV, index=False)

    print(f"[SUCESSO] Arquivo gerado com sucesso em: {PATH_OUTPUT_CSV}")
    print(f"[INFO] Dimensões do dataset: {df_features.shape}")
    print(df_features.head())


if __name__ == "__main__":
    main()