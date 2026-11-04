#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Etapa 08.3: Extração de Características e Rotulagem de Superpíxeis
==============================================================================
"""

import sys
import os
import gc
import warnings
import duckdb

# Compatibilidade DuckDB e geobr no Python 3.13
if not hasattr(duckdb.DuckDBPyRelation, "fetchone"):
    duckdb.DuckDBPyRelation.fetchone = lambda self: self.fetchall()[0]

import numpy as np
import cv2
import pandas as pd
from scipy import ndimage
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 08.3-extracaoCaracteristicas.py <code_muni> <ano_inicio> <ano_fim>")
        sys.exit(1)

    code_muni = str(sys.argv[1])
    ano_inicio = str(sys.argv[2])
    ano_fim = str(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    # 1. Estrutura de diretórios em data/processing
    dir_processing = os.path.join(project_root, "data", "processing")
    dir_proc_rgb = os.path.join(dir_processing, "rgb")
    dir_proc_masks_npy = os.path.join(dir_processing, "masks", "npy_truth_masks")
    dir_proc_slics = os.path.join(dir_processing, "slics")
    dir_proc_csv = os.path.join(dir_processing, "csv")

    os.makedirs(dir_proc_csv, exist_ok=True)

    # 2. Resolução dos caminhos dos arquivos gerados nas etapas anteriores
    path_rgb = os.path.join(dir_proc_rgb, f"{code_muni}_rgb_{ano_fim}.png")
    path_slic = os.path.join(dir_proc_slics, f"mask_slic_{code_muni}_{ano_fim}.npy")
    path_mask_gt = os.path.join(dir_proc_masks_npy, f"{code_muni}_GT_U_{ano_fim}.npy")

    # Fallback para GT_MapBiomas caso GT_U não exista
    if not os.path.exists(path_mask_gt):
        path_mask_gt = os.path.join(dir_proc_masks_npy, f"{code_muni}_GT_MapBiomas_{ano_fim}.npy")

    output_csv = os.path.join(dir_proc_csv, f"features_{code_muni}_{ano_fim}.csv")

    # Validação dos arquivos de entrada
    if not os.path.exists(path_rgb):
        print(f"❌ [ERRO CRÍTICO] Imagem RGB não encontrada: {path_rgb}")
        sys.exit(1)
    if not os.path.exists(path_slic):
        print(f"❌ [ERRO CRÍTICO] Matriz SLIC (.npy) não encontrada: {path_slic}")
        sys.exit(1)
    if not os.path.exists(path_mask_gt):
        print(f"❌ [ERRO CRÍTICO] Máscara GT (.npy) não encontrada: {path_mask_gt}")
        sys.exit(1)

    print("=" * 80, flush=True)
    print(f"🚀 SCRIPT 08.3 - EXTRAÇÃO DE CARACTERÍSTICAS E ROTULAGEM DE SUPERPÍXEIS ({ano_fim})", flush=True)
    print("=" * 80, flush=True)

    # 3. Carregando Imagem RGB, Matriz SLIC e Ground Truth
    print("1. Carregando dados de imagem, superpíxeis e máscara de referência...", flush=True)
    bgr_img = cv2.imread(path_rgb)
    if bgr_img is None:
        print(f"❌ [ERRO CRÍTICO] Falha ao ler imagem RGB: {path_rgb}")
        sys.exit(1)

    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    img_slic = np.load(path_slic)
    truth_mask = np.load(path_mask_gt)

    # 4. Processamento das estatísticas dos superpíxeis ativos (> 0)
    unique_labels = np.unique(img_slic[img_slic > 0])
    num_superpixels = len(unique_labels)
    print(f"2. Extraindo atributos espectrais de {num_superpixels:,} superpíxeis...", flush=True)

    dataset = {'sp_id': unique_labels}

    num_channels = rgb_img.shape[2] if rgb_img.ndim == 3 else 1
    for c in range(num_channels):
        channel_data = rgb_img[:, :, c] if num_channels > 1 else rgb_img
        dataset[f'mean_ch{c+1}'] = ndimage.mean(channel_data, labels=img_slic, index=unique_labels)
        dataset[f'std_ch{c+1}'] = ndimage.standard_deviation(channel_data, labels=img_slic, index=unique_labels)

    # 5. Atribuição do rótulo majoritário (Ground Truth)
    print("3. Associando classe predominante por votação majoritária...", flush=True)
    majority_classes = ndimage.labeled_comprehension(
        truth_mask,
        img_slic,
        unique_labels,
        lambda x: np.bincount(x).argmax() if len(x) > 0 else 0,
        int,
        0
    )
    dataset['target_class'] = majority_classes

    # 6. Exportação do DataFrame para CSV
    df_features = pd.DataFrame(dataset)
    df_features.to_csv(output_csv, index=False)

    print(f"\n✅ SUCESSO! Script 08.3 concluído.")
    print(f" -> Tabela salva em: {output_csv}")
    print(f" -> Dimensões da tabela: {df_features.shape}")
    print(df_features.head())

if __name__ == "__main__":
    main()