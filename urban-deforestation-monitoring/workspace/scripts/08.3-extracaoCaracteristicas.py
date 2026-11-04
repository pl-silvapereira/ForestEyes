#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Etapa 08.3: Extração de Características e Rotulagem de Superpíxeis (Otimizado)
==============================================================================
"""

import sys
import os
import warnings
import duckdb

if not hasattr(duckdb.DuckDBPyRelation, "fetchone"):
    duckdb.DuckDBPyRelation.fetchone = lambda self: self.fetchall()[0]

import numpy as np
import cv2
import pandas as pd
from scipy import ndimage
from scipy.sparse import coo_matrix
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

    dir_processing = os.path.join(project_root, "data", "processing")
    dir_proc_rgb = os.path.join(dir_processing, "rgb")
    dir_proc_masks_npy = os.path.join(dir_processing, "masks", "npy_truth_masks")
    dir_proc_slics = os.path.join(dir_processing, "slics")
    dir_proc_csv = os.path.join(dir_processing, "csv")

    os.makedirs(dir_proc_csv, exist_ok=True)

    path_rgb = os.path.join(dir_proc_rgb, f"{code_muni}_rgb_{ano_fim}.png")
    path_slic = os.path.join(dir_proc_slics, f"mask_slic_{code_muni}_{ano_fim}.npy")
    path_mask_gt = os.path.join(dir_proc_masks_npy, f"{code_muni}_GT_U_{ano_fim}.npy")

    if not os.path.exists(path_mask_gt):
        path_mask_gt = os.path.join(dir_proc_masks_npy, f"{code_muni}_GT_MapBiomas_{ano_fim}.npy")

    output_csv = os.path.join(dir_proc_csv, f"features_{code_muni}_{ano_fim}.csv")

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
    print(f"🚀 SCRIPT 08.3 - EXTRAÇÃO DE CARACTERÍSTICAS E ROTULAGEM (OTIMIZADO) ({ano_fim})", flush=True)
    print("=" * 80, flush=True)

    print("1. Carregando dados na memória...", flush=True)
    bgr_img = cv2.imread(path_rgb)
    if bgr_img is None:
        print(f"❌ [ERRO CRÍTICO] Falha ao ler imagem RGB: {path_rgb}")
        sys.exit(1)

    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    img_slic = np.load(path_slic)
    truth_mask = np.load(path_mask_gt)

    unique_labels = np.unique(img_slic[img_slic > 0])
    num_superpixels = len(unique_labels)
    print(f"2. Processando estatísticas espectrais de {num_superpixels:,} superpíxeis (Vetorizado)...", flush=True)

    dataset = {'sp_id': unique_labels}
    num_channels = rgb_img.shape[2] if rgb_img.ndim == 3 else 1

    # Cálculo vetorizado de Média e Desvio Padrão
    for c in range(num_channels):
        channel_data = rgb_img[:, :, c].astype(np.float64) if num_channels > 1 else rgb_img.astype(np.float64)
        
        mean_val = ndimage.mean(channel_data, labels=img_slic, index=unique_labels)
        mean_sq = ndimage.mean(channel_data**2, labels=img_slic, index=unique_labels)
        
        var_val = np.maximum(0.0, mean_sq - (mean_val ** 2))
        std_val = np.sqrt(var_val)

        dataset[f'mean_ch{c+1}'] = mean_val
        dataset[f'std_ch{c+1}'] = std_val

    print("3. Executando votação majoritária de classes via Matrizes Esparsas C-Level...", flush=True)
    mask_flat = img_slic.ravel()
    gt_flat = truth_mask.ravel()

    # Considera apenas pixels dentro de superpíxeis válidos (> 0)
    valid_idx = mask_flat > 0
    mask_valid = mask_flat[valid_idx]
    gt_valid = gt_flat[valid_idx].astype(np.int64)

    # Ajuste para garantir índices não negativos no Ground Truth
    gt_min = int(gt_valid.min())
    if gt_min < 0:
        gt_valid -= gt_min

    max_sp = int(unique_labels.max())
    max_gt = int(gt_valid.max())

    # Constrói matriz de contagem rápida: [sp_id, gt_class] = número_de_pixels
    counts = coo_matrix(
        (np.ones_like(mask_valid, dtype=np.int32), (mask_valid, gt_valid)),
        shape=(max_sp + 1, max_gt + 1)
    ).tocsr()

    # argmax por linha encontra a classe majoritária em milissegundos
    majority_classes = counts.argmax(axis=1).A1[unique_labels]
    
    if gt_min < 0:
        majority_classes += gt_min

    dataset['target_class'] = majority_classes

    # Salva o arquivo CSV final
    df_features = pd.DataFrame(dataset)
    df_features.to_csv(output_csv, index=False)

    print(f"\n✅ SUCESSO! Processamento concluído em poucos segundos.")
    print(f" -> Tabela salva em: {output_csv}")
    print(f" -> Total de superpíxeis extraídos: {len(df_features):,}")
    print(df_features.head())

if __name__ == "__main__":
    main()