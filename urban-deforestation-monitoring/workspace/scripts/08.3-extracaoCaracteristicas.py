#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Etapa 08.3: Extração de Características e Rotulagem (Processamento em Blocos)
==============================================================================
"""

import sys
import os
import gc
import warnings
import duckdb

if not hasattr(duckdb.DuckDBPyRelation, "fetchone"):
    duckdb.DuckDBPyRelation.fetchone = lambda self: self.fetchall()[0]

import numpy as np
import cv2
import pandas as pd
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
    print(f"🚀 SCRIPT 08.3 - EXTRAÇÃO DE CARACTERÍSTICAS (PROCESSAMENTO EM BLOCOS) ({ano_fim})", flush=True)
    print("=" * 80, flush=True)

    print("1. Carregando matrizes na memória...", flush=True)
    bgr_img = cv2.imread(path_rgb)
    if bgr_img is None:
        print(f"❌ [ERRO CRÍTICO] Falha ao ler imagem RGB: {path_rgb}")
        sys.exit(1)

    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    del bgr_img
    gc.collect()

    img_slic = np.load(path_slic)
    truth_mask = np.load(path_mask_gt)

    height, width = img_slic.shape[:2]
    max_sp = int(img_slic.max())

    unique_labels = np.unique(img_slic[img_slic > 0])
    num_superpixels = len(unique_labels)

    print(f"2. Processando {num_superpixels:,} superpíxeis em blocos de linhas...", flush=True)

    # Tamanho do bloco (1.000 linhas por vez)
    block_size = 1000
    num_blocks = int(np.ceil(height / block_size))

    num_channels = rgb_img.shape[2] if rgb_img.ndim == 3 else 1
    pixel_counts = np.zeros(max_sp + 1, dtype=np.int64)
    sum_ch = np.zeros((num_channels, max_sp + 1), dtype=np.float64)
    sum_sq_ch = np.zeros((num_channels, max_sp + 1), dtype=np.float64)

    pair_counts_dict = {}

    gt_min_val = int(truth_mask.min())
    gt_offset = -gt_min_val if gt_min_val < 0 else 0

    # Iteração em blocos
    for b in range(num_blocks):
        r_start = b * block_size
        r_end = min((b + 1) * block_size, height)

        sub_slic = img_slic[r_start:r_end, :].ravel()
        valid = sub_slic > 0
        if not np.any(valid):
            continue

        sub_slic_valid = sub_slic[valid]

        # Contagem de pixels
        pixel_counts += np.bincount(sub_slic_valid, minlength=max_sp + 1)

        # Estatísticas espectrais por canal
        sub_rgb = rgb_img[r_start:r_end, :, :]
        for c in range(num_channels):
            ch_vals = sub_rgb[:, :, c].ravel()[valid].astype(np.float64) if num_channels > 1 else sub_rgb.ravel()[valid].astype(np.float64)
            sum_ch[c] += np.bincount(sub_slic_valid, weights=ch_vals, minlength=max_sp + 1)
            sum_sq_ch[c] += np.bincount(sub_slic_valid, weights=ch_vals ** 2, minlength=max_sp + 1)

        # Votação majoritária do Ground Truth
        sub_gt = truth_mask[r_start:r_end, :].ravel()[valid].astype(np.int64) + gt_offset
        pair_code = (sub_slic_valid.astype(np.int64) << 32) | sub_gt
        unq_codes, cnts = np.unique(pair_code, return_counts=True)

        for code, count in zip(unq_codes, cnts):
            pair_counts_dict[code] = pair_counts_dict.get(code, 0) + count

        pct = int(((b + 1) / num_blocks) * 100)
        print(f"   -> [Bloco {b+1}/{num_blocks}] ({pct}%) processado com sucesso.", flush=True)

    # Liberação das matrizes da memória
    del rgb_img, truth_mask
    gc.collect()

    print("3. Calculando médias, desvios padrão e classes majoritárias...", flush=True)
    counts_safe = np.maximum(1, pixel_counts[unique_labels])

    dataset = {'sp_id': unique_labels}

    for c in range(num_channels):
        mean_val = sum_ch[c][unique_labels] / counts_safe
        mean_sq = sum_sq_ch[c][unique_labels] / counts_safe

        var_val = np.maximum(0.0, mean_sq - (mean_val ** 2))
        std_val = np.sqrt(var_val)

        dataset[f'mean_ch{c+1}'] = mean_val
        dataset[f'std_ch{c+1}'] = std_val

    # Reconstrução da matriz para votação de classe
    if pair_counts_dict:
        codes = np.array(list(pair_counts_dict.keys()), dtype=np.int64)
        counts = np.array(list(pair_counts_dict.values()), dtype=np.int32)

        sp_ids = codes >> 32
        gt_shifted = codes & 0xFFFFFFFF
        max_gt_shifted = int(gt_shifted.max())

        counts_matrix = coo_matrix(
            (counts, (sp_ids, gt_shifted)),
            shape=(max_sp + 1, max_gt_shifted + 1)
        ).tocsr()

        majority_shifted = counts_matrix.argmax(axis=1).A1[unique_labels]
        majority_classes = majority_shifted - gt_offset
    else:
        majority_classes = np.zeros(len(unique_labels), dtype=int)

    dataset['target_class'] = majority_classes

    # Salvamento do CSV
    print("4. Salvando arquivo CSV final...", flush=True)
    df_features = pd.DataFrame(dataset)
    df_features.to_csv(output_csv, index=False)

    print(f"\n✅ SUCESSO! Script 08.3 concluído.")
    print(f" -> Tabela salva em: {output_csv}")
    print(f" -> Total de superpíxeis extraídos: {len(df_features):,}")
    print(df_features.head())

if __name__ == "__main__":
    main()