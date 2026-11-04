#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Etapa 08.4: Cálculo de Homogeneidade (HoR) e Filtragem de Segmentos
==============================================================================
Descrição:
  - Calcula a Taxa de Homogeneidade (HoR) para cada superpíxel do SLIC.
  - Filtra superpíxeis por tamanho mínimo (70px) e limite de HoR (0.7).
  - Atualiza a máscara SLIC atribuindo valor -1 aos segmentos rejeitados.
  - Exporta DataFrames para classes 'forest' e 'non forest' ordenados por HoR.
==============================================================================
"""

import sys
import os
import gc
import warnings
import duckdb

# Compatibilidade DuckDB / geobr
if not hasattr(duckdb.DuckDBPyRelation, "fetchone"):
    duckdb.DuckDBPyRelation.fetchone = lambda self: self.fetchall()[0]

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)

# Configuração dos Limiares Globais
MIN_HOR_THRESHOLD = 0.7
MIN_SIZE_SEG_THRESHOLD = 70
MAX_SIZE_SEG_THRESHOLD = 10000

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 08.4-filtrarCalcularHoR.py <code_muni> <ano_inicio> <ano_fim>")
        sys.exit(1)

    code_muni = str(sys.argv[1])
    ano_inicio = str(sys.argv[2])
    ano_fim = str(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    # 1. Resolução da estrutura de diretórios
    dir_processing = os.path.join(project_root, "data", "processing")
    dir_proc_masks_npy = os.path.join(dir_processing, "masks", "npy_truth_masks")
    dir_proc_slics = os.path.join(dir_processing, "slics")
    dir_csv_root = os.path.join(dir_processing, "csv")
    
    dir_csv_forest = os.path.join(dir_csv_root, "forest")
    dir_csv_non_forest = os.path.join(dir_csv_root, "non-forest")

    os.makedirs(dir_csv_forest, exist_ok=True)
    os.makedirs(dir_csv_non_forest, exist_ok=True)

    # Resolução dos arquivos de entrada
    path_slic = os.path.join(dir_proc_slics, f"mask_slic_{code_muni}_{ano_fim}.npy")
    path_mask_gt = os.path.join(dir_proc_masks_npy, f"{code_muni}_GT_U_{ano_fim}.npy")

    if not os.path.exists(path_mask_gt):
        path_mask_gt = os.path.join(dir_proc_masks_npy, f"{code_muni}_GT_MapBiomas_{ano_fim}.npy")

    if not os.path.exists(path_slic):
        print(f"❌ [ERRO CRÍTICO] Matriz SLIC (.npy) não encontrada: {path_slic}")
        sys.exit(1)
    if not os.path.exists(path_mask_gt):
        print(f"❌ [ERRO CRÍTICO] Máscara GT (.npy) não encontrada: {path_mask_gt}")
        sys.exit(1)

    print("=" * 80, flush=True)
    print(f"🚀 SCRIPT 08.4 - CÁLCULO DE HoR E FILTRAGEM DE SEGMENTOS ({ano_fim})", flush=True)
    print("=" * 80, flush=True)

    print("1. Carregando matrizes SLIC e Ground Truth...", flush=True)
    img_slic = np.load(path_slic)
    truth_mask = np.load(path_mask_gt)

    flat_slic = img_slic.ravel()
    flat_gt = truth_mask.ravel()

    # Considera apenas pixels com rótulo de superpíxel válido (> 0)
    valid_idx = flat_slic > 0
    slic_valid = flat_slic[valid_idx]
    gt_valid = flat_gt[valid_idx]

    unique_labels = np.unique(slic_valid)
    max_label = int(unique_labels.max())
    total_segs = len(unique_labels)

    print(f"2. Processando HoR e dimensões para {total_segs:,} superpíxeis...", flush=True)

    # Contagem de pixels totais e de não-fundo (>0) por segmento
    pixel_counts = np.bincount(slic_valid, minlength=max_label + 1)
    np_counts = np.bincount(slic_valid, weights=(gt_valid > 0).astype(np.int64), minlength=max_label + 1)

    # Contagem de pixels de Floresta (Classe 2) e Não-Floresta (Classe 1)
    nfp_counts = np.bincount(slic_valid, weights=(gt_valid == 2).astype(np.int64), minlength=max_label + 1)
    nnp_counts = np.bincount(slic_valid, weights=(gt_valid == 1).astype(np.int64), minlength=max_label + 1)

    # Extração de vetores alinhados com unique_labels
    NP = np_counts[unique_labels]
    NFP = nfp_counts[unique_labels]
    NNP = nnp_counts[unique_labels]
    size_segs = pixel_counts[unique_labels]

    # Cálculo do HoR (evitando divisão por zero)
    max_pixels = np.maximum(NFP, NNP)
    hor_array = np.where(NP > 0, max_pixels / NP, 0.0)

    # Determinação da classe majoritária
    # 2: forest | 1: non forest | outros: not analyzed / background
    class_str = np.where(NFP > NNP, "forest", np.where(NNP > NFP, "non forest", "other"))

    # Métricas Globais
    good_mask = (size_segs > MIN_SIZE_SEG_THRESHOLD) & (hor_array > MIN_HOR_THRESHOLD) & np.isin(class_str, ["forest", "non forest"])
    bad_hor_mask = hor_array <= MIN_HOR_THRESHOLD
    bad_size_mask = (size_segs <= MIN_SIZE_SEG_THRESHOLD) | (NP == 0)
    perfect_mask = hor_array == 1.0

    good_segs = int(np.sum(good_mask))
    bad_segs = total_segs - good_segs
    bad_by_hor = int(np.sum(bad_hor_mask))
    bad_by_size = int(np.sum(bad_size_mask))
    n_perfect_segs = int(np.sum(perfect_mask))
    pct_perfect = (n_perfect_segs / total_segs * 100) if total_segs > 0 else 0.0

    region_tag = f"x{code_muni}"
    print(f"\n📊 RESUMO DE SEGMENTAÇÃO ({region_tag}):")
    print(f"   min_size_seg: {MIN_SIZE_SEG_THRESHOLD} | max_size_seg: {MAX_SIZE_SEG_THRESHOLD}")
    print(f"   total_segs = {total_segs} | good_segs = {good_segs} | bad_segs = {bad_segs}")
    print(f"   bad_segs por HoR = {bad_by_hor} | bad_segs por tamanho = {bad_by_size}")
    print(f"   perfect_segs = {n_perfect_segs} ({pct_perfect:.2f}%)\n")

    # 3. Atualizando os IDs dos segmentos ruins para -1 na matriz SLIC
    bad_labels = unique_labels[~good_mask]
    if len(bad_labels) > 0:
        print(f"3. Atualizando {len(bad_labels):,} segmentos ruins para -1 no SLIC...", flush=True)
        mask_to_bad = np.isin(img_slic, bad_labels)
        img_slic[mask_to_bad] = -1
        np.save(path_slic, img_slic)

    # 4. Construção dos DataFrames para Forest e Non-Forest
    print("4. Gerando e exportando DataFrames filtrados...", flush=True)

    df_all = pd.DataFrame({
        "Segment_label": unique_labels,
        "HoR": hor_array,
        "Class": class_str,
        "Size": size_segs,
        "Region": region_tag
    })

    df_good = df_all[good_mask].copy()

    df_forest = df_good[df_good["Class"] == "forest"].sort_values(by="HoR", ascending=False)
    df_non_forest = df_good[df_good["Class"] == "non forest"].sort_values(by="HoR", ascending=False)

    out_csv_forest = os.path.join(dir_csv_forest, f"df_forest_sort_{code_muni}_{ano_fim}.csv")
    out_csv_non_forest = os.path.join(dir_csv_non_forest, f"df_non-forest_sort_{code_muni}_{ano_fim}.csv")

    df_forest.to_csv(out_csv_forest, index=False)
    df_non_forest.to_csv(out_csv_non_forest, index=False)

    # 5. Relatório estatístico por contagem de segmentos e píxeis
    forest_count = len(df_forest)
    non_forest_count = len(df_non_forest)
    print(f"   -> across {region_tag}: forest = {forest_count} | non forest = {non_forest_count}")

    good_labels = df_good["Segment_label"].values
    good_pixel_idx = np.isin(flat_slic, good_labels)
    good_gt_pixels = flat_gt[good_pixel_idx]

    forest_pix_count = int(np.sum(good_gt_pixels == 2))
    non_forest_pix_count = int(np.sum(good_gt_pixels == 1))
    ratio = (forest_pix_count / non_forest_pix_count) if non_forest_pix_count > 0 else 0.0

    print(f"   -> across {region_tag}: forest pix = {forest_pix_count:,} | non forest pix = {non_forest_pix_count:,} | forest / non-forest = {ratio:.3f}")

    print(f"\n✅ SUCESSO! Script 08.4 concluído.")
    print(f" -> Tabela Forest salva em: {out_csv_forest}")
    print(f" -> Tabela Non-Forest salva em: {out_csv_non_forest}")

if __name__ == "__main__":
    main()