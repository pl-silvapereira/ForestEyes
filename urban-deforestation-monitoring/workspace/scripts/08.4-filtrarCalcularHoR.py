import sys
import os
import gc
import warnings
import duckdb

if not hasattr(duckdb.DuckDBPyRelation, "fetchone"):
    duckdb.DuckDBPyRelation.fetchone = lambda self: self.fetchall()[0]

import numpy as np
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)

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

    dir_processing = os.path.join(project_root, "data", "processing")
    dir_proc_masks_npy = os.path.join(dir_processing, "masks", "npy_truth_masks")
    dir_proc_slics = os.path.join(dir_processing, "slics")
    dir_csv_root = os.path.join(dir_processing, "csv")
    
    dir_csv_forest = os.path.join(dir_csv_root, "forest")
    dir_csv_non_forest = os.path.join(dir_csv_root, "non-forest")

    os.makedirs(dir_csv_forest, exist_ok=True)
    os.makedirs(dir_csv_non_forest, exist_ok=True)

    path_slic = os.path.join(dir_proc_slics, f"mask_slic_{code_muni}_{ano_fim}.npy")
    path_mask_gt = os.path.join(dir_proc_masks_npy, f"{code_muni}_GT_3Classes_{ano_fim}.npy")

    if not os.path.exists(path_mask_gt):
        path_mask_gt = os.path.join(dir_proc_masks_npy, f"{code_muni}_GT_U_{ano_fim}.npy")

    if not os.path.exists(path_slic) or not os.path.exists(path_mask_gt):
        print("❌ [ERRO CRÍTICO] Matrizes .npy de entrada não encontradas para o script 08.4.")
        sys.exit(1)

    print("=" * 80, flush=True)
    print(f"🚀 SCRIPT 08.4 - CÁLCULO DE HoR E CORREÇÃO DE MAPEAMENTO DE CLASSE ({ano_fim})", flush=True)
    print("=" * 80, flush=True)

    img_slic = np.load(path_slic)
    truth_mask = np.load(path_mask_gt)

    flat_slic = img_slic.ravel()
    flat_gt = truth_mask.ravel()

    valid_idx = flat_slic > 0
    slic_valid = flat_slic[valid_idx]
    gt_valid = flat_gt[valid_idx]

    unique_labels = np.unique(slic_valid)
    max_label = int(unique_labels.max())
    total_segs = len(unique_labels)

    pixel_counts = np.bincount(slic_valid, minlength=max_label + 1)
    
    # --------------------------------------------------------------------------
    # MAPEAMENTO CORRIGIDO DAS CLASSES DO SHAPEFILE:
    # GT = 1 -> Floresta (Verde)
    # GT = 2 -> Nao_Floresta (Vermelho)
    # --------------------------------------------------------------------------
    nfp_counts = np.bincount(slic_valid, weights=(gt_valid == 1).astype(np.int64), minlength=max_label + 1)  # Floresta
    nnp_counts = np.bincount(slic_valid, weights=(gt_valid == 2).astype(np.int64), minlength=max_label + 1)  # Nao_Floresta

    NFP = nfp_counts[unique_labels]
    NNP = nnp_counts[unique_labels]
    size_segs = pixel_counts[unique_labels]

    NP = NFP + NNP
    max_pixels = np.maximum(NFP, NNP)
    hor_array = np.where(NP > 0, max_pixels / NP, 0.0)

    # Definição semântica correta
    class_str = np.where(NFP > NNP, "forest", np.where(NNP > NFP, "non forest", "other"))

    good_mask = (size_segs >= MIN_SIZE_SEG_THRESHOLD) & (size_segs <= MAX_SIZE_SEG_THRESHOLD) & (hor_array >= MIN_HOR_THRESHOLD) & np.isin(class_str, ["forest", "non forest"])
    
    good_segs = int(np.sum(good_mask))
    bad_segs = total_segs - good_segs

    print(f"\n📊 RESUMO DA FILTRAGEM HOR:")
    print(f" -> Total Superpixels: {total_segs:,}")
    print(f" -> Superpixels Aprovados (HoR >= 0.7): {good_segs:,}")
    print(f" -> Superpixels Rejeitados: {bad_segs:,}")

    # Atualiza segmentos rejeitados para -1
    bad_labels = unique_labels[~good_mask]
    if len(bad_labels) > 0:
        mask_to_bad = np.isin(img_slic, bad_labels)
        img_slic[mask_to_bad] = -1
        np.save(path_slic, img_slic)

    region_tag = f"x{code_muni}"
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

    print(f"\n✅ SUCESSO! Script 08.4 concluído.")
    print(f" -> Total Floresta: {len(df_forest):,} | Salvo em: {out_csv_forest}")
    print(f" -> Total Nao_Floresta: {len(df_non_forest):,} | Salvo em: {out_csv_non_forest}")

if __name__ == "__main__":
    main()