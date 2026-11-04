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
    path_mask_gt = os.path.join(dir_proc_masks_npy, f"{code_muni}_GT_3Classes_{ano_fim}.npy")

    if not os.path.exists(path_mask_gt):
        path_mask_gt = os.path.join(dir_proc_masks_npy, f"{code_muni}_GT_U_{ano_fim}.npy")

    output_csv = os.path.join(dir_proc_csv, f"features_{code_muni}_{ano_fim}.csv")

    if not os.path.exists(path_rgb) or not os.path.exists(path_slic) or not os.path.exists(path_mask_gt):
        print("❌ [ERRO CRÍTICO] Arquivos de entrada ausentes para a etapa 08.3.")
        sys.exit(1)

    print("=" * 80, flush=True)
    print(f"🚀 SCRIPT 08.3 - EXTRAÇÃO DE CARACTERÍSTICAS EM BLOCOS ({ano_fim})", flush=True)
    print("=" * 80, flush=True)

    bgr_img = cv2.imread(path_rgb)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    del bgr_img

    img_slic = np.load(path_slic)
    truth_mask = np.load(path_mask_gt)

    height, width = img_slic.shape[:2]
    max_sp = int(img_slic.max())

    unique_labels = np.unique(img_slic[img_slic > 0])
    num_superpixels = len(unique_labels)

    print(f"Processando {num_superpixels:,} superpixels...", flush=True)

    block_size = 1000
    num_blocks = int(np.ceil(height / block_size))
    num_channels = rgb_img.shape[2]

    pixel_counts = np.zeros(max_sp + 1, dtype=np.int64)
    sum_ch = np.zeros((num_channels, max_sp + 1), dtype=np.float64)
    sum_sq_ch = np.zeros((num_channels, max_sp + 1), dtype=np.float64)

    pair_counts_dict = {}

    for b in range(num_blocks):
        r_start = b * block_size
        r_end = min((b + 1) * block_size, height)

        sub_slic = img_slic[r_start:r_end, :].ravel()
        valid = sub_slic > 0
        if not np.any(valid):
            continue

        sub_slic_valid = sub_slic[valid]
        pixel_counts += np.bincount(sub_slic_valid, minlength=max_sp + 1)

        sub_rgb = rgb_img[r_start:r_end, :, :]
        for c in range(num_channels):
            ch_vals = sub_rgb[:, :, c].ravel()[valid].astype(np.float64)
            sum_ch[c] += np.bincount(sub_slic_valid, weights=ch_vals, minlength=max_sp + 1)
            sum_sq_ch[c] += np.bincount(sub_slic_valid, weights=ch_vals ** 2, minlength=max_sp + 1)

        sub_gt = truth_mask[r_start:r_end, :].ravel()[valid].astype(np.int64)
        pair_code = (sub_slic_valid.astype(np.int64) << 32) | sub_gt
        unq_codes, cnts = np.unique(pair_code, return_counts=True)

        for code, count in zip(unq_codes, cnts):
            pair_counts_dict[code] = pair_counts_dict.get(code, 0) + count

    del rgb_img, truth_mask
    gc.collect()

    counts_safe = np.maximum(1, pixel_counts[unique_labels])
    dataset = {'sp_id': unique_labels, 'size_px': counts_safe}

    for c in range(num_channels):
        mean_val = sum_ch[c][unique_labels] / counts_safe
        mean_sq = sum_sq_ch[c][unique_labels] / counts_safe
        var_val = np.maximum(0.0, mean_sq - (mean_val ** 2))
        std_val = np.sqrt(var_val)

        dataset[f'mean_ch{c+1}'] = mean_val
        dataset[f'std_ch{c+1}'] = std_val

    if pair_counts_dict:
        codes = np.array(list(pair_counts_dict.keys()), dtype=np.int64)
        counts = np.array(list(pair_counts_dict.values()), dtype=np.int32)
        sp_ids = codes >> 32
        gt_vals = codes & 0xFFFFFFFF
        max_gt_val = int(gt_vals.max())

        counts_matrix = coo_matrix((counts, (sp_ids, gt_vals)), shape=(max_sp + 1, max_gt_val + 1)).tocsr()
        majority_classes = counts_matrix.argmax(axis=1).A1[unique_labels]
    else:
        majority_classes = np.zeros(len(unique_labels), dtype=int)

    dataset['target_class'] = majority_classes
    df_features = pd.DataFrame(dataset)
    df_features.to_csv(output_csv, index=False)

    print(f"\n✅ SUCESSO! Script 08.3 concluído. Tabela salva em: {output_csv}")

if __name__ == "__main__":
    main()