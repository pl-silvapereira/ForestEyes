#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Etapa 08.5: Geração de Chips Multi-Composição e Tarefa Unificada (Alta Performance)
==============================================================================
"""

import sys
import os
import gc
import shutil
import warnings
import duckdb

if not hasattr(duckdb.DuckDBPyRelation, "fetchone"):
    duckdb.DuckDBPyRelation.fetchone = lambda self: self.fetchall()[0]

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import find_objects
from tqdm import tqdm
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)

ZOOM_PARAM = 2.5
BORDER = 60  # Borda de contexto (120m no terreno)

def obter_cor_contorno():
    return (0, 255, 255)  # Amarelo Fluorescente em BGR

def carregar_imagem_bgr(caminho):
    if not os.path.exists(caminho):
        return None
    img = cv2.imread(caminho)
    if img is not None and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def processar_superpixels_otimizado(code_muni, ano_fim, classification, path_slic, path_csv, dir_processing, dir_workflow):
    if not os.path.exists(path_csv):
        print(f"⚠️ [AVISO] CSV não encontrado: {path_csv}. Pulando classe '{classification}'...")
        return

    # 1. Carrega as 3 composições base para a RAM
    path_rgb = os.path.join(dir_processing, "rgb", f"{code_muni}_rgb_{ano_fim}.png")
    path_ndvi = os.path.join(dir_processing, "ndvi", f"{code_muni}_ndvi_{ano_fim}.png")
    path_nir = os.path.join(dir_processing, "nir", f"{code_muni}_nir_{ano_fim}.png")

    img_rgb = carregar_imagem_bgr(path_rgb)
    img_ndvi = carregar_imagem_bgr(path_ndvi)
    img_nir = carregar_imagem_bgr(path_nir)

    if img_rgb is None or img_ndvi is None or img_nir is None:
        print(f"❌ [ERRO CRÍTICO] Imagens base (RGB, NDVI, NIR) ausentes em {dir_processing}.")
        return

    img_slic = np.load(path_slic)
    df = pd.read_csv(path_csv)

    if df.empty:
        print(f"ℹ️ CSV vazio para a classe '{classification}': {path_csv}")
        return

    unique_segments_sort = df["Segment_label"].values
    hors_sort = df["HoR"].values
    num_segs = len(unique_segments_sort)

    print(f"\n⚡ Precalculando Bounding Boxes de todos os superpixels...", flush=True)
    slices = find_objects(img_slic)

    color_bgr = obter_cor_contorno()
    nrows, ncols = img_rgb.shape[:2]
    max_label_slic = len(slices)

    dir_class_target = os.path.join(dir_workflow, f"AE-X{code_muni}", classification)
    os.makedirs(dir_class_target, exist_ok=True)

    print(f"📸 Processando {num_segs:,} superpixels | Classe: '{classification}'", flush=True)

    for i, seg_val in enumerate(tqdm(unique_segments_sort, desc=f" Processando {classification}")):
        hor_seg = hors_sort[i]
        seg_int = int(seg_val)

        if seg_int < 1 or seg_int > max_label_slic:
            continue

        loc = slices[seg_int - 1]
        if loc is None:
            continue

        min_r, max_r = loc[0].start, loc[0].stop
        min_c, max_c = loc[1].start, loc[1].stop

        first_row = max(0, min_r - BORDER)
        last_row = min(nrows - 1, max_r + BORDER)
        first_col = max(0, min_c - BORDER)
        last_col = min(ncols - 1, max_c + BORDER)

        sub_slic = img_slic[first_row:last_row + 1, first_col:last_col + 1]
        sub_mask = (sub_slic == seg_int).astype(np.uint8)

        contours, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        task_dir_name = f"task_{seg_int}"
        task_dir_path = os.path.join(dir_class_target, task_dir_name)
        os.makedirs(task_dir_path, exist_ok=True)

        fn_rgb = f"0_task_{seg_int}_HoR_{hor_seg:.4f}_rgb.png"
        fn_ndvi = f"1_task_{seg_int}_HoR_{hor_seg:.4f}_ndvi.png"
        fn_nir = f"2_task_{seg_int}_HoR_{hor_seg:.4f}_nir.png"

        # RGB
        crop_rgb = img_rgb[first_row:last_row + 1, first_col:last_col + 1].copy()
        cv2.drawContours(crop_rgb, contours, -1, color_bgr, thickness=2)
        zoom_rgb = cv2.resize(crop_rgb, None, fx=ZOOM_PARAM, fy=ZOOM_PARAM, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(task_dir_path, fn_rgb), zoom_rgb)

        # NDVI
        crop_ndvi = img_ndvi[first_row:last_row + 1, first_col:last_col + 1].copy()
        cv2.drawContours(crop_ndvi, contours, -1, color_bgr, thickness=2)
        zoom_ndvi = cv2.resize(crop_ndvi, None, fx=ZOOM_PARAM, fy=ZOOM_PARAM, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(task_dir_path, fn_ndvi), zoom_ndvi)

        # NIR
        crop_nir = img_nir[first_row:last_row + 1, first_col:last_col + 1].copy()
        cv2.drawContours(crop_nir, contours, -1, color_bgr, thickness=2)
        zoom_nir = cv2.resize(crop_nir, None, fx=ZOOM_PARAM, fy=ZOOM_PARAM, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(task_dir_path, fn_nir), zoom_nir)

        # MANIFEST.CSV
        manifest_path = os.path.join(task_dir_path, "manifest.csv")
        with open(manifest_path, "w", encoding="utf-8") as f_man:
            f_man.write("id,0_rgb,1_ndvi,2_nir\n")
            f_man.write(f"{task_dir_name},{fn_rgb},{fn_ndvi},{fn_nir}\n")

    del img_rgb, img_ndvi, img_nir, img_slic, df, slices
    gc.collect()

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 08.5-gerarImagensSegmentosZooniverse.py <code_muni> <ano_inicio> <ano_fim>")
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
    dir_proc_slics = os.path.join(dir_processing, "slics")
    dir_csv_root = os.path.join(dir_processing, "csv")
    
    dir_csv_forest = os.path.join(dir_csv_root, "forest")
    dir_csv_non_forest = os.path.join(dir_csv_root, "non-forest")

    # Caminho final exigido no PROJECT_ROOT/data/out_workflow/cbers_workflow
    dir_workflow_final = os.path.join(project_root, "data", "out_workflow", "cbers_workflow")
    os.makedirs(dir_workflow_final, exist_ok=True)

    # Usa buffer rápido local se estiver no Google Colab e sincroniza ao final
    em_colab = os.path.exists("/content")
    dir_execucao = "/content/out_workflow/cbers_workflow" if em_colab else dir_workflow_final

    path_slic = os.path.join(dir_proc_slics, f"mask_slic_{code_muni}_{ano_fim}.npy")

    if not os.path.exists(path_slic):
        print(f"❌ [ERRO CRÍTICO] Matriz SLIC (.npy) não encontrada: {path_slic}")
        sys.exit(1)

    classifications = ['forest', 'non-forest']

    print("=" * 80, flush=True)
    print(f"🚀 SCRIPT 08.5 - GERAÇÃO ULTRA-RÁPIDA MULTI-COMPOSIÇÃO ({ano_fim})", flush=True)
    print(f"📂 Destino Final: {dir_workflow_final}", flush=True)
    print("=" * 80, flush=True)

    for classification in classifications:
        path_csv = os.path.join(
            dir_csv_forest if classification == 'forest' else dir_csv_non_forest,
            f"df_{classification}_sort_{code_muni}_{ano_fim}.csv"
        )
        processar_superpixels_otimizado(
            code_muni, ano_fim, classification, path_slic, path_csv,
            dir_processing, dir_execucao
        )

    if em_colab and os.path.abspath(dir_execucao) != os.path.abspath(dir_workflow_final):
        print(f"\n🔄 Transferindo tarefas geradas para o Google Drive em:\n   {dir_workflow_final} ...", flush=True)
        shutil.copytree(dir_execucao, dir_workflow_final, dirs_exist_ok=True)

    print(f"\n🎉 PROCESSO CONCLUÍDO COM SUCESSO!")
    print(f" -> Tarefas disponíveis em: {dir_workflow_final}")

if __name__ == "__main__":
    main()