#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Etapa 08.5: Geração Paralela (12 Threads x 1 Item por Rodada) + Disparo 08.6
==============================================================================
Descrição:
 - Opera diretamente em PROJECT_ROOT/data/out_workflow/ (sem usar /content).
 - A cada iteração, seleciona 12 superpixels pendentes e abre 12 threads
   simultâneas (1 item por thread).
 - Aguarda todas as 12 threads finalizarem (geração de RGB, NDVI, NIR +
   execução do Script 08.6) antes de iniciar a próxima rodada de 12.
==============================================================================
"""

import sys
import os
import gc
import time
import warnings
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
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
BORDER = 60        # Borda de contexto (120m no terreno)
NUM_THREADS = 12   # 12 threads paralelas com 1 superpixel por thread a cada rodada


def carregar_modulo_086(diretorio_scripts, project_root):
    """Carrega o Script 08.6 como módulo Python para chamada direta."""
    candidatos = [
        os.path.join(diretorio_scripts, "08.6-gerarManifestoZooniverse.py"),
        os.path.join(diretorio_scripts, "08.6-gerarManifestoZooniverse_3.py"),
        os.path.join(project_root, "scripts", "08.6-gerarManifestoZooniverse.py"),
    ]
    caminho_086 = next((p for p in candidatos if os.path.exists(p)), None)
    if not caminho_086:
        raise FileNotFoundError(f"Script 08.6 não encontrado nos caminhos esperados: {candidatos}")

    spec = importlib.util.spec_from_file_location("mod_086", caminho_086)
    mod_086 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod_086)
    return mod_086


def carregar_imagem_bgr(caminho):
    if not os.path.exists(caminho):
        return None
    img = cv2.imread(caminho)
    if img is not None and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def salvar_png_com_retry(caminho, imagem, max_tentativas=5):
    """Grava imagem PNG no Google Drive com retentativa em caso de oscilação FUSE."""
    for tentativa in range(1, max_tentativas + 1):
        try:
            sucesso = cv2.imwrite(caminho, imagem)
            if sucesso and os.path.exists(caminho):
                return True
        except OSError:
            pass
        time.sleep(1.5 * tentativa)
    raise IOError(f"Falha ao gravar arquivo no Google Drive após {max_tentativas} tentativas: {caminho}")


def processar_unico_superpixel(
    seg_int, hor_seg, loc,
    img_rgb, img_ndvi, img_nir, img_slic,
    nrows, ncols, dir_staging_class,
    project_root, code_muni, classification, mod_086
):
    """Processa 1 superpixel: gera RGB, NDVI, NIR e chama o Script 08.6."""
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
        return False

    color_bgr = (0, 255, 255)  # Amarelo Fluorescente em BGR

    # 1. Pasta temporária exclusiva do segmento no Drive
    dir_seg_temp = os.path.join(dir_staging_class, f"seg_{seg_int}")
    os.makedirs(dir_seg_temp, exist_ok=True)

    fn_rgb = f"0_task_{seg_int}_HoR_{hor_seg:.4f}_rgb.png"
    fn_ndvi = f"1_task_{seg_int}_HoR_{hor_seg:.4f}_ndvi.png"
    fn_nir = f"2_task_{seg_int}_HoR_{hor_seg:.4f}_nir.png"

    composicoes = [
        (img_rgb, fn_rgb),
        (img_ndvi, fn_ndvi),
        (img_nir, fn_nir),
    ]

    # 2. Gera as 3 composições recortadas e ampliadas
    for img_base, fn_out in composicoes:
        crop = img_base[first_row:last_row + 1, first_col:last_col + 1].copy()
        cv2.drawContours(crop, contours, -1, color_bgr, thickness=2)
        zoom = cv2.resize(crop, None, fx=ZOOM_PARAM, fy=ZOOM_PARAM, interpolation=cv2.INTER_CUBIC)
        salvar_png_com_retry(os.path.join(dir_seg_temp, fn_out), zoom)

    # 3. Chama o Script 08.6 para mover para task_{seg_int}, apagar seg_{seg_int} e criar manifest.csv
    return mod_086.processar_tarefa_segmento(project_root, code_muni, classification, seg_int, hor_seg)


def processar_superpixels_12_por_vez(
    project_root, code_muni, ano_fim, classification,
    path_slic, path_csv, dir_processing, base_out_workflow, mod_086
):
    if not os.path.exists(path_csv):
        print(f"⚠️ [AVISO] CSV não encontrado: {path_csv}. Pulando classe '{classification}'...")
        return

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

    print(f"\n⚡ Precalculando Bounding Boxes ({classification})...", flush=True)
    slices = find_objects(img_slic)
    nrows, ncols = img_rgb.shape[:2]
    max_label_slic = len(slices)

    dir_staging_class = os.path.join(base_out_workflow, "all_segs", f"AE-X{code_muni}", classification)
    dir_final_class = os.path.join(base_out_workflow, "cbers_workflow", f"AE-X{code_muni}", classification)
    os.makedirs(dir_staging_class, exist_ok=True)
    os.makedirs(dir_final_class, exist_ok=True)

    print(f"🔍 Verificando tarefas já concluídas no Google Drive...", flush=True)
    pastas_existentes = set(os.listdir(dir_final_class))

    tarefas_pendentes = []
    for seg_val, hor_seg in zip(df["Segment_label"].values, df["HoR"].values):
        seg_int = int(seg_val)
        if seg_int < 1 or seg_int > max_label_slic:
            continue
        loc = slices[seg_int - 1]
        if loc is None:
            continue
        if f"task_{seg_int}" in pastas_existentes:
            if os.path.exists(os.path.join(dir_final_class, f"task_{seg_int}", "manifest.csv")):
                continue
        tarefas_pendentes.append((seg_int, float(hor_seg), loc))

    total_pendentes = len(tarefas_pendentes)
    print(f"📸 Classe '{classification}': {total_pendentes:,} pendentes | Rodadas de {NUM_THREADS} threads (1 item/thread)", flush=True)

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor, \
         tqdm(total=total_pendentes, desc=f" {classification}") as pbar:

        for inicio in range(0, total_pendentes, NUM_THREADS):
            grupo_12 = tarefas_pendentes[inicio : inicio + NUM_THREADS]

            futures = [
                executor.submit(
                    processar_unico_superpixel,
                    seg_int, hor_seg, loc,
                    img_rgb, img_ndvi, img_nir, img_slic,
                    nrows, ncols, dir_staging_class,
                    project_root, code_muni, classification, mod_086
                )
                for seg_int, hor_seg, loc in grupo_12
            ]

            # Aguarda todas as 12 threads terminarem antes de abrir as próximas 12
            for future in as_completed(futures):
                future.result()

            pbar.update(len(grupo_12))

            if (inicio // NUM_THREADS) % 100 == 0:
                gc.collect()

    try:
        os.rmdir(dir_staging_class)
    except OSError:
        pass

    del img_rgb, img_ndvi, img_nir, img_slic, df, slices, tarefas_pendentes
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
    diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
    if not project_root:
        project_root = os.path.dirname(diretorio_scripts)

    mod_086 = carregar_modulo_086(diretorio_scripts, project_root)

    dir_processing = os.path.join(project_root, "data", "processing")
    dir_proc_slics = os.path.join(dir_processing, "slics")
    dir_csv_root = os.path.join(dir_processing, "csv")

    dir_csv_forest = os.path.join(dir_csv_root, "forest")
    dir_csv_non_forest = os.path.join(dir_csv_root, "non-forest")

    base_out_workflow = os.path.join(project_root, "data", "out_workflow")
    os.makedirs(base_out_workflow, exist_ok=True)

    path_slic = os.path.join(dir_proc_slics, f"mask_slic_{code_muni}_{ano_fim}.npy")
    if not os.path.exists(path_slic):
        print(f"❌ [ERRO CRÍTICO] Matriz SLIC (.npy) não encontrada: {path_slic}")
        sys.exit(1)

    classifications = ['non-forest', 'forest']

    print("=" * 80, flush=True)
    print(f"🚀 SCRIPT 08.5 - EXECUÇÃO PARALELA ({NUM_THREADS} THREADS x 1 ITEM/RODADA)", flush=True)
    print(f"📂 Destino Workflow: {os.path.join(base_out_workflow, 'cbers_workflow')}", flush=True)
    print("=" * 80, flush=True)

    for classification in classifications:
        path_csv = os.path.join(
            dir_csv_forest if classification == 'forest' else dir_csv_non_forest,
            f"df_{classification}_sort_{code_muni}_{ano_fim}.csv"
        )
        processar_superpixels_12_por_vez(
            project_root, code_muni, ano_fim, classification,
            path_slic, path_csv, dir_processing, base_out_workflow, mod_086
        )

    for pasta_vazia in [
        os.path.join(base_out_workflow, "all_segs", f"AE-X{code_muni}"),
        os.path.join(base_out_workflow, "all_segs")
    ]:
        try:
            os.rmdir(pasta_vazia)
        except OSError:
            pass

    print(f"\n🎉 PROCESSO CONCLUÍDO COM SUCESSO!")
    print(f" -> Tarefas e manifestos salvos em: {os.path.join(base_out_workflow, 'cbers_workflow')}")


if __name__ == "__main__":
    main()