#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Etapa 08.8: Auditoria e Reparo Pontual de Tarefas Vazias ou Incompletas
==============================================================================
Descrição:
 - Compara a lista oficial de superpixels aprovados (CSVs do Script 08.4)
   com as pastas task_{Segment_label} em PROJECT_ROOT/data/out_workflow/cbers_workflow.
 - Identifica pastas vazias, pastas sem os 3 recortes (RGB, NDVI, NIR),
   pastas sem manifest.csv ou pastas ausentes.
 - Recorta e salva pontualmente apenas os itens faltantes, populando as
   pastas correspondentes de forma sequencial e segura no Google Drive.
==============================================================================
"""

import sys
import os
import gc
import time
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


def carregar_imagem_bgr(caminho):
    if not os.path.exists(caminho):
        return None
    img = cv2.imread(caminho)
    if img is not None and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def gravar_bytes_com_retry(caminho, dados_bytes, max_tentativas=5):
    """Grava arquivo no Google Drive com retentativa em caso de oscilação FUSE."""
    pasta_pai = os.path.dirname(caminho)
    for tentativa in range(1, max_tentativas + 1):
        try:
            os.makedirs(pasta_pai, exist_ok=True)
            with open(caminho, "wb") as f_out:
                f_out.write(dados_bytes)
            if os.path.exists(caminho) and os.path.getsize(caminho) > 0:
                return True
        except OSError:
            pass
        time.sleep(1.5 * tentativa)
    raise IOError(f"Falha ao gravar arquivo no Drive após {max_tentativas} tentativas: {caminho}")


def verificar_tarefa_completa(task_dir_path, fn_rgb, fn_ndvi, fn_nir):
    """Retorna True apenas se a pasta contiver os 3 PNGs não-vazios e o manifest.csv."""
    if not os.path.isdir(task_dir_path):
        return False

    arquivos_obrigatorios = [fn_rgb, fn_ndvi, fn_nir, "manifest.csv"]
    for arq in arquivos_obrigatorios:
        caminho_arq = os.path.join(task_dir_path, arq)
        if not os.path.exists(caminho_arq) or os.path.getsize(caminho_arq) == 0:
            return False
    return True


def auditar_e_reparar_classe(
    code_muni, ano_fim, classification,
    path_slic, path_csv, dir_processing, dir_workflow_base
):
    if not os.path.exists(path_csv):
        print(f"⚠️ [AVISO] CSV não encontrado: {path_csv}. Pulando classe '{classification}'...", flush=True)
        return

    df = pd.read_csv(path_csv)
    if df.empty:
        print(f"ℹ️ CSV vazio para a classe '{classification}': {path_csv}", flush=True)
        return

    dir_final_class = os.path.join(dir_workflow_base, f"AE-X{code_muni}", classification)
    os.makedirs(dir_final_class, exist_ok=True)

    print(f"\n🔍 Auditando integridade das pastas em '{classification}' ({len(df):,} superpixels esperados)...", flush=True)

    # Lista rápida das pastas existentes no diretório da classe
    pastas_no_disco = set(os.listdir(dir_final_class))
    pendentes_reparo = []
    qtd_pastas_vazias_ou_incompletas = 0
    qtd_pastas_ausentes = 0

    for seg_val, hor_seg in zip(df["Segment_label"].values, df["HoR"].values):
        seg_int = int(seg_val)
        hor_float = float(hor_seg)
        task_name = f"task_{seg_int}"
        task_dir = os.path.join(dir_final_class, task_name)

        fn_rgb = f"0_task_{seg_int}_HoR_{hor_float:.4f}_rgb.png"
        fn_ndvi = f"1_task_{seg_int}_HoR_{hor_float:.4f}_ndvi.png"
        fn_nir = f"2_task_{seg_int}_HoR_{hor_float:.4f}_nir.png"

        if task_name not in pastas_no_disco:
            qtd_pastas_ausentes += 1
            pendentes_reparo.append((seg_int, hor_float, fn_rgb, fn_ndvi, fn_nir))
        elif not verificar_tarefa_completa(task_dir, fn_rgb, fn_ndvi, fn_nir):
            qtd_pastas_vazias_ou_incompletas += 1
            pendentes_reparo.append((seg_int, hor_float, fn_rgb, fn_ndvi, fn_nir))

    print(f" 📊 Diagnóstico '{classification}':", flush=True)
    print(f"    -> Pastas íntegras (3 composições + manifest.csv): {len(df) - len(pendentes_reparo):,}", flush=True)
    print(f"    -> Pastas vazias ou incompletas encontradas:       {qtd_pastas_vazias_ou_incompletas:,}", flush=True)
    print(f"    -> Pastas ausentes no diretório:                   {qtd_pastas_ausentes:,}", flush=True)

    if not pendentes_reparo:
        print(f" ✅ Nenhuma pendência encontrada na classe '{classification}'!", flush=True)
        return

    # Carrega as imagens base e a matriz SLIC apenas se houver itens para reparar
    print(f"\n📥 Carregando imagens base (RGB, NDVI, NIR) e matriz SLIC para reparar {len(pendentes_reparo):,} tarefas...", flush=True)
    path_rgb = os.path.join(dir_processing, "rgb", f"{code_muni}_rgb_{ano_fim}.png")
    path_ndvi = os.path.join(dir_processing, "ndvi", f"{code_muni}_ndvi_{ano_fim}.png")
    path_nir = os.path.join(dir_processing, "nir", f"{code_muni}_nir_{ano_fim}.png")

    img_rgb = carregar_imagem_bgr(path_rgb)
    img_ndvi = carregar_imagem_bgr(path_ndvi)
    img_nir = carregar_imagem_bgr(path_nir)

    if img_rgb is None or img_ndvi is None or img_nir is None:
        print(f"❌ [ERRO CRÍTICO] Imagens base (RGB, NDVI, NIR) ausentes em {dir_processing}.", flush=True)
        return

    img_slic = np.load(path_slic)
    slices = find_objects(img_slic)
    nrows, ncols = img_rgb.shape[:2]
    max_label_slic = len(slices)
    color_bgr = (0, 255, 255)  # Amarelo Fluorescente em BGR

    reparados = 0
    for idx, (seg_int, hor_float, fn_rgb, fn_ndvi, fn_nir) in enumerate(
        tqdm(pendentes_reparo, desc=f" Reparando {classification}")
    ):
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
        task_dir_path = os.path.join(dir_final_class, task_dir_name)
        os.makedirs(task_dir_path, exist_ok=True)

        # Gera e grava apenas as composições que estiverem faltando ou vazias na pasta
        composicoes = [
            (img_rgb, fn_rgb),
            (img_ndvi, fn_ndvi),
            (img_nir, fn_nir),
        ]

        for img_base, fn_out in composicoes:
            caminho_png = os.path.join(task_dir_path, fn_out)
            if os.path.exists(caminho_png) and os.path.getsize(caminho_png) > 0:
                continue

            crop = img_base[first_row:last_row + 1, first_col:last_col + 1].copy()
            cv2.drawContours(crop, contours, -1, color_bgr, thickness=2)
            zoom = cv2.resize(crop, None, fx=ZOOM_PARAM, fy=ZOOM_PARAM, interpolation=cv2.INTER_CUBIC)

            ok, buffer_png = cv2.imencode(".png", zoom)
            if ok:
                gravar_bytes_com_retry(caminho_png, buffer_png.tobytes())

        # Gera/atualiza o manifest.csv da tarefa
        manifest_path = os.path.join(task_dir_path, "manifest.csv")
        conteudo_manifest = (
            "id,0_rgb,1_ndvi,2_nir\n"
            f"{task_dir_name},{fn_rgb},{fn_ndvi},{fn_nir}\n"
        ).encode("utf-8")
        gravar_bytes_com_retry(manifest_path, conteudo_manifest)

        reparados += 1
        if (idx + 1) % 500 == 0:
            gc.collect()

    print(f" ✅ Reparo concluído para '{classification}': {reparados:,} pastas populadas com sucesso!", flush=True)

    del img_rgb, img_ndvi, img_nir, img_slic, df, slices
    gc.collect()


def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 08.8-repararTarefasVaziasZooniverse.py <code_muni> <ano_inicio> <ano_fim>")
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

    dir_workflow_base = os.path.join(project_root, "data", "out_workflow", "cbers_workflow")
    path_slic = os.path.join(dir_proc_slics, f"mask_slic_{code_muni}_{ano_fim}.npy")

    if not os.path.exists(path_slic):
        print(f"❌ [ERRO CRÍTICO] Matriz SLIC (.npy) não encontrada: {path_slic}")
        sys.exit(1)

    classifications = ["non-forest", "forest"]

    print("=" * 80, flush=True)
    print(f"🛠️  SCRIPT 08.8 - AUDITORIA E REPARO DE TAREFAS VAZIAS/AUSENTES ({ano_fim})", flush=True)
    print(f"📂 Diretório Alvo: {dir_workflow_base}", flush=True)
    print("=" * 80, flush=True)

    for classification in classifications:
        path_csv = os.path.join(
            dir_csv_forest if classification == "forest" else dir_csv_non_forest,
            f"df_{classification}_sort_{code_muni}_{ano_fim}.csv"
        )
        auditar_e_reparar_classe(
            code_muni, ano_fim, classification,
            path_slic, path_csv, dir_processing, dir_workflow_base
        )

    print("\n🎉 AUDITORIA E REPARO CONCLUÍDOS COM SUCESSO!")


if __name__ == "__main__":
    main()