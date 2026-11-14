#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Etapa 08.5: Geração Multi-Composição em Micro-Lotes + Empacotamento no Drive
==============================================================================
Descrição:
 - Processa em micro-lotes de 10.000 superpixels (ocupa < 1 GB no Colab).
 - Para cada superpixel, gera RGB, NDVI e NIR no mesmo loop e aciona o
   Script 08.6 para montar task_{seg_int} e o manifest.csv unificado.
 - Ao concluir cada micro-lote, empacota diretamente no Google Drive
   (PROJECT_ROOT/data/out_workflow/cbers_workflow) e limpa o disco local.
 - Possui checkpoint automático: pula lotes .zip já salvos no Drive.
==============================================================================
"""

import sys
import os
import gc
import time
import shutil
import zipfile
import warnings
import importlib.util
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
BORDER = 60          # Borda de contexto (120m no terreno)
MICRO_BATCH = 10000  # 10 mil tarefas por pacote (~700 MB temporários no Colab)


def carregar_modulo_086(diretorio_scripts, project_root):
    """Carrega o Script 08.6 como módulo Python para chamada direta."""
    candidatos = [
        os.path.join(diretorio_scripts, "08.6-gerarManifestoZooniverse.py"),
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


def compactar_lote_para_drive(dir_origem_tasks, zip_destino_drive, max_tentativas=5):
    """Grava o arquivo .zip diretamente no Google Drive com retentativa."""
    os.makedirs(os.path.dirname(zip_destino_drive), exist_ok=True)
    zip_temp_drive = zip_destino_drive + ".tmp"

    for tentativa in range(1, max_tentativas + 1):
        try:
            with zipfile.ZipFile(zip_temp_drive, "w", compression=zipfile.ZIP_STORED) as zf:
                for root, _, files in os.walk(dir_origem_tasks):
                    for fname in files:
                        full_path = os.path.join(root, fname)
                        rel_path = os.path.relpath(full_path, dir_origem_tasks)
                        zf.write(full_path, rel_path)
            if os.path.exists(zip_destino_drive):
                os.remove(zip_destino_drive)
            os.rename(zip_temp_drive, zip_destino_drive)
            return True
        except OSError:
            if os.path.exists(zip_temp_drive):
                try:
                    os.remove(zip_temp_drive)
                except OSError:
                    pass
            time.sleep(2.0 * tentativa)
    raise IOError(f"Falha ao gravar pacote ZIP no Google Drive: {zip_destino_drive}")


def processar_superpixels_micro_lotes(
    code_muni, ano_fim, classification,
    path_slic, path_csv, dir_processing, dir_workflow_drive, mod_086
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

    # Filtra todos os segmentos válidos do CSV
    itens_validos = []
    for seg_val, hor_seg in zip(df["Segment_label"].values, df["HoR"].values):
        seg_int = int(seg_val)
        if 1 <= seg_int <= max_label_slic and slices[seg_int - 1] is not None:
            itens_validos.append((seg_int, float(hor_seg), slices[seg_int - 1]))

    total_itens = len(itens_validos)
    total_lotes = (total_itens + MICRO_BATCH - 1) // MICRO_BATCH

    dir_drive_class = os.path.join(dir_workflow_drive, f"AE-X{code_muni}", classification)
    os.makedirs(dir_drive_class, exist_ok=True)

    # Usa /content se estiver no Colab para velocidade local, limpando a cada micro-lote
    raiz_temp = "/content/tmp_foresteyes_batch" if os.path.exists("/content") else os.path.join(dir_workflow_drive, "_tmp_batch")

    print(f"📸 Classe '{classification}': {total_itens:,} superpixels divididos em {total_lotes} pacotes de até {MICRO_BATCH:,}", flush=True)

    color_bgr = (0, 255, 255)  # Amarelo Fluorescente em BGR

    for idx_lote in range(total_lotes):
        lote_num = idx_lote + 1
        zip_nome = f"AE-X{code_muni}_{classification}_lote_{lote_num:03d}.zip"
        zip_caminho_drive = os.path.join(dir_drive_class, zip_nome)

        # Checkpoint: se o pacote ZIP deste lote já existe no Drive, pula instantaneamente
        if os.path.exists(zip_caminho_drive) and os.path.getsize(zip_caminho_drive) > 0:
            print(f" ⏭️ Lote [{lote_num}/{total_lotes}] já salvo no Drive ({zip_nome}). Pulando...", flush=True)
            continue

        # Limpa qualquer resquício local anterior antes de iniciar o lote
        if os.path.exists(raiz_temp):
            shutil.rmtree(raiz_temp, ignore_errors=True)
        os.makedirs(raiz_temp, exist_ok=True)

        inicio = idx_lote * MICRO_BATCH
        fim = min(inicio + MICRO_BATCH, total_itens)
        lote_atual = itens_validos[inicio:fim]

        desc_barra = f" Lote [{lote_num}/{total_lotes}] ({classification})"
        for seg_int, hor_seg, loc in tqdm(lote_atual, desc=desc_barra):
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

            # 1. Pasta temporária do segmento dentro do micro-lote
            dir_seg_temp = os.path.join(raiz_temp, "all_segs", f"AE-X{code_muni}", classification, f"seg_{seg_int}")
            os.makedirs(dir_seg_temp, exist_ok=True)

            fn_rgb = f"0_task_{seg_int}_HoR_{hor_seg:.4f}_rgb.png"
            fn_ndvi = f"1_task_{seg_int}_HoR_{hor_seg:.4f}_ndvi.png"
            fn_nir = f"2_task_{seg_int}_HoR_{hor_seg:.4f}_nir.png"

            # 2. Gera as 3 composições no mesmo loop do superpixel
            for img_base, fn_out in [(img_rgb, fn_rgb), (img_ndvi, fn_ndvi), (img_nir, fn_nir)]:
                crop = img_base[first_row:last_row + 1, first_col:last_col + 1].copy()
                cv2.drawContours(crop, contours, -1, color_bgr, thickness=2)
                zoom = cv2.resize(crop, None, fx=ZOOM_PARAM, fy=ZOOM_PARAM, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(dir_seg_temp, fn_out), zoom)

            # 3. Antes de ir para o próximo superpixel, chama o Script 08.6
            mod_086.processar_tarefa_segmento(raiz_temp, code_muni, classification, seg_int, hor_seg)

        # 4. Grava o micro-lote inteiro como 1 arquivo ZIP direto no Google Drive
        pasta_tasks_lote = os.path.join(raiz_temp, "cbers_workflow", f"AE-X{code_muni}", classification)
        if os.path.exists(pasta_tasks_lote):
            print(f" 📦 Salvando {zip_nome} diretamente no Google Drive...", flush=True)
            compactar_lote_para_drive(pasta_tasks_lote, zip_caminho_drive)

        # 5. Libera o espaço do micro-lote imediatamente antes do próximo
        shutil.rmtree(raiz_temp, ignore_errors=True)
        gc.collect()

    del img_rgb, img_ndvi, img_nir, img_slic, df, slices, itens_validos
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

    dir_workflow_drive = os.path.join(project_root, "data", "out_workflow", "cbers_workflow")
    os.makedirs(dir_workflow_drive, exist_ok=True)

    path_slic = os.path.join(dir_proc_slics, f"mask_slic_{code_muni}_{ano_fim}.npy")
    if not os.path.exists(path_slic):
        print(f"❌ [ERRO CRÍTICO] Matriz SLIC (.npy) não encontrada: {path_slic}")
        sys.exit(1)

    classifications = ['non-forest', 'forest']

    print("=" * 80, flush=True)
    print(f"🚀 SCRIPT 08.5 - MICRO-LOTES ({MICRO_BATCH}/lote) + 08.6 + PACOTES NO DRIVE ({ano_fim})", flush=True)
    print(f"📂 Destino Final: {dir_workflow_drive}", flush=True)
    print("=" * 80, flush=True)

    for classification in classifications:
        path_csv = os.path.join(
            dir_csv_forest if classification == 'forest' else dir_csv_non_forest,
            f"df_{classification}_sort_{code_muni}_{ano_fim}.csv"
        )
        processar_superpixels_micro_lotes(
            code_muni, ano_fim, classification,
            path_slic, path_csv, dir_processing, dir_workflow_drive, mod_086
        )

    print(f"\n🎉 PROCESSO CONCLUÍDO COM SUCESSO!")
    print(f" -> Pacotes salvos em: {dir_workflow_drive}")


if __name__ == "__main__":
    main()