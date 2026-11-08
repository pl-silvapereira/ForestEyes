#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Etapa 08.5: Geração Exclusiva da Campanha Piloto (100 Superpixels) Zooniverse
==============================================================================
Descrição:
 - Processa APENAS a Fase 1 (100 superpixels balanceados):
   * 25 Floresta (HoR = 1.0) + 25 Floresta (HoR ~ 0.7)
   * 25 Não-Floresta (HoR = 1.0) + 25 Não-Floresta (HoR ~ 0.7)
 - Desenha contornos em amarelo fluorescente (2px), aplica zoom de 2.5x e gera os chips.
 - Chama automaticamente o Script 08.6 ao finalizar e encerra a execução.
==============================================================================
"""

import sys
import os
import gc
import subprocess
import warnings
import duckdb

# Compatibilidade DuckDB / geobr
if not hasattr(duckdb.DuckDBPyRelation, "fetchone"):
    duckdb.DuckDBPyRelation.fetchone = lambda self: self.fetchall()[0]

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)

ZOOM_PARAM = 2.5
BORDER = 60  # Borda de contexto de 60px (120m no terreno)

PREFIX_MAP = {
    'rgb': 0,
    'ndvi': 1,
    'nir': 2
}

def obter_cor_contorno():
    # Amarelo Fluorescente Vivo em BGR (Alto contraste sobre RGB, NDVI e NIR)
    return (0, 255, 255)

def selecionar_amostras_piloto(df):
    """
    Seleciona 50 superpixels balanceados por classe (Total de 100 no projeto):
    - 25 de máxima pureza (HoR = 1.0 - topo do CSV ordenado descendentemente)
    - 25 de limiar (HoR próximo de 0.7 - base do CSV ordenado descendentemente)
    """
    total = len(df)
    if total <= 50:
        return df.copy()

    df_top25 = df.head(25)
    df_bottom25 = df.tail(25)
    return pd.concat([df_top25, df_bottom25]).drop_duplicates(subset=["Segment_label"])

def processar_chips_piloto(code_muni, ano_fim, classification, composition, path_slic, path_csv, dir_processing, dir_output):
    if not os.path.exists(path_csv):
        print(f"⚠️ CSV não encontrado: {path_csv}. Pulando classe {classification}...")
        return

    dir_comp = os.path.join(dir_processing, composition)
    path_img_comp = os.path.join(dir_comp, f"{code_muni}_{composition}_{ano_fim}.png")
    
    if not os.path.exists(path_img_comp):
        path_img_comp = os.path.join(dir_processing, "rgb", f"{code_muni}_rgb_{ano_fim}.png")

    if not os.path.exists(path_img_comp):
        print(f"⚠️ Imagem de entrada ausente: {path_img_comp}. Pulando...")
        return

    img = cv2.imread(path_img_comp)
    if img is None:
        return

    img_slic = np.load(path_slic)
    df_raw = pd.read_csv(path_csv)

    if df_raw.empty:
        return

    df = selecionar_amostras_piloto(df_raw)
    if df.empty:
        return

    unique_segments_sort = df["Segment_label"].values
    hors_sort = df["HoR"].values
    num_segs = len(unique_segments_sort)

    print(f"\n📸 Gerando Chips Piloto | Classe: '{classification}' | {num_segs} superpixels | Composição: '{composition.upper()}'", flush=True)

    prefix = PREFIX_MAP.get(composition.lower(), 0)
    color_bgr = obter_cor_contorno()

    output_dir_full = os.path.join(dir_output, f"AE-X{code_muni}", f"{classification}_sort_hor", composition)
    os.makedirs(output_dir_full, exist_ok=True)

    nrows, ncols = img.shape[:2]

    for i, seg_val in enumerate(tqdm(unique_segments_sort, desc=f" {classification}/{composition}")):
        hor_seg = hors_sort[i]
        seg_mask = (img_slic == seg_val).astype(np.uint8)
        
        rows, cols = np.where(seg_mask > 0)
        if len(rows) == 0:
            continue

        min_r, max_r = rows.min(), rows.max()
        min_c, max_c = cols.min(), cols.max()

        first_row = max(0, min_r - BORDER)
        last_row = min(nrows - 1, max_r + BORDER)
        first_col = max(0, min_c - BORDER)
        last_col = min(ncols - 1, max_c + BORDER)

        newimg = img.copy()
        contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(newimg, contours, -1, color_bgr, thickness=2)

        subimg = newimg[first_row:last_row + 1, first_col:last_col + 1]

        if subimg.size > 0:
            zoomimg = cv2.resize(subimg, None, fx=ZOOM_PARAM, fy=ZOOM_PARAM, interpolation=cv2.INTER_CUBIC)
            filename = f"{prefix}_p_{i:02d}_task_{int(seg_val)}_HoR_{hor_seg:.4f}_{composition}.png"
            output_filepath = os.path.join(output_dir_full, filename)
            cv2.imwrite(output_filepath, zoomimg)

    del img, img_slic, df, df_raw
    gc.collect()

def disparar_script_086(script_086_path, code_muni, ano_inicio, ano_fim):
    print("\n🔄 Disparando execução automática do Script 08.6 (Geração do Manifesto Zooniverse)...", flush=True)
    if os.path.exists(script_086_path):
        cmd = [sys.executable, script_086_path, code_muni, ano_inicio, ano_fim]
        res = subprocess.run(cmd)
        if res.returncode == 0:
            print("✅ Script 08.6 concluído com sucesso!")
        else:
            print(f"⚠️ Erro ao executar Script 08.6: Código {res.returncode}")
    else:
        print(f"⚠️ Script 08.6 não localizado em: {script_086_path}")

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

    dir_output_zooniverse = os.path.join(project_root, "out_workflow", "all_segs")
    path_slic = os.path.join(dir_proc_slics, f"mask_slic_{code_muni}_{ano_fim}.npy")

    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    script_086_path = os.path.join(diretorio_atual, "08.6-gerarManifestoZooniverse.py")
    if not os.path.exists(script_086_path):
        script_086_path = os.path.join(project_root, "scripts", "08.6-gerarManifestoZooniverse.py")

    if not os.path.exists(path_slic):
        print(f"❌ [ERRO CRÍTICO] Matriz SLIC (.npy) não encontrada: {path_slic}")
        sys.exit(1)

    classifications = ['forest', 'non-forest']
    compositions = ['rgb', 'ndvi', 'nir']

    print("=" * 80, flush=True)
    print(f"🚀 EXECUÇÃO EXCLUSIVA - CAMPANHA PILOTO (100 SUPERPIXELS) ({ano_fim})", flush=True)
    print("=" * 80, flush=True)

    for classification in classifications:
        path_csv = os.path.join(dir_csv_forest if classification == 'forest' else dir_csv_non_forest, f"df_{classification}_sort_{code_muni}_{ano_fim}.csv")
        for composition in compositions:
            processar_chips_piloto(code_muni, ano_fim, classification, composition, path_slic, path_csv, dir_processing, dir_output_zooniverse)

    # Dispara a geração de manifestos do Zooniverse e encerra
    disparar_script_086(script_086_path, code_muni, ano_inicio, ano_fim)

    print(f"\n🎉 EXECUÇÃO PILOTO CONCLUÍDA! 100 superpixels processados e manifestos gerados com sucesso.")

if __name__ == "__main__":
    main()