#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Etapa 08.5: Geração Amostrada (Piloto) e Completa de Chips Zooniverse
==============================================================================
Fluxo:
 - Fase 1: Gera os 100 chips piloto e dispara a 1ª chamada do Script 08.6.
 - Fase 2: Gera os chips restantes (imagem 101 ao fim) e dispara a 2ª chamada do 08.6.
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
BORDER = 120

PREFIX_MAP = {
    'ndvi': 0,
    'rgb': 0
}

def obter_cor_contorno(composition):
    return (20, 70, 140)

def selecionar_amostras(df, modo='pilot'):
    """
    Fase 1 ('pilot'): 50 superpixels balanceados por classe.
    Fase 2 ('remaining'): Todos os superpixels restantes (imagem 101 em diante).
    """
    total = len(df)
    if modo == 'pilot':
        if total <= 50:
            return df.copy()
        df_top25 = df.head(25)
        df_bottom25 = df.tail(25)
        return pd.concat([df_top25, df_bottom25]).drop_duplicates(subset=["Segment_label"])
    elif modo == 'remaining':
        if total <= 50:
            return pd.DataFrame()  # Já processados no piloto
        # Extrai os elementos entre o topo 25 e os últimos 25
        return df.iloc[25:total-25].copy()

def processar_chips(code_muni, ano_fim, classification, composition, path_slic, path_csv, dir_img, dir_output, modo='pilot'):
    if not os.path.exists(path_csv):
        print(f"⚠️ [AVISO] CSV não encontrado: {path_csv}. Pulando classe {classification}...")
        return

    path_img_comp = os.path.join(dir_img, f"{code_muni}_{composition}_{ano_fim}.png")
    if not os.path.exists(path_img_comp):
        path_img_comp = os.path.join(dir_img, f"{code_muni}_rgb_{ano_fim}.png")
        
    if not os.path.exists(path_img_comp):
        print(f"⚠️ [AVISO] Imagem não encontrada: {path_img_comp}. Pulando...")
        return

    img = cv2.imread(path_img_comp)
    if img is None:
        return

    img_slic = np.load(path_slic)
    df_raw = pd.read_csv(path_csv)

    if df_raw.empty:
        return

    df = selecionar_amostras(df_raw, modo=modo)
    if df.empty:
        print(f"ℹ️ Sem registros para a fase '{modo}' na classe '{classification}'.")
        return

    unique_segments_sort = df["Segment_label"].values
    hors_sort = df["HoR"].values
    num_segs = len(unique_segments_sort)

    tag_fase = "PILOTO (100 Imagens)" if modo == 'pilot' else "EXECUÇÃO COMPLETA (Da imagem 101 ao fim)"
    print(f"\n📸 [{tag_fase}] Classe: '{classification}' | Quantidade: {num_segs} chips", flush=True)

    prefix = PREFIX_MAP.get(composition.lower(), 0)
    color_bgr = obter_cor_contorno(composition)

    output_dir_full = os.path.join(dir_output, f"AE-X{code_muni}", f"{classification}_sort_hor", composition)
    os.makedirs(output_dir_full, exist_ok=True)

    nrows, ncols = img.shape[:2]

    for i, seg_val in enumerate(tqdm(unique_segments_sort, desc=f" {classification}/{composition} ({modo})")):
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
        cv2.drawContours(newimg, contours, -1, color_bgr, thickness=1)

        subimg = newimg[first_row:last_row + 1, first_col:last_col + 1]

        if subimg.size > 0:
            zoomimg = cv2.resize(subimg, None, fx=ZOOM_PARAM, fy=ZOOM_PARAM, interpolation=cv2.INTER_CUBIC)
            idx_label = f"pilot_{i:02d}" if modo == 'pilot' else f"rem_{i:05d}"
            filename = f"{prefix}_{idx_label}_task_{int(seg_val)}_HoR_{hor_seg:.4f}_{composition}.png"
            output_filepath = os.path.join(output_dir_full, filename)
            cv2.imwrite(output_filepath, zoomimg)

    del img, img_slic, df, df_raw
    gc.collect()

def disparar_script_086(script_086_path, code_muni, ano_inicio, ano_fim, fase_nome):
    print(f"\n🔄 Disparando execução do Script 08.6 ({fase_nome})...", flush=True)
    if os.path.exists(script_086_path):
        cmd = [sys.executable, script_086_path, code_muni, ano_inicio, ano_fim]
        res = subprocess.run(cmd)
        if res.returncode == 0:
            print(f"✅ Script 08.6 executado com sucesso ({fase_nome})!")
        else:
            print(f"⚠️ Erro ao executar Script 08.6 ({fase_nome}): Código {res.returncode}")
    else:
        print(f"⚠️ Script 08.6 não encontrado em: {script_086_path}")

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
    dir_proc_rgb = os.path.join(dir_processing, "rgb")
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
    compositions = ['rgb']

    # =========================================================================
    # FASE 1: EXECUÇÃO DO PILOTO (100 IMAGENS) E 1ª CHAMADA DO SCRIPT 08.6
    # =========================================================================
    print("=" * 80, flush=True)
    print(f"🚀 FASE 1 - GERAÇÃO PILOTO (100 IMAGENS) ({ano_fim})", flush=True)
    print("=" * 80, flush=True)

    for classification in classifications:
        path_csv = os.path.join(dir_csv_forest if classification == 'forest' else dir_csv_non_forest, f"df_{classification}_sort_{code_muni}_{ano_fim}.csv")
        for composition in compositions:
            processar_chips(code_muni, ano_fim, classification, composition, path_slic, path_csv, dir_proc_rgb, dir_output_zooniverse, modo='pilot')

    disparar_script_086(script_086_path, code_muni, ano_inicio, ano_fim, "1ª Chamada - Piloto")

    # =========================================================================
    # FASE 2: EXECUÇÃO DOS RESTANTES (IMAGEM 101 AO FIM) E 2ª CHAMADA DO SCRIPT 08.6
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print(f"🚀 FASE 2 - GERAÇÃO DOS SUPERPIXELS RESTANTES (IMAGEM 101 AO FIM) ({ano_fim})", flush=True)
    print("=" * 80, flush=True)

    for classification in classifications:
        path_csv = os.path.join(dir_csv_forest if classification == 'forest' else dir_csv_non_forest, f"df_{classification}_sort_{code_muni}_{ano_fim}.csv")
        for composition in compositions:
            processar_chips(code_muni, ano_fim, classification, composition, path_slic, path_csv, dir_proc_rgb, dir_output_zooniverse, modo='remaining')

    disparar_script_086(script_086_path, code_muni, ano_inicio, ano_fim, "2ª Chamada - Completa")

    print(f"\n🎉 PROCESSO COMPLETO FINALIZADO! Todos os recortes e tarefas foram gerados.")

if __name__ == "__main__":
    main()