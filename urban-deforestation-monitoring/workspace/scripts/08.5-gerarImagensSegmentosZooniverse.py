#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Etapa 08.5: Geração de Imagens de Segmentos (Chips) para o Zooniverse
==============================================================================
Descrição:
  - Recorta os superpíxeis filtrados (floresta e não-floresta) com margem de contexto.
  - Desenha o contorno/fronteira do segmento em destaque.
  - Aplica o zoom necessário (2.5x) para adequação à plataforma Zooniverse.
  - Exporta os arquivos de imagem (.png) com nomenclatura padronizada.
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

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)

# Configurações Padrão da Campanha Zooniverse
ZOOM_PARAM = 2.5   # Zoom de 2.5x para imagens Sentinel-2 / CBERS-4A
BORDER = 120       # Margem em pixels ao redor do segmento para incluir contexto espacial

# Mapeamento de prefixos por composição para ordenação no Zooniverse
PREFIX_MAP = {
    'ndvi': 0,
    '8114': 1,
    '41112': 2,
    '1184': 3,
    'rgb': 0
}

def obter_cor_contorno(composition):
    """Retorna a cor BGR para destacar a borda do segmento conforme a composição."""
    if composition == '8114':
        # Azul escuro para composição avermelhada (BGR)
        return (160, 64, 64)
    else:
        # Destaque avermelhado/castanho para as demais composições (BGR)
        return (20, 70, 140)

def processar_classe_e_composicao(code_muni, ano_fim, classification, composition, path_slic, path_csv, dir_img, dir_output):
    """Gera os recorte de imagem (chips) para uma determinada classe e composição."""
    if not os.path.exists(path_csv):
        print(f"⚠️ [AVISO] Arquivo CSV não encontrado: {path_csv}. Pulando classe {classification}...")
        return

    # Busca a imagem da composição específica ou recorre à imagem RGB padrão
    path_img_comp = os.path.join(dir_img, f"{code_muni}_{composition}_{ano_fim}.png")
    if not os.path.exists(path_img_comp):
        path_img_comp = os.path.join(dir_img, f"{code_muni}_rgb_{ano_fim}.png")
    
    if not os.path.exists(path_img_comp):
        print(f"⚠️ [AVISO] Imagem não encontrada em {path_img_comp}. Pulando composição {composition}...")
        return

    print(f"\n📸 Processando Classe: '{classification}' | Composição: '{composition}'", flush=True)

    # Carregamento da Imagem de entrada e da Máscara SLIC
    img = cv2.imread(path_img_comp)
    if img is None:
        print(f"❌ Falha ao ler a imagem: {path_img_comp}")
        return

    img_slic = np.load(path_slic)
    df = pd.read_csv(path_csv)

    if df.empty:
        print(f"⚠️ CSV vazio: {path_csv}")
        return

    unique_segments_sort = df["Segment_label"].values
    hors_sort = df["HoR"].values
    num_segs = len(unique_segments_sort)

    prefix = PREFIX_MAP.get(composition.lower(), 0)
    color_bgr = obter_cor_contorno(composition)

    output_dir_full = os.path.join(dir_output, f"AE-X{code_muni}", f"{classification}_sort_hor", composition)
    os.makedirs(output_dir_full, exist_ok=True)

    nrows, ncols = img.shape[:2]
    
    print(f"   -> Gerando {num_segs:,} imagens em: {output_dir_full}", flush=True)

    for i, seg_val in enumerate(tqdm(unique_segments_sort, desc=f"   {classification}/{composition}")):
        hor_seg = hors_sort[i]
        
        # Máscara binária do segmento atual
        seg_mask = (img_slic == seg_val).astype(np.uint8)
        
        rows, cols = np.where(seg_mask > 0)
        if len(rows) == 0:
            continue

        # Delimitação da caixa envolvente (bounding box)
        min_r, max_r = rows.min(), rows.max()
        min_c, max_c = cols.min(), cols.max()

        # Aplicação da margem de contexto (border) com controle de limites
        first_row = max(0, min_r - BORDER)
        last_row = min(nrows - 1, max_r + BORDER)
        first_col = max(0, min_c - BORDER)
        last_col = min(ncols - 1, max_c + BORDER)

        # Cópia da imagem original para desenho do contorno
        newimg = img.copy()

        # Extração vetorizada do contorno externo do segmento (C-Level via OpenCV)
        contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(newimg, contours, -1, color_bgr, thickness=1)

        # Recorte da sub-imagem com margem de contexto
        subimg = newimg[first_row:last_row + 1, first_col:last_col + 1]

        if subimg.size > 0:
            # Aplicação do Zoom
            zoomimg = cv2.resize(subimg, None, fx=ZOOM_PARAM, fy=ZOOM_PARAM, interpolation=cv2.INTER_CUBIC)
            
            # Formatação do nome do arquivo padrão para a tarefa Zooniverse
            filename = f"{prefix}_{i:02d}_task_{int(seg_val)}_HoR_{hor_seg:.4f}_{composition}.png"
            output_filepath = os.path.join(output_dir_full, filename)
            
            cv2.imwrite(output_filepath, zoomimg)

    del img, img_slic, df
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

    # Resolução de diretórios
    dir_processing = os.path.join(project_root, "data", "processing")
    dir_proc_rgb = os.path.join(dir_processing, "rgb")
    dir_proc_slics = os.path.join(dir_processing, "slics")
    dir_csv_root = os.path.join(dir_processing, "csv")
    
    dir_csv_forest = os.path.join(dir_csv_root, "forest")
    dir_csv_non_forest = os.path.join(dir_csv_root, "non-forest")

    dir_output_zooniverse = os.path.join(project_root, "out_workflow", "all_segs")

    path_slic = os.path.join(dir_proc_slics, f"mask_slic_{code_muni}_{ano_fim}.npy")

    if not os.path.exists(path_slic):
        print(f"❌ [ERRO CRÍTICO] Matriz SLIC (.npy) não encontrada: {path_slic}")
        sys.exit(1)

    print("=" * 80, flush=True)
    print(f"🚀 SCRIPT 08.5 - GERAÇÃO DE IMAGENS DE SEGMENTOS PARA ZOONIVERSE ({ano_fim})", flush=True)
    print("=" * 80, flush=True)

    # Classes a processar
    classifications = ['forest', 'non-forest']
    
    # Composições a serem processadas (ajuste se houver outras composições na pasta)
    compositions = ['rgb']

    for classification in classifications:
        if classification == 'forest':
            path_csv = os.path.join(dir_csv_forest, f"df_forest_sort_{code_muni}_{ano_fim}.csv")
        else:
            path_csv = os.path.join(dir_csv_non_forest, f"df_non-forest_sort_{code_muni}_{ano_fim}.csv")

        for composition in compositions:
            processar_classe_e_composicao(
                code_muni=code_muni,
                ano_fim=ano_fim,
                classification=classification,
                composition=composition,
                path_slic=path_slic,
                path_csv=path_csv,
                dir_img=dir_proc_rgb,
                dir_output=dir_output_zooniverse
            )

    print(f"\n✅ SUCESSO! Script 08.5 concluído com êxito.")
    print(f" -> Imagens salvas em: {dir_output_zooniverse}")

if __name__ == "__main__":
    main()