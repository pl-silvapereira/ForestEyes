#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Etapa 08.6: Estruturação da Tarefa do Segmento e Manifesto Zooniverse
==============================================================================
Descrição:
 - Recebe um segmento individual (ou executa varredura completa).
 - Cria a pasta `task_{seg_val}` com as 3 composições (RGB, NDVI, NIR).
 - Escreve o arquivo `manifest.csv` unificado para a tarefa contendo as 
   3 composições como colunas do mesmo manifesto.
==============================================================================
"""

import sys
import os
import shutil
from glob import glob
import pandas as pd
from dotenv import load_dotenv

def processar_tarefa_segmento(project_root, code_muni, classification, seg_val, hor_seg=None):
    """Cria a pasta task_{seg_val} e gera o manifesto unificado das 3 composições."""
    dir_seg_temp = os.path.join(project_root, "out_workflow", "all_segs", f"AE-X{code_muni}", classification, f"seg_{seg_val}")
    
    if not os.path.exists(dir_seg_temp):
        return False

    task_dir_name = f"task_{seg_val}"
    target_task_dir = os.path.join(project_root, "out_workflow", "cbers_workflow", f"AE-X{code_muni}", classification, task_dir_name)
    os.makedirs(target_task_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(dir_seg_temp) if f.endswith('.png')])
    if not files:
        return False

    # Move os 3 recortes PNG para a pasta definitiva da tarefa
    for file_name in files:
        src_file = os.path.join(dir_seg_temp, file_name)
        dst_file = os.path.join(target_task_dir, file_name)
        shutil.move(src_file, dst_file)

    # Remove o diretório temporário do segmento
    try:
        os.rmdir(dir_seg_temp)
    except OSError:
        pass

    # --- MONTAGEM DO MANIFEST.CSV UNIFICADO DAS 3 COMPOSIÇÕES ---
    task_files = sorted([f for f in os.listdir(target_task_dir) if f.endswith('.png')])
    data_dict = {"id": task_dir_name}

    for f_name in task_files:
        if "rgb" in f_name.lower():
            data_dict["0_rgb"] = f_name
        elif "ndvi" in f_name.lower():
            data_dict["1_ndvi"] = f_name
        elif "nir" in f_name.lower():
            data_dict["2_nir"] = f_name

    dataframe = pd.DataFrame([data_dict])
    manifest_path = os.path.join(target_task_dir, "manifest.csv")
    dataframe.to_csv(manifest_path, index=False)
    return True

def varredura_completa(project_root, code_muni):
    """Executa a verificação e geração de manifestos para todas as tarefas existentes."""
    workflow_path = os.path.join(project_root, "out_workflow", "cbers_workflow", f"AE-X{code_muni}")
    tasks = glob(os.path.join(workflow_path, "*", "task_*"))
    total_manifests = 0

    for task_dir in tasks:
        task_id = os.path.basename(task_dir)
        files = sorted([f for f in os.listdir(task_dir) if f.endswith('.png')])
        if not files:
            continue

        data_dict = {"id": task_id}
        for f_name in files:
            if "rgb" in f_name.lower():
                data_dict["0_rgb"] = f_name
            elif "ndvi" in f_name.lower():
                data_dict["1_ndvi"] = f_name
            elif "nir" in f_name.lower():
                data_dict["2_nir"] = f_name

        dataframe = pd.DataFrame([data_dict])
        manifest_path = os.path.join(task_dir, "manifest.csv")
        dataframe.to_csv(manifest_path, index=False)
        total_manifests += 1

    print(f"✅ Total de {total_manifests} manifestos verificados/gerados com sucesso!")

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 08.6-gerarManifestoZooniverse.py <code_muni> <ano_inicio> <ano_fim> [classification] [seg_val] [hor_seg]")
        sys.exit(1)

    code_muni = str(sys.argv[1])
    ano_inicio = str(sys.argv[2])
    ano_fim = str(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    # Chamada individual por segmento feita pelo Script 08.5
    if len(sys.argv) >= 6:
        classification = str(sys.argv[4])
        seg_val = str(sys.argv[5])
        hor_seg = str(sys.argv[6]) if len(sys.argv) >= 7 else None
        processar_tarefa_segmento(project_root, code_muni, classification, seg_val, hor_seg)
    else:
        # Modo de varredura geral
        varredura_completa(project_root, code_muni)

if __name__ == "__main__":
    main()