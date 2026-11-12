#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Etapa 08.6: Estruturação da Tarefa do Segmento e Manifesto Zooniverse
==============================================================================
Descrição:
 - Opera diretamente em PROJECT_ROOT/data/out_workflow/.
 - Move as 3 composições (RGB, NDVI, NIR) da pasta temporária seg_{seg_val}
   para a pasta definitiva task_{seg_val}, removendo a pasta temporária.
 - Gera o arquivo manifest.csv unificado contendo as 3 composições da tarefa.
==============================================================================
"""

import sys
import os
import time
import shutil
from glob import glob
import pandas as pd
from dotenv import load_dotenv


def mover_com_retry(src_file, dst_file, max_tentativas=5):
    """Move arquivo no Google Drive com retentativa para evitar falhas FUSE."""
    for tentativa in range(1, max_tentativas + 1):
        try:
            shutil.move(src_file, dst_file)
            return True
        except OSError:
            time.sleep(1.5 * tentativa)
    raise IOError(f"Falha ao mover arquivo no Drive após {max_tentativas} tentativas: {src_file}")


def processar_tarefa_segmento(project_root, code_muni, classification, seg_val, hor_seg=None):
    """Cria a pasta task_{seg_val}, move as 3 composições e gera o manifest.csv."""
    base_out = os.path.join(project_root, "data", "out_workflow")
    dir_seg_temp = os.path.join(base_out, "all_segs", f"AE-X{code_muni}", classification, f"seg_{seg_val}")

    if not os.path.exists(dir_seg_temp):
        return False

    task_dir_name = f"task_{seg_val}"
    target_task_dir = os.path.join(base_out, "cbers_workflow", f"AE-X{code_muni}", classification, task_dir_name)
    os.makedirs(target_task_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(dir_seg_temp) if f.endswith('.png')])
    if not files:
        return False

    # Move os 3 recortes PNG para a pasta definitiva da tarefa
    for file_name in files:
        src_file = os.path.join(dir_seg_temp, file_name)
        dst_file = os.path.join(target_task_dir, file_name)
        if os.path.exists(dst_file):
            os.remove(dst_file)
        mover_com_retry(src_file, dst_file)

    # Remove o diretório temporário do segmento para economizar espaço e inodes
    try:
        os.rmdir(dir_seg_temp)
    except OSError:
        pass

    # Montagem do manifest.csv unificado das 3 composições
    task_files = sorted([f for f in os.listdir(target_task_dir) if f.endswith('.png')])
    fn_rgb, fn_ndvi, fn_nir = "", "", ""

    for f_name in task_files:
        lower_name = f_name.lower()
        if "rgb" in lower_name:
            fn_rgb = f_name
        elif "ndvi" in lower_name:
            fn_ndvi = f_name
        elif "nir" in lower_name:
            fn_nir = f_name

    manifest_path = os.path.join(target_task_dir, "manifest.csv")
    for tentativa in range(1, 5):
        try:
            with open(manifest_path, "w", encoding="utf-8") as f_man:
                f_man.write("id,0_rgb,1_ndvi,2_nir\n")
                f_man.write(f"{task_dir_name},{fn_rgb},{fn_ndvi},{fn_nir}\n")
            return True
        except OSError:
            time.sleep(1.5 * tentativa)

    return False


def varredura_completa(project_root, code_muni):
    """Executa a verificação e geração de manifestos para todas as tarefas existentes."""
    workflow_path = os.path.join(project_root, "data", "out_workflow", "cbers_workflow", f"AE-X{code_muni}")

    if not os.path.exists(workflow_path):
        print(f"⚠️ Pasta não encontrada: {workflow_path}")
        sys.exit(1)

    tasks = glob(os.path.join(workflow_path, "*", "task_*"))
    total_manifests = 0

    for task_dir in tasks:
        task_id = os.path.basename(task_dir)
        files = sorted([f for f in os.listdir(task_dir) if f.endswith('.png')])
        if not files:
            continue

        data_dict = {"id": task_id}
        for f_name in files:
            lower_name = f_name.lower()
            if "rgb" in lower_name:
                data_dict["0_rgb"] = f_name
            elif "ndvi" in lower_name:
                data_dict["1_ndvi"] = f_name
            elif "nir" in lower_name:
                data_dict["2_nir"] = f_name

        dataframe = pd.DataFrame([data_dict])
        manifest_path = os.path.join(task_dir, "manifest.csv")
        dataframe.to_csv(manifest_path, index=False)
        total_manifests += 1

    print(f"✅ Total de {total_manifests:,} manifestos verificados/gerados com sucesso em:\n   {workflow_path}")


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

    if len(sys.argv) >= 6:
        classification = str(sys.argv[4])
        seg_val = str(sys.argv[5])
        hor_seg = str(sys.argv[6]) if len(sys.argv) >= 7 else None
        processar_tarefa_segmento(project_root, code_muni, classification, seg_val, hor_seg)
    else:
        varredura_completa(project_root, code_muni)


if __name__ == "__main__":
    main()