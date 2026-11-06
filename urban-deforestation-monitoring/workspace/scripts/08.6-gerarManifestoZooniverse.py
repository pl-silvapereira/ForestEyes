#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Etapa 08.6: Organização Dinâmica de Tarefas e Manifesto Zooniverse
==============================================================================
Descrição:
 - Detecta automaticamente as imagens disponíveis em `out_workflow/all_segs/`.
 - Aloca as imagens dinamicamente em pastas de tarefas (10 imagens por tarefa).
 - Incrementa a numeração `task_001`, `task_002`, ... sem sobrescrever as existentes.
 - Gera o arquivo `manifest.csv` para cada tarefa criada ou atualizada.
==============================================================================
"""

import sys
import os
import shutil
import re
from glob import glob
import pandas as pd
from dotenv import load_dotenv

def rename_files_list(file_list):
    """Garante a formatação uniforme de dois dígitos (_1_ -> _01_)."""
    for file_path in file_list:
        nome_figura = os.path.basename(file_path)
        novo_nome = re.sub(r'_(\d{1})_', r'_0\1_', nome_figura)
        if novo_nome != nome_figura:
            os.rename(file_path, os.path.join(os.path.dirname(file_path), novo_nome))

def get_next_task_id(workflow_path):
    """Encontra o maior ID de tarefa existente na pasta do workflow e retorna o próximo número inteiro."""
    existing_tasks = glob(os.path.join(workflow_path, '*', '*', 'task_*'))
    max_id = 0
    for t_path in existing_tasks:
        folder_name = os.path.basename(t_path)
        num_str = folder_name.replace('task_', '')
        if num_str.isdigit():
            max_id = max(max_id, int(num_str))
    return max_id + 1

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 08.6-gerarManifestoZooniverse.py <code_muni> <ano_inicio> <ano_fim>")
        sys.exit(1)

    code_muni = str(sys.argv[1])
    ano_inicio = str(sys.argv[2])
    ano_fim = str(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    print("=" * 80, flush=True)
    print(f"🚀 SCRIPT 08.6 - ALOCAÇÃO DINÂMICA E GERAÇÃO DE MANIFESTO ZOONIVERSE ({ano_fim})", flush=True)
    print("=" * 80, flush=True)

    IMAGES_PER_TASK = 10  # Grupo padrão de 10 sujeitos por tarefa no Zooniverse

    all_segs_path = sorted(glob(os.path.join(project_root, 'out_workflow', 'all_segs', '*')))
    workflow_path = os.path.join(project_root, 'out_workflow', 'cbers_workflow')
    os.makedirs(workflow_path, exist_ok=True)

    # 1. Alocação dinâmica de arquivos das imagens em out_workflow/all_segs
    for area_path in all_segs_path:
        area_name = os.path.basename(area_path)
        
        for classification in ['forest', 'non-forest']:
            comp_dirs = sorted(glob(os.path.join(area_path, f'{classification}_sort_hor', '*')))
            tasks_class_path = os.path.join(workflow_path, area_name, classification)
            os.makedirs(tasks_class_path, exist_ok=True)

            if not comp_dirs:
                continue

            for comp_dir in comp_dirs:
                seg_files = sorted(glob(os.path.join(comp_dir, '*.png')))
                if not seg_files:
                    continue

                rename_files_list(seg_files)
                seg_files = sorted(glob(os.path.join(comp_dir, '*.png')))

                # Move os arquivos em lotes de 10 para novas tarefas
                for i in range(0, len(seg_files), IMAGES_PER_TASK):
                    chunk = seg_files[i:i + IMAGES_PER_TASK]
                    next_id = get_next_task_id(workflow_path)
                    task_dir = os.path.join(tasks_class_path, f'task_{next_id:03d}')
                    os.makedirs(task_dir, exist_ok=True)

                    for file_path in chunk:
                        shutil.move(file_path, task_dir)

    # 2. Varredura de todas as tarefas para geração e atualização dos manifestos
    scenes_path = sorted(glob(os.path.join(workflow_path, '*')))

    total_manifests = 0
    for scene in scenes_path:
        class_paths = sorted(glob(os.path.join(scene, '*')))
        
        for class_path in class_paths:
            tasks = sorted(glob(os.path.join(class_path, 'task_*')))
            
            for task in tasks:
                files = sorted([f for f in os.listdir(task) if f.endswith('.png')])
                if not files:
                    continue
                    
                task_id = os.path.basename(task).replace('task_', '')
                data_dict = {"id": task_id}
                for file_name in files:
                    prefix = file_name.split('_')[0]
                    comp_name = file_name.replace('.png', '').split('_')[-1]
                    col_name = f"{prefix}_{comp_name}"
                    data_dict[col_name] = file_name
                    
                dataframe = pd.DataFrame([data_dict])
                manifest_path = os.path.join(task, 'manifest.csv')
                dataframe.to_csv(manifest_path, index=False)
                total_manifests += 1
                
    print(f'✅ Total de {total_manifests} manifestos gerados/atualizados em: {workflow_path}')

if __name__ == "__main__":
    main()