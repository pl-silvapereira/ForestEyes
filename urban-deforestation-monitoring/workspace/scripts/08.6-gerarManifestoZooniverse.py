#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Etapa 08.6: Estruturação de Tarefas e Manifesto Zooniverse (Execução Manual)
==============================================================================
Descrição:
 - Agrupa todos os chips PNG salvos em `out_workflow/all_segs/` em tarefas de 10 itens.
 - Move os arquivos para a estrutura `out_workflow/cbers_workflow/{area}/{classe}/task_{id}/`.
 - Compila o arquivo `manifest.csv` para cada pasta de tarefa gerada.
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
    """Garante a padronização de dígitos nos nomes dos arquivos."""
    for file_path in file_list:
        nome_figura = os.path.basename(file_path)
        novo_nome = re.sub(r'_(\d{1})_', r'_0\1_', nome_figura)
        if novo_nome != nome_figura:
            os.rename(file_path, os.path.join(os.path.dirname(file_path), novo_nome))

def get_next_task_id(workflow_path):
    """Retorna o próximo ID de tarefa disponível na estrutura do workflow."""
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
    print(f"🚀 SCRIPT 08.6 - MONTAGEM DE TAREFAS E MANIFESTOS ZOONIVERSE ({ano_fim})", flush=True)
    print("=" * 80, flush=True)

    IMAGES_PER_TASK = 10

    all_segs_path = sorted(glob(os.path.join(project_root, 'out_workflow', 'all_segs', '*')))
    workflow_path = os.path.join(project_root, 'out_workflow', 'cbers_workflow')
    os.makedirs(workflow_path, exist_ok=True)

    if not all_segs_path:
        print(f"⚠️ Nenhuma área encontrada em: {os.path.join(project_root, 'out_workflow', 'all_segs')}")
        print("Certifique-se de ter executado o Script 08.5 antes de rodar o 08.6.")
        sys.exit(1)

    # 1. Agrupamento dinâmico e movimentação dos arquivos por tarefa
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

                for i in range(0, len(seg_files), IMAGES_PER_TASK):
                    chunk = seg_files[i:i + IMAGES_PER_TASK]
                    next_id = get_next_task_id(workflow_path)
                    task_dir = os.path.join(tasks_class_path, f'task_{next_id:03d}')
                    os.makedirs(task_dir, exist_ok=True)

                    for file_path in chunk:
                        shutil.move(file_path, task_dir)

    # 2. Geração dos arquivos manifest.csv
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
                    parts = file_name.replace('.png', '').split('_')
                    prefix = parts[0]
                    comp_name = parts[-1]
                    col_name = f"{prefix}_{comp_name}"
                    data_dict[col_name] = file_name
                    
                dataframe = pd.DataFrame([data_dict])
                manifest_path = os.path.join(task, 'manifest.csv')
                dataframe.to_csv(manifest_path, index=False)
                total_manifests += 1
                
    print(f'✅ Concluído! Total de {total_manifests} manifestos gerados com sucesso em:\n   {workflow_path}')

if __name__ == "__main__":
    main()