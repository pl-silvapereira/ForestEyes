import sys
import os
import shutil
import re
from glob import glob
import pandas as pd

def rename_files_list(file_list):
    """Garante padronização de dígitos numéricos nos nomes dos arquivos."""
    for file_path in file_list:
        nome_figura = os.path.basename(file_path)
        novo_nome = re.sub(r'_(\d{1})_', r'_0\1_', nome_figura)
        if novo_nome != nome_figura:
            os.rename(file_path, os.path.join(os.path.dirname(file_path), novo_nome))

def main():
    TASKS_TO_CREATE_BY_CLASS_BY_SA = 10

    all_segs_path = sorted(glob('out_workflow/all_segs/*'))
    workflow_path = 'out_workflow/cbers_workflow'

    if not os.path.exists(workflow_path):
        os.makedirs(workflow_path)
        print(f'Diretório "{workflow_path}" criado com sucesso!')

    task_id_to_create = 1

    for area_path in all_segs_path:
        area_name = os.path.basename(area_path)
        
        tasks_forest_path = os.path.join(workflow_path, area_name, 'forest')
        tasks_non_forest_path = os.path.join(workflow_path, area_name, 'non-forest')
        
        num_tasks_per_group = int(TASKS_TO_CREATE_BY_CLASS_BY_SA / 2)
        
        for _ in range(num_tasks_per_group):
            dir_hor_1 = os.path.join(tasks_forest_path, f'task_{task_id_to_create:03d}')
            dir_hor_07 = os.path.join(tasks_forest_path, f'task_{task_id_to_create + 100:03d}')
            os.makedirs(dir_hor_1, exist_ok=True)
            os.makedirs(dir_hor_07, exist_ok=True)
            task_id_to_create += 1

        for _ in range(num_tasks_per_group):
            dir_hor_1 = os.path.join(tasks_non_forest_path, f'task_{task_id_to_create:03d}')
            dir_hor_07 = os.path.join(tasks_non_forest_path, f'task_{task_id_to_create + 100:03d}')
            os.makedirs(dir_hor_1, exist_ok=True)
            os.makedirs(dir_hor_07, exist_ok=True)
            task_id_to_create += 1

    for area_path in all_segs_path:
        area_name = os.path.basename(area_path)
        
        for classification in ['forest', 'non-forest']:
            comp_dirs = sorted(glob(os.path.join(area_path, f'{classification}_sort_hor', '*')))
            target_tasks = sorted(glob(os.path.join(workflow_path, area_name, classification, 'task_*')))
            
            if not comp_dirs:
                continue
                
            for comp_dir in comp_dirs:
                seg_files = sorted(glob(os.path.join(comp_dir, '*.png')))
                rename_files_list(seg_files)
                seg_files = sorted(glob(os.path.join(comp_dir, '*.png')))
                
                for task_idx, file_path in enumerate(seg_files):
                    if task_idx < len(target_tasks):
                        target_dir = target_tasks[task_idx]
                        shutil.move(file_path, target_dir)

    def get_task_id(path):
        return os.path.basename(path).replace('task_', '')

    scenes_path = sorted(glob(os.path.join(workflow_path, '*')))

    for scene in scenes_path:
        class_paths = sorted(glob(os.path.join(scene, '*')))
        
        for class_path in class_paths:
            tasks = sorted(glob(os.path.join(class_path, 'task_*')))
            
            for task in tasks:
                files = sorted([f for f in os.listdir(task) if f.endswith('.png')])
                if not files:
                    continue
                    
                task_id = get_task_id(task)
                data_dict = {"id": task_id}
                for file_name in files:
                    prefix = file_name.split('_')[0]
                    comp_name = file_name.replace('.png', '').split('_')[-1]
                    col_name = f"{prefix}_{comp_name}"
                    data_dict[col_name] = file_name
                    
                dataframe = pd.DataFrame([data_dict])
                dataframe.to_csv(os.path.join(task, 'manifest.csv'), index=False)
                
    print(f'✅ Manifesto Zooniverse gerado com sucesso para a estrutura CBERS!')

if __name__ == "__main__":
    main()