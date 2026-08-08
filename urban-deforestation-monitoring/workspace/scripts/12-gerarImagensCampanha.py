import os
import sys
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from skimage.measure import label, regionprops
from scipy.ndimage import binary_dilation, binary_fill_holes, find_objects, zoom, binary_closing
from dotenv import load_dotenv

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 12-gerarImagensCampanha.py <code_muni> <ano_inicio> <ano_fim>")
        sys.exit(1)

    code_muni = int(sys.argv[1])
    ano_inicio = str(sys.argv[2])
    ano_fim = str(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    segmentation_dir = os.path.join(project_root, "data", "output", "mask", "segmentation")
    campaign_dir = os.path.join(project_root, "data", "output", "mask", "campaign")
    os.makedirs(campaign_dir, exist_ok=True)

    path_labels = os.path.join(segmentation_dir, f"{code_muni}_segmentation_labels_{ano_inicio}_vs_{ano_fim}.tif")
    path_sat = os.path.join(
        project_root, "data", "output", "pansharpening", "geopolitic-RGBN", 
        ano_fim, f"{code_muni}_{ano_fim}_CBERS_TRUE_COLOR_CLIPPED.tif"
    )

    if not os.path.exists(path_labels) or not os.path.exists(path_sat):
        print(f"[ERRO CRÍTICO] Arquivos de segmentação ou satélite não encontrados.")
        sys.exit(1)

    print("=" * 115)
    print(f"🎯 GERANDO PATCHES: RESOLUÇÃO NÍTIDA (720px) E FILTRO DE SUPERPIXELS FRAGMENTADOS")
    print(f"📍 MUNICÍPIO: {code_muni} | PERÍODO: {ano_inicio} vs {ano_fim}")
    print("=" * 115)

    print("Lendo raster de labels e imagem de satélite...")
    with rasterio.open(path_labels) as src_lab:
        labels = src_lab.read(1).astype(np.int32)

    with rasterio.open(path_sat) as src_sat:
        sat_data = src_sat.read()

    ids_unicos = np.unique(labels)
    ids_unicos = ids_unicos[ids_unicos > 0]

    print(f"Total de superpixels detectados: {len(ids_unicos)}. Processando e filtrando...")

    slices = find_objects(labels)
    estatisticas_lista = []

    # PREPARAÇÃO DA BASE DE DADOS E ESTATÍSTICAS (Sem alteração na lógica original de atributos)
    for sp_id in ids_unicos:
        slc = slices[sp_id - 1]
        if slc is None:
            continue
        
        mask_sp_small = (labels[slc] == sp_id)
        mask_sp_small = binary_closing(mask_sp_small, structure=np.ones((3,3)))

        labeled_mask, num_features = label(mask_sp_small, return_num=True)
        if num_features == 0: continue
        
        props = regionprops(labeled_mask)
        largest_comp = max(props, key=lambda r: r.area)
        
        clean_mask_small = (labeled_mask == largest_comp.label)
        clean_mask_small = binary_fill_holes(clean_mask_small)
        area = np.sum(clean_mask_small)
        
        if area < 100 or area > 2000:
            continue
        
        sat_slice = sat_data[:, slc[0], slc[1]]
        
        if sat_slice.shape[0] >= 3:
            r = sat_slice[0][clean_mask_small].astype(np.float32)
            g = sat_slice[1][clean_mask_small].astype(np.float32)
            media_ver = np.mean(g)
            media_verm = np.mean(r)
            is_floresta = media_ver > media_verm
        else:
            is_floresta = True

        classe = "Floresta" if is_floresta else "Nao_Floresta"
        std_val = np.std(r) if sat_slice.shape[0] >= 3 else 10
        hor_simulado = max(70.0, min(100.0, 100.0 - (std_val * 0.5)))
        
        estatisticas_lista.append([sp_id, area, round(hor_simulado, 2), classe])

    df = pd.DataFrame(estatisticas_lista, columns=['ID_Segmento', 'Quantidade_Pixels', 'Taxa_HoR', 'Classe_Majoritaria'])

    if df.empty:
        print("\n❌ ERRO CRÍTICO: Nenhum superpixel sobreviveu aos filtros.")
        sys.exit(1)

    df['Taxa_HoR'] = pd.to_numeric(df['Taxa_HoR'], errors='coerce')
    df = df.dropna(subset=['Taxa_HoR'])

    df_floresta = df[df['Classe_Majoritaria'] == 'Floresta']
    df_nao_floresta = df[df['Classe_Majoritaria'] == 'Nao_Floresta']

    # Filas ordenadas para garantir a extração sequencial
    filas_processamento = {
        'Floresta_Perfeito': df_floresta.sort_values(by='Taxa_HoR', ascending=False),
        'Floresta_Imperfeito': df_floresta.iloc[(df_floresta['Taxa_HoR'] - 70.0).abs().argsort()],
        'Nao_Floresta_Perfeito': df_nao_floresta.sort_values(by='Taxa_HoR', ascending=False),
        'Nao_Floresta_Imperfeito': df_nao_floresta.iloc[(df_nao_floresta['Taxa_HoR'] - 70.0).abs().argsort()]
    }

    # Normalização de Contraste da Imagem
    h_img, w_img = sat_data.shape[1], sat_data.shape[2]
    rgb_normalized = np.zeros((3, h_img, w_img), dtype=np.uint8)
    
    for b_idx in range(min(3, sat_data.shape[0])):
        b_data = sat_data[b_idx].astype(np.float32)
        p2, p98 = np.percentile(b_data[b_data > 0], (2, 98)) if np.any(b_data > 0) else (0, 1)
        if p98 > p2:
            norm = np.clip((b_data - p2) / (p98 - p2), 0, 1) * 255.0
        else:
            norm = np.clip(b_data, 0, 255)
        rgb_normalized[b_idx] = norm.astype(np.uint8)

    print("Renderizando exatamente 100 imagens da campanha...")
    
    contador = 1
    processed_ids = set()

    for nome_fila, fila_df in filas_processamento.items():
        salvos_nesta_categoria = 0
        classe_str = "Floresta" if "Nao_Floresta" not in nome_fila else "Nao_Floresta"
        tipo_str = "Perfeito" if "Perfeito" in nome_fila else "Imperfeito"
        
        for idx, row in fila_df.iterrows():
            if salvos_nesta_categoria >= 25:
                break # Meta de 25 imagens por categoria atingida
            
            sp_id = int(row['ID_Segmento'])
            hor = row['Taxa_HoR']

            if sp_id in processed_ids:
                continue

            slc = slices[sp_id - 1]
            if slc is None: continue
            
            mask_sp_small = (labels[slc] == sp_id)
            mask_sp_small = binary_closing(mask_sp_small, structure=np.ones((3,3)))
            
            # --- NOVO FILTRO: DESCARTAR SE HOUVER MAIS DE UM PEDAÇO DO SUPERPIXEL ---
            labeled_mask, num_features = label(mask_sp_small, return_num=True)
            if num_features > 1:
                # Se após o tratamento, o superpixel tem feições fragmentadas (2 ou mais polígonos)
                # O script PULA para o próximo ID, evitando imagens poluídas.
                continue

            props = regionprops(labeled_mask)
            largest_comp = max(props, key=lambda r: r.area)
            clean_mask_small = (labeled_mask == largest_comp.label)
            clean_mask_small = binary_fill_holes(clean_mask_small)
            
            y_indices_small, x_indices_small = np.where(clean_mask_small)
            
            offset_y = slc[0].start
            offset_x = slc[1].start
            
            cy = offset_y + (y_indices_small.max() + y_indices_small.min()) // 2
            cx = offset_x + (x_indices_small.max() + x_indices_small.min()) // 2
            
            h_obj = y_indices_small.max() - y_indices_small.min()
            w_obj = x_indices_small.max() - x_indices_small.min()

            padding = 25
            half_size = max(35, max(h_obj, w_obj) // 2 + padding)
            
            ymin, ymax = max(0, cy - half_size), min(h_img, cy + half_size)
            xmin, xmax = max(0, cx - half_size), min(w_img, cx + half_size)

            patch = rgb_normalized[:, ymin:ymax, xmin:xmax]
            patch_rgb = np.moveaxis(patch, 0, -1)

            # Filtro Anti-Nuvem
            cloud_mask = (patch_rgb[..., 0] > 220) & (patch_rgb[..., 1] > 220) & (patch_rgb[..., 2] > 220)
            if (np.sum(cloud_mask) / cloud_mask.size) > 0.15:
                continue 

            patch_labels = np.zeros((ymax - ymin, xmax - xmin), dtype=bool)
            y_final = (y_indices_small + offset_y) - ymin
            x_final = (x_indices_small + offset_x) - xmin
            
            valid = (y_final >= 0) & (y_final < patch_labels.shape[0]) & (x_final >= 0) & (x_final < patch_labels.shape[1])
            patch_labels[y_final[valid], x_final[valid]] = True

            # --- REDIMENSIONAMENTO PARA 520x520 (E NITIDEZ APRIMORADA) ---
            target_size = 520
            zoom_y = target_size / patch_rgb.shape[0]
            zoom_x = target_size / patch_rgb.shape[1]

            patch_rgb_zoomed = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            for c in range(3):
                # order=3 (Bicúbico) remove o aspecto quadriculado ruim e suaviza a imagem de satélite
                patch_rgb_zoomed[..., c] = zoom(patch_rgb[..., c], (zoom_y, zoom_x), order=3)

            # --- LINHA AMARELA 30% MAIS GROSSA ---
            # order=0 para a máscara para manter as bordas matemáticas exatas
            patch_labels_zoomed = zoom(patch_labels, (zoom_y, zoom_x), order=0)
            borders = find_boundaries(patch_labels_zoomed, mode='inner')
            # Aumentado para iterations=2 para deixar a linha visivelmente mais encorpada/grossa
            borders_dilated = binary_dilation(borders, iterations=2) 
            patch_rgb_zoomed[borders_dilated] = [255, 255, 0]

            nome_arquivo = f"target_{contador:03d}_{classe_str}_{tipo_str}_HoR_{hor:.1f}_ID_{sp_id}.png"
            caminho_png = os.path.join(campaign_dir, nome_arquivo)

            plt.imsave(caminho_png, patch_rgb_zoomed)
            processed_ids.add(sp_id)
            salvos_nesta_categoria += 1
            contador += 1

    print(f"\n[SUCESSO] {contador-1} imagens de alta qualidade geradas em:\n-> {campaign_dir}")

if __name__ == "__main__":
    main()