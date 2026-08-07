import os
import sys
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from skimage.measure import label, regionprops
from scipy.ndimage import binary_dilation, binary_fill_holes
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
    print(f"🎯 GERANDO PATCHES: 100 IMAGENS, ZOOM DINÂMICO E NITIDEZ MÁXIMA")
    print(f"📍 MUNICÍPIO: {code_muni} | PERÍODO: {ano_inicio} vs {ano_fim}")
    print("=" * 115)

    print("Lendo raster de labels e imagem de satélite...")
    with rasterio.open(path_labels) as src_lab:
        labels = src_lab.read(1)

    with rasterio.open(path_sat) as src_sat:
        sat_data = src_sat.read()

    ids_unicos = np.unique(labels)
    ids_unicos = ids_unicos[ids_unicos > 0]

    print(f"Total de superpixels detectados: {len(ids_unicos)}. Filtrando e calculando métricas...")

    estatisticas_lista = []
    for sp_id in ids_unicos:
        mask_sp = (labels == sp_id)
        
        # Filtro Rigoroso: Isola apenas a maior massa conectada e preenche buracos internos
        labeled_mask, num_features = label(mask_sp, return_num=True)
        if num_features == 0: continue
        
        props = regionprops(labeled_mask)
        largest_comp = max(props, key=lambda r: r.area)
        
        # Removemos os fragmentos distantes e usamos apenas o polígono principal
        clean_mask = (labeled_mask == largest_comp.label)
        clean_mask = binary_fill_holes(clean_mask)
        
        area = np.sum(clean_mask)
        
        # Filtro de tamanho para evitar superpixels minúsculos (invisíveis) ou gigantescos (metade da imagem)
        if area < 100 or area > 2000:
            continue
        
        if sat_data.shape[0] >= 3:
            r = sat_data[0][clean_mask].astype(np.float32)
            g = sat_data[1][clean_mask].astype(np.float32)
            media_ver = np.mean(g)
            media_verm = np.mean(r)
            is_floresta = media_ver > media_verm
        else:
            is_floresta = True

        classe = "Floresta" if is_floresta else "Nao_Floresta"
        std_val = np.std(r) if sat_data.shape[0] >= 3 else 10
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

    print(f"Candidatos Floresta: {len(df_floresta)} | Candidatos Não-Floresta: {len(df_nao_floresta)}")

    # ====================================================================
    # SELEÇÃO EXATA DE 100 IMAGENS (50 Floresta / 50 Não-Floresta)
    # ====================================================================
    # Floresta: 25 Perfeitos e 25 Imperfeitos
    perf_f = df_floresta.nlargest(25, 'Taxa_HoR').copy()
    perf_f['Tipo_Selecao'] = 'Perfeito (100%)'
    
    imp_f = df_floresta.iloc[(df_floresta['Taxa_HoR'] - 70.0).abs().argsort()].head(25).copy()
    imp_f['Tipo_Selecao'] = 'Imperfeito (~70%)'

    # Não-Floresta: 25 Perfeitos e 25 Imperfeitos
    perf_nf = df_nao_floresta.nlargest(25, 'Taxa_HoR').copy()
    perf_nf['Tipo_Selecao'] = 'Perfeito (100%)'
    
    imp_nf = df_nao_floresta.iloc[(df_nao_floresta['Taxa_HoR'] - 70.0).abs().argsort()].head(25).copy()
    imp_nf['Tipo_Selecao'] = 'Imperfeito (~70%)'

    df_campanha = pd.concat([perf_f, imp_f, perf_nf, imp_nf])
    
    print(f"Seleção concluída. Total de imagens a gerar: {len(df_campanha)}/100")

    h_img, w_img = sat_data.shape[1], sat_data.shape[2]
    rgb_normalized = np.zeros((3, h_img, w_img), dtype=np.uint8)
    
    # Contraste aprimorado
    for b_idx in range(min(3, sat_data.shape[0])):
        b_data = sat_data[b_idx].astype(np.float32)
        p2, p98 = np.percentile(b_data[b_data > 0], (2, 98)) if np.any(b_data > 0) else (0, 1)
        if p98 > p2:
            norm = np.clip((b_data - p2) / (p98 - p2), 0, 1) * 255.0
        else:
            norm = np.clip(b_data, 0, 255)
        rgb_normalized[b_idx] = norm.astype(np.uint8)

    contador = 1
    for idx, row in df_campanha.iterrows():
        sp_id = int(row['ID_Segmento'])
        classe = row['Classe_Majoritaria']
        tipo = row['Tipo_Selecao']
        hor = row['Taxa_HoR']

        mask_sp = (labels == sp_id)
        
        # Isolar Absoluto: Garante apenas 1 polígono desenhado por imagem
        labeled_mask = label(mask_sp)
        props = regionprops(labeled_mask)
        largest_comp = max(props, key=lambda r: r.area)
        clean_mask = (labeled_mask == largest_comp.label)
        clean_mask = binary_fill_holes(clean_mask)
        
        y_indices, x_indices = np.where(clean_mask)
        
        # Centralização Geométrica (Bounding Box Real do Objeto)
        cy = (y_indices.max() + y_indices.min()) // 2
        cx = (x_indices.max() + x_indices.min()) // 2
        h_obj = y_indices.max() - y_indices.min()
        w_obj = x_indices.max() - x_indices.min()

        # ====================================================================
        # ZOOM DINÂMICO
        # Define uma margem de exatamente 30 pixels ao redor do superpixel. 
        # Isso dá um super zoom nos pequenos e enquadra os grandes perfeitamente.
        # ====================================================================
        padding = 30
        half_size = max(40, max(h_obj, w_obj) // 2 + padding)
        
        ymin, ymax = max(0, cy - half_size), min(h_img, cy + half_size)
        xmin, xmax = max(0, cx - half_size), min(w_img, cx + half_size)

        patch = rgb_normalized[:, ymin:ymax, xmin:xmax]
        patch_rgb = np.moveaxis(patch, 0, -1)
        patch_labels = clean_mask[ymin:ymax, xmin:xmax]

        # Desenho do contorno limpo (1 único polígono por imagem)
        borders = find_boundaries(patch_labels, mode='inner')
        borders_dilated = binary_dilation(borders, iterations=1)
        patch_rgb[borders_dilated] = [255, 255, 0]

        nome_arquivo = f"target_{contador:03d}_{classe}_{tipo.split()[0]}_HoR_{hor:.1f}_ID_{sp_id}.png"
        caminho_png = os.path.join(campaign_dir, nome_arquivo)

        # Renderização com Nitidez Extrema (interpolation='nearest')
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        ax.imshow(patch_rgb, interpolation='nearest')
        ax.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(caminho_png, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.close()
        contador += 1

    print(f"\n[SUCESSO] {contador-1} imagens salvas com sucesso em:\n-> {campaign_dir}")

if __name__ == "__main__":
    main()