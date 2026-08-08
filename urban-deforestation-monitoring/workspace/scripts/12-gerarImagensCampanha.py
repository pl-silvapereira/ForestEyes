import os
import sys
import numpy as np
import pandas as pd
import rasterio
import imageio
from skimage.segmentation import find_boundaries
from skimage.measure import label, regionprops
from scipy.ndimage import binary_dilation, binary_closing, find_objects, zoom
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
    path_sat = os.path.join(project_root, "data", "output", "pansharpening", "geopolitic-RGBN", ano_fim, f"{code_muni}_{ano_fim}_CBERS_TRUE_COLOR_CLIPPED.tif")
    path_csv = os.path.join(segmentation_dir, f"{code_muni}_tabela_segmentos_{ano_inicio}_vs_{ano_fim}.csv")

    if not os.path.exists(path_labels) or not os.path.exists(path_sat) or not os.path.exists(path_csv):
        print("\n❌ Erro: Arquivos de entrada não encontrados. Verifique os caminhos.")
        sys.exit(1)

    print("\n📂 Carregando dados e metadados...")
    df = pd.read_csv(path_csv)

    # Separação das pools de dados (sem limitar a 50 no início, pois algumas serão descartadas)
    is_floresta = df['classe'].str.contains('Floresta', case=False, na=False)
    df_floresta = df[is_floresta].sample(frac=1, random_state=42).reset_index(drop=True)
    df_nao_floresta = df[~is_floresta].sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"📊 Total disponível: {len(df_floresta)} Floresta | {len(df_nao_floresta)} Não-Floresta")

    salvos_floresta = 0
    salvos_nao_floresta = 0
    alvo_por_classe = 50

    with rasterio.open(path_labels) as src_lab:
        labels_arr = src_lab.read(1)

    slices = find_objects(labels_arr)
    processed_ids = set()

    with rasterio.open(path_sat) as src_sat:
        
        # Função interna para processar e tentar salvar um patch
        def tentar_gerar_patch(row, categoria_atual):
            superpixel_id = int(row['id']) if 'id' in row else int(row.get('label_id', 0))

            if superpixel_id in processed_ids or superpixel_id <= 0 or superpixel_id > len(slices):
                return False
            
            sl = slices[superpixel_id - 1]
            if sl is None:
                return False

            ymin, ymax = sl[0].start, sl[0].stop
            xmin, xmax = sl[1].start, sl[1].stop

            pad = 20
            h_img, w_img = labels_arr.shape
            ymin_p, ymax_p = max(0, ymin - pad), min(h_img, ymax + pad)
            xmin_p, xmax_p = max(0, xmin - pad), min(w_img, xmax + pad)

            window = rasterio.windows.Window(xmin_p, ymin_p, xmax_p - xmin_p, ymax_p - ymin_p)
            patch_sat_raw = src_sat.read(window=window)
            
            if patch_sat_raw.shape[0] >= 3:
                patch_rgb = np.stack([patch_sat_raw[2], patch_sat_raw[1], patch_sat_raw[0]], axis=-1)
            else:
                patch_rgb = np.stack([patch_sat_raw[0], patch_sat_raw[0], patch_sat_raw[0]], axis=-1)

            patch_rgb = np.nan_to_num(patch_rgb).astype(np.float32)
            
            p_min, p_max = np.percentile(patch_rgb, (2, 98))
            if p_max > p_min:
                patch_rgb = np.clip((patch_rgb - p_min) / (p_max - p_min) * 255, 0, 255)
            patch_rgb = patch_rgb.astype(np.uint8)

            # ☁️ 1. FILTRO ANTI-NUVEM:
            # Se mais de 15% da imagem for extremamente clara (RGB > 220), ignorar a amostra
            cloud_mask = (patch_rgb[..., 0] > 220) & (patch_rgb[..., 1] > 220) & (patch_rgb[..., 2] > 220)
            if (np.sum(cloud_mask) / cloud_mask.size) > 0.15:
                return False # Pula esta amostra por excesso de nuvem/brilho branco

            patch_labels_sub = labels_arr[ymin_p:ymax_p, xmin_p:xmax_p]
            mask_sp = (patch_labels_sub == superpixel_id)

            if not np.any(mask_sp):
                return False

            # 🔗 2. UNIFICAÇÃO E SEGMENTO ÚNICO
            # Fecha buracos e une fragmentos próximos (distância de até 7 pixels)
            mask_closed = binary_closing(mask_sp, structure=np.ones((7,7)))
            
            # Isola rigorosamente o maior objeto para garantir que haverá apenas 1 contorno
            labeled_mask, num_features = label(mask_closed)
            if num_features > 1:
                regions = regionprops(labeled_mask)
                largest_region = max(regions, key=lambda r: r.area)
                mask_final = (labeled_mask == largest_region.label)
            else:
                mask_final = mask_closed

            # 📏 3. REDIMENSIONAMENTO ANTES DO CONTORNO (Para linha mais fina)
            h_orig, w_orig = patch_rgb.shape[:2]
            target_size = 1024
            zoom_y, zoom_x = target_size / h_orig, target_size / w_orig

            patch_rgb_zoomed = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            for c in range(3):
                patch_rgb_zoomed[..., c] = zoom(patch_rgb[..., c], (zoom_y, zoom_x), order=0)
            
            # Amplia a máscara binária final
            mask_zoomed = zoom(mask_final, (zoom_y, zoom_x), order=0)

            # 🖌️ 4. DESENHO DA LINHA FINA
            # Encontra a borda na imagem já em alta resolução e engrossa levemente (2 pixels)
            borders = find_boundaries(mask_zoomed, mode='inner')
            borders_dilated = binary_dilation(borders, iterations=2)
            patch_rgb_zoomed[borders_dilated] = [255, 255, 0]

            # Salvar o arquivo
            classe_str = str(row.get('classe', 'Desconhecido')).replace(" ", "_")
            tipo_str = str(row.get('tipo', 'Padrao')).split()[0]
            hor_val = float(row.get('hor', 0.0))
            
            # Organização dos nomes
            status_perfeicao = "Perfeito" if hor_val >= 95.0 else "Imperfeito"
            nome_arquivo = f"target_F{salvos_floresta+1:03d}_{classe_str}_{status_perfeicao}_HoR_{hor_val:.1f}_ID_{superpixel_id}.png"
            if categoria_atual == "Nao_Floresta":
                nome_arquivo = f"target_NF{salvos_nao_floresta+1:03d}_{classe_str}_{status_perfeicao}_HoR_{hor_val:.1f}_ID_{superpixel_id}.png"

            caminho_saida = os.path.join(campaign_dir, nome_arquivo)
            imageio.imwrite(caminho_saida, patch_rgb_zoomed)
            
            processed_ids.add(superpixel_id)
            return True

        # Processar as amostras de Floresta
        print("\n🌲 Iniciando geração: FLORESTA")
        for idx, row in df_floresta.iterrows():
            if salvos_floresta >= alvo_por_classe: break
            if tentar_gerar_patch(row, "Floresta"):
                salvos_floresta += 1
                sys.stdout.write(f"\r✅ Progresso Floresta: [{salvos_floresta}/{alvo_por_classe}]")
                sys.stdout.flush()

        # Processar as amostras de Não-Floresta
        print("\n\n🏙️ Iniciando geração: NÃO-FLORESTA")
        for idx, row in df_nao_floresta.iterrows():
            if salvos_nao_floresta >= alvo_por_classe: break
            if tentar_gerar_patch(row, "Nao_Floresta"):
                salvos_nao_floresta += 1
                sys.stdout.write(f"\r✅ Progresso Não-Floresta: [{salvos_nao_floresta}/{alvo_por_classe}]")
                sys.stdout.flush()

    print(f"\n\n🎉 Sucesso absoluto! {salvos_floresta + salvos_nao_floresta} imagens únicas e limpas geradas em: {campaign_dir}")

if __name__ == "__main__":
    main()