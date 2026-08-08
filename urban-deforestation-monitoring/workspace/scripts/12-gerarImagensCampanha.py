import os
import sys
import numpy as np
import pandas as pd
import rasterio
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation, find_objects, zoom
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
    
    # 🎯 Ajustado para apontar para a imagem correta do Pan-sharpening / Recorte geopolítico
    path_sat = os.path.join(
        project_root, "data", "output", "pansharpening", "geopolitic-RGBN", 
        ano_fim, f"{code_muni}_{ano_fim}_CBERS_TRUE_COLOR_CLIPPED.tif"
    )
    
    path_csv = os.path.join(segmentation_dir, f"{code_muni}_tabela_segmentos_{ano_inicio}_vs_{ano_fim}.csv")

    # Diagnóstico rápido de caminhos para validação imediata
    print("Verificando caminhos dos arquivos:")
    print(f" - Labels: {path_labels} (Existe? {os.path.exists(path_labels)})")
    print(f" - Satélite: {path_sat} (Existe? {os.path.exists(path_sat)})")
    print(f" - CSV: {path_csv} (Existe? {os.path.exists(path_csv)})")

    if not os.path.exists(path_labels) or not os.path.exists(path_sat) or not os.path.exists(path_csv):
        print("\n❌ Erro: Um ou mais arquivos de entrada acima não foram encontrados.")
        sys.exit(1)

    print("\n📂 Carregando dados e metadados...")
    df = pd.read_csv(path_csv)

    # Filtrar exatamente 50 de Floresta e 50 de Não_Floresta de forma balanceada
    if 'classe' in df.columns and 'tipo' in df.columns:
        df_floresta = df[df['classe'].str.contains('Floresta', case=False, na=False)].head(50)
        df_nao_floresta = df[~df['classe'].str.contains('Floresta', case=False, na=False)].head(50)
    else:
        df_floresta = df.iloc[:50]
        df_nao_floresta = df.iloc[50:100]

    df_amostras = pd.concat([df_floresta, df_nao_floresta]).reset_index(drop=True)
    print(f"🎯 Total de amostras selecionadas: {len(df_amostras)} (Alvo: 50 Floresta, 50 Não-Floresta)")

    with rasterio.open(path_labels) as src_lab:
        labels_arr = src_lab.read(1)

    slices = find_objects(labels_arr)

    with rasterio.open(path_sat) as src_sat:
        processed_ids = set()
        contador = 0

        for idx, row in df_amostras.iterrows():
            superpixel_id = int(row['id']) if 'id' in row else int(row.get('label_id', idx + 1))

            # Remoção de duplicidade: garante que o mesmo superpixel não seja exportado duas vezes
            if superpixel_id in processed_ids:
                continue
            processed_ids.add(superpixel_id)

            if superpixel_id <= 0 or superpixel_id > len(slices) or slices[superpixel_id - 1] is None:
                continue

            sl = slices[superpixel_id - 1]
            ymin, ymax = sl[0].start, sl[0].stop
            xmin, xmax = sl[1].start, sl[1].stop

            pad = 20
            h_img, w_img = labels_arr.shape
            ymin_p = max(0, ymin - pad)
            ymax_p = min(h_img, ymax + pad)
            xmin_p = max(0, xmin - pad)
            xmax_p = min(w_img, xmax + pad)

            window = rasterio.windows.Window(xmin_p, ymin_p, xmax_p - xmin_p, ymax_p - ymin_p)
            patch_sat_raw = src_sat.read(window=window)  # [Bands, H, W]
            
            if patch_sat_raw.shape[0] >= 3:
                patch_rgb = np.stack([patch_sat_raw[2], patch_sat_raw[1], patch_sat_raw[0]], axis=-1)
            else:
                patch_rgb = np.stack([patch_sat_raw[0], patch_sat_raw[0], patch_sat_raw[0]], axis=-1)

            patch_rgb = np.nan_to_num(patch_rgb).astype(np.float32)
            
            p_min, p_max = np.percentile(patch_rgb, (2, 98))
            if p_max > p_min:
                patch_rgb = np.clip((patch_rgb - p_min) / (p_max - p_min) * 255, 0, 255)
            patch_rgb = patch_rgb.astype(np.uint8)

            patch_labels_sub = labels_arr[ymin_p:ymax_p, xmin_p:xmax_p]
            mask_sp = (patch_labels_sub == superpixel_id)

            if not np.any(mask_sp):
                continue

            # Contorno amarelo de alta visibilidade
            borders = find_boundaries(mask_sp, mode='inner')
            borders_dilated = binary_dilation(borders, iterations=1)
            patch_rgb[borders_dilated] = [255, 255, 0]

            # Redimensionamento inteligente para exatos 1024x1024 pixels mantendo a nitidez
            h_orig, w_orig = patch_rgb.shape[:2]
            target_size = 1024
            zoom_y = target_size / h_orig
            zoom_x = target_size / w_orig

            patch_rgb_zoomed = np.zeros((target_size, target_size, 3), dtype=patch_rgb.dtype)
            for c in range(3):
                patch_rgb_zoomed[..., c] = zoom(patch_rgb[..., c], (zoom_y, zoom_x), order=0)

            classe_str = str(row.get('classe', 'Desconhecido'))
            tipo_str = str(row.get('tipo', 'Padrao'))
            hor_val = float(row.get('hor', 0.0))

            nome_arquivo = f"target_{contador+1:03d}_{classe_str}_{tipo_str}_HoR_{hor_val:.1f}_ID_{superpixel_id}.png"
            caminho_saida = os.path.join(campaign_dir, nome_arquivo)

            import imageio
            imageio.imwrite(caminho_saida, patch_rgb_zoomed)
            
            contador += 1
            print(f"✅ [{contador}/100] Salvo: {nome_arquivo} (1024x1024 px)")

            if contador >= 100:
                break

    print(f"\n🎉 Processo concluído! 100 imagens únicas geradas em: {campaign_dir}")

if __name__ == "__main__":
    main()