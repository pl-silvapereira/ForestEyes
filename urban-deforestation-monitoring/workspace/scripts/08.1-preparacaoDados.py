import sys
import os
import gc
import duckdb

# Correção de compatibilidade entre DuckDB e geobr no Python 3.13
if not hasattr(duckdb.DuckDBPyRelation, "fetchone"):
    duckdb.DuckDBPyRelation.fetchone = lambda self: self.fetchall()[0]

import numpy as np
import rasterio
from PIL import Image
import cv2
from dotenv import load_dotenv

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 08.1-preparacaoDados.py <code_muni> <ano_inicio> <ano_fim>")
        sys.exit(1)

    code_muni = str(sys.argv[1])
    ano_inicio = str(sys.argv[2])
    ano_fim = str(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    # 1. Criação da estrutura de pastas em data/processing
    dir_processing = os.path.join(project_root, "data", "processing")
    dir_proc_rgb = os.path.join(dir_processing, "rgb")
    dir_proc_ndvi = os.path.join(dir_processing, "ndvi")
    dir_proc_masks_png = os.path.join(dir_processing, "masks", "png_truth_masks")
    dir_proc_masks_npy = os.path.join(dir_processing, "masks", "npy_truth_masks")

    for folder in [dir_proc_rgb, dir_proc_ndvi, dir_proc_masks_png, dir_proc_masks_npy]:
        os.makedirs(folder, exist_ok=True)

    # Diretórios de entrada das etapas anteriores
    dir_pan = os.path.join(project_root, "data", "output", "pansharpening", ano_fim)
    dir_gt = os.path.join(project_root, "data", "output", "ground_truth", ano_fim)

    path_cbers_2m = os.path.join(dir_pan, f"{code_muni}_{ano_fim}_CBERS_TRUE_COLOR_2M.tif")
    path_gt_mb = os.path.join(dir_gt, f"{code_muni}_GT_MapBiomas_{ano_fim}.tif")
    path_gt_u = os.path.join(dir_gt, f"{code_muni}_GT_U_{ano_fim}.tif")
    path_gt_m = os.path.join(dir_gt, f"{code_muni}_GT_M_{ano_fim}.tif")

    if not os.path.exists(path_cbers_2m):
        print(f"[ERRO CRÍTICO] Imagem CBERS-4A 2m não encontrada: {path_cbers_2m}")
        sys.exit(1)

    print("=" * 80)
    print(f"🚀 SCRIPT 08.1 - PREPARAÇÃO DE COMPOSIÇÕES, NDVI E MÁSCARAS NPY/PNG ({ano_fim})")
    print("=" * 80)

    # 2. Leitura da imagem CBERS-4A 2m (B1=Red, B2=Green, B3=Blue, B4=NIR)
    print("1. Carregando bandas espectrais do CBERS-4A WPM (2m)...")
    with rasterio.open(path_cbers_2m) as src:
        red = src.read(1).astype(np.float32)
        green = src.read(2).astype(np.float32)
        blue = src.read(3).astype(np.float32)
        nir = src.read(4).astype(np.float32) if src.count >= 4 else None
        
        height, width = src.height, src.width

    # 3. Geração da Composição RGB (Cor Natural em 8 bits)
    print("2. Gerando composição RGB de alta resolução...")
    def normalizar_8bit(banda):
        p2, p98 = np.percentile(banda[banda > 0], (2, 98)) if np.any(banda > 0) else (0, 1)
        if p98 == p2: p98 = p2 + 1
        return np.clip((banda - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)

    r_norm = normalizar_8bit(red)
    g_norm = normalizar_8bit(green)
    b_norm = normalizar_8bit(blue)

    rgb_composite = np.dstack((r_norm, g_norm, b_norm))
    output_rgb_path = os.path.join(dir_proc_rgb, f"{code_muni}_rgb_{ano_fim}.png")
    
    img_rgb = Image.fromarray(rgb_composite)
    img_rgb.save(output_rgb_path)
    print(f"   -> Composição RGB salva em: {output_rgb_path}")

    # 4. Geração do NDVI (Normalized Difference Vegetation Index)
    if nir is not None:
        print("3. Calculando NDVI e normalizando para 8 bits (0-255)...")
        denom = nir + red
        denom[denom == 0] = 1e-6
        ndvi = (nir - red) / denom
        
        # Mapeamento do intervalo [-1.0, 1.0] para [0, 255]
        ndvi_scaled = np.interp(ndvi, (-1.0, 1.0), (0, 255)).astype(np.uint8)
        
        output_ndvi_path = os.path.join(dir_proc_ndvi, f"{code_muni}_ndvi_{ano_fim}.png")
        img_ndvi = Image.fromarray(ndvi_scaled)
        img_ndvi.save(output_ndvi_path)
        print(f"   -> NDVI salvo em: {output_ndvi_path}")

    # 5. Exportação das Máscaras de Verdade Terrestre (PNG e NPY)
    print("4. Convertendo conjuntos verdade (GT-MapBiomas, GT-U, GT-M) para .png e .npy...")
    gt_dict = {
        'GT_MapBiomas': path_gt_mb,
        'GT_U': path_gt_u,
        'GT_M': path_gt_m
    }

    for name_gt, path_gt in gt_dict.items():
        if os.path.exists(path_gt):
            with rasterio.open(path_gt) as src_gt:
                mask_data = src_gt.read(1).astype(np.uint8)
                
                # Salva em .npy
                out_npy = os.path.join(dir_proc_masks_npy, f"{code_muni}_{name_gt}_{ano_fim}.npy")
                np.save(out_npy, mask_data)
                
                # Salva em .png
                out_png = os.path.join(dir_proc_masks_png, f"{code_muni}_{name_gt}_{ano_fim}.png")
                cv2.imwrite(out_png, mask_data)
                
                print(f"   -> {name_gt}: Salvo NPY ({out_npy}) e PNG ({out_png})")

    print("\n✅ SUCESSO! Script 08.1 concluído. Dados preparados em data/processing/")

if __name__ == "__main__":
    main()