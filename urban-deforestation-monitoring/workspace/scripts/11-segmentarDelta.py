import os
import sys
import rasterio
from rasterio.windows import Window
from skimage.segmentation import slic, find_boundaries
import numpy as np
from dotenv import load_dotenv
import gc

def main():
    # Uso correto: python 11-segmentarDelta.py <code_muni> <ano_inicio> <ano_fim>
    # Exemplo: python 11-segmentarDelta.py 3549904 2023 2024
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 11-segmentarDelta.py <code_muni> <ano_inicio> <ano_fim>")
        print("Exemplo: python 11-segmentarDelta.py 3549904 2023 2024")
        sys.exit(1)

    code_muni = int(sys.argv[1])
    ano_inicio = str(sys.argv[2])
    ano_fim = str(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    # Nova subpasta solicitada: data/output/mask/segmentation/
    segmentation_dir = os.path.join(project_root, "data", "output", "mask", "segmentation")
    os.makedirs(segmentation_dir, exist_ok=True)

    # 1. Caminho da imagem de satélite original recortada (fundo completo clipped)
    path_sat_original = os.path.join(
        project_root, "data", "output", "pansharpening", "geopolitic-RGBN", 
        ano_fim, f"{code_muni}_{ano_fim}_CBERS_TRUE_COLOR_CLIPPED.tif"
    )

    # 2. Caminho da máscara do Script 10 (define onde estão os pontos de mudança)
    path_mask_black = os.path.join(
        project_root, "data", "output", "mask", "black", 
        f"{code_muni}_masked_black_transparent_{ano_inicio}_vs_{ano_fim}.tif"
    )

    if not os.path.exists(path_sat_original):
        print(f"[ERRO CRÍTICO] Imagem de satélite original não encontrada em:\n-> {path_sat_original}")
        sys.exit(1)

    if not os.path.exists(path_mask_black):
        print(f"[ERRO CRÍTICO] Máscara do Script 10 não encontrada em:\n-> {path_mask_black}")
        print("Execute o Script 10 primeiro.")
        sys.exit(1)

    print("=" * 115)
    print(f"🧩 GERANDO SEGMENTAÇÃO APENAS NOS TRECHOS DE MUDANÇA (ANO: {ano_fim})")
    print(f"📍 MUNICÍPIO: {code_muni} | PERÍODO: {ano_inicio} vs {ano_fim}")
    print("=" * 115)

    with rasterio.open(path_sat_original) as src_sat:
        meta_sat = src_sat.meta.copy()
        height = src_sat.height
        width = src_sat.width

    with rasterio.open(path_mask_black) as src_mask:
        mask_bands = src_mask.read()
        # O canal Alpha do Script 10: 0 nos pontos de mudança, 255 (ou opaco) no fundo
        # Logo, a máscara booleana de onde vamos segmentar é onde Alpha == 0 (ou onde RGB > 0)
        if mask_bands.shape[0] >= 4:
            alpha_channel = mask_bands[3]
            # Onde o canal alpha for 0, significa que há o ponto de mudança do Script 08/09
            mask_mudanca = (alpha_channel == 0)
        else:
            # Fallback caso tenha sido lido de outra forma: onde não for preto absoluto
            r_m, g_m, b_m = mask_bands[0], mask_bands[1], mask_bands[2]
            mask_mudanca = (r_m > 0) | (g_m > 0) | (b_m > 0)

    # Arquivos de saída na pasta segmentation
    saida_labels = os.path.join(segmentation_dir, f"{code_muni}_segmentation_labels_{ano_inicio}_vs_{ano_fim}.tif")
    saida_visual = os.path.join(segmentation_dir, f"{code_muni}_segmentation_visual_{ano_inicio}_vs_{ano_fim}.tif")

    TILE_SIZE = 1500
    global_id_offset = 0

    # Metadados para o raster de labels (matriz matemática de IDs dos superpixels)
    meta_labels = meta_sat.copy()
    meta_labels.update({"dtype": rasterio.int32, "count": 1, "nodata": 0})

    # Metadados para o raster visual (imagem colorida RGB do satélite inteiro + contornos amarelos nos trechos de mudança)
    meta_vis = meta_sat.copy()
    meta_vis.update({"dtype": rasterio.uint8, "count": 3, "nodata": 0, "photometric": "RGB"})

    print(f"Iniciando segmentação SLIC restrita às áreas de mudança (blocos de {TILE_SIZE}px)...")

    with rasterio.open(path_sat_original) as sat_src:
        with rasterio.open(saida_labels, "w", **meta_labels) as dst_labels:
            with rasterio.open(saida_visual, "w", **meta_vis) as dst_vis:
                
                n_rows = int(np.ceil(height / TILE_SIZE))
                n_cols = int(np.ceil(width / TILE_SIZE))
                total_blocos = n_rows * n_cols
                bloco_atual = 0

                for row in range(0, height, TILE_SIZE):
                    for col in range(0, width, TILE_SIZE):
                        bloco_atual += 1
                        
                        win_h = min(TILE_SIZE, height - row)
                        win_w = min(TILE_SIZE, width - col)
                        window = Window(col, row, win_w, win_h)
                        
                        # Extrai a máscara de mudança do tile atual
                        mask_tile = mask_mudanca[row:row+win_h, col:col+win_w]
                        
                        # Lê o bloco da imagem original de satélite (fundo clipped completo)
                        sat_data = sat_src.read(window=window)
                        sat_img_tile = np.moveaxis(sat_data, 0, -1).astype(np.float32)
                        
                        # Prepara a imagem RGB em 8-bits (0-255) para visualização
                        rgb_tile = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                        num_bandas = min(3, sat_img_tile.shape[2])
                        
                        for i in range(num_bandas):
                            band_max = np.percentile(sat_img_tile[:,:,i], 99)
                            if band_max > 0:
                                rgb_tile[:,:,i] = np.clip((sat_img_tile[:,:,i] / band_max) * 255.0, 0, 255).astype(np.uint8)
                        
                        # Se o tile não possuir nenhum ponto de mudança do Script 08, mantém o satélite limpo e pula o SLIC
                        if not np.any(mask_tile):
                            dst_labels.write(np.zeros((win_h, win_w), dtype=np.int32), 1, window=window)
                            for b in range(3):
                                dst_vis.write(rgb_tile[:, :, b], b+1, window=window)
                            continue
                        
                        print(f"[{bloco_atual:03d}/{total_blocos}] Segmentando trechos de mudança com SLIC...")
                        
                        # Proporção de segmentos para o bloco
                        n_seg_tile = max(30, int(1500 * ((win_h * win_w) / (TILE_SIZE * TILE_SIZE))))

                        # Executa o SLIC restrito estritamente à máscara de mudança (mask_tile)
                        segments = slic(
                            sat_img_tile, 
                            n_segments=n_seg_tile, 
                            compactness=10.0, 
                            mask=mask_tile, 
                            convert2lab=False, 
                            max_num_iter=5,
                            start_label=1
                        )
                        
                        segments[~mask_tile] = 0
                        
                        max_id_local = segments.max()
                        if max_id_local > 0:
                            segments[segments > 0] += global_id_offset
                            global_id_offset += max_id_local
                        
                        # Desenha contornos amarelos (R=255, G=255, B=0) apenas onde houve segmentação de mudança
                        borders = find_boundaries(segments, mode='outer')
                        rgb_tile[borders] = [255, 255, 0]
                        
                        # Grava os resultados no disco
                        dst_labels.write(segments.astype(np.int32), 1, window=window)
                        for b in range(3):
                            dst_vis.write(rgb_tile[:, :, b], b+1, window=window)

    print(f"\n[SUCESSO] Processo de segmentação finalizado!")
    print(f"📊 Raster de Labels (IDs dos Superpixels): {saida_labels}")
    print(f"🖼️ Raster Visual (Satélite Clipped + Contornos de Mudança): {saida_visual}")

if __name__ == "__main__":
    main()