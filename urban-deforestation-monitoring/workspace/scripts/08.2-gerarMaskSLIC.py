import sys
import os
import gc
import warnings
import duckdb

# Correção de compatibilidade entre DuckDB e geobr no Python 3.13
if not hasattr(duckdb.DuckDBPyRelation, "fetchone"):
    duckdb.DuckDBPyRelation.fetchone = lambda self: self.fetchall()[0]

import numpy as np
import cv2
from skimage.segmentation import slic
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message=".*One of the clusters is empty.*")
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 08.2-gerarMaskSLIC.py <code_muni> <ano_inicio> <ano_fim>")
        sys.exit(1)

    code_muni = str(sys.argv[1])
    ano_inicio = str(sys.argv[2])
    ano_fim = str(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    # 1. Estrutura de diretórios em data/processing
    dir_processing = os.path.join(project_root, "data", "processing")
    dir_proc_rgb = os.path.join(dir_processing, "rgb")
    dir_proc_masks_npy = os.path.join(dir_processing, "masks", "npy_truth_masks")
    dir_proc_slics = os.path.join(dir_processing, "slics")
    
    os.makedirs(dir_proc_slics, exist_ok=True)

    path_rgb = os.path.join(dir_proc_rgb, f"{code_muni}_rgb_{ano_fim}.png")
    path_mask_gt = os.path.join(dir_proc_masks_npy, f"{code_muni}_GT_U_{ano_fim}.npy")
    
    # Fallback para GT_MapBiomas caso GT_U ainda não esteja presente
    if not os.path.exists(path_mask_gt):
        path_mask_gt = os.path.join(dir_proc_masks_npy, f"{code_muni}_GT_MapBiomas_{ano_fim}.npy")

    if not os.path.exists(path_rgb):
        print(f"[ERRO CRÍTICO] Imagem RGB não encontrada: {path_rgb}")
        sys.exit(1)
    if not os.path.exists(path_mask_gt):
        print(f"[ERRO CRÍTICO] Máscara NPY não encontrada: {path_mask_gt}")
        sys.exit(1)

    print("=" * 80, flush=True)
    print(f"🚀 SCRIPT 08.2 - GERAÇÃO E ARMAZENAMENTO DE SUPERPIXELS MASKSLIC (.NPY) ({ano_fim})", flush=True)
    print("=" * 80, flush=True)

    # 2. Carregando a imagem RGB e a Máscara Verdade (.NPY)
    print("1. Carregando imagem RGB e máscara de referência...", flush=True)
    bgr_img = cv2.imread(path_rgb)
    if bgr_img is None:
        print(f"[ERRO CRÍTICO] Falha ao ler a imagem RGB: {path_rgb}")
        sys.exit(1)

    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    truth_mask = np.load(path_mask_gt)

    height, width, _ = rgb_img.shape
    print(f"   -> Dimensões da imagem: {width}x{height} pixels", flush=True)

    # 3. Definição do domínio ativo para MaskSLIC
    # Considera ativas as áreas de Floresta (1), Não-Floresta (2) e Indefinido (3)
    binary_mask = (truth_mask > 0).astype(np.uint8)

    # Parâmetros MaskSLIC
    qps = 250.0  # Densidade de pixels por superpixel alvo
    max_iter = 10
    tile_size = 1024

    total_pixels_ativos = np.sum(binary_mask)
    n_segments_estimado = max(int(total_pixels_ativos / qps), 10)
    avg_segment_size = total_pixels_ativos / n_segments_estimado if n_segments_estimado > 0 else qps

    print(f"2. Parâmetros da Segmentação MaskSLIC:")
    print(f"   -> Pixels ativos na máscara: {total_pixels_ativos:,}")
    print(f"   -> Estimativa total de superpixels: ~{n_segments_estimado:,}")
    print(f"   -> Tamanho médio estimado do segmento: {avg_segment_size:.2f} px", flush=True)

    # 4. Execução do MaskSLIC por blocos (Tiles 1024x1024) para evitar sobrecarga de memória
    print("3. Executando MaskSLIC em blocos (Tiles 1024x1024)...", flush=True)
    slic_map = np.zeros((height, width), dtype=np.int32)
    
    total_blocos = int(np.ceil(height / tile_size)) * int(np.ceil(width / tile_size))
    bloco_atual = 0
    current_label_offset = 0

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            bloco_atual += 1
            h_win = min(tile_size, height - y)
            w_win = min(tile_size, width - x)

            rgb_tile = rgb_img[y:y+h_win, x:x+w_win]
            mask_tile = binary_mask[y:y+h_win, x:x+w_win]
            pixels_ativos_tile = np.sum(mask_tile)

            if pixels_ativos_tile > 0:
                n_seg_tile = max(int(pixels_ativos_tile / qps), 1)
                
                # Executa SLIC com espaço de cor CIELAB habilitado
                slic_tile = slic(
                    rgb_tile,
                    n_segments=n_seg_tile,
                    max_num_iter=max_iter,
                    compactness=10,
                    mask=(mask_tile == 1),
                    start_label=1,
                    convert2lab=True
                )

                # Desloca os rótulos para garantir IDs únicos globais na matriz final
                seg_mask = slic_tile > 0
                if np.any(seg_mask):
                    slic_tile_offset = np.zeros_like(slic_tile, dtype=np.int32)
                    slic_tile_offset[seg_mask] = slic_tile[seg_mask] + current_label_offset
                    slic_map[y:y+h_win, x:x+w_win] = slic_tile_offset
                    current_label_offset += int(slic_tile.max())

            if bloco_atual % 20 == 0 or bloco_atual == total_blocos:
                print(f"  ⏩ Processando blocos MaskSLIC: [{bloco_atual}/{total_blocos}]", flush=True)

    # 5. Salvando a matriz de superpixels (.NPY)
    print("4. Salvando matriz de rótulos dos superpixels...", flush=True)
    output_npy_mask_slic = os.path.join(dir_proc_slics, f"mask_slic_{code_muni}_{ano_fim}.npy")
    np.save(output_npy_mask_slic, slic_map)

    num_superpixels_gerados = len(np.unique(slic_map[slic_map > 0]))
    print(f"   -> Total de superpixels únicos gerados: {num_superpixels_gerados:,}")
    print(f"   -> Matriz NPY salva com sucesso em: {output_npy_mask_slic}", flush=True)

    del rgb_img, truth_mask, binary_mask, slic_map
    gc.collect()

    print("\n✅ SUCESSO! Script 08.2 concluído. Próximo passo: 08.3.")

if __name__ == "__main__":
    main()