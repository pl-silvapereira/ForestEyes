import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from skimage.segmentation import slic, find_boundaries
from scipy.ndimage import label, find_objects
from dotenv import load_dotenv
import gc
import time

def executar_mapa_tematico_ultrarrapido():
    load_dotenv()
    ROOT = os.getenv('PROJECT_ROOT')
    if not ROOT:
        print("❌ Erro: Variável 'PROJECT_ROOT' não encontrada no .env.")
        return

    dir_mapbiomas = os.path.join(ROOT, 'data', 'MapBiomas')
    dir_output = os.path.join(ROOT, 'data', 'Output')

    imagem_sat_path = os.path.join(dir_output, "02_SJC_Recortado_MapBiomas.tif")
    saida_visual = os.path.join(dir_output, "15_SJC_Mapa_Verde_Vermelho_Segmentado.tif")

    if not os.path.exists(imagem_sat_path):
        print(f"❌ Erro: Arquivo {imagem_sat_path} não encontrado.")
        return

    busca = glob.glob(os.path.join(dir_mapbiomas, "*2021*coverage*10m*.tif"))
    if not busca:
        busca = glob.glob(os.path.join(dir_mapbiomas, "*coverage_10m*.tif"))
    mapbiomas_path = busca[0]

    # -------------------------------------------------------------
    # 1. PREPARAÇÃO DA BASE
    # -------------------------------------------------------------
    print("1/4 - Preparando metadados e alinhando classes...")
    with rasterio.open(imagem_sat_path) as sat_src:
        meta_sat = sat_src.meta.copy()
        height, width = sat_src.height, sat_src.width
        sat_transform, sat_crs = sat_src.transform, sat_src.crs

    with rasterio.open(mapbiomas_path) as mb_src:
        mb_aligned = np.zeros((height, width), dtype=np.uint8)
        reproject(
            source=rasterio.band(mb_src, 1),
            destination=mb_aligned,
            src_transform=mb_src.transform,
            src_crs=mb_src.crs,
            dst_transform=sat_transform,
            dst_crs=sat_crs,
            resampling=Resampling.nearest
        )

    # -------------------------------------------------------------
    # 2. DEFINIÇÃO DAS CORES (VERDE, VERMELHO, PRETO)
    # -------------------------------------------------------------
    print("2/4 - Criando o Fundo Sólido (Floresta/Não-Floresta/Máscara)...")
    ids_floresta = [1, 3, 4, 5, 6, 49]
    #ids_nao_floresta = [10, 11, 12, 32, 50, 13]
    ids_nao_floresta = [9, 10, 11, 12, 13, 29, 32, 50]

    mask_floresta = np.isin(mb_aligned, ids_floresta)
    mask_nao_floresta = np.isin(mb_aligned, ids_nao_floresta)
    mask_segmentacao = mask_floresta | mask_nao_floresta

    rgb_out = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_out[mask_floresta] = [0, 255, 0]      # Verde
    rgb_out[mask_nao_floresta] = [255, 0, 0]  # Vermelho
    
    del mb_aligned
    gc.collect()

    # -------------------------------------------------------------
    # 3. SEGMENTAÇÃO INTELIGENTE (BYPASS DE MICRO-ILHAS)
    # -------------------------------------------------------------
    print("3/4 - Mapeando as ilhas de vegetação...")
    
    ilhas, num_ilhas = label(mask_segmentacao)
    caixas = find_objects(ilhas)
    
    total_pixels_alvo = np.sum(mask_segmentacao)
    #ALVO_SUPERPIXELS = 15000
    ALVO_SUPERPIXELS = 1000
    
    # Descobre o tamanho médio esperado de 1 superpixel
    tamanho_medio_sp = max(100, int(total_pixels_alvo / ALVO_SUPERPIXELS))
    
    print(f"-> {num_ilhas} ilhas isoladas encontradas.")
    print(f"-> Tamanho médio de 1 superpixel: ~{tamanho_medio_sp} pixels.")
    print("-> Iniciando processamento...")

    ilhas_processadas = 0
    ilhas_ignoradas_ruido = 0
    ilhas_bypass_rapido = 0
    ilhas_slic_pesado = 0
    
    start_time = time.time()

    with rasterio.open(imagem_sat_path) as sat_src:
        for i, slc in enumerate(caixas):
            if slc is None:
                continue

            patch_mask = (ilhas[slc] == (i + 1))
            patch_area = np.sum(patch_mask)

            # 1. Filtro de Ruído: Se for quase invisível, nem contorna
            if patch_area < 50:
                ilhas_ignoradas_ruido += 1
                ilhas_processadas += 1
                continue

            # 2. BYPASS RÁPIDO: Se a ilha for menor que 1 superpixel, a ilha inteira é o superpixel!
            if patch_area <= tamanho_medio_sp:
                ilhas_bypass_rapido += 1
                # Simula um superpixel sem rodar o algoritmo ou ler a imagem CBERS
                seg_patch = np.zeros_like(patch_mask, dtype=np.int32)
                seg_patch[patch_mask] = 1
            
            # 3. SLIC PESADO: Rodar a IA apenas nas matas e florestas grandes
            else:
                ilhas_slic_pesado += 1
                window = Window.from_slices(slc[0], slc[1])
                patch_sat = sat_src.read((1,2,3), window=window)
                patch_sat = np.moveaxis(patch_sat, 0, -1).astype(np.float32)
                
                for b in range(patch_sat.shape[2]):
                    p99 = np.percentile(patch_sat[:,:,b], 99)
                    if p99 > 0:
                        patch_sat[:,:,b] = np.clip((patch_sat[:,:,b] / p99) * 255.0, 0, 255.0)
                patch_sat = patch_sat.astype(np.uint8)

                n_seg_patch = max(2, int(patch_area / tamanho_medio_sp))

                seg_patch = slic(
                    patch_sat, 
                    n_segments=n_seg_patch, 
                    compactness=10.0, 
                    mask=patch_mask, 
                    convert2lab=False, 
                    enforce_connectivity=False, 
                    max_num_iter=5,
                    start_label=1
                )

            # Desenha a borda da ilha processada (Amarelo)
            borders = find_boundaries(seg_patch, mode='inner', background=0)
            rgb_patch = rgb_out[slc]
            rgb_patch[borders] = [255, 255, 0]
            rgb_out[slc] = rgb_patch

            ilhas_processadas += 1
            if ilhas_processadas % 1000 == 0:
                print(f"   [{ilhas_processadas}/{num_ilhas}] Processadas... (Bypass: {ilhas_bypass_rapido} | SLIC: {ilhas_slic_pesado})")

    tempo_total = round(time.time() - start_time, 1)
    
    del ilhas, caixas
    gc.collect()

    print(f"\n✅ Passo 3 Concluído em {tempo_total} segundos!")
    print(f"Estatísticas: {ilhas_bypass_rapido} resolvidas instantaneamente, e apenas {ilhas_slic_pesado} precisaram do CBERS.")

    # -------------------------------------------------------------
    # 4. SALVAR RESULTADO
    # -------------------------------------------------------------
    print("4/4 - Salvando a imagem final...")
    meta_sat.update({
        "dtype": rasterio.uint8, 
        "count": 3, 
        "nodata": None, 
        "photometric": "RGB"
    })
    
    with rasterio.open(saida_visual, "w", **meta_sat) as dst_vis:
        for b in range(3):
            dst_vis.write(rgb_out[:, :, b], b+1)

    print(f"🎉 Processo perfeito concluído! Arquivo salvo em:\n{saida_visual}")

if __name__ == "__main__":
    executar_mapa_tematico_ultrarrapido()