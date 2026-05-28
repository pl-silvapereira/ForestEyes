import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from skimage.segmentation import slic, find_boundaries
from scipy.ndimage import label, find_objects
from dotenv import load_dotenv
import gc
import time

def gerar_metricas_segmentacao():
    load_dotenv()
    ROOT = os.getenv('PROJECT_ROOT')
    if not ROOT:
        print("❌ Erro: Variável 'PROJECT_ROOT' não encontrada no .env.")
        return

    dir_mapbiomas = os.path.join(ROOT, 'data', 'MapBiomas')
    dir_output = os.path.join(ROOT, 'data', 'Output')

    imagem_sat_path = os.path.join(dir_output, "02_SJC_Recortado_MapBiomas.tif")
    
    # ---- NOVAS SAÍDAS ----
    saida_visual = os.path.join(dir_output, "17_SJC_Mapa_Segmentado.tif")
    saida_npy = os.path.join(dir_output, "17_SJC_Matriz_Segmentacao.npy")
    saida_csv = os.path.join(dir_output, "17_SJC_Estatisticas_Superpixels.csv")

    busca = glob.glob(os.path.join(dir_mapbiomas, "*2021*coverage*10m*.tif"))
    if not busca:
        busca = glob.glob(os.path.join(dir_mapbiomas, "*coverage_10m*.tif"))
    mapbiomas_path = busca[0]

    # -------------------------------------------------------------
    # 1. ALINHAMENTO DO MAPBIOMAS
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
    # 2. DEFINIÇÃO DAS MÁSCARAS
    # -------------------------------------------------------------
    ids_floresta = [1, 3, 4, 5, 6, 49]
    ids_nao_floresta = [10, 11, 12, 32, 50, 13]
    
    mask_floresta = np.isin(mb_aligned, ids_floresta)
    mask_nao_floresta = np.isin(mb_aligned, ids_nao_floresta)
    mask_segmentacao = mask_floresta | mask_nao_floresta

    rgb_out = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_out[mask_floresta] = [0, 255, 0]      
    rgb_out[mask_nao_floresta] = [255, 0, 0]  
    
    del mb_aligned
    gc.collect()

    # -------------------------------------------------------------
    # 3. SEGMENTAÇÃO E EXTRAÇÃO DE MÉTRICAS (VETORIZADA)
    # -------------------------------------------------------------
    print("2/4 - Mapeando ilhas e extraindo métricas matemáticas...")
    
    ilhas, num_ilhas = label(mask_segmentacao)
    caixas = find_objects(ilhas)
    
    total_pixels_alvo = np.sum(mask_segmentacao)
    ALVO_SUPERPIXELS = 15000
    tamanho_medio_sp = max(100, int(total_pixels_alvo / ALVO_SUPERPIXELS))
    
    # Estruturas Globais
    global_segments = np.zeros((height, width), dtype=np.int32)
    global_id_offset = 0
    estatisticas_lista = []
    
    ilhas_processadas = 0
    start_time = time.time()

    with rasterio.open(imagem_sat_path) as sat_src:
        for i, slc in enumerate(caixas):
            if slc is None:
                continue

            patch_mask = (ilhas[slc] == (i + 1))
            patch_area = np.sum(patch_mask)

            if patch_area < 50:
                continue

            if patch_area <= tamanho_medio_sp:
                seg_patch = np.zeros_like(patch_mask, dtype=np.int32)
                seg_patch[patch_mask] = 1
            else:
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
                    patch_sat, n_segments=n_seg_patch, compactness=10.0, 
                    mask=patch_mask, convert2lab=False, enforce_connectivity=False, 
                    max_num_iter=5, start_label=1
                )

            # Injetar o ID Global e aplicar os contornos
            max_id_local = seg_patch.max()
            if max_id_local > 0:
                # Ajusta os IDs locais para não repetirem globalmente
                mask_validos = (seg_patch > 0)
                seg_patch[mask_validos] += global_id_offset
                
                # --- CÁLCULO VETORIZADO DE MÉTRICAS (RÁPIDO) ---
                patch_f = mask_floresta[slc]
                patch_nf = mask_nao_floresta[slc]
                
                # Achata as matrizes para contagem
                flat_seg = seg_patch[mask_validos]
                flat_f = patch_f[mask_validos]
                flat_nf = patch_nf[mask_validos]
                
                # Bincount conta tudo em milissegundos
                areas = np.bincount(flat_seg)
                counts_f = np.bincount(flat_seg, weights=flat_f)
                counts_nf = np.bincount(flat_seg, weights=flat_nf)
                
                ids_presentes = np.unique(flat_seg)
                
                for sp_id in ids_presentes:
                    area_sp = areas[sp_id]
                    qtd_flor = counts_f[sp_id]
                    qtd_nflor = counts_nf[sp_id]
                    
                    if qtd_flor >= qtd_nflor:
                        classe = "Floresta"
                        hor = (qtd_flor / area_sp) * 100
                    else:
                        classe = "Nao_Floresta"
                        hor = (qtd_nflor / area_sp) * 100
                        
                    estatisticas_lista.append([sp_id, area_sp, round(hor, 2), classe])
                
                global_id_offset += max_id_local
                
                # Salva na matriz global .npy
                global_patch = global_segments[slc]
                global_patch[mask_validos] = seg_patch[mask_validos]
                global_segments[slc] = global_patch

            borders = find_boundaries(seg_patch, mode='inner', background=0)
            rgb_patch = rgb_out[slc]
            rgb_patch[borders] = [255, 255, 0]
            rgb_out[slc] = rgb_patch

            ilhas_processadas += 1
            if ilhas_processadas % 1000 == 0:
                print(f"   [{ilhas_processadas}] ilhas processadas... Superpixels mapeados: {global_id_offset}")

    tempo_total = round(time.time() - start_time, 1)
    print(f"\n✅ Segmentação e Extração concluídas em {tempo_total} segundos!")

    # -------------------------------------------------------------
    # 4. SALVAMENTO DOS ARQUIVOS (NPY, CSV, TIF)
    # -------------------------------------------------------------
    print("3/4 - Salvando arquivos (.npy, .csv, .tif)...")
    
    # Salva Numpy Array
    np.save(saida_npy, global_segments)
    
    # Salva DataFrame e CSV
    df_stats = pd.DataFrame(estatisticas_lista, columns=['ID_Segmento', 'Quantidade_Pixels', 'Taxa_HoR', 'Classe_Majoritaria'])
    df_stats.to_csv(saida_csv, index=False, sep=';', encoding='utf-8')
    
    # Salva TIF
    meta_sat.update({"dtype": rasterio.uint8, "count": 3, "nodata": None, "photometric": "RGB"})
    with rasterio.open(saida_visual, "w", **meta_sat) as dst_vis:
        for b in range(3):
            dst_vis.write(rgb_out[:, :, b], b+1)

    # -------------------------------------------------------------
    # 5. CÁLCULO E EXIBIÇÃO DOS RELATÓRIOS ESTATÍSTICOS
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("4/4 - RELATÓRIO ESTATÍSTICO DE SUPERPIXELS")
    print("="*50)
    
    qtd_flor = (df_stats['Classe_Majoritaria'] == 'Floresta').sum()
    qtd_nflor = (df_stats['Classe_Majoritaria'] == 'Nao_Floresta').sum()
    
    print(f"🌲 Quantidade de Segmentos (Floresta): {qtd_flor}")
    print(f"🍂 Quantidade de Segmentos (Não Floresta): {qtd_nflor}")
    print("-" * 50)
    
    print("📏 QUANTIDADE DE PIXELS (Área):")
    print(f"   - Média:   {df_stats['Quantidade_Pixels'].mean():.2f}")
    print(f"   - Mediana: {df_stats['Quantidade_Pixels'].median():.2f}")
    print(f"   - Desvio Padrão: {df_stats['Quantidade_Pixels'].std():.2f}")
    print(f"   - Maior Segmento: {df_stats['Quantidade_Pixels'].max()} pixels")
    print(f"   - Menor Segmento: {df_stats['Quantidade_Pixels'].min()} pixels")
    print("-" * 50)
    
    print("🎯 TAXA DE HOMOGENEIDADE (HoR %):")
    print(f"   - Média:   {df_stats['Taxa_HoR'].mean():.2f}%")
    print(f"   - Mediana: {df_stats['Taxa_HoR'].median():.2f}%")
    print(f"   - Desvio Padrão: {df_stats['Taxa_HoR'].std():.2f}%")
    print("="*50)

if __name__ == "__main__":
    gerar_metricas_segmentacao()