import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from skimage.segmentation import slic, find_boundaries
from dotenv import load_dotenv
import gc
import time

def gerar_metricas_segmentacao_tiling():
    load_dotenv()
    ROOT = os.getenv('PROJECT_ROOT')
    if not ROOT:
        print("❌ Erro: Variável 'PROJECT_ROOT' não encontrada no .env.")
        return

    dir_mapbiomas = os.path.join(ROOT, 'data', 'MapBiomas')
    dir_output = os.path.join(ROOT, 'data', 'Output')

    imagem_sat_path = os.path.join(dir_output, "02_SJC_Recortado_MapBiomas.tif")
    
    # Arquivos que serão gerados
    saida_visual = os.path.join(dir_output, "17_SJC_Mapa_Segmentado.tif")
    saida_npy = os.path.join(dir_output, "17_SJC_Matriz_Segmentacao.npy")
    saida_csv = os.path.join(dir_output, "17_SJC_Estatisticas_Superpixels.csv")

    busca = glob.glob(os.path.join(dir_mapbiomas, "*2021*coverage*10m*.tif"))
    if not busca:
        busca = glob.glob(os.path.join(dir_mapbiomas, "*coverage_10m*.tif"))
    mapbiomas_path = busca[0]

    # -------------------------------------------------------------
    # 1. PREPARAÇÃO DOS METADADOS E MAPBIOMAS
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

    # Identificação das classes
    ids_floresta = [1, 3, 4, 5, 6, 49]
    ids_nao_floresta = [10, 11, 12, 32, 50, 13]
    
    mask_floresta = np.isin(mb_aligned, ids_floresta)
    mask_nao_floresta = np.isin(mb_aligned, ids_nao_floresta)
    
    area_total_mascara = np.sum(mask_floresta | mask_nao_floresta)
    ALVO_SUPERPIXELS = 15000
    
    del mb_aligned
    gc.collect()

    # -------------------------------------------------------------
    # 2. PROCESSAMENTO EM BLOCOS (À PROVA DE TRAVAMENTO)
    # -------------------------------------------------------------
    TILE_SIZE = 2000  # Processa quadrados de 2000x2000 pixels (Usa pouquíssima RAM)
    
    global_segments = np.zeros((height, width), dtype=np.int32)
    global_id_offset = 0
    estatisticas_lista = []

    meta_vis = meta_sat.copy()
    meta_vis.update({"dtype": rasterio.uint8, "count": 3, "nodata": None, "photometric": "RGB"})

    print("2/4 - Iniciando segmentação e extração de métricas por blocos...")
    
    n_rows = int(np.ceil(height / TILE_SIZE))
    n_cols = int(np.ceil(width / TILE_SIZE))
    total_blocos = n_rows * n_cols
    bloco_atual = 0
    start_time = time.time()

    with rasterio.open(imagem_sat_path) as sat_src:
        with rasterio.open(saida_visual, "w", **meta_vis) as dst_vis:
            
            for row in range(0, height, TILE_SIZE):
                for col in range(0, width, TILE_SIZE):
                    bloco_atual += 1
                    
                    win_h = min(TILE_SIZE, height - row)
                    win_w = min(TILE_SIZE, width - col)
                    window = Window(col, row, win_w, win_h)
                    
                    # Extrai as máscaras locais do bloco atual
                    tile_f = mask_floresta[row:row+win_h, col:col+win_w]
                    tile_nf = mask_nao_floresta[row:row+win_h, col:col+win_w]
                    tile_mask = tile_f | tile_nf
                    
                    # Prepara a imagem visual (Tudo começa Preto = Máscara)
                    rgb_tile = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                    rgb_tile[tile_f] = [0, 255, 0]      # Verde Sólido
                    rgb_tile[tile_nf] = [255, 0, 0]     # Vermelho Sólido
                    
                    area_tile = np.sum(tile_mask)

                    # Se existir algo para segmentar neste bloco
                    if area_tile > 0:
                        # Lê os pixels reais da imagem apenas do tamanho deste bloco (MUITO LEVE)
                        patch_sat = sat_src.read((1,2,3), window=window)
                        patch_sat = np.moveaxis(patch_sat, 0, -1).astype(np.float32)
                        
                        # Normaliza brilho
                        for b in range(patch_sat.shape[2]):
                            p99 = np.percentile(patch_sat[:,:,b], 99)
                            if p99 > 0:
                                patch_sat[:,:,b] = np.clip((patch_sat[:,:,b] / p99) * 255.0, 0, 255.0)
                        patch_sat = patch_sat.astype(np.uint8)

                        # Calcula quantidade justa de superpixels baseada no tamanho da área deste bloco
                        n_seg_patch = max(2, int(ALVO_SUPERPIXELS * (area_tile / area_total_mascara)))

                        seg_patch = slic(
                            patch_sat, n_segments=n_seg_patch, compactness=10.0, 
                            mask=tile_mask, convert2lab=False, enforce_connectivity=False, 
                            max_num_iter=5, start_label=1
                        )
                        seg_patch[~tile_mask] = 0
                        
                        # Ajusta os IDs para nunca repetirem em blocos diferentes
                        mask_validos = (seg_patch > 0)
                        if np.any(mask_validos):
                            seg_patch[mask_validos] += global_id_offset
                            
                            # CÁLCULO DE MÉTRICAS (VETORIZADO E INSTANTÂNEO)
                            seg_validos = seg_patch[mask_validos]
                            f_validos = tile_f[mask_validos]
                            nf_validos = tile_nf[mask_validos]
                            
                            areas = np.bincount(seg_validos)
                            counts_f = np.bincount(seg_validos, weights=f_validos)
                            counts_nf = np.bincount(seg_validos, weights=nf_validos)
                            
                            ids_presentes = np.unique(seg_validos)
                            for sp_id in ids_presentes:
                                a = areas[sp_id]
                                cf = counts_f[sp_id]
                                cnf = counts_nf[sp_id]
                                
                                classe = "Floresta" if cf >= cnf else "Nao_Floresta"
                                maioria = max(cf, cnf)
                                hor = (maioria / a) * 100
                                
                                estatisticas_lista.append([sp_id, a, round(hor, 2), classe])
                            
                            global_id_offset += seg_patch.max() - global_id_offset
                            
                            # Salva os IDs na matriz Global
                            global_segments[row:row+win_h, col:col+win_w] = seg_patch

                        # Desenha as bordas amarelas no bloco
                        borders = find_boundaries(seg_patch, mode='inner', background=0)
                        rgb_tile[borders] = [255, 255, 0]

                    # Escreve o bloco diretamente no arquivo .tif do HD
                    for b in range(3):
                        dst_vis.write(rgb_tile[:, :, b], b+1, window=window)
                        
                    print(f"   Bloco [{bloco_atual:03d}/{total_blocos}] concluído... (Superpixels mapeados: {global_id_offset})")

    # -------------------------------------------------------------
    # 3. SALVAR MATRIZ E CSV
    # -------------------------------------------------------------
    print("\n3/4 - Salvando matriz de dados e planilha CSV...")
    np.save(saida_npy, global_segments)
    
    df_stats = pd.DataFrame(estatisticas_lista, columns=['ID_Segmento', 'Quantidade_Pixels', 'Taxa_HoR', 'Classe_Majoritaria'])
    # Limpa possíveis IDs vazios gerados pela matemática
    df_stats = df_stats[df_stats['Quantidade_Pixels'] > 0]
    df_stats.to_csv(saida_csv, index=False, sep=';', encoding='utf-8')

    tempo_total = round(time.time() - start_time, 1)

    # -------------------------------------------------------------
    # 4. RELATÓRIO ESTATÍSTICO
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print(f"✅ SCRIPT FINALIZADO EM {tempo_total} SEGUNDOS!")
    print("="*50)
    print("RELATÓRIO ESTATÍSTICO DE SUPERPIXELS:")
    print("-" * 50)
    
    qtd_flor = (df_stats['Classe_Majoritaria'] == 'Floresta').sum()
    qtd_nflor = (df_stats['Classe_Majoritaria'] == 'Nao_Floresta').sum()
    
    print(f"🌲 Segmentos de Floresta: {qtd_flor}")
    print(f"🍂 Segmentos Não Floresta: {qtd_nflor}")
    print("-" * 50)
    print("📏 TAMANHO (PIXELS):")
    print(f"   Média:  {df_stats['Quantidade_Pixels'].mean():.2f}")
    print(f"   Mediana:{df_stats['Quantidade_Pixels'].median():.2f}")
    print(f"   Desvio: {df_stats['Quantidade_Pixels'].std():.2f}")
    print(f"   Maior:  {df_stats['Quantidade_Pixels'].max()}")
    print(f"   Menor:  {df_stats['Quantidade_Pixels'].min()}")
    print("-" * 50)
    print("🎯 TAXA DE HOMOGENEIDADE (HoR %):")
    print(f"   Média:  {df_stats['Taxa_HoR'].mean():.2f}%")
    print(f"   Mediana:{df_stats['Taxa_HoR'].median():.2f}%")
    print(f"   Desvio: {df_stats['Taxa_HoR'].std():.2f}%")
    print("="*50)

if __name__ == "__main__":
    gerar_metricas_segmentacao_tiling()