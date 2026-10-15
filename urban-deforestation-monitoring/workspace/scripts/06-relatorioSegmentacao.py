import sys
import os
import gc
import glob
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
import rasterio.warp
from rasterio.features import rasterize
from skimage.segmentation import slic, find_boundaries
from skimage.measure import regionprops_table
import matplotlib.pyplot as plt
from dotenv import load_dotenv

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 06-relatorioSegmentacao.py <code_muni> <ano_inicio> <ano_fim>")
        sys.exit(1)

    code_muni = str(sys.argv[1])
    ano_inicio = str(sys.argv[2])
    ano_fim = str(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    # Diretórios (Estrutura do Professor)
    output_seg_dir = os.path.join(project_root, "data", "output", "segmentation", ano_fim)
    output_report_dir = os.path.join(project_root, "reports")
    
    dir_forest = os.path.join(output_report_dir, "dataframes_csv", "forest")
    dir_non_forest = os.path.join(output_report_dir, "dataframes_csv", "non-forest")
    
    os.makedirs(output_report_dir, exist_ok=True)
    os.makedirs(dir_forest, exist_ok=True)
    os.makedirs(dir_non_forest, exist_ok=True)

    # Ficheiros de Entrada
    path_shp_limite = os.path.join(output_seg_dir, f"{code_muni}_Segmentar_vs_NaoSegmentar_{ano_fim}.shp")
    path_cbers = os.path.join(project_root, "data", "output", "pansharpening", ano_fim, f"{code_muni}_{ano_fim}_CBERS_TRUE_COLOR_2M.tif")
    
    mb_dir = os.path.join(project_root, "data", "input", "MapBiomas", ano_fim)
    mb_files = glob.glob(os.path.join(mb_dir, "mapbiomas_lulc_10m_*.tif"))
    
    if not os.path.exists(path_shp_limite) or not os.path.exists(path_cbers):
        print("[ERRO CRÍTICO] Arquivos base (CBERS ou Shapefile) não encontrados.", flush=True)
        sys.exit(1)
    if not mb_files:
        print(f"[ERRO CRÍTICO] Imagem MapBiomas não encontrada no diretório: {mb_dir}", flush=True)
        sys.exit(1)
        
    path_mapbiomas = mb_files[0]
    
    # Classes oficiais do MapBiomas que representam Floresta (Natural e Plantada)
    CLASSES_FLORESTA_MAPBIOMAS = [1, 3, 4, 5, 9, 49]

    # Ficheiros de Saída
    path_csv_forest = os.path.join(dir_forest, f"df_forest_sort_{code_muni}_{ano_fim}.csv")
    path_csv_non_forest = os.path.join(dir_non_forest, f"df_non-forest_sort_{code_muni}_{ano_fim}.csv")
    relatorio_txt_path = os.path.join(output_report_dir, f"{code_muni}_Relatorio_MaskSLIC_HoR_{ano_fim}.txt")

    print("=" * 80, flush=True)
    print(f"📄 SCRIPT 06 - ARQUITETURA OBIA + MAPBIOMAS (METODOLOGIA PROFESSOR) - {ano_fim}", flush=True)
    print("=" * 80, flush=True)

    # Limiares Académicos exigidos pelo professor
    min_Hor = 0.7
    min_size_seg_threshold = 630
    
    # Valores XML fixos para manter a Imagem Original Radiométrica
    min_r, max_r = 56, 1342
    min_g, max_g = 74, 1050
    min_b, max_b = 101, 1267

    print(f"1. A ler metadados do CBERS e alinhar com o MapBiomas ({os.path.basename(path_mapbiomas)})...", flush=True)
    with rasterio.open(path_cbers) as src:
        cbers_meta = src.meta.copy()
        cbers_crs = src.crs
        cbers_transform = src.transform
        cbers_shape = (src.height, src.width)
        num_bands = src.count

    print("2. A rasterizar os limites do município...", flush=True)
    gdf_muni = gpd.read_file(path_shp_limite).to_crs(cbers_crs)
    geom_seg = gdf_muni[gdf_muni['class_name'] == 'Segmentar'].geometry
    geom_nao_seg = gdf_muni[gdf_muni['class_name'] == 'Nao_Segmentar'].geometry

    mask_seg = rasterize([(geom, 1) for geom in geom_seg], out_shape=cbers_shape, transform=cbers_transform, fill=0, dtype=np.uint8)
    mask_nao_seg = rasterize([(geom, 1) for geom in geom_nao_seg], out_shape=cbers_shape, transform=cbers_transform, fill=0, dtype=np.uint8)
    mask_muni = ((mask_seg == 1) | (mask_nao_seg == 1)).astype(np.uint8)
    
    qps = 1500.0 # Quantidade de Pixeis Base

    print(f"\n3. Iniciando Segmentação, Cruzamento com MapBiomas e Extração de Matrizes...", flush=True)
    
    temp_tif_slic = f"/content/temp_slic_{code_muni}_{ano_fim}.tif"
    temp_tif_ids = f"/content/temp_ids_{code_muni}_{ano_fim}.tif"
    
    path_tif_slic = os.path.join(output_seg_dir, f"{code_muni}_MaskSLIC_ContornosBrancos_{ano_fim}.tif")
    path_tif_ids = os.path.join(output_seg_dir, f"{code_muni}_IDs_Superpixels_{ano_fim}.tif") # <--- NOVO: Ouro para a IA
    
    # Metadados para o TIF dos IDs
    meta_ids = cbers_meta.copy()
    meta_ids.update(dtype=rasterio.uint32, count=1, nodata=0)

    tile_size = 1024 
    h, w = cbers_shape
    total_blocos = int(np.ceil(h / tile_size)) * int(np.ceil(w / tile_size))
    
    df_list = []
    global_seg_offset = 0
    bloco_atual = 0

    with rasterio.open(path_cbers) as src, rasterio.open(path_mapbiomas) as src_mb, \
         rasterio.open(temp_tif_slic, 'w', **cbers_meta) as dst, rasterio.open(temp_tif_ids, 'w', **meta_ids) as dst_ids:
        
        dst.set_band_description(1, "Red")
        dst.set_band_description(2, "Green")
        dst.set_band_description(3, "Blue")
        if num_bands >= 4: dst.set_band_description(4, "NIR")
        
        dst.update_tags(1, STATISTICS_MINIMUM=str(min_r), STATISTICS_MAXIMUM=str(max_r))
        dst.update_tags(2, STATISTICS_MINIMUM=str(min_g), STATISTICS_MAXIMUM=str(max_g))
        dst.update_tags(3, STATISTICS_MINIMUM=str(min_b), STATISTICS_MAXIMUM=str(max_b))

        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                bloco_atual += 1
                w_janela, h_janela = min(tile_size, w - x), min(tile_size, h - y)
                window = Window(x, y, w_janela, h_janela)
                win_transform = rasterio.windows.transform(window, cbers_transform)
                
                mask_muni_tile = mask_muni[y:y+h_janela, x:x+w_janela]
                
                rgbnir_raw = src.read(window=window) 
                matriz_ids = np.zeros((h_janela, w_janela), dtype=np.uint32)

                if np.sum(mask_muni_tile) > 0 and np.max(rgbnir_raw) > 0:
                    qts_tile = int(np.sum(mask_muni_tile) / qps)
                    if qts_tile > 0:
                        rgbnir_ch_last = np.moveaxis(rgbnir_raw, 0, -1)
                        rgb_slic = rgbnir_ch_last[:, :, :3].astype(np.float32)
                        tile_max = rgb_slic.max()
                        if tile_max > 0: rgb_slic /= tile_max 

                        segmentos_tile = slic(rgb_slic, n_segments=qts_tile, compactness=10, mask=(mask_muni_tile == 1), start_label=1, max_num_iter=5)
                        
                        if np.any(segmentos_tile > 0):
                            validos = segmentos_tile > 0
                            segmentos_tile[validos] += global_seg_offset
                            
                            # -------------------------------------------------------------
                            # REPROJEÇÃO DO MAPBIOMAS PARA O CBERS NO BLOCO
                            # -------------------------------------------------------------
                            mb_block = np.empty((h_janela, w_janela), dtype=np.uint8)
                            rasterio.warp.reproject(
                                source=rasterio.band(src_mb, 1),
                                destination=mb_block,
                                src_transform=src_mb.transform,
                                src_crs=src_mb.crs,
                                dst_transform=win_transform,
                                dst_crs=cbers_crs,
                                resampling=rasterio.warp.Resampling.nearest
                            )
                            # Cria máscara do MapBiomas (1 = Floresta, 0 = Resto)
                            mask_gt = np.isin(mb_block, CLASSES_FLORESTA_MAPBIOMAS).astype(np.uint8)
                            
                            # Extração
                            props = regionprops_table(segmentos_tile, intensity_image=rgbnir_ch_last, properties=('label', 'area', 'intensity_mean'))
                            props_mask = regionprops_table(segmentos_tile, intensity_image=mask_gt, properties=('label', 'intensity_mean'))
                            
                            df_tile = pd.DataFrame({
                                'Segment_label': props['label'],
                                'Size': props['area'],
                                'Prop_Floresta': props_mask['intensity_mean'],
                                'Mean_R': props['intensity_mean-0'],
                                'Mean_G': props['intensity_mean-1'],
                                'Mean_B': props['intensity_mean-2']
                            })
                            if num_bands >= 4:
                                df_tile['Mean_NIR'] = props['intensity_mean-3']
                                
                            df_tile['Region'] = f"x{code_muni}"
                            
                            # Lógica HoR do Professor
                            df_tile['HoR'] = np.where(df_tile['Prop_Floresta'] >= 0.5, df_tile['Prop_Floresta'], 1.0 - df_tile['Prop_Floresta'])
                            df_tile['Class'] = np.where(df_tile['Prop_Floresta'] >= 0.5, 'forest', 'non forest')
                            
                            # Identifica segmentos inválidos
                            bad_mask = (df_tile['HoR'] <= min_Hor) | (df_tile['Size'] <= min_size_seg_threshold)
                            bad_ids_local = df_tile[bad_mask]['Segment_label'].values
                            
                            # Oculta bordas brancas dos segmentos maus (valor de fundo -1)
                            segmentos_visuais = np.where(np.isin(segmentos_tile, bad_ids_local), -1, segmentos_tile)
                            
                            # Guarda apenas os IDs BONS para a Inteligência Artificial no Script 07
                            matriz_ids = np.where(np.isin(segmentos_tile, bad_ids_local), 0, segmentos_tile).astype(np.uint32)
                            
                            df_list.append(df_tile)
                            global_seg_offset = df_tile['Segment_label'].max()

                            contornos_tile = find_boundaries(segmentos_visuais, mode='inner', background=-1)
                            rgbnir_raw[0][contornos_tile] = max_r
                            rgbnir_raw[1][contornos_tile] = max_g
                            rgbnir_raw[2][contornos_tile] = max_b
                        
                        print(f"   ✅ [Bloco {bloco_atual}/{total_blocos}] -> Segmentado e Cruzado com MapBiomas.", flush=True)
                else:
                    if bloco_atual % 20 == 0:
                        print(f"   ⏩ [Bloco {bloco_atual}/{total_blocos}] -> Ignorando áreas vazias...", flush=True)

                dst.write(rgbnir_raw, window=window)
                dst_ids.write(matriz_ids, 1, window=window)
                
                del rgbnir_raw, mask_muni_tile
                gc.collect()

    shutil.move(temp_tif_slic, path_tif_slic)
    shutil.move(temp_tif_ids, path_tif_ids)
    print(f"\n✅ Imagem Visual guardada em: {path_tif_slic}")
    print(f"✅ Matriz Base de Inteligência Artificial gerada em: {path_tif_ids}")

    print("\n4. Gerando DataFrames ordenados e Auditoria Académica...", flush=True)
    df_all = pd.concat(df_list, ignore_index=True)
    
    # Contabilização rigorosa baseada no Notebook do Professor
    total_segs = len(df_all)
    bad_by_hor = (df_all['HoR'] <= min_Hor).sum()
    bad_by_size = (df_all['Size'] <= min_size_seg_threshold).sum()
    
    good_mask = (df_all['HoR'] > min_Hor) & (df_all['Size'] > min_size_seg_threshold)
    good_segs = good_mask.sum()
    bad_segs = total_segs - good_segs
    
    n_perfect_segs = (df_all['HoR'] == 1.0).sum()
    perc_perfect = (n_perfect_segs / total_segs * 100) if total_segs > 0 else 0.0

    auditoria_str = f"x{code_muni}: total_segs = {total_segs} | good_segs = {good_segs} | bad_segs = {bad_segs} | bad_segs because of HoR = {bad_by_hor} | bad_segs because of size = {bad_by_size} | perfect_segs = {n_perfect_segs} ({perc_perfect:.2f}%)"
    print("\n" + auditoria_str)

    # Separação e filtragem rigorosa para os CSVs
    df_good = df_all[good_mask].copy()
    
    df_forest = df_good[df_good['Class'] == 'forest'].sort_values(by='HoR', ascending=False)
    df_non_forest = df_good[df_good['Class'] == 'non forest'].sort_values(by='HoR', ascending=False)
    
    colunas_finais = ['Segment_label', 'HoR', 'Class', 'Size', 'Region', 'Mean_R', 'Mean_G', 'Mean_B']
    if num_bands >= 4: colunas_finais.append('Mean_NIR')

    df_forest[colunas_finais].to_csv(path_csv_forest, index=False, float_format='%.6f')
    df_non_forest[colunas_finais].to_csv(path_csv_non_forest, index=False, float_format='%.6f')
    
    print(f"✅ DataFrames organizados e guardados nas pastas 'forest' e 'non-forest'.")

    # Atualiza o arquivo de Texto
    with open(relatorio_txt_path, 'w', encoding='utf-8') as f:
        f.write("=========================================================\n")
        f.write(f" RELATÓRIO DE SEGMENTAÇÃO OBIA E AVALIAÇÃO HoR - {ano_fim}\n")
        f.write("=========================================================\n")
        f.write(f"Gabarito Verdade: Arquivo Raster MapBiomas 10m\n")
        f.write(f"min_size_seg: {min_size_seg_threshold}       max_size_seg: 10000 \n\n")
        f.write(auditoria_str + "\n")
        f.write("\n=========================================================\n")

if __name__ == "__main__":
    main()