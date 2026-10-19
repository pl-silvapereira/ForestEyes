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

    # Diretórios
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
    CLASSES_FLORESTA_MAPBIOMAS = [1, 3, 4, 5, 9, 49]

    # Ficheiros de Saída
    path_csv_forest = os.path.join(dir_forest, f"df_forest_sort_{code_muni}_{ano_fim}.csv")
    path_csv_non_forest = os.path.join(dir_non_forest, f"df_non-forest_sort_{code_muni}_{ano_fim}.csv")
    relatorio_txt_path = os.path.join(output_report_dir, f"{code_muni}_Relatorio_MaskSLIC_HoR_{ano_fim}.txt")
    relatorio_tabela_csv = os.path.join(output_report_dir, f"{code_muni}_Relatorio_Tabela_Detalhada_{ano_fim}.csv")

    print("=" * 80, flush=True)
    print(f"📄 SCRIPT 06 - OBIA URBANO (CBERS 2M - ESCALA ~1000m²) - {ano_fim}", flush=True)
    print("=" * 80, flush=True)

    # Limiares e Variáveis
    min_Hor = 0.7
    min_size_seg_threshold = 100  # 100 pixels = 400 m² (limiar de descarte de ruído)
    qps = 250.0  # Média de ~250 pixels por segmento
    
    min_r, max_r = 56, 1342
    min_g, max_g = 74, 1050
    min_b, max_b = 101, 1267

    with rasterio.open(path_cbers) as src:
        cbers_meta = src.meta.copy()
        cbers_crs = src.crs
        cbers_transform = src.transform
        cbers_shape = (src.height, src.width)
        num_bands = src.count

    gdf_muni = gpd.read_file(path_shp_limite).to_crs(cbers_crs)
    geom_seg = gdf_muni[gdf_muni['class_name'] == 'Segmentar'].geometry
    mask_seg = rasterize([(geom, 1) for geom in geom_seg], out_shape=cbers_shape, transform=cbers_transform, fill=0, dtype=np.uint8)
    
    temp_tif_slic = f"/content/temp_slic_{code_muni}_{ano_fim}.tif"
    temp_tif_ids = f"/content/temp_ids_{code_muni}_{ano_fim}.tif"
    
    path_tif_slic = os.path.join(output_seg_dir, f"{code_muni}_MaskSLIC_ContornosBrancos_{ano_fim}.tif")
    path_tif_ids = os.path.join(output_seg_dir, f"{code_muni}_IDs_Superpixels_{ano_fim}.tif") 

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
                
                mask_seg_tile = mask_seg[y:y+h_janela, x:x+w_janela]
                rgbnir_raw = src.read(window=window) 
                matriz_ids = np.zeros((h_janela, w_janela), dtype=np.uint32)

                if np.sum(mask_seg_tile) > 0 and np.max(rgbnir_raw) > 0:
                    qts_tile = int(np.sum(mask_seg_tile) / qps)
                    if qts_tile > 0:
                        rgbnir_ch_last = np.moveaxis(rgbnir_raw, 0, -1)
                        rgb_slic = rgbnir_ch_last[:, :, :3].astype(np.float32)
                        tile_max = rgb_slic.max()
                        if tile_max > 0: rgb_slic /= tile_max 

                        segmentos_tile = slic(rgb_slic, n_segments=qts_tile, compactness=10, mask=(mask_seg_tile == 1), start_label=1, max_num_iter=5)
                        
                        if np.any(segmentos_tile > 0):
                            validos = segmentos_tile > 0
                            segmentos_tile[validos] += global_seg_offset
                            
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
                            mask_gt = np.isin(mb_block, CLASSES_FLORESTA_MAPBIOMAS).astype(np.uint8)
                            
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
                            df_tile['HoR'] = np.where(df_tile['Prop_Floresta'] >= 0.5, df_tile['Prop_Floresta'], 1.0 - df_tile['Prop_Floresta'])
                            df_tile['Class'] = np.where(df_tile['Prop_Floresta'] >= 0.5, 'floresta', 'nao floresta')
                            
                            bad_mask = (df_tile['HoR'] <= min_Hor) | (df_tile['Size'] <= min_size_seg_threshold)
                            bad_ids_local = df_tile[bad_mask]['Segment_label'].values
                            
                            segmentos_visuais = np.where(np.isin(segmentos_tile, bad_ids_local), -1, segmentos_tile)
                            matriz_ids = np.where(np.isin(segmentos_tile, bad_ids_local), 0, segmentos_tile).astype(np.uint32)
                            
                            df_list.append(df_tile)
                            global_seg_offset = df_tile['Segment_label'].max()

                            contornos_tile = find_boundaries(segmentos_visuais, mode='inner', background=-1)
                            rgbnir_raw[0][contornos_tile] = max_r
                            rgbnir_raw[1][contornos_tile] = max_g
                            rgbnir_raw[2][contornos_tile] = max_b

                dst.write(rgbnir_raw, window=window)
                dst_ids.write(matriz_ids, 1, window=window)
                del rgbnir_raw, mask_seg_tile
                gc.collect()

    shutil.move(temp_tif_slic, path_tif_slic)
    shutil.move(temp_tif_ids, path_tif_ids)

    print("\n4. Calculando métricas e montando o Relatório em Tabela...", flush=True)
    df_all = pd.concat(df_list, ignore_index=True)
    
    total_segs = len(df_all)
    bad_by_hor = (df_all['HoR'] <= min_Hor).sum()
    bad_by_size = (df_all['Size'] <= min_size_seg_threshold).sum()
    good_mask = (df_all['HoR'] > min_Hor) & (df_all['Size'] > min_size_seg_threshold)
    good_segs = good_mask.sum()
    bad_segs = total_segs - good_segs
    n_perfect_segs = (df_all['HoR'] == 1.0).sum()
    perc_perfect = (n_perfect_segs / total_segs * 100) if total_segs > 0 else 0.0

    auditoria_str = f"x{code_muni}: total_segs = {total_segs} | good_segs = {good_segs} | bad_segs = {bad_segs} | bad_segs because of HoR = {bad_by_hor} | bad_segs because of size = {bad_by_size} | perfect_segs = {n_perfect_segs} ({perc_perfect:.2f}%)"

    df_good = df_all[good_mask].copy()
    
    # ATUALIZAÇÃO CHAVE: Cria a métrica física de metros quadrados, mas também preserva o 'Size' em pixels
    df_good['Size_m2'] = df_good['Size'] * 4.0

    def categoria_hor_exata(h):
        if h >= 0.9999: return 'Perfeito'
        elif h > 0.7: return 'Util'
        else: return 'Nao Util'

    df_good['Categoria_HoR'] = df_good['HoR'].apply(categoria_hor_exata)

    linhas_tabela = []
    for classe in ['floresta', 'nao floresta']:
        df_c = df_good[df_good['Class'] == classe]
        
        if len(df_c) > 0:
            max_m2 = df_c['Size_m2'].max()
            min_m2 = df_c['Size_m2'].min()
            mean_m2 = df_c['Size_m2'].mean()
            
            df_perf = df_c[df_c['Categoria_HoR'] == 'Perfeito']
            df_util = df_c[df_c['Categoria_HoR'] == 'Util']
            df_nao_util = df_c[df_c['Categoria_HoR'] == 'Nao Util']
            
            linhas_tabela.append({
                'Classe': classe,
                'Qtd_Total_Segmentos': len(df_c),
                'Tamanho_Max_m2': max_m2,
                'Tamanho_Min_m2': min_m2,
                'Tamanho_Medio_m2': mean_m2,
                'Qtd_HoR_Perfeito': len(df_perf),
                'Qtd_HoR_Util': len(df_util),
                'Taxa_Media_HoR_Util': df_util['HoR'].mean() if len(df_util) > 0 else 0.0,
                'Qtd_HoR_Nao_Util': len(df_nao_util),
                'Taxa_Media_HoR_Nao_Util': df_nao_util['HoR'].mean() if len(df_nao_util) > 0 else 0.0
            })
        else:
            linhas_tabela.append({
                'Classe': classe, 'Qtd_Total_Segmentos': 0, 'Tamanho_Max_m2': 0, 'Tamanho_Min_m2': 0, 'Tamanho_Medio_m2': 0,
                'Qtd_HoR_Perfeito': 0, 'Qtd_HoR_Util': 0, 'Taxa_Media_HoR_Util': 0.0, 'Qtd_HoR_Nao_Util': 0, 'Taxa_Media_HoR_Nao_Util': 0.0
            })

    df_resumo_tabela = pd.DataFrame(linhas_tabela)
    df_resumo_tabela.to_csv(relatorio_tabela_csv, index=False, float_format='%.4f')

    df_forest = df_good[df_good['Class'] == 'floresta'].sort_values(by='HoR', ascending=False)
    df_non_forest = df_good[df_good['Class'] == 'nao floresta'].sort_values(by='HoR', ascending=False)
    
    # GARANTIA DE COMPATIBILIDADE: Exportamos tanto Size quanto Size_m2
    colunas_finais = ['Segment_label', 'HoR', 'Class', 'Size', 'Size_m2', 'Region', 'Mean_R', 'Mean_G', 'Mean_B']
    if num_bands >= 4: colunas_finais.append('Mean_NIR')

    df_forest[colunas_finais].to_csv(path_csv_forest, index=False, float_format='%.6f')
    df_non_forest[colunas_finais].to_csv(path_csv_non_forest, index=False, float_format='%.6f')

    with open(relatorio_txt_path, 'w', encoding='utf-8') as f:
        f.write("=========================================================================================\n")
        f.write(f" RELATÓRIO ESTATÍSTICO DE SEGMENTAÇÃO OBIA E HoR - {ano_fim}\n")
        f.write("=========================================================================================\n")
        f.write(f"Município: {code_muni} | Gabarito: MapBiomas 10m\n")
        f.write(f"Critério HoR: Perfeito (=100%) | Útil (<100% e >70%) | Não Útil (<=70%)\n")
        f.write(f"Filtros: min_size = {min_size_seg_threshold} px ({min_size_seg_threshold*4} m2) | qps = {qps}\n\n")
        f.write(auditoria_str + "\n\n")
        f.write("TABELA RESUMO DE SEGMENTOS (MÉTRICAS POR CLASSE):\n")
        f.write("-" * 105 + "\n")
        
        for _, row in df_resumo_tabela.iterrows():
            f.write(f"Classe: {row['Classe'].upper()}\n")
            f.write(f"  - Segmentos Válidos: {row['Qtd_Total_Segmentos']}\n")
            f.write(f"  - Tamanho (Max / Min / Médio): {row['Tamanho_Max_m2']:.2f} m² | {row['Tamanho_Min_m2']:.2f} m² | {row['Tamanho_Medio_m2']:.2f} m²\n")
            f.write(f"  - HoR Perfeito (= 100%): {row['Qtd_HoR_Perfeito']} segmentos\n")
            f.write(f"  - HoR Útil (< 100% e > 70%): {row['Qtd_HoR_Util']} segmentos (Taxa Média: {row['Taxa_Media_HoR_Util']*100:.2f}%)\n")
            f.write(f"  - HoR Não Útil (<= 70%): {row['Qtd_HoR_Nao_Util']} segmentos (Taxa Média: {row['Taxa_Media_HoR_Nao_Util']*100:.2f}%)\n")
            f.write("-" * 105 + "\n")
            
    print(f"✅ Relatório em Tabela gerado com sucesso em: {relatorio_tabela_csv}")
    print(f"✅ Relatório TXT atualizado em: {relatorio_txt_path}")

if __name__ == "__main__":
    main()