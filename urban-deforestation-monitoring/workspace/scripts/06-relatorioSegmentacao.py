import sys
import os
import gc
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
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

    output_seg_dir = os.path.join(project_root, "data", "output", "segmentation", ano_fim)
    output_report_dir = os.path.join(project_root, "reports")
    os.makedirs(output_report_dir, exist_ok=True)

    path_shp_binario = os.path.join(output_seg_dir, f"{code_muni}_Segmentar_vs_NaoSegmentar_{ano_fim}.shp")
    path_cbers = os.path.join(project_root, "data", "output", "pansharpening", ano_fim, f"{code_muni}_{ano_fim}_CBERS_TRUE_COLOR_2M.tif")
    
    path_csv_obia = os.path.join(output_report_dir, f"{code_muni}_Atributos_OBIA_{ano_fim}.csv")
    relatorio_txt_path = os.path.join(output_report_dir, f"{code_muni}_Relatorio_MaskSLIC_{ano_fim}.txt")

    if not os.path.exists(path_shp_binario) or not os.path.exists(path_cbers):
        print("[ERRO CRÍTICO] Arquivos base não encontrados.", flush=True)
        sys.exit(1)

    print("=" * 80, flush=True)
    print(f"📄 SCRIPT 06 - EXTRAÇÃO OBIA E SUPERPIXELS ({ano_fim})", flush=True)
    print("=" * 80, flush=True)

    # 1. VALORES EXATOS EXTRAÍDOS DO XML FORNECIDO
    min_r, max_r = 56, 1342
    min_g, max_g = 74, 1050
    min_b, max_b = 101, 1267

    print("1. A ler metadados da imagem CBERS...", flush=True)
    with rasterio.open(path_cbers) as src:
        cbers_meta = src.meta.copy()
        cbers_crs = src.crs
        cbers_transform = src.transform
        cbers_shape = (src.height, src.width)
        num_bands = src.count
        
        pixel_res_x = abs(cbers_transform[0])
        pixel_res_y = abs(cbers_transform[4])

    print("2. A rasterizar áreas do Shapefile...", flush=True)
    gdf_binario = gpd.read_file(path_shp_binario).to_crs(cbers_crs)
    geom_seg = gdf_binario[gdf_binario['class_name'] == 'Segmentar'].geometry
    geom_nao_seg = gdf_binario[gdf_binario['class_name'] == 'Nao_Segmentar'].geometry

    mask_seg = rasterize([(geom, 1) for geom in geom_seg], out_shape=cbers_shape, transform=cbers_transform, fill=0, dtype=np.uint8)
    mask_nao_seg = rasterize([(geom, 1) for geom in geom_nao_seg], out_shape=cbers_shape, transform=cbers_transform, fill=0, dtype=np.uint8)
    
    # Máscara total (Todo o município) para extrair todos os atributos
    mask_muni = ((mask_seg == 1) | (mask_nao_seg == 1)).astype(np.uint8)
    qps = 62.5

    print(f"\n3. Iniciando SLIC Global e Extração de Atributos OBIA (Random Forest/SVM)...", flush=True)
    temp_tif_local = f"/content/temp_slic_{code_muni}_{ano_fim}.tif"
    path_tif_slic = os.path.join(output_seg_dir, f"{code_muni}_MaskSLIC_ContornosBrancos_{ano_fim}.tif")
    
    tile_size = 1024 
    h, w = cbers_shape
    total_blocos = int(np.ceil(h / tile_size)) * int(np.ceil(w / tile_size))
    
    df_list = []
    global_seg_offset = 0
    bloco_atual = 0

    with rasterio.open(path_cbers) as src, rasterio.open(temp_tif_local, 'w', **cbers_meta) as dst:
        
        dst.set_band_description(1, "Red")
        dst.set_band_description(2, "Green")
        dst.set_band_description(3, "Blue")
        if num_bands >= 4: dst.set_band_description(4, "NIR")
        
        # INSERÇÃO MANUAL COM OS DADOS EXATOS DO XML
        dst.update_tags(1, STATISTICS_MINIMUM=str(min_r), STATISTICS_MAXIMUM=str(max_r))
        dst.update_tags(2, STATISTICS_MINIMUM=str(min_g), STATISTICS_MAXIMUM=str(max_g))
        dst.update_tags(3, STATISTICS_MINIMUM=str(min_b), STATISTICS_MAXIMUM=str(max_b))

        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                bloco_atual += 1
                h_janela = min(tile_size, h - y)
                w_janela = min(tile_size, w - x)
                window = Window(x, y, w_janela, h_janela)
                
                mask_muni_tile = mask_muni[y:y+h_janela, x:x+w_janela]
                qtp_muni_tile = np.sum(mask_muni_tile)
                
                rgbnir_raw = src.read(window=window) 
                
                if qtp_muni_tile > 0:
                    qts_tile = int(qtp_muni_tile / qps)
                    if qts_tile > 0:
                        rgbnir_ch_last = np.moveaxis(rgbnir_raw, 0, -1)
                        
                        rgb_slic = rgbnir_ch_last[:, :, :3].astype(np.float32)
                        tile_max = rgb_slic.max()
                        if tile_max > 0: rgb_slic /= tile_max 
                            
                        segmentos_tile = slic(rgb_slic, n_segments=qts_tile, compactness=10, mask=(mask_muni_tile == 1), start_label=1, max_num_iter=5)
                        
                        if np.any(segmentos_tile > 0):
                            mask_seg_tile = mask_seg[y:y+h_janela, x:x+w_janela]
                            
                            validos = segmentos_tile > 0
                            segmentos_tile[validos] += global_seg_offset
                            
                            # Extração de estatísticas
                            props = regionprops_table(segmentos_tile, intensity_image=rgbnir_ch_last, properties=('label', 'area', 'intensity_mean'))
                            props_mask = regionprops_table(segmentos_tile, intensity_image=mask_seg_tile, properties=('label', 'intensity_mean'))
                            
                            df_tile = pd.DataFrame({
                                'ID_Segmento': props['label'],
                                'Area_px': props['area'],
                                'Prop_Floresta': props_mask['intensity_mean'],
                                'Mean_R': props['intensity_mean-0'],
                                'Mean_G': props['intensity_mean-1'],
                                'Mean_B': props['intensity_mean-2']
                            })
                            if num_bands >= 4:
                                df_tile['Mean_NIR'] = props['intensity_mean-3']
                                
                            df_list.append(df_tile)
                            global_seg_offset = df_tile['ID_Segmento'].max()

                        # Usa os valores máximos reais extraídos do XML para a borda branca 
                        contornos_tile = find_boundaries(segmentos_tile, mode='inner')
                        rgbnir_raw[0][contornos_tile] = max_r
                        rgbnir_raw[1][contornos_tile] = max_g
                        rgbnir_raw[2][contornos_tile] = max_b
                        
                        print(f"   ✅ [Bloco {bloco_atual}/{total_blocos}] -> Extraídos IDs e atributos.", flush=True)
                else:
                    if bloco_atual % 20 == 0:
                        print(f"   ⏩ [Bloco {bloco_atual}/{total_blocos}] -> Ignorando áreas vazias...", flush=True)

                dst.write(rgbnir_raw, window=window)
                del rgbnir_raw, mask_muni_tile
                gc.collect()

    shutil.move(temp_tif_local, path_tif_slic)
    print(f"\n✅ Imagem pronta e configurada para o QGIS em: {path_tif_slic}", flush=True)

    print("\n4. Calculando HoR, Classes e finalizando Relatórios OBIA...", flush=True)
    df_all = pd.concat(df_list, ignore_index=True)
    
    # Cálculos OBIA
    df_all['Classe'] = np.where(df_all['Prop_Floresta'] >= 0.5, 'Floresta', 'Nao_Floresta')
    df_all['HoR'] = np.where(df_all['Prop_Floresta'] >= 0.5, df_all['Prop_Floresta'], 1.0 - df_all['Prop_Floresta'])
    df_all['HoR_perc'] = df_all['HoR'] * 100.0

    def classificar_grupo(hor):
        if hor >= 99.9: return 'Perfeitos'
        elif hor >= 70.0: return 'Uteis'
        else: return 'Nao Uteis'
        
    df_all['Grupo'] = df_all['HoR_perc'].apply(classificar_grupo)
    
    # Exporta CSV
    colunas_finais = ['ID_Segmento', 'Classe', 'HoR_perc', 'Grupo', 'Area_px', 'Mean_R', 'Mean_G', 'Mean_B']
    if num_bands >= 4: colunas_finais.append('Mean_NIR')
    df_all[colunas_finais].to_csv(path_csv_obia, index=False, float_format='%.2f')
    print(f"✅ Arquivo CSV (Random Forest/SVM) salvo em: {path_csv_obia}")

    # Gera Relatório Agrupado TXT
    resumo = df_all.groupby('Grupo').agg(
        Qtd=('ID_Segmento', 'count'),
        Area_Media=('Area_px', 'mean'),
        HoR_Medio=('HoR_perc', 'mean')
    ).reindex(['Perfeitos', 'Uteis', 'Nao Uteis'], fill_value=0)

    with open(relatorio_txt_path, 'w', encoding='utf-8') as f:
        f.write("=========================================================\n")
        f.write(f" RELATÓRIO DE SEGMENTAÇÃO OBIA E AVALIAÇÃO HoR - {ano_fim}\n")
        f.write("=========================================================\n")
        f.write(f"Município / Código: {code_muni}\n\n")
        f.write("1. PARÂMETROS BASE:\n")
        f.write(f" - Resolução do Pixel: {pixel_res_x:.1f}m x {pixel_res_y:.1f}m\n")
        f.write(f" - QPS Alvo: {qps:.1f} px\n\n")
        
        f.write("2. ESTATÍSTICAS DOS SUPERPIXELS (AVALIAÇÃO HoR)\n")
        f.write(f" - Segmentos Perfeitos (HoR = 100%): {resumo.loc['Perfeitos', 'Qtd']:,} segs | Tamanho Médio: {resumo.loc['Perfeitos', 'Area_Media']:.1f} px | HoR Médio: {resumo.loc['Perfeitos', 'HoR_Medio']:.1f}%\n")
        f.write(f" - Segmentos Úteis (70% <= HoR < 100%): {resumo.loc['Uteis', 'Qtd']:,} segs | Tamanho Médio: {resumo.loc['Uteis', 'Area_Media']:.1f} px | HoR Médio: {resumo.loc['Uteis', 'HoR_Medio']:.1f}%\n")
        f.write(f" - Segmentos Não Úteis (HoR < 70%): {resumo.loc['Nao Uteis', 'Qtd']:,} segs | Tamanho Médio: {resumo.loc['Nao Uteis', 'Area_Media']:.1f} px | HoR Médio: {resumo.loc['Nao Uteis', 'HoR_Medio']:.1f}%\n")
        f.write("=========================================================\n")
    print(f"✅ Relatório TXT gerado em: {relatorio_txt_path}", flush=True)

    print("\n🎉 Todos os relatórios finalizados! A estrutura OBIA está pronta para a classificação.")

if __name__ == "__main__":
    main()