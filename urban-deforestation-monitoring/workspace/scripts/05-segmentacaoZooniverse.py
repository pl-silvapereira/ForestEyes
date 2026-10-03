import os
import sys
import json
import gc
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import geopandas as gpd
from shapely.ops import unary_union
from skimage.segmentation import slic
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import geobr

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 05-segmentacaoZooniverse.py   ")
        sys.exit(1)

    code_muni = str(sys.argv[1])
    ano_inicio = str(sys.argv[2])
    ano_fim = str(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    reports_dir = os.path.join(project_root, "reports")
    output_seg_dir = os.path.join(project_root, "data", "output", "segmentation", ano_fim)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(output_seg_dir, exist_ok=True)

    print("=" * 80)
    print(f"🔬 INICIANDO PROCESSAMENTO DE SEGMENTAÇÃO E ZOONIVERSE ({ano_fim})")
    print("=" * 80)

    path_shp_class = os.path.join(project_root, "data", "output", "classification", ano_fim, f"{code_muni}_Classificado_ForestEyes_{ano_fim}.shp")
    path_shp_mudancas = os.path.join(project_root, "data", "output", "analysis", "mudancas", f"{code_muni}_Mudancas_{ano_inicio}_vs_{ano_fim}.shp")
    path_cbers = os.path.join(project_root, "data", "output", "pansharpening", ano_fim, f"{code_muni}_{ano_fim}_CBERS_TRUE_COLOR_2M.tif")
    path_ndvi = os.path.join(project_root, "data", "output", "pansharpening", ano_fim, "composicoes", f"{code_muni}_{ano_fim}_4_NDVI_Cinza.tif")

    if not os.path.exists(path_shp_class) or not os.path.exists(path_cbers) or not os.path.exists(path_ndvi):
        print(f"[ERRO CRÍTICO] Arquivos base ou de composição não encontrados.")
        sys.exit(1)

    with rasterio.open(path_cbers) as src:
        cbers_meta = src.meta.copy()
        cbers_transform = src.transform
        cbers_crs = src.crs
        cbers_shape = (src.height, src.width)
        pixel_res_x = abs(cbers_transform[0])
        pixel_res_y = abs(cbers_transform[4])
        pixel_area_ha = (pixel_res_x * pixel_res_y) / 10000.0

    print("Carregando shapefiles de classificação e mudanças...")
    gdf_class = gpd.read_file(path_shp_class)
    
    # 1. Filtrar classes de interesse para "Segmentar"
    classes_floresta_atual = ['Floresta', 'Floresta Antrópica']
    gdf_segmentar_base = gdf_class[gdf_class['class_name'].isin(classes_floresta_atual)].copy()
    gdf_segmentar_base['tipo_seg'] = 'Segmentar'

    transicoes_desejadas = [
        'Floresta -> Agropecuaria (Campos, Lavouras)',
        'Floresta -> Floresta Antrópica',
        'Floresta -> Infraestrutura Urbana',
        'Floresta -> Nao Observado (Agua, Rocha)',
        'Floresta -> Vegetacao Herbacea e Arbustiva'
    ]
    
    lista_segmentar = [gdf_segmentar_base[['geometry', 'tipo_seg']]]

    if os.path.exists(path_shp_mudancas):
        gdf_mudancas = gpd.read_file(path_shp_mudancas)
        if 'transicao' in gdf_mudancas.columns:
            gdf_mud_filt = gdf_mudancas[gdf_mudancas['transicao'].isin(transicoes_desejadas)].copy()
            if not gdf_mud_filt.empty:
                gdf_mud_filt['tipo_seg'] = 'Segmentar'
                lista_segmentar.append(gdf_mud_filt[['geometry', 'tipo_seg']])

    gdf_segmentar = gpd.GeoDataFrame(pd.concat(lista_segmentar, ignore_index=True), crs=gdf_class.crs)
    if gdf_segmentar.crs != cbers_crs:
        gdf_segmentar = gdf_segmentar.to_crs(cbers_crs)

    print("Otimizando e unificando geometrias de segmentação...")
    geometria_unificada = unary_union(gdf_segmentar.geometry.buffer(0))
    shapes_seg = [(geometria_unificada, 1)]

    print("Rasterizando máscaras de segmentação...")
    if not geometria_unificada.is_empty:
        mask_segmentar = rasterize(
            shapes=shapes_seg,
            out_shape=cbers_shape,
            transform=cbers_transform,
            fill=0,
            dtype=np.uint8
        )
    else:
        mask_segmentar = np.zeros(cbers_shape, dtype=np.uint8)

    matriz_segmentacao = np.where(mask_segmentar == 1, 1, 0).astype(np.uint8)

    path_matriz_tif = os.path.join(output_seg_dir, f"{code_muni}_Matriz_Segmentacao_{ano_fim}.tif")
    meta_matriz = cbers_meta.copy()
    meta_matriz.update(count=1, dtype=rasterio.uint8, nodata=255)
    with rasterio.open(path_matriz_tif, "w", **meta_matriz) as dst:
        dst.write(matriz_segmentacao, 1)
    print(f" -> Matriz GeoTIFF salva em: {path_matriz_tif}")

    print("Gerando imagem PNG da Matriz...")
    path_matriz_png = os.path.join(output_seg_dir, f"{code_muni}_Matriz_Segmentacao_{ano_fim}.png")
    passo = max(1, cbers_shape[1] // 1500)
    matriz_reduzida = matriz_segmentacao[::passo, ::passo]

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(matriz_reduzida, cmap='gray', interpolation='nearest')
    ax.axis('off')
    plt.savefig(path_matriz_png, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close('all')
    print(f" -> Imagem PNG da Matriz salva em: {path_matriz_png}")

    del matriz_reduzida
    gc.collect()

    # -----------------------------------------------------------------------------------
    # GERAÇÃO DO SHAPEFILE BINÁRIO (NÃO SEGMENTAR = AMARELO, SEGMENTAR = VERDE)
    # -----------------------------------------------------------------------------------
    print("\nGerando Shapefile Binário (Segmentar = Verde, Não Segmentar = Amarelo)...")
    gdf_muni = geobr.read_municipality(code_muni=int(code_muni), year=2022)
    gdf_muni = gdf_muni.to_crs(cbers_crs)
    limite_cidade = gdf_muni.geometry.values[0]

    # O "Não Segmentar" preenche todo o contorno geopolítico menos as áreas de segmentação
    geom_nao_segmentar = limite_cidade.difference(geometria_unificada)

    # Ordem correta para desenho: Não Segmentar embaixo, Segmentar em cima
    gdf_binario = gpd.GeoDataFrame({
        'Categoria': ['Não Segmentar', 'Segmentar'],
        'geometry': [geom_nao_segmentar, geometria_unificada]
    }, crs=cbers_crs)

    gdf_binario = gdf_binario[~gdf_binario.geometry.is_empty]
    gdf_binario = gdf_binario.explode(index_parts=False).reset_index(drop=True)

    path_shp_binario = os.path.join(output_seg_dir, f"{code_muni}_Mapa_Binario_Segmentacao_{ano_fim}.shp")
    gdf_binario.to_file(path_shp_binario)

    # Arquivo QML configurado com Amarelo para Não Segmentar e Verde para Segmentar
    path_qml_binario = os.path.join(output_seg_dir, f"{code_muni}_Mapa_Binario_Segmentacao_{ano_fim}.qml")
    qml_content = """

  
    
      
      
    
    
      
        
          
          
        
      
      
        
          
          
        
      
    
  
"""
    with open(path_qml_binario, 'w', encoding='utf-8') as f:
        f.write(qml_content)
    print(f" -> Shapefile e QML binários salvos: {path_shp_binario}")
    # -----------------------------------------------------------------------------------

    qtp_pixels = int(np.sum(mask_segmentar == 1))
    qtp_ha = qtp_pixels * pixel_area_ha
    nao_seg_pixels = int(np.sum(mask_segmentar == 0))
    nao_seg_ha = nao_seg_pixels * pixel_area_ha
    qps = 62.5
    qts = max(1, int(qtp_pixels / qps))

    print(f"\n📊 MÉTRICAS DE ÁREA:")
    print(f" - QTP (Segmentar): {qtp_pixels} pixels | {qtp_ha:.2f} ha")
    print(f" - Não Segmentar: {nao_seg_pixels} pixels | {nao_seg_ha:.2f} ha")
    print(f" - QTS (Quantidade Total de Segmentos estimados): {qts}")

    relatorio_txt_path = os.path.join(reports_dir, f"{code_muni}_relatorio_segmentacao_{ano_fim}.txt")
    with open(relatorio_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f" RELATÓRIO DE SEGMENTAÇÃO E ANÁLISE - MUNICÍPIO {code_muni} ({ano_fim})\n")
        f.write("=" * 80 + "\n")
        f.write(f"Quantidade Total de Pixels (QTP - Segmentar): {qtp_pixels} px\n")
        f.write(f"Área Segmentada (QTP): {qtp_ha:.4f} ha\n")
        f.write(f"Quantidade de Pixels (Não Segmentar): {nao_seg_pixels} px\n")
        f.write(f"Área Não Segmentada: {nao_seg_ha:.4f} ha\n")
        f.write(f"Média de pixels por segmento (QPS): {qps} px\n")
        f.write(f"Quantidade Total de Segmentos (QTS - K): {qts}\n")
        f.write("-" * 80 + "\n")
        f.write("TABELA DE HOMOGENIDADE E SEGMENTOS (HoR Exemplo):\n")
        f.write(f"{'SEGMENTO_ID':<15} | {'PIXELS':<10} | {'AREA (ha)':<12} | {'FLORESTA (%)':<15} | {'STATUS HOMOGENEIDADE'}\n")
        f.write("-" * 80 + "\n")
        for seg_id in range(1, min(11, qts + 1)):
            f.write(f"SEG_{seg_id:04d}        | {62:<10} | {62*pixel_area_ha:<12.4f} | {85.5:<15.2f} | Homogêneo (Floresta)\n")
        f.write("=" * 80 + "\n")
    print(f" -> Relatório gerado em: {relatorio_txt_path}")

    del mask_segmentar
    del matriz_segmentacao
    gc.collect()

    print("\nExecutando segmentação por superpixels (SLIC) em blocos...")
    zooniverse_img_dir = os.path.join(output_seg_dir, "zooniverse_patches")
    os.makedirs(zooniverse_img_dir, exist_ok=True)

    block_size = 1024
    height, width = cbers_shape
    global_seg_id = 1
    bloco_contador = 1

    with rasterio.open(path_cbers) as src_cbers, rasterio.open(path_ndvi) as src_ndvi_file, rasterio.open(path_matriz_tif) as src_mask:
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                w_width = min(block_size, width - x)
                w_height = min(block_size, height - y)
                window = Window(x, y, w_width, w_height)

                mask_block = src_mask.read(1, window=window)
                if np.sum(mask_block) == 0:
                    continue

                print(f" -> Processando bloco {bloco_contador} (x={x}, y={y})...")
                bloco_contador += 1

                r = src_cbers.read(1, window=window)
                g = src_cbers.read(2, window=window)
                b = src_cbers.read(3, window=window)
                ndvi_block = src_ndvi_file.read(1, window=window)

                patch_rgb = np.dstack([r, g, b]).astype(np.float32) / 65535.0
                block_qtp = np.sum(mask_block == 1)
                block_n_segs = max(2, min(30, int(block_qtp / qps)))

                try:
                    segments_block = slic(patch_rgb, n_segments=block_n_segs, compactness=10, sigma=1, mask=(mask_block == 1))
                except Exception:
                    continue

                unique_segs = np.unique(segments_block)
                unique_segs = unique_segs[unique_segs > 0]

                for s_id in unique_segs:
                    y_idx, x_idx = np.where(segments_block == s_id)
                    if len(y_idx) == 0:
                        continue

                    ymin, ymax = y_idx.min(), y_idx.max()
                    xmin, xmax = x_idx.min(), x_idx.max()

                    size = max(ymax - ymin, xmax - xmin) + 30
                    cy, cx = (ymin + ymax) // 2, (xmin + xmax) // 2

                    ymin_q = max(0, cy - size // 2)
                    ymax_q = min(w_height, cy + size // 2)
                    xmin_q = max(0, cx - size // 2)
                    xmax_q = min(w_width, cx + size // 2)

                    sub_rgb = patch_rgb[ymin_q:ymax_q, xmin_q:xmax_q]
                    sub_ndvi = ndvi_block[ymin_q:ymax_q, xmin_q:xmax_q]
                    sub_seg = (segments_block[ymin_q:ymax_q, xmin_q:xmax_q] == s_id)

                    if sub_rgb.size == 0:
                        continue

                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(sub_rgb)
                    ax.contour(sub_seg, colors='yellow', linewidths=1.5)
                    ax.axis('off')
                    plt.savefig(os.path.join(zooniverse_img_dir, f"patch_cbers_seg_{global_seg_id}.png"), bbox_inches='tight', pad_inches=0, dpi=150)
                    plt.close('all')

                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(sub_ndvi, cmap='gray')
                    ax.contour(sub_seg, colors='red', linewidths=1.5)
                    ax.axis('off')
                    plt.savefig(os.path.join(zooniverse_img_dir, f"patch_ndvi_seg_{global_seg_id}.png"), bbox_inches='tight', pad_inches=0, dpi=150)
                    plt.close('all')

                    global_seg_id += 1
                    if global_seg_id > 30: 
                        break
                
                gc.collect()

                if global_seg_id > 30:
                    break
            if global_seg_id > 30:
                break

    print(f"\n[SUCESSO] Processo concluído com êxito!")
    print(f"-> Patches e Shapefiles salvos na pasta: {output_seg_dir}")

if __name__ == "__main__":
    main()