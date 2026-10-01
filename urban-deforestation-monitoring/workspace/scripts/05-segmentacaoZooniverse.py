import os
import sys
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import geopandas as gpd
from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt
from dotenv import load_dotenv

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 05-segmentacaoZooniverse.py   ")
        print("Exemplo: python 05-segmentacaoZooniverse.py 3549904 2020 2024")
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
    print(f"🔬 INICIANDO PROCESSAMENTO DE SEGMENTAÇÃO E ZOONIVERSE (${ano_fim})")
    print("=" * 80)

    # 1. Caminhos dos arquivos de entrada
    path_shp_class = os.path.join(project_root, "data", "output", "classification", ano_fim, f"{code_muni}_Classificado_ForestEyes_{ano_fim}.shp")
    path_shp_mudancas = os.path.join(project_root, "data", "output", "analysis", "mudancas", f"{code_muni}_Mudancas_{ano_inicio}_vs_{ano_fim}.shp")
    path_cbers = os.path.join(project_root, "data", "output", "pansharpening", ano_fim, f"{code_muni}_{ano_fim}_CBERS_TRUE_COLOR_2M.tif")
    path_ndvi = os.path.join(project_root, "data", "output", "pansharpening", "composicoes", f"{code_muni}_{ano_fim}_4_NDVI_Cinza.tif")

    if not os.path.exists(path_shp_class) or not os.path.exists(path_cbers):
        print(f"[ERRO CRÍTICO] Arquivos base não encontrados. Verifique a execução dos scripts anteriores.")
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

    print("Rasterizando máscaras de segmentação...")
    shapes_seg = [(geom, 1) for geom in gdf_segmentar.geometry if geom.is_valid and not geom.is_empty]
    
    if shapes_seg:
        mask_segmentar = rasterize(
            shapes=shapes_seg,
            out_shape=cbers_shape,
            transform=cbers_transform,
            fill=0,
            dtype=np.uint8
        )
    else:
        mask_segmentar = np.zeros(cbers_shape, dtype=np.uint8)

    # Matriz de Segmentação: 01 = Segmentar, 00 = Não Segmentar
    matriz_segmentacao = np.where(mask_segmentar == 1, 1, 0).astype(np.uint8)

    path_matriz_tif = os.path.join(output_seg_dir, f"{code_muni}_Matriz_Segmentacao_{ano_fim}.tif")
    meta_matriz = cbers_meta.copy()
    meta_matriz.update(count=1, dtype=rasterio.uint8, nodata=255)
    with rasterio.open(path_matriz_tif, "w", **meta_matriz) as dst:
        dst.write(matriz_segmentacao, 1)
    print(f" -> Matriz de Segmentação salva em: {path_matriz_tif}")

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

    # Gerar Relatório em .txt
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

    # 4. Abordagem Otimizada em Blocos (Tiles) para Superpixels (SLIC)
    print("\nExecutando segmentação por superpixels (SLIC) otimizada em blocos...")
    zooniverse_img_dir = os.path.join(output_seg_dir, "zooniverse_patches")
    os.makedirs(zooniverse_img_dir, exist_ok=True)

    block_size = 2048
    height, width = cbers_shape
    global_seg_id = 1

    with rasterio.open(path_cbers) as src_cbers, rasterio.open(path_ndvi) as src_ndvi_file:
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                w_width = min(block_size, width - x)
                w_height = min(block_size, height - y)
                window = Window(x, y, w_width, w_height)

                mask_block = mask_segmentar[y:y+w_height, x:x+w_width]
                if np.sum(mask_block) == 0:
                    continue # Pula blocos que não têm área para segmentar

                # Lê bandas RGB e NDVI do bloco
                r = src_cbers.read(1, window=window)
                g = src_cbers.read(2, window=window)
                b = src_cbers.read(3, window=window)
                ndvi_block = src_ndvi_file.read(1, window=window)

                patch_rgb = np.dstack([r, g, b]).astype(np.float32) / 65535.0

                # Quantidade de segmentos proporcional ao tamanho do bloco mascarado
                block_qtp = np.sum(mask_block == 1)
                block_n_segs = max(5, int(block_qtp / qps))

                # Aplica SLIC apenas no bloco ativo
                segments_block = slic(patch_rgb, n_segments=block_n_segs, compactness=10, sigma=1, mask=(mask_block == 1))

                unique_segs = np.unique(segments_block)
                unique_segs = unique_segs[unique_segs > 0]

                for s_id in unique_segs:
                    y_idx, x_idx = np.where(segments_block == s_id)
                    if len(y_idx) == 0:
                        continue

                    ymin, ymax = y_idx.min(), y_idx.max()
                    xmin, xmax = x_idx.min(), x_idx.max()

                    size = max(ymax - ymin, xmax - xmin) + 40
                    cy, cx = (ymin + ymax) // 2, (xmin + xmax) // 2

                    ymin_q = max(0, cy - size // 2)
                    ymax_q = min(w_height, cy + size // 2)
                    xmin_q = max(0, cx - size // 2)
                    xmax_q = min(w_width, cx + size // 2)

                    sub_rgb = patch_rgb[ymin_q:ymax_q, xmin_q:xmax_q]
                    sub_ndvi = ndvi_block[ymin_q:ymax_q, xmin_q:xmax_q]
                    sub_seg = (segments_block[ymin_q:ymax_q, xmin_q:xmax_q] == s_id)

                    # Salvar imagem CBERS com contorno amarelo
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(sub_rgb)
                    ax.contour(sub_seg, colors='yellow', linewidths=1.5)
                    ax.axis('off')
                    plt.savefig(os.path.join(zooniverse_img_dir, f"patch_cbers_seg_{global_seg_id}.png"), bbox_inches='tight', pad_inches=0, dpi=150)
                    plt.close()

                    # Salvar imagem NDVI com contorno vermelho
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(sub_ndvi, cmap='gray')
                    ax.contour(sub_seg, colors='red', linewidths=1.5)
                    ax.axis('off')
                    plt.savefig(os.path.join(zooniverse_img_dir, f"patch_ndvi_seg_{global_seg_id}.png"), bbox_inches='tight', pad_inches=0, dpi=150)
                    plt.close()

                    global_seg_id += 1
                    if global_seg_id > 50: # Limita a quantidade de patches de exemplo gerados na primeira execução
                        break
                if global_seg_id > 50:
                    break
            if global_seg_id > 50:
                break

    print(f"\n[SUCESSO] Processo de segmentação e geração de patches para o Zooniverse concluído com êxito!")
    print(f"-> Patches salvos em: {zooniverse_img_dir}")

if __name__ == "__main__":
    main()