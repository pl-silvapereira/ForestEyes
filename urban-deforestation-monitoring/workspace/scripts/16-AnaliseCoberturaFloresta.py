import os
import sys
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from skimage.segmentation import slic, find_boundaries
from scipy.ndimage import find_objects, binary_dilation
from dotenv import load_dotenv

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 16-AnaliseCoberturaFloresta2023_2024.py <code_muni> <ano_base> <ano_alvo>")
        print("Exemplo: python 16-AnaliseCoberturaFloresta2023_2024.py 3549904 2023 2024")
        sys.exit(1)

    code_muni = int(sys.argv[1])
    ano_base = str(sys.argv[2])   # Ex: 2023
    ano_alvo = str(sys.argv[3])   # Ex: 2024

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    # Diretórios de entrada e saída
    class_dir = os.path.join(project_root, "data", "output", "classification")
    reports_dir = os.path.join(project_root, "reports")
    segmentation_dir = os.path.join(project_root, "data", "output", "mask", "segmentation")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(segmentation_dir, exist_ok=True)

    path_shp_base = os.path.join(class_dir, ano_base, f"{code_muni}_Classificado_ForestEyes_{ano_base}.shp")
    path_shp_alvo = os.path.join(class_dir, ano_alvo, f"{code_muni}_Classificado_ForestEyes_{ano_alvo}.shp")
    path_sat = os.path.join(
        project_root, "data", "output", "pansharpening", "geopolitic-RGBN", 
        ano_alvo, f"{code_muni}_{ano_alvo}_CBERS_TRUE_COLOR_CLIPPED.tif"
    )

    if not os.path.exists(path_shp_base) or not os.path.exists(path_shp_alvo) or not os.path.exists(path_sat):
        print("[ERRO CRÍTICO] Shapefiles de classificação ou imagem de satélite do ano alvo não encontrados.")
        sys.exit(1)

    print("=" * 115)
    print(f"🌲 SCRIPT 16 (REFATORADO): ANÁLISE DE FLORESTAS E SUPERPIXELS ({ano_base} vs {ano_alvo})")
    print(f"📍 MUNICÍPIO: {code_muni}")
    print("=" * 115)

    # 1. Carregamento e Reprojecão rigorosa (Lógica inspirada no Script 07)
    print(f"Carregando e padronizando bases de {ano_base} e {ano_alvo}...")
    df1 = gpd.read_file(path_shp_base)
    df2 = gpd.read_file(path_shp_alvo)

    utm_crs = df1.estimate_utm_crs()
    df1 = df1.to_crs(utm_crs)
    df2 = df2.to_crs(utm_crs)

    # Filtrar apenas geometrias de floresta no ano base
    col_cls = 'class_name' if 'class_name' in df1.columns else 'class_id'
    if col_cls == 'class_name':
        floresta_base = df1[df1['class_name'].str.contains('Floresta', case=False, na=False)].copy()
    else:
        floresta_base = df1[df1['class_id'] == 3].copy()

    floresta_base['area_ha'] = floresta_base.geometry.area / 10000.0
    area_corte = floresta_base['area_ha'].quantile(0.25)
    grandes_florestas = floresta_base[floresta_base['area_ha'] >= area_corte]
    print(f"-> {len(grandes_florestas)} grandes fragmentos florestais isolados em {ano_base}.")

    # 2. Cruzamento Espacial Preciso (Overlay idêntico ao Script 07)
    print("Executando cruzamento espacial (Overlay) para rastrear transições exatas...")
    df1_sub = grandes_florestas[['geometry']].copy()
    df1_sub['cat_ano1'] = 'Floresta'
    
    df2_sub = df2[['geometry']].copy()
    if 'class_name' in df2.columns:
        df2_sub['cat_ano2'] = df2['class_name'].apply(lambda x: 'Floresta' if 'Floresta' in str(x) else 'Nao_Floresta')
    else:
        df2_sub['cat_ano2'] = df2['class_id'].apply(lambda x: 'Floresta' if x == 3 else 'Nao_Floresta')

    overlap = gpd.overlay(df1_sub, df2_sub, how='intersection', keep_geom_type=True)
    overlap['status_alvo'] = overlap['cat_ano2'] # 'Floresta' (Permanente) ou 'Nao_Floresta' (Supressão)
    overlap['area_m2'] = overlap.geometry.area

    tot_permanente = overlap[overlap['status_alvo'] == 'Floresta']['area_m2'].sum() / 10000.0
    tot_supressao = overlap[overlap['status_alvo'] == 'Nao_Floresta']['area_m2'].sum() / 10000.0

    print(f"📊 Estatísticas Validadas de Transição:")
    print(f"   - Permaneceu Floresta ({ano_base} -> {ano_alvo}): {tot_permanente:.2f} ha")
    print(f"   - Converteu para Não-Floresta / Supressão: {tot_supressao:.2f} ha")

    # 3. Leitura do Raster e Máscara Anti-Nuvens (Lógica do Script 09)
    print("Processando imagem raster e aplicando filtro anti-nuvem...")
    with rasterio.open(path_sat) as src:
        sat_meta = src.meta.copy()
        sat_img = src.read()
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width

    overlap_raster_crs = overlap.to_crs(crs)

    if sat_img.shape[0] >= 3:
        r, g, b = sat_img[0], sat_img[1], sat_img[2]
        max_val = np.max(sat_img)
        limiar_nuvem = max_val * 0.82 if max_val > 255 else 215
        is_cloud = (r >= limiar_nuvem) & (g >= limiar_nuvem) & (b >= limiar_nuvem)
    else:
        is_cloud = np.zeros((height, width), dtype=bool)

    shapes_interesse = [(geom, 1) for geom in overlap_raster_crs.geometry]
    mask_interesse = rasterize(shapes_interesse, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8)

    mask_valida = (mask_interesse == 1) & (~is_cloud)

    # 4. Segmentação por Superpixels (SLIC)
    print("Executando segmentação por superpixels (SLIC) nas zonas validadas...")
    rgb_norm = np.zeros((3, height, width), dtype=np.float32)
    for b in range(min(3, sat_img.shape[0])):
        band = sat_img[b].astype(np.float32)
        p2, p98 = np.percentile(band[band > 0], (2, 98)) if np.any(band > 0) else (0, 1)
        rgb_norm[b] = np.clip((band - p2) / (p98 - p2), 0, 1)

    sat_slic_input = np.moveaxis(rgb_norm, 0, -1)
    
    segments = slic(
        sat_slic_input,
        n_segments=max(50, int((height * width) / 5000)),
        compactness=15.0,
        mask=mask_valida,
        convert2lab=False,
        max_num_iter=10,
        start_label=1
    )
    segments[~mask_valida] = 0

    # 5. Avaliação de Qualidade (HoR, Tamanho e Contagem por Classe)
    print("Calculando métricas de HoR e contagem de segmentos...")
    shapes_classe_alvo = []
    for _, row in overlap_raster_crs.iterrows():
        val_cls = 1 if row['status_alvo'] == 'Floresta' else 2
        shapes_classe_alvo.append((row.geometry, val_cls))

    raster_classe_alvo = rasterize(
        shapes_classe_alvo, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8
    )

    ids_unicos = np.unique(segments)
    ids_unicos = ids_unicos[ids_unicos > 0]
    slices = find_objects(segments)

    metricas_segmentos = []
    pixel_area_m2 = abs(transform[0]) * abs(transform[4])

    for sp_id in ids_unicos:
        slc = slices[sp_id - 1]
        if slc is None: continue

        mask_sp = (segments[slc] == sp_id)
        classe_pixels = raster_classe_alvo[slc][mask_sp]
        classe_pixels = classe_pixels[classe_pixels > 0]
        
        if len(classe_pixels) == 0:
            continue

        npixels = len(classe_pixels)
        n_floresta = np.sum(classe_pixels == 1)
        n_nao_floresta = np.sum(classe_pixels == 2)

        if n_floresta >= n_nao_floresta:
            cls_majoritaria = 'Floresta'
            max_pixels = n_floresta
        else:
            cls_majoritaria = 'Nao_Floresta'
            max_pixels = n_nao_floresta

        hor = max_pixels / npixels

        metricas_segmentos.append({
            'segment_id': sp_id,
            'npixels': npixels,
            'area_m2': npixels * pixel_area_m2,
            'hor': hor,
            'classe': cls_majoritaria
        })

    df_metricas = pd.DataFrame(metricas_segmentos)
    if df_metricas.empty:
        print("⚠️ Nenhum segmento válido gerado.")
        sys.exit(0)

    # 6. Geração do Relatório Textual
    relatorio_path = os.path.join(reports_dir, f"{code_muni}_relatorio_qualidade_segmentacao_{ano_base}_vs_{ano_alvo}.txt")
    
    resumo_linhas = [
        "=" * 115,
        f"RELATÓRIO DE QUALIDADE DA SEGMENTAÇÃO E TRANSIÇÃO FLORESTAL ({ano_base} vs {ano_alvo})",
        f"Município IBGE: {code_muni}",
        "=" * 115,
        f"Área Permanente (Mantida): {tot_permanente:.2f} ha",
        f"Área Convertida (Supressão): {tot_supressao:.2f} ha",
        "-" * 115,
        f"{'CLASSE':<15} | {'QTD SEGMENTOS':<15} | {'MÉDIA PÍXELS':<15} | {'MÉDIA ÁREA (m²)':<18} | {'MEDIANA HoR':<15} | {'HoR MÍN/MÁX':<20}",
        "-" * 115
    ]

    for cls_name in ['Floresta', 'Nao_Floresta']:
        subset = df_metricas[df_metricas['classe'] == cls_name]
        qtd = len(subset)
        if qtd > 0:
            media_px = subset['npixels'].mean()
            media_area = subset['area_m2'].mean()
            med_hor = subset['hor'].median()
            min_hor = subset['hor'].min()
            max_hor = subset['hor'].max()
            linha = f"{cls_name:<15} | {qtd:<15} | {media_px:<15.1f} | {media_area:<18.1f} | {med_hor:<15.2f} | {min_hor:.2f} / {max_hor:.2f}"
        else:
            linha = f"{cls_name:<15} | {0:<15} | {0:<15.1f} | {0.0:<18.1f} | {0.0:<15.2f} | 0.00 / 0.00"
        resumo_linhas.append(linha)

    resumo_linhas.append("=" * 115)
    with open(relatorio_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(resumo_linhas))

    print("\n".join(resumo_linhas))
    print(f"\n[SUCESSO] Relatório salvo em:\n-> {relatorio_path}")

    # 7. Geração do Mapa Geopolítico com Cores Corrigidas (Floresta=Vermelho, Não-Floresta=Azul)
    print("\nGerando mapa geopolítico global com superpixels coloridos (Floresta = Vermelho, Não-Floresta = Azul)...")
    mapa_classes = dict(zip(df_metricas['segment_id'], df_metricas['classe']))

    rgb_full = np.zeros((3, height, width), dtype=np.uint8)
    for b in range(min(3, sat_img.shape[0])):
        band = sat_img[b].astype(np.float32)
        p2, p98 = np.percentile(band[band > 0], (2, 98)) if np.any(band > 0) else (0, 1)
        rgb_full[b] = np.clip((band - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)

    img_visual = np.moveaxis(rgb_full, 0, -1).copy()

    for sp_id in ids_unicos:
        slc = slices[sp_id - 1]
        if slc is None: continue
        
        mask_sp = (segments[slc] == sp_id)
        if not np.any(mask_sp): continue

        classe = mapa_classes.get(sp_id, 'Floresta')
        
        # 🎯 Regra de Cores Exata:
        # Floresta (Permanente) = Vermelho [255, 0, 0]
        # Não-Floresta (Supressão) = Azul [0, 0, 255]
        if classe == 'Floresta':
            cor_borda = np.array([255, 0, 0], dtype=np.uint8)
        else:
            cor_borda = np.array([0, 0, 255], dtype=np.uint8)

        # Borda com dilatação morfológica para alta visibilidade no QGIS
        borda_sp = find_boundaries(mask_sp, mode='outer')
        borda_grossa = binary_dilation(borda_sp, iterations=2)

        sub_img = img_visual[slc[0], slc[1]]
        sub_img[borda_grossa] = cor_borda

    mapa_saida_path = os.path.join(segmentation_dir, f"{code_muni}_mapa_superpixels_coloridos_{ano_base}_vs_{ano_alvo}.tif")

    sat_meta.update({"dtype": rasterio.uint8, "count": 3, "photometric": "RGB"})
    with rasterio.open(mapa_saida_path, "w", **sat_meta) as dst:
        for b in range(3):
            dst.write(img_visual[..., b], b + 1)

    print(f"✅ Mapa global salvo em:\n-> {mapa_saida_path}")

if __name__ == "__main__":
    main()