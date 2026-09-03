import os
import sys
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize, shapes
import geopandas as gpd
import shapely.geometry
from skimage.segmentation import slic, find_boundaries
from scipy.ndimage import find_objects, binary_dilation
from dotenv import load_dotenv

def main():
    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    # Diretórios de saída
    output_dir = os.path.join(project_root, "data", "output", "analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Caminhos de entrada especificados
    path_mapbiomas = os.path.join(project_root, "data", "input", "MapBiomas", "2023", "mapbiomas_lulc_10m_sao_jose_dos_campos_2023.tif")
    path_cbers = os.path.join(project_root, "data", "output", "pansharpening", "geopolitic-RGBN", "2024", "3549904_2024_CBERS_TRUE_COLOR_CLIPPED.tif")
    path_shp_2023 = os.path.join(project_root, "data", "output", "classification", "2023", "3549904_Classificado_ForestEyes_2023.shp")
    path_shp_2024 = os.path.join(project_root, "data", "output", "classification", "2024", "3549904_Classificado_ForestEyes_2024.shp")

    for p in [path_mapbiomas, path_cbers, path_shp_2023, path_shp_2024]:
        if not os.path.exists(p):
            print(f"[ERRO CRÍTICO] Arquivo não encontrado: {p}")
            sys.exit(1)

    print("=" * 115)
    print("🌲 ANÁLISE ESTRATÉGICA AVANÇADA DE FLORESTA (MAPBIOMAS 2023 + CBERS 2024)")
    print("=" * 115)

    # =========================================================================
    # PASSO 1: Máscara ROI MapBiomas 2023 (Apenas Vegetação / Floresta)
    # =========================================================================
    print("Passo 1: Processando raster do MapBiomas 2023 para isolar classe de Floresta...")
    with rasterio.open(path_mapbiomas) as src_mb:
        mb_meta = src_mb.meta.copy()
        mb_data = src_mb.read(1)
        mb_transform = src_mb.transform
        mb_crs = src_mb.crs

    mask_floresta_mb = np.isin(mb_data, [3, 4, 5, 6])

    # =========================================================================
    # PASSO 2: Projeção na imagem CBERS 2024 (Área de Floresta Transparente)
    # =========================================================================
    print("Passo 2: Projetando máscara na imagem CBERS 2024 e gerando analysis_florestation_2024.tif...")
    with rasterio.open(path_cbers) as src_cbers:
        cbers_meta = src_cbers.meta.copy()
        cbers_img = src_cbers.read()
        cbers_transform = src_cbers.transform
        cbers_crs = src_cbers.crs
        height, width = src_cbers.height, src_cbers.width

    mask_reprojected = np.zeros((height, width), dtype=np.uint8)
    reproject(
        source=mask_floresta_mb.astype(np.uint8),
        destination=mask_reprojected,
        src_transform=mb_transform,
        src_crs=mb_crs,
        dst_transform=cbers_transform,
        dst_crs=cbers_crs,
        resampling=Resampling.nearest
    )

    is_forest_2024_grid = (mask_reprojected == 1)

    rgb_normalized = np.zeros((3, height, width), dtype=np.uint8)
    for b in range(min(3, cbers_img.shape[0])):
        band = cbers_img[b].astype(np.float32)
        p2, p98 = np.percentile(band[band > 0], (2, 98)) if np.any(band > 0) else (0, 1)
        rgb_normalized[b] = np.clip((band - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)

    out_step2_bands = np.zeros((4, height, width), dtype=np.uint8)
    for b in range(3):
        out_step2_bands[b] = np.where(is_forest_2024_grid, rgb_normalized[b], 0)
    out_step2_bands[3] = np.where(is_forest_2024_grid, 255, 0)

    step2_meta = cbers_meta.copy()
    step2_meta.update({"count": 4, "dtype": rasterio.uint8, "nodata": 0})

    path_step2_out = os.path.join(output_dir, "analysis_florestation_2024.tif")
    with rasterio.open(path_step2_out, "w", **step2_meta) as dst:
        dst.write(out_step2_bands)

    # =========================================================================
    # PASSO 3: Geração do Shapefile (.shp) de Classificação 2023 + Raster .tif
    # =========================================================================
    print("Passo 3: Criando Shapefile e Raster baseados nas classes de 2023 (Floresta=#006400, Demais=#A9A9A9)...")
    gdf_2023 = gpd.read_file(path_shp_2023)
    if gdf_2023.crs != cbers_crs:
        gdf_2023 = gdf_2023.to_crs(cbers_crs)

    # Atribuir cores e nomes baseados na categoria [3]
    class_names = []
    hex_colors = []
    for _, row in gdf_2023.iterrows():
        c_name = str(row.get('class_name', row.get('class_id', '')))
        if '3' in c_name or 'Floresta' in c_name:
            class_names.append('Floresta')
            hex_colors.append('#006400')  # Verde escuro
        else:
            class_names.append('Outros')
            hex_colors.append('#A9A9A9')  # Cinza

    gdf_2023['analise_cls'] = class_names
    gdf_2023['cor_hex'] = hex_colors

    # Salvar o .shp intermediário solicitado
    path_shp_out = os.path.join(output_dir, "analysis_florestation_2024_based_class.shp")
    gdf_2023.to_file(path_shp_out)
    print(f"-> Shapefile salvo em: {path_shp_out}")

    # Rasterizar o Shapefile gerado para criar analysis_florestation_2024_based_class.tif
    shapes_class_3 = []
    for _, row in gdf_2023.iterrows():
        val = 1 if row['analise_cls'] == 'Floresta' else 2
        shapes_class_3.append((row.geometry, val))

    raster_class_3 = rasterize(
        shapes_class_3, out_shape=(height, width), transform=cbers_transform, fill=2, dtype=np.uint8
    )

    img_class_3 = np.zeros((3, height, width), dtype=np.uint8)
    cond_floresta = (raster_class_3 == 1) & is_forest_2024_grid
    cond_demais = (raster_class_3 == 2) & is_forest_2024_grid

    for b, val in enumerate([0, 100, 0]): # #006400
        img_class_3[b] = np.where(cond_floresta, val, img_class_3[b])
    for b, val in enumerate([169, 169, 169]): # #A9A9A9
        img_class_3[b] = np.where(cond_demais, val, img_class_3[b])

    # Nuvens e Sombras pintadas de preto
    if cbers_img.shape[0] >= 3:
        r, g, b = cbers_img[0], cbers_img[1], cbers_img[2]
        max_val = np.max(cbers_img)
        limiar_nuvem = max_val * 0.82 if max_val > 255 else 215
        is_cloud_or_shadow = (r >= limiar_nuvem) & (g >= limiar_nuvem) & (b >= limiar_nuvem)
    else:
        is_cloud_or_shadow = np.zeros((height, width), dtype=bool)

    for b in range(3):
        img_class_3[b] = np.where(is_cloud_or_shadow, 0, img_class_3[b])

    step3_bands = np.array([
        img_class_3[0],
        img_class_3[1],
        img_class_3[2],
        np.where(is_cloud_or_shadow | ~is_forest_2024_grid, 0, 255)
    ], dtype=np.uint8)

    path_step3_out = os.path.join(output_dir, "analysis_florestation_2024_based_class.tif")
    with rasterio.open(path_step3_out, "w", **step2_meta) as dst:
        dst.write(step3_bands)
    print(f"-> Raster TIFF salvo em: {path_step3_out}")

    # =========================================================================
    # PASSO 5 & 6: Segmentação com Cores nos Segmentos (Amarelo = Floresta, Vermelho = Não-Floresta)
    # =========================================================================
    print("Passo 5 & 6: Executando segmentação SLIC nas áreas verdes com preenchimento/contorno colorido...")
    gdf_2024 = gpd.read_file(path_shp_2024)
    if gdf_2024.crs != cbers_crs:
        gdf_2024 = gdf_2024.to_crs(cbers_crs)

    shapes_2024 = []
    for _, row in gdf_2024.iterrows():
        c_name = str(row.get('class_name', row.get('class_id', '')))
        val_cls = 1 if ('3' in c_name or 'Floresta' in c_name) else 2
        shapes_2024.append((row.geometry, val_cls))

    raster_2024 = rasterize(
        shapes_2024, out_shape=(height, width), transform=cbers_transform, fill=2, dtype=np.uint8
    )

    # Executar SLIC apenas onde é a classe verde de 2023 (cond_floresta)
    slic_input = np.moveaxis(rgb_normalized, 0, -1).astype(np.float32)
    segments = slic(
        slic_input,
        n_segments=max(50, int((height * width) / 4000)),
        compactness=15.0,
        mask=cond_floresta,
        convert2lab=False,
        max_num_iter=10,
        start_label=1
    )
    segments[~cond_floresta] = 0

    ids_unicos = np.unique(segments)
    ids_unicos = ids_unicos[ids_unicos > 0]
    slices = find_objects(segments)

    # Preparar matrizes visuais para Passo 5 (CBERS Colorido) e Passo 6 (Escala de Cinza / NDVI)
    img_seg_transposed_color = np.moveaxis(rgb_normalized.copy(), 0, -1)
    
    gray_bg = np.mean(rgb_normalized, axis=0).astype(np.uint8)
    gray_stack = np.array([gray_bg, gray_bg, gray_bg], dtype=np.uint8)
    img_seg_transposed_gray = np.moveaxis(gray_stack, 0, -1)

    for sp_id in ids_unicos:
        slc = slices[sp_id - 1]
        if slc is None: continue
        mask_sp = (segments[slc] == sp_id)
        if not np.any(mask_sp): continue

        pixels_cls = raster_2024[slc][mask_sp]
        pixels_cls = pixels_cls[pixels_cls > 0]
        if len(pixels_cls) == 0: continue

        is_floresta_24 = np.sum(pixels_cls == 1) >= np.sum(pixels_cls == 2)
        
        # Amarelo [255, 255, 0] para Floresta [3], Vermelho [255, 0, 0] para Diferente
        cor_segmento = [255, 255, 0] if is_floresta_24 else [255, 0, 0]

        borda_sp = find_boundaries(mask_sp, mode='outer')
        borda_grossa = binary_dilation(borda_sp, iterations=1)

        # Aplicar cor translúcida/preenchimento suave + borda forte nos segmentos
        for target_img in [img_seg_transposed_color, img_seg_transposed_gray]:
            sub_img = target_img[slc[0], slc[1]]
            # Preenchimento translúcido leve (mistura 30% da cor com 70% da imagem base)
            sub_img[mask_sp] = (0.7 * sub_img[mask_sp].astype(float) + 0.3 * np.array(cor_segmento)).astype(np.uint8)
            # Borda sólida colorida para destaque absoluto
            sub_img[borda_grossa] = cor_segmento

    # Salvar Passo 5 (Sobre CBERS Colorido)
    img_seg_final_color = np.moveaxis(img_seg_transposed_color, -1, 0)
    step5_bands = np.array([
        img_seg_final_color[0], img_seg_final_color[1], img_seg_final_color[2],
        np.where(cond_floresta, 255, 0)
    ], dtype=np.uint8)

    path_step5_out = os.path.join(output_dir, "analysis_florestation_2024_segmentation_class.tif")
    with rasterio.open(path_step5_out, "w", **step2_meta) as dst:
        dst.write(step5_bands)
    print(f"-> Salvo (Passo 5): {path_step5_out}")

    # Salvar Passo 6 (Sobre Escala de Cinza / NDVI)
    img_seg_final_gray = np.moveaxis(img_seg_transposed_gray, -1, 0)
    step6_bands = np.array([
        img_seg_final_gray[0], img_seg_final_gray[1], img_seg_final_gray[2],
        np.where(cond_floresta, 255, 0)
    ], dtype=np.uint8)

    path_step6_out = os.path.join(output_dir, "analysis_florestation_2024_segmentation_class_ndvi.tif")
    with rasterio.open(path_step6_out, "w", **step2_meta) as dst:
        dst.write(step6_bands)
    print(f"-> Salvo (Passo 6): {path_step6_out}")
    print("\n[SUCESSO] Processo concluído com Shapefile e Superpixels coloridos gerados com sucesso!")

if __name__ == "__main__":
    main()