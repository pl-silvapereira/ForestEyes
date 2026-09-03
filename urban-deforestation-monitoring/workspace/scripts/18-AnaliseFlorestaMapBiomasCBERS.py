import os
import sys
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
import geopandas as gpd
from skimage.segmentation import slic, find_boundaries
from scipy.ndimage import find_objects, binary_dilation
from dotenv import load_dotenv

def main():
    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    # Diretórios de entrada e saída
    output_dir = os.path.join(project_root, "data", "output", "analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Caminhos definidos conforme especificação
    path_mapbiomas = os.path.join(project_root, "data", "input", "MapBiomas", "2023", "mapbiomas_lulc_10m_sao_jose_dos_campos_2023.tif")
    path_cbers = os.path.join(project_root, "data", "output", "pansharpening", "geopolitic-RGBN", "2024", "3549904_2024_CBERS_TRUE_COLOR_CLIPPED.tif")
    path_shp = os.path.join(project_root, "data", "output", "classification", "2024", "3549904_Classificado_ForestEyes_2024.shp")

    if not os.path.exists(path_mapbiomas) or not os.path.exists(path_cbers) or not os.path.exists(path_shp):
        print("[ERRO CRÍTICO] Um ou mais arquivos de entrada não foram encontrados nos caminhos especificados.")
        sys.exit(1)

    print("=" * 115)
    print("🌲 EXECUTANDO ANÁLISE ESTRATÉGICA DE FLORESTA (MAPBIOMAS 2023 + CBERS 2024)")
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

    # No MapBiomas (Brasil Coleção), a classe 3 corresponde geralmente a Formação Florestal (Floresta)
    # Criamos a máscara binária: 1 onde é floresta (valor 3 ou conjunto florestal), 0 caso contrário (nuvens, sombra, demais classes)
    mask_floresta_mb = np.isin(mb_data, [3, 4, 5, 6]) # Coleta abrangente de formações florestais/savânicas

    # =========================================================================
    # PASSO 2: Projeção na imagem CBERS 2024 com transparência / fundo preto
    # =========================================================================
    print("Passo 2: Projetando máscara na imagem CBERS 2024 e gerando analysis_florestation_2024.tif...")
    with rasterio.open(path_cbers) as src_cbers:
        cbers_meta = src_cbers.meta.copy()
        cbers_img = src_cbers.read()
        cbers_transform = src_cbers.transform
        cbers_crs = src_cbers.crs
        height, width = src_cbers.height, src_cbers.width

    # Reprojetar/Redimensionar a máscara do MapBiomas para o grid exato do CBERS 2024
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

    # Pré-normalização global de contraste do CBERS para visualização fiel
    rgb_normalized = np.zeros((3, height, width), dtype=np.uint8)
    for b in range(min(3, cbers_img.shape[0])):
        band = cbers_img[b].astype(np.float32)
        p2, p98 = np.percentile(band[band > 0], (2, 98)) if np.any(band > 0) else (0, 1)
        rgb_normalized[b] = np.clip((band - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)

    # Aplicar máscara: dentro da floresta revela a imagem do satélite, fora fica preto com canal alfa transparente/opaco
    out_step2_bands = np.zeros((4, height, width), dtype=np.uint8)
    for b in range(3):
        out_step2_bands[b] = np.where(is_forest_2024_grid, rgb_normalized[b], 0)
    
    # Canal Alpha: 255 (opaco) na floresta revelada, 0 (transparente/preto) fora
    out_step2_bands[3] = np.where(is_forest_2024_grid, 255, 0)

    step2_meta = cbers_meta.copy()
    step2_meta.update({"count": 4, "dtype": rasterio.uint8, "nodata": 0})

    path_step2_out = os.path.join(output_dir, "analysis_florestation_2024.tif")
    with rasterio.open(path_step2_out, "w", **step2_meta) as dst:
        dst.write(out_step2_bands)

    print(f"-> Salvo em: {path_step2_out}")

    # =========================================================================
    # PASSO 3: Pintar com base no Shapefile de Classificação 2024
    # =========================================================================
    print("Passo 3: Pintando classes do Shapefile e gerando analysis_florestation_2024_based_class.tif...")
    gdf_class = gpd.read_file(path_shp)
    if gdf_class.crs != cbers_crs:
        gdf_class = gdf_class.to_crs(cbers_crs)

    # Mapear classes para valores inteiros e cores RGB
    # Floresta = Verde [0, 255, 0], Demais classes recebem outras cores distintas
    classes_unicas = gdf_class['class_name'].unique() if 'class_name' in gdf_class.columns else gdf_class['class_id'].unique()
    
    shapes_class = []
    for idx, row in gdf_class.iterrows():
        c_name = row['class_name'] if 'class_name' in gdf_class.columns else str(row['class_id'])
        # Atribuir IDs numéricos por categoria
        if 'Floresta' in str(c_name):
            val_id = 1 # Verde
        elif 'Agropecuaria' in str(c_name) or 'Pastagem' in str(c_name):
            val_id = 2 # Amarelo/Laranja
        elif 'Urbana' in str(c_name) or 'Infraestrutura' in str(c_name):
            val_id = 3 # Vermelho/Roxo
        else:
            val_id = 4 # Ciano/Outros
        shapes_class.append((row.geometry, val_id))

    raster_classes = rasterize(
        shapes_class, out_shape=(height, width), transform=cbers_transform, fill=0, dtype=np.uint8
    )

    img_class_colored = np.zeros((3, height, width), dtype=np.uint8)
    # Cores associadas:
    # 1: Floresta -> Verde [0, 255, 0]
    # 2: Agropecuária -> Laranja [255, 165, 0]
    # 3: Urbana -> Magenta [255, 0, 255]
    # 4: Outros -> Ciano [0, 255, 255]
    color_map = {
        1: [0, 255, 0],
        2: [255, 165, 0],
        3: [255, 0, 255],
        4: [0, 255, 255]
    }

    # Restringir coloração à máscara de floresta gerada no passo 2
    for cid, rgb in color_map.items():
        cond = (raster_classes == cid) & is_forest_2024_grid
        for b in range(3):
            img_class_colored[b] = np.where(cond, rgb[b], img_class_colored[b])

    step3_bands = np.array([
        img_class_colored[0],
        img_class_colored[1],
        img_class_colored[2],
        np.where(is_forest_2024_grid, 255, 0)
    ], dtype=np.uint8)

    path_step3_out = os.path.join(output_dir, "analysis_florestation_2024_based_class.tif")
    with rasterio.open(path_step3_out, "w", **step2_meta) as dst:
        dst.write(step3_bands)

    print(f"-> Salvo em: {path_step3_out}")

    # =========================================================================
    # PASSO 4: Segmentação de Superpixels restrita às áreas de floresta
    # =========================================================================
    print("Passo 4: Executando segmentação SLIC nas áreas de floresta e gerando analysis_florestation_2024_segementation.tif...")
    
    # Executar SLIC restrito exclusivamente à máscara de floresta validada
    slic_input = np.moveaxis(rgb_normalized, 0, -1).astype(np.float32)
    segments = slic(
        slic_input,
        n_segments=max(50, int((height * width) / 4000)),
        compactness=15.0,
        mask=is_forest_2024_grid,
        convert2lab=False,
        max_num_iter=10,
        start_label=1
    )
    segments[~is_forest_2024_grid] = 0

    # Desenhar contornos dos superpixels sobre a imagem original CBERS
    img_seg_visual = rgb_normalized.copy()
    img_seg_transposed = np.moveaxis(img_seg_visual, 0, -1)

    ids_unicos = np.unique(segments)
    ids_unicos = ids_unicos[ids_unicos > 0]
    slices = find_objects(segments)

    for sp_id in ids_unicos:
        slc = slices[sp_id - 1]
        if slc is None: continue
        mask_sp = (segments[slc] == sp_id)
        if not np.any(mask_sp): continue

        borda_sp = find_boundaries(mask_sp, mode='outer')
        borda_grossa = binary_dilation(borda_sp, iterations=1)
        
        sub_img = img_seg_transposed[slc[0], slc[1]]
        sub_img[borda_grossa] = [255, 255, 0] # Contornos amarelos de alta visibilidade

    img_seg_final = np.moveaxis(img_seg_transposed, -1, 0)
    step4_bands = np.array([
        img_seg_final[0],
        img_seg_final[1],
        img_seg_final[2],
        np.where(is_forest_2024_grid, 255, 0)
    ], dtype=np.uint8)

    path_step4_out = os.path.join(output_dir, "analysis_florestation_2024_segementation.tif")
    with rasterio.open(path_step4_out, "w", **step2_meta) as dst:
        dst.write(step4_bands)

    print(f"-> Salvo em: {path_step4_out}")
    print("\n[SUCESSO] Todas as etapas do novo script foram concluídas com êxito!")

if __name__ == "__main__":
    main()