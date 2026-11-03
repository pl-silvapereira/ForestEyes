import sys
import os
import duckdb

# Correção de compatibilidade entre DuckDB e geobr no Python 3.13
if not hasattr(duckdb.DuckDBPyRelation, "fetchone"):
    duckdb.DuckDBPyRelation.fetchone = lambda self: self.fetchall()[0]

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from skimage.segmentation import slic
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from dotenv import load_dotenv

def gerar_estilo_qml_gt_u(caminho_qml):
    """Gera QML para GT-U: Verde (Vegetação), Vermelho (Não-Vegetação), Azul (Indefinido), Preto (Mascara)."""
    conteudo = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.0" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer type="paletted" opacity="1" alphaBand="-1" band="1">
      <colorPalette>
        <paletteEntry value="0" color="#000000" label="Mascara / Fora"/>
        <paletteEntry value="1" color="#33a02c" label="Vegetacao (Verde)"/>
        <paletteEntry value="2" color="#e31a1c" label="Nao-Vegetacao (Vermelho)"/>
        <paletteEntry value="3" color="#1f78b4" label="Indefinido (Azul)"/>
      </colorPalette>
    </rasterrenderer>
  </pipe>
  <blendMode>0</blendMode>
</qgis>"""
    with open(caminho_qml, 'w', encoding='utf-8') as f:
        f.write(conteudo)

def gerar_estilo_qml_gt_m(caminho_qml):
    """Gera QML para GT-MapBiomas e GT-M: Verde (Vegetação), Vermelho (Não-Vegetação), Preto (Mascara)."""
    conteudo = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.0" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer type="paletted" opacity="1" alphaBand="-1" band="1">
      <colorPalette>
        <paletteEntry value="0" color="#000000" label="Mascara / Fora"/>
        <paletteEntry value="1" color="#33a02c" label="Vegetacao (Verde)"/>
        <paletteEntry value="2" color="#e31a1c" label="Nao-Vegetacao (Vermelho)"/>
      </colorPalette>
    </rasterrenderer>
  </pipe>
  <blendMode>0</blendMode>
</qgis>"""
    with open(caminho_qml, 'w', encoding='utf-8') as f:
        f.write(conteudo)

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 07-gerarConjuntosVerdade.py <code_muni> <ano_inicio> <ano_fim>")
        sys.exit(1)

    code_muni = str(sys.argv[1])
    ano_inicio = str(sys.argv[2])
    ano_fim = str(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    dir_input_mb = os.path.join(project_root, "data", "input", "MapBiomas", ano_fim)
    dir_pan = os.path.join(project_root, "data", "output", "pansharpening", ano_fim)
    dir_seg = os.path.join(project_root, "data", "output", "segmentation", ano_fim)
    dir_reports = os.path.join(project_root, "reports")
    dir_gt_out = os.path.join(project_root, "data", "output", "ground_truth", ano_fim)
    
    os.makedirs(dir_gt_out, exist_ok=True)
    os.makedirs(dir_reports, exist_ok=True)

    path_cbers_2m = os.path.join(dir_pan, f"{code_muni}_{ano_fim}_CBERS_TRUE_COLOR_2M.tif")
    path_shp_3classes = os.path.join(dir_seg, f"{code_muni}_Segmentacao_3Classes_{ano_fim}.shp")
    
    mb_files = [f for f in os.listdir(dir_input_mb) if f.endswith(f"_{ano_fim}.tif")] if os.path.exists(dir_input_mb) else []
    if not mb_files:
        print(f"[ERRO CRÍTICO] Raster do MapBiomas não encontrado em {dir_input_mb}")
        sys.exit(1)
    path_mapbiomas = os.path.join(dir_input_mb, mb_files[0])

    print("=" * 80)
    print(f"🎯 SCRIPT 07 - GERAÇÃO DOS CONJUNTOS VERDADE (VETORIZADO HORT/GT) ({ano_fim})")
    print("=" * 80)

    # 1. Carregar grade de referência CBERS 2m
    print("1. Lendo grade de referência espacial da cena CBERS-4A (2m)...", flush=True)
    with rasterio.open(path_cbers_2m) as src_cbers:
        meta_cbers = src_cbers.meta.copy()
        cbers_crs = src_cbers.crs
        cbers_transform = src_cbers.transform
        height, width = src_cbers.height, src_cbers.width
        rgb_raw = src_cbers.read([1, 2, 3])

    # 2. Binarizar MapBiomas (GT-MapBiomas)
    print("2. Gerando conjunto GT-MapBiomas (Binarização: 1=Vegetação, 2=Não-Vegetação)...", flush=True)
    gt_mapbiomas_2m = np.zeros((height, width), dtype=np.uint8)

    with rasterio.open(path_mapbiomas) as src_mb:
        mb_data_raw = np.zeros((height, width), dtype=src_mb.dtypes[0])
        reproject(
            source=rasterio.band(src_mb, 1),
            destination=mb_data_raw,
            src_transform=src_mb.transform,
            src_crs=src_mb.crs,
            dst_transform=cbers_transform,
            dst_crs=cbers_crs,
            resampling=Resampling.nearest
        )
        
        mask_veg = (mb_data_raw == 3) | (mb_data_raw == 4) | (mb_data_raw == 5) | (mb_data_raw == 49)
        mask_nonveg = (mb_data_raw > 0) & (~mask_veg)
        
        gt_mapbiomas_2m[mask_veg] = 1
        gt_mapbiomas_2m[mask_nonveg] = 2

    # 3. Gerar Superpixels via MaskSLIC
    print("3. Executando MaskSLIC sobre áreas 'Floresta' e 'Nao_Floresta'...", flush=True)
    gdf_3classes = gpd.read_file(path_shp_3classes).to_crs(cbers_crs)
    gdf_seg_areas = gdf_3classes[gdf_3classes['class_name'].isin(['Floresta', 'Nao_Floresta'])]
    
    mask_seg_domain = rasterize(
        [(geom, 1) for geom in gdf_seg_areas.geometry],
        out_shape=(height, width), transform=cbers_transform, fill=0, dtype=np.uint8
    )

    rgb_norm = np.moveaxis(rgb_raw, 0, -1).astype(np.float32)
    max_val = rgb_norm.max()
    if max_val > 0: 
        rgb_norm /= max_val

    num_segments = max(int(np.sum(mask_seg_domain) / 250.0), 10)
    
    superpixels = slic(
        rgb_norm, n_segments=num_segments, compactness=10,
        mask=(mask_seg_domain == 1), start_label=1, max_num_iter=5
    )

    # 4. Cálculo Vetorizado da Taxa de Homogeneidade (HoR), GT-U e GT-M (Sem Laço For)
    print("4. Calculando HoR, GT-U (HoR >= 0.7) e GT-M (HoR > 0.5) via histograma vetorizado...", flush=True)
    sp_flat = superpixels.ravel()
    gt_flat = gt_mapbiomas_2m.ravel()

    valid = sp_flat > 0
    sp_v = sp_flat[valid]
    gt_v = gt_flat[valid]

    max_lbl = int(sp_v.max()) if len(sp_v) > 0 else 0

    if max_lbl > 0:
        count_veg = np.bincount(sp_v[gt_v == 1], minlength=max_lbl + 1)
        count_nonveg = np.bincount(sp_v[gt_v == 2], minlength=max_lbl + 1)
        count_total = np.bincount(sp_v, minlength=max_lbl + 1)

        safe_total = np.where(count_total == 0, 1, count_total)
        prop_veg = count_veg / safe_total
        prop_nonveg = count_nonveg / safe_total
        hor = np.maximum(prop_veg, prop_nonveg)

        # Vetorização das Regras GT-U (1=Veg, 2=Não-Veg, 3=Indefinido)
        gt_u_map = np.zeros(max_lbl + 1, dtype=np.uint8)
        mask_active = count_total > 0
        gt_u_map[mask_active & (hor < 0.7)] = 3
        gt_u_map[mask_active & (hor >= 0.7) & (prop_veg >= 0.7)] = 1
        gt_u_map[mask_active & (hor >= 0.7) & (prop_veg < 0.7)] = 2

        # Vetorização das Regras GT-M (1=Veg, 2=Não-Veg)
        gt_m_map = np.zeros(max_lbl + 1, dtype=np.uint8)
        gt_m_map[mask_active & (prop_veg > 0.5)] = 1
        gt_m_map[mask_active & (prop_veg <= 0.5)] = 2

        gt_u = gt_u_map[superpixels]
        gt_m = gt_m_map[superpixels]
    else:
        gt_u = np.zeros((height, width), dtype=np.uint8)
        gt_m = np.zeros((height, width), dtype=np.uint8)

    # 5. Salvar Rasters GeoTIFF e QML
    print("5. Gravando rasters e estilos QML no disco...", flush=True)
    meta_out = meta_cbers.copy()
    meta_out.update(count=1, dtype=rasterio.uint8, nodata=0)

    path_gt_mb = os.path.join(dir_gt_out, f"{code_muni}_GT_MapBiomas_{ano_fim}.tif")
    path_gt_u = os.path.join(dir_gt_out, f"{code_muni}_GT_U_{ano_fim}.tif")
    path_gt_m = os.path.join(dir_gt_out, f"{code_muni}_GT_M_{ano_fim}.tif")

    with rasterio.open(path_gt_mb, 'w', **meta_out) as dst: dst.write(gt_mapbiomas_2m, 1)
    with rasterio.open(path_gt_u, 'w', **meta_out) as dst: dst.write(gt_u, 1)
    with rasterio.open(path_gt_m, 'w', **meta_out) as dst: dst.write(gt_m, 1)

    gerar_estilo_qml_gt_m(os.path.join(dir_gt_out, f"{code_muni}_GT_MapBiomas_{ano_fim}.qml"))
    gerar_estilo_qml_gt_u(os.path.join(dir_gt_out, f"{code_muni}_GT_U_{ano_fim}.qml"))
    gerar_estilo_qml_gt_m(os.path.join(dir_gt_out, f"{code_muni}_GT_M_{ano_fim}.qml"))

    # 6. Gerar Figura 8 (4 Painéis ABNT)
    print("6. Gerando painel comparativo da Figura 8...", flush=True)
    center_y, center_x = height // 2, width // 2
    crop_h, crop_w = min(1200, height), min(2400, width)
    
    y1 = max(0, center_y - crop_h // 2)
    y2 = min(height, y1 + crop_h)
    x1 = max(0, center_x - crop_w // 2)
    x2 = min(width, x1 + crop_w)

    rgb_crop = np.moveaxis(rgb_raw[:, y1:y2, x1:x2], 0, -1).astype(np.float32)
    p2, p98 = np.percentile(rgb_crop[rgb_crop > 0], (2, 98))
    rgb_crop = np.clip((rgb_crop - p2) / (p98 - p2), 0, 1)

    gt_mb_crop = gt_mapbiomas_2m[y1:y2, x1:x2]
    gt_u_crop = gt_u[y1:y2, x1:x2]
    gt_m_crop = gt_m[y1:y2, x1:x2]

    cmap_m_mb = ListedColormap(['#000000', '#33a02c', '#e31a1c'])
    cmap_u = ListedColormap(['#000000', '#33a02c', '#e31a1c', '#1f78b4'])

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=300)
    plt.subplots_adjust(wspace=0.08, hspace=0.15)

    axes[0, 0].imshow(rgb_crop)
    axes[0, 0].set_title("(a) Recorte da cena orbital CBERS-4A WPM (2m RGB)", fontsize=11, fontweight='bold', pad=8)
    
    axes[0, 1].imshow(gt_mb_crop, cmap=cmap_m_mb, vmin=0, vmax=2)
    axes[0, 1].set_title("(b) GT-MapBiomas (Binarizado: Vegetação vs Não-Vegetação)", fontsize=11, fontweight='bold', pad=8)

    axes[1, 0].imshow(gt_u_crop, cmap=cmap_u, vmin=0, vmax=3)
    axes[1, 0].set_title("(c) GT-U (Verdade com Indefinido - HoR < 0,7 em Azul)", fontsize=11, fontweight='bold', pad=8)

    axes[1, 1].imshow(gt_m_crop, cmap=cmap_m_mb, vmin=0, vmax=2)
    axes[1, 1].set_title("(d) GT-M (Verdade com Maioria - HoR > 0,5 Estrito)", fontsize=11, fontweight='bold', pad=8)

    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1.0)
        ax.set_xticks([])
        ax.set_yticks([])

    patch_veg = mpatches.Patch(color='#33a02c', label='Vegetação')
    patch_nonveg = mpatches.Patch(color='#e31a1c', label='Não-Vegetação (Urbano/Solo Expsto)')
    patch_undef = mpatches.Patch(color='#1f78b4', label='Indefinido (HoR < 0,7)')

    fig.legend(
        handles=[patch_veg, patch_nonveg, patch_undef],
        loc='lower center', bbox_to_anchor=(0.5, 0.02),
        ncol=3, frameon=True, facecolor='white', edgecolor='black', fontsize=10
    )

    path_figura8 = os.path.join(dir_reports, f"Figura_8_Conjuntos_Verdade_{code_muni}_{ano_fim}.png")
    plt.savefig(path_figura8, bbox_inches='tight', pad_inches=0.2)
    plt.close()

    print(f"\n✅ SUCESSO! Conjuntos verdade concluídos em instantes:")
    print(f"   -> Imagem da Figura 8: {path_figura8}")
    print(f"   -> Raster GT-MapBiomas: {path_gt_mb}")
    print(f"   -> Raster GT-U: {path_gt_u}")
    print(f"   -> Raster GT-M: {path_gt_m}")

if __name__ == "__main__":
    main()