import sys
import os
import gc
import warnings
import duckdb

# Correção de compatibilidade entre DuckDB e geobr no Python 3.13
if not hasattr(duckdb.DuckDBPyRelation, "fetchone"):
    duckdb.DuckDBPyRelation.fetchone = lambda self: self.fetchall()[0]

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from skimage.segmentation import slic
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message=".*One of the clusters is empty.*")
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

def gerar_estilo_qml_gt_u(caminho_qml):
    """Gera QML para GT-U: Verde (Floresta), Vermelho (Não-Floresta), Azul (Indefinido), Preto (Não Segmentar)."""
    conteudo = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.0" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer type="paletted" opacity="1" alphaBand="-1" band="1">
      <colorPalette>
        <paletteEntry value="0" color="#000000" label="Nao Segmentar (Preto)"/>
        <paletteEntry value="1" color="#33a02c" label="Floresta (Verde)"/>
        <paletteEntry value="2" color="#e31a1c" label="Nao Floresta (Vermelho)"/>
        <paletteEntry value="3" color="#1f78b4" label="Indefinido (Azul)"/>
      </colorPalette>
    </rasterrenderer>
  </pipe>
  <blendMode>0</blendMode>
</qgis>"""
    with open(caminho_qml, 'w', encoding='utf-8') as f:
        f.write(conteudo)

def gerar_estilo_qml_gt_m(caminho_qml):
    """Gera QML para GT-MapBiomas e GT-M: Verde (Floresta), Vermelho (Não-Floresta), Preto (Não Segmentar)."""
    conteudo = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.0" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer type="paletted" opacity="1" alphaBand="-1" band="1">
      <colorPalette>
        <paletteEntry value="0" color="#000000" label="Nao Segmentar (Preto)"/>
        <paletteEntry value="1" color="#33a02c" label="Floresta (Verde)"/>
        <paletteEntry value="2" color="#e31a1c" label="Nao Floresta (Vermelho)"/>
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
    
    if not os.path.exists(path_shp_3classes):
        print(f"[ERRO CRÍTICO] Máscara de segmentação não encontrada: {path_shp_3classes}")
        sys.exit(1)

    mb_files = [f for f in os.listdir(dir_input_mb) if f.endswith(f"_{ano_fim}.tif")] if os.path.exists(dir_input_mb) else []
    if not mb_files:
        print(f"[ERRO CRÍTICO] Raster do MapBiomas não encontrado em {dir_input_mb}")
        sys.exit(1)
    path_mapbiomas = os.path.join(dir_input_mb, mb_files[0])

    print("=" * 80, flush=True)
    print(f"🎯 SCRIPT 07 - GERAÇÃO DOS CONJUNTOS VERDADE (PADRÃO FORESTEYES) ({ano_fim})", flush=True)
    print("=" * 80, flush=True)

    print("1. Lendo grade espacial da cena CBERS-4A (2m)...", flush=True)
    with rasterio.open(path_cbers_2m) as src_cbers:
        meta_cbers = src_cbers.meta.copy()
        cbers_crs = src_cbers.crs
        cbers_transform = src_cbers.transform
        height, width = src_cbers.height, src_cbers.width

    print("2. Rasterizando área de interesse (Floresta + Nao_Floresta)...", flush=True)
    gdf_3classes = gpd.read_file(path_shp_3classes).to_crs(cbers_crs)
    gdf_seg_domain = gdf_3classes[gdf_3classes['class_name'].isin(['Floresta', 'Nao_Floresta'])]
    
    mask_seg_domain = rasterize(
        [(geom, 1) for geom in gdf_seg_domain.geometry],
        out_shape=(height, width), transform=cbers_transform, fill=0, dtype=np.uint8
    )

    print("3. Binarizando MapBiomas: 1=Floresta, 2=Não-Floresta (Matriz Completa)...", flush=True)
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
        
        # Classes de Formação Florestal/Mata Nativa do MapBiomas (Coleção 10m/Coleção 8/9) -> 1 (Floresta)
        # Qualquer outra classe de uso (Urbano, Agro, Pasto, Solo Expsto, Agua) -> 2 (Não-Floresta)
        mask_floresta_nat = (mb_data_raw == 3) | (mb_data_raw == 4) | (mb_data_raw == 5) | (mb_data_raw == 49)
        mask_uso_geral = (mb_data_raw > 0) & (~mask_floresta_nat)
        
        # Aplicação no domínio de análise
        gt_mapbiomas_2m[mask_floresta_nat & (mask_seg_domain == 1)] = 1
        gt_mapbiomas_2m[mask_uso_geral & (mask_seg_domain == 1)] = 2
        
        del mb_data_raw, mask_floresta_nat, mask_uso_geral
        gc.collect()

    print("4. Executando MaskSLIC e calculando HoR (GT-U com Indefinido e GT-M)...", flush=True)
    gt_u = np.zeros((height, width), dtype=np.uint8)
    gt_m = np.zeros((height, width), dtype=np.uint8)

    tile_size = 1024
    total_blocos = int(np.ceil(height / tile_size)) * int(np.ceil(width / tile_size))
    bloco_atual = 0

    # Ajuste fino: 400 pixels por superpixel (~400m² em 2m de resolução) para permitir capturar superpixels de borda
    qps = 400.0

    with rasterio.open(path_cbers_2m) as src_cbers:
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                bloco_atual += 1
                h_win = min(tile_size, height - y)
                w_win = min(tile_size, width - x)
                window = Window(x, y, w_win, h_win)

                mask_tile = mask_seg_domain[y:y+h_win, x:x+w_win]
                qtp_tile = np.sum(mask_tile)

                if qtp_tile > 0:
                    qts_tile = int(qtp_tile / qps)
                    if qts_tile > 0:
                        rgb_tile = src_cbers.read([1, 2, 3], window=window)
                        rgb_ch_last = np.moveaxis(rgb_tile, 0, -1).astype(np.float32)
                        tile_max = rgb_ch_last.max()
                        if tile_max > 0:
                            rgb_ch_last /= tile_max

                        # SLIC com compacidade 15 para melhor adaptação às bordas espectrais
                        seg_tile = slic(
                            rgb_ch_last, n_segments=qts_tile, compactness=15,
                            mask=(mask_tile == 1), start_label=1, max_num_iter=10
                        )

                        gt_mb_tile = gt_mapbiomas_2m[y:y+h_win, x:x+w_win]

                        sp_v = seg_tile[seg_tile > 0]
                        gt_v = gt_mb_tile[seg_tile > 0]
                        max_lbl = seg_tile.max()

                        if max_lbl > 0 and len(sp_v) > 0:
                            count_veg = np.bincount(sp_v[gt_v == 1], minlength=max_lbl + 1)
                            count_nonveg = np.bincount(sp_v[gt_v == 2], minlength=max_lbl + 1)
                            count_total = np.bincount(sp_v, minlength=max_lbl + 1)

                            safe_total = np.where(count_total == 0, 1, count_total)
                            prop_veg = count_veg / safe_total
                            prop_nonveg = count_nonveg / safe_total
                            hor = np.maximum(prop_veg, prop_nonveg)

                            # GT-U: HoR < 0.7 -> 3 (Indefinido em Azul)
                            # GT-U: HoR >= 0.7 -> 1 (Floresta/Verde) ou 2 (Não-Floresta/Vermelho)
                            gt_u_map = np.zeros(max_lbl + 1, dtype=np.uint8)
                            mask_active = count_total > 0
                            gt_u_map[mask_active & (hor < 0.7)] = 3
                            gt_u_map[mask_active & (hor >= 0.7) & (prop_veg >= 0.7)] = 1
                            gt_u_map[mask_active & (hor >= 0.7) & (prop_veg < 0.7)] = 2

                            # GT-M: Maioria simples (prop_veg > 0.5 -> 1, senão 2)
                            gt_m_map = np.zeros(max_lbl + 1, dtype=np.uint8)
                            gt_m_map[mask_active & (prop_veg > 0.5)] = 1
                            gt_m_map[mask_active & (prop_veg <= 0.5)] = 2

                            gt_u[y:y+h_win, x:x+w_win] = gt_u_map[seg_tile]
                            gt_m[y:y+h_win, x:x+w_win] = gt_m_map[seg_tile]

                if bloco_atual % 20 == 0 or bloco_atual == total_blocos:
                    print(f"  ⏩ Bloco [{bloco_atual}/{total_blocos}] concluído.", flush=True)

    print("5. Salvando rasters dos Conjuntos Verdade (GT) e QMLs...", flush=True)
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

    print("6. Gerando painel comparativo da Figura 8 (4 Painéis ABNT)...", flush=True)
    center_y, center_x = height // 2, width // 2
    crop_h, crop_w = min(1200, height), min(2400, width)
    
    y1 = max(0, center_y - crop_h // 2)
    y2 = min(height, y1 + crop_h)
    x1 = max(0, center_x - crop_w // 2)
    x2 = min(width, x1 + crop_w)

    with rasterio.open(path_cbers_2m) as src_cbers:
        rgb_raw_crop = src_cbers.read([1, 2, 3], window=Window(x1, y1, crop_w, crop_h))

    rgb_crop = np.moveaxis(rgb_raw_crop, 0, -1).astype(np.float32)
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
    axes[0, 1].set_title("(b) GT-MapBiomas (Floresta vs Não-Floresta)", fontsize=11, fontweight='bold', pad=8)

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

    patch_veg = mpatches.Patch(color='#33a02c', label='Floresta')
    patch_nonveg = mpatches.Patch(color='#e31a1c', label='Não-Floresta')
    patch_undef = mpatches.Patch(color='#1f78b4', label='Indefinido (HoR < 0,7)')
    patch_noseg = mpatches.Patch(color='#000000', label='Não Segmentar')

    fig.legend(
        handles=[patch_veg, patch_nonveg, patch_undef, patch_noseg],
        loc='lower center', bbox_to_anchor=(0.5, 0.02),
        ncol=4, frameon=True, facecolor='white', edgecolor='black', fontsize=10
    )

    path_figura8 = os.path.join(dir_reports, f"Figura_8_Conjuntos_Verdade_{code_muni}_{ano_fim}.png")
    plt.savefig(path_figura8, bbox_inches='tight', pad_inches=0.2)
    plt.close()

    print(f"\n✅ CONJUNTOS VERDADE ATUALIZADOS COM SUCESSO:")
    print(f"   -> Raster GT-MapBiomas: {path_gt_mb}")
    print(f"   -> Raster GT-U (com Indefinido Azul): {path_gt_u}")
    print(f"   -> Raster GT-M: {path_gt_m}")
    print(f"   -> Figura 8 (4 Painéis ABNT): {path_figura8}")

if __name__ == "__main__":
    main()