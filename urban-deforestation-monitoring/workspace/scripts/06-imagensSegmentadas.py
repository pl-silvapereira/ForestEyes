import sys
import os
import gc
import shutil
import warnings
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from skimage.segmentation import slic, find_boundaries
from dotenv import load_dotenv

# Silenciar avisos inofensivos do SciPy/scikit-image durante a convergência do SLIC em tiles
warnings.filterwarnings("ignore", message=".*One of the clusters is empty.*")
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

def gerar_estilo_qml_raster(caminho_qml, min_r=56, max_r=1342, min_g=74, max_g=1050, min_b=101, max_b=1267):
    """Gera o arquivo de estilo .qml para o raster no QGIS com Brilho = 50 
    e os limites Mínimo e Máximo configurados para cada banda."""
    conteudo_qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.0" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer type="multibandcolor" redBand="1" greenBand="2" blueBand="3" opacity="1" alphaBand="-1">
      <rasterTransparency/>
      <redContrastEnhancement>
        <minValue>{min_r}</minValue>
        <maxValue>{max_r}</maxValue>
        <algorithm>StretchToMinimumMaximum</algorithm>
      </redContrastEnhancement>
      <greenContrastEnhancement>
        <minValue>{min_g}</minValue>
        <maxValue>{max_g}</maxValue>
        <algorithm>StretchToMinimumMaximum</algorithm>
      </greenContrastEnhancement>
      <blueContrastEnhancement>
        <minValue>{min_b}</minValue>
        <maxValue>{max_b}</maxValue>
        <algorithm>StretchToMinimumMaximum</algorithm>
      </blueContrastEnhancement>
    </rasterrenderer>
    <brightnesscontrast brightness="50" contrast="0" gamma="1"/>
    <huesaturation colorizeGreen="128" colorizeOn="0" colorizeRed="255" colorizeBlue="128" colorizeStrength="100" saturation="0" grayscaleMode="0"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>"""
    with open(caminho_qml, 'w', encoding='utf-8') as f:
        f.write(conteudo_qml)

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 06-imagensSegmentadas.py <code_muni> <ano_inicio> <ano_fim>")
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
    os.makedirs(output_seg_dir, exist_ok=True)

    path_shp_3classes = os.path.join(output_seg_dir, f"{code_muni}_Segmentacao_3Classes_{ano_fim}.shp")
    path_cbers = os.path.join(project_root, "data", "output", "pansharpening", ano_fim, f"{code_muni}_{ano_fim}_CBERS_TRUE_COLOR_2M.tif")
    path_tif_slic = os.path.join(output_seg_dir, f"{code_muni}_MaskSLIC_Contornos_{ano_fim}.tif")
    path_qml_slic = os.path.join(output_seg_dir, f"{code_muni}_MaskSLIC_Contornos_{ano_fim}.qml")

    if not os.path.exists(path_shp_3classes) or not os.path.exists(path_cbers):
        print("[ERRO CRÍTICO] Arquivos base não encontrados (Shapefile 3 classes ou CBERS).", flush=True)
        sys.exit(1)

    print("=" * 80, flush=True)
    print(f"🖼️ SCRIPT 06 - GERAÇÃO DE IMAGENS SEGMENTADAS (FLORESTA vs NÃO FLORESTA) ({ano_fim})", flush=True)
    print("=" * 80, flush=True)

    # Valores de estatísticas radiométricas da cena CBERS-4A
    min_r, max_r = 56, 1342
    min_g, max_g = 74, 1050
    min_b, max_b = 101, 1267

    print("1. Lendo metadados da imagem CBERS-4A...", flush=True)
    with rasterio.open(path_cbers) as src:
        cbers_meta = src.meta.copy()
        cbers_crs = src.crs
        cbers_transform = src.transform
        cbers_shape = (src.height, src.width)
        num_bands = src.count

    print("2. Rasterizando as classes 'Floresta' e 'Nao_Floresta' do Shapefile de 3 classes...", flush=True)
    gdf_3classes = gpd.read_file(path_shp_3classes).to_crs(cbers_crs)
    
    gdf_floresta = gdf_3classes[gdf_3classes['class_name'] == 'Floresta']
    gdf_nao_floresta = gdf_3classes[gdf_3classes['class_name'] == 'Nao_Floresta']

    mask_floresta = rasterize(
        [(geom, 1) for geom in gdf_floresta.geometry],
        out_shape=cbers_shape, transform=cbers_transform, fill=0, dtype=np.uint8
    ) if not gdf_floresta.empty else np.zeros(cbers_shape, dtype=np.uint8)

    mask_nao_floresta = rasterize(
        [(geom, 1) for geom in gdf_nao_floresta.geometry],
        out_shape=cbers_shape, transform=cbers_transform, fill=0, dtype=np.uint8
    ) if not gdf_nao_floresta.empty else np.zeros(cbers_shape, dtype=np.uint8)

    qps = 250.0  # Pixels por Superpixel alvo

    print(f"\n3. Processando SLIC por blocos com contornos diferenciados...", flush=True)
    temp_tif_local = f"/content/temp_slic_{code_muni}_{ano_fim}.tif"
    
    tile_size = 1024 
    h, w = cbers_shape
    total_blocos = int(np.ceil(h / tile_size)) * int(np.ceil(w / tile_size))
    
    bloco_atual = 0

    with rasterio.open(path_cbers) as src, rasterio.open(temp_tif_local, 'w', **cbers_meta) as dst:
        
        dst.set_band_description(1, "Red")
        dst.set_band_description(2, "Green")
        dst.set_band_description(3, "Blue")
        if num_bands >= 4:
            dst.set_band_description(4, "NIR")
        
        dst.update_tags(1, STATISTICS_MINIMUM=str(min_r), STATISTICS_MAXIMUM=str(max_r))
        dst.update_tags(2, STATISTICS_MINIMUM=str(min_g), STATISTICS_MAXIMUM=str(max_g))
        dst.update_tags(3, STATISTICS_MINIMUM=str(min_b), STATISTICS_MAXIMUM=str(max_b))

        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                bloco_atual += 1
                h_janela = min(tile_size, h - y)
                w_janela = min(tile_size, w - x)
                window = Window(x, y, w_janela, h_janela)
                
                mask_flor_tile = mask_floresta[y:y+h_janela, x:x+w_janela]
                mask_nflor_tile = mask_nao_floresta[y:y+h_janela, x:x+w_janela]
                
                rgbnir_raw = src.read(window=window)

                rgbnir_ch_last = np.moveaxis(rgbnir_raw, 0, -1)
                rgb_slic = rgbnir_ch_last[:, :, :3].astype(np.float32)
                tile_max = rgb_slic.max()
                if tile_max > 0:
                    rgb_slic /= tile_max

                processed_any = False

                # 3a. Segmentação da Classe Floresta -> Contorno Verde Fluorescente
                qtp_flor = np.sum(mask_flor_tile)
                if qtp_flor > 0:
                    qts_flor = int(qtp_flor / qps)
                    if qts_flor > 0:
                        seg_flor = slic(
                            rgb_slic, n_segments=qts_flor, compactness=10, 
                            mask=(mask_flor_tile == 1), start_label=1, max_num_iter=5
                        )
                        if np.any(seg_flor > 0):
                            contornos_flor = find_boundaries(seg_flor, mode='inner')
                            rgbnir_raw[0][contornos_flor] = 0        # Red = 0
                            rgbnir_raw[1][contornos_flor] = max_g    # Green = max (Verde Fluorescente)
                            rgbnir_raw[2][contornos_flor] = 0        # Blue = 0
                            processed_any = True

                # 3b. Segmentação da Classe Nao_Floresta -> Contorno Vermelho
                qtp_nflor = np.sum(mask_nflor_tile)
                if qtp_nflor > 0:
                    qts_nflor = int(qtp_nflor / qps)
                    if qts_nflor > 0:
                        seg_nflor = slic(
                            rgb_slic, n_segments=qts_nflor, compactness=10, 
                            mask=(mask_nflor_tile == 1), start_label=1, max_num_iter=5
                        )
                        if np.any(seg_nflor > 0):
                            contornos_nflor = find_boundaries(seg_nflor, mode='inner')
                            rgbnir_raw[0][contornos_nflor] = max_r   # Red = max (Vermelho)
                            rgbnir_raw[1][contornos_nflor] = 0       # Green = 0
                            rgbnir_raw[2][contornos_nflor] = 0       # Blue = 0
                            processed_any = True

                if processed_any:
                    print(f"  ✅ [Bloco {bloco_atual}/{total_blocos}] -> Superpixels desenhados (Verde: Floresta / Vermelho: Não Floresta).", flush=True)
                else:
                    if bloco_atual % 20 == 0:
                        print(f"  ⏩ [Bloco {bloco_atual}/{total_blocos}] -> Sem superpixels para segmentar...", flush=True)

                dst.write(rgbnir_raw, window=window)
                del rgbnir_raw, mask_flor_tile, mask_nflor_tile
                gc.collect()

    shutil.move(temp_tif_local, path_tif_slic)

    print("4. Gerando estilo QML com Brilho = 50 e Min/Max para o QGIS...")
    gerar_estilo_qml_raster(path_qml_slic, min_r, max_r, min_g, max_g, min_b, max_b)

    print(f"\n✅ Processamento concluído com sucesso:")
    print(f"   -> Imagem TIF: {path_tif_slic}")
    print(f"   -> Estilo QML: {path_qml_slic}")

if __name__ == "__main__":
    main()