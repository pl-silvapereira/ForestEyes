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

    path_shp_binario = os.path.join(output_seg_dir, f"{code_muni}_Segmentar_vs_NaoSegmentar_{ano_fim}.shp")
    path_cbers = os.path.join(project_root, "data", "output", "pansharpening", ano_fim, f"{code_muni}_{ano_fim}_CBERS_TRUE_COLOR_2M.tif")
    path_tif_slic = os.path.join(output_seg_dir, f"{code_muni}_MaskSLIC_ContornosBrancos_{ano_fim}.tif")

    if not os.path.exists(path_shp_binario) or not os.path.exists(path_cbers):
        print("[ERRO CRÍTICO] Arquivos base não encontrados.", flush=True)
        sys.exit(1)

    print("=" * 80, flush=True)
    print(f"🖼️ SCRIPT 06 - GERAÇÃO DE IMAGENS SEGMENTADAS (SUPERPIXELS) ({ano_fim})", flush=True)
    print("=" * 80, flush=True)

    # Valores de estatísticas radiométricas (extraídos da cena CBERS-4A)
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

    print("2. Rasterizando EXCLUSIVAMENTE a classe 'Segmentar' do Shapefile...", flush=True)
    gdf_binario = gpd.read_file(path_shp_binario).to_crs(cbers_crs)
    
    # Filtro estrito: apenas feições rotuladas como 'Segmentar'
    gdf_segmentar = gdf_binario[gdf_binario['class_name'] == 'Segmentar']
    
    if gdf_segmentar.empty:
        print("[AVISO] Nenhum polígono com a classe 'Segmentar' foi encontrado.")
        sys.exit(0)

    geom_seg = gdf_segmentar.geometry
    mask_seg = rasterize([(geom, 1) for geom in geom_seg], out_shape=cbers_shape, transform=cbers_transform, fill=0, dtype=np.uint8)
    
    qps = 250.0  # Quantidade de Pixels por Superpixel alvo

    print(f"\n3. Processando SLIC por blocos restrito à máscara 'Segmentar'...", flush=True)
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
        
        # Tags de estatísticas de contraste radiométrico
        dst.update_tags(1, STATISTICS_MINIMUM=str(min_r), STATISTICS_MAXIMUM=str(max_r))
        dst.update_tags(2, STATISTICS_MINIMUM=str(min_g), STATISTICS_MAXIMUM=str(max_g))
        dst.update_tags(3, STATISTICS_MINIMUM=str(min_b), STATISTICS_MAXIMUM=str(max_b))

        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                bloco_atual += 1
                h_janela = min(tile_size, h - y)
                w_janela = min(tile_size, w - x)
                window = Window(x, y, w_janela, h_janela)
                
                mask_seg_tile = mask_seg[y:y+h_janela, x:x+w_janela]
                qtp_seg_tile = np.sum(mask_seg_tile)
                
                rgbnir_raw = src.read(window=window) 

                if qtp_seg_tile > 0:
                    qts_tile = int(qtp_seg_tile / qps)
                    if qts_tile > 0:
                        rgbnir_ch_last = np.moveaxis(rgbnir_raw, 0, -1)
                        
                        rgb_slic = rgbnir_ch_last[:, :, :3].astype(np.float32)
                        tile_max = rgb_slic.max()
                        if tile_max > 0: 
                            rgb_slic /= tile_max 

                        # Aplicar SLIC unicamente sobre pixels da classe 'Segmentar'
                        segmentos_tile = slic(
                            rgb_slic, 
                            n_segments=qts_tile, 
                            compactness=10, 
                            mask=(mask_seg_tile == 1), 
                            start_label=1, 
                            max_num_iter=5
                        )
                        
                        if np.any(segmentos_tile > 0):
                            # Desenhar bordas brancas somente nos contornos dos superpixels gerados na área 'Segmentar'
                            contornos_tile = find_boundaries(segmentos_tile, mode='inner')
                            rgbnir_raw[0][contornos_tile] = max_r
                            rgbnir_raw[1][contornos_tile] = max_g
                            rgbnir_raw[2][contornos_tile] = max_b
                            
                            print(f"  ✅ [Bloco {bloco_atual}/{total_blocos}] -> Segmentados superpixels sobre classe 'Segmentar'.", flush=True)
                        else:
                            if bloco_atual % 20 == 0:
                                print(f"  ⏩ [Bloco {bloco_atual}/{total_blocos}] -> Bloco sem superpixels válidos...", flush=True)
                else:
                    if bloco_atual % 20 == 0:
                        print(f"  ⏩ [Bloco {bloco_atual}/{total_blocos}] -> Fora da classe 'Segmentar'...", flush=True)

                dst.write(rgbnir_raw, window=window)
                del rgbnir_raw, mask_seg_tile
                gc.collect()

    shutil.move(temp_tif_local, path_tif_slic)
    print(f"\n✅ Imagem raster de superpixels (restrita à classe 'Segmentar') gerada em:")
    print(f"   -> {path_tif_slic}")
    print("\n🎉 Geração de imagem segmentada concluída!")

if __name__ == "__main__":
    main()