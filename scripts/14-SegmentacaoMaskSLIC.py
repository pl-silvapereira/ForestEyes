import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from skimage.segmentation import slic
from dotenv import load_dotenv

def executar_maskslic():
    # 1. Carrega as configurações do ambiente (.env)
    load_dotenv()
    ROOT = os.getenv('PROJECT_ROOT')
    if not ROOT:
        print("❌ Erro: Variável 'PROJECT_ROOT' não encontrada no .env.")
        return

    dir_mapbiomas = os.path.join(ROOT, 'data', 'MapBiomas')
    dir_output = os.path.join(ROOT, 'data', 'Output')

    # Definição dos caminhos dos arquivos
    imagem_sat_path = os.path.join(dir_output, "02_SJC_Recortado_MapBiomas.tif")
    saida_segments = os.path.join(dir_output, "03_SJC_Superpixels_MaskSLIC.tif")

    if not os.path.exists(imagem_sat_path):
        print(f"❌ Erro: A imagem recortada do Script 06 não foi encontrada em: {imagem_sat_path}")
        return

    # Busca o arquivo original do MapBiomas para extrair as classes guia
    mapbiomas_files = glob.glob(os.path.join(dir_mapbiomas, "*2021*coverage*10m*.tif"))
    if not mapbiomas_files:
        mapbiomas_files = glob.glob(os.path.join(dir_mapbiomas, "*coverage_10m*.tif"))
    if not mapbiomas_files:
        print("❌ Erro: Arquivo base do MapBiomas não encontrado.")
        return
    mapbiomas_path = mapbiomas_files[0]

    print(f"🔍 Imagem Alvo: {os.path.basename(imagem_sat_path)}")
    print(f"🔍 Alinhando com a base: {os.path.basename(mapbiomas_path)}")

    # -------------------------------------------------------------
    # 2. LEITURA DA IMAGEM DE ALTA RESOLUÇÃO
    # -------------------------------------------------------------
    with rasterio.open(imagem_sat_path) as sat_src:
        meta_saida = sat_src.meta.copy()
        sat_data = sat_src.read()  # Formato rasterio: (Bandas, Altura, Largura)
        
        # O Scikit-Image espera a matriz no formato (Altura, Largura, Bandas)
        sat_img_slic = np.moveaxis(sat_data, 0, -1)

    # -------------------------------------------------------------
    # 3. REPROJEÇÃO E REAMOSTRAGEM DA MÁSCARA DO MAPBIOMAS
    # -------------------------------------------------------------
    print("Redimensionando a grade do MapBiomas para a resolução da imagem...")
    with rasterio.open(mapbiomas_path) as mb_src:
        mb_resampled = np.zeros((meta_saida['height'], meta_saida['width']), dtype=np.uint8)
        reproject(
            source=rasterio.band(mb_src, 1),
            destination=mb_resampled,
            src_transform=mb_src.transform,
            src_crs=mb_src.crs,
            dst_transform=meta_saida['transform'],
            dst_crs=meta_saida['crs'],
            resampling=Resampling.nearest
        )

    # Definição das classes que farão parte da análise (conforme seus requisitos anteriores)
    agropecuaria = [14, 15, 18, 19, 39, 20, 40, 62, 41, 36, 46, 47, 35, 48, 9, 21]
    infra_urbana = [24]
    agua_rocha = [26, 33, 31, 29] 
    ruido_descartadas = [27]
    ids_mascara = agropecuaria + infra_urbana + agua_rocha + ruido_descartadas

    # Criação da máscara booleana exigida pelo MaskSLIC
    mask_boolean = np.isin(mb_resampled, ids_mascara)

    # -------------------------------------------------------------
    # 4. EXECUÇÃO DO ALGORITMO MASKSLIC
    # -------------------------------------------------------------
    print("Executando a segmentação MaskSLIC dentro das regiões de interesse...")
    
    # n_segments: quantidade alvo de superpixels
    # compactness: equilíbrio entre a forma geométrica (valores altos) e adaptação às bordas reais (valores baixos)
    segments = slic(
        sat_img_slic, 
        n_segments=5000, 
        compactness=10.0, 
        mask=mask_boolean, 
        start_label=1
    )

    # Onde não havia máscara (Florestas/Veg Herbácea), definimos o ID do segmento como 0 (Nodata)
    segments[~mask_boolean] = 0

    # -------------------------------------------------------------
    # 5. SALVANDO O RASTER DE SUPERPIXELS
    # -------------------------------------------------------------
    print("Gravando o arquivo de segmentos (.tif)...")
    meta_saida.update({
        "dtype": rasterio.int32,  # IDs de superpixels frequentemente passam de 255
        "count": 1,
        "nodata": 0
    })

    with rasterio.open(saida_segments, "w", **meta_saida) as dst:
        dst.write(segments.astype(np.int32), 1)

    print(f"🎉 Sucesso! Superpixels gerados e salvos em:\n{saida_segments}")

if __name__ == "__main__":
    executar_maskslic()