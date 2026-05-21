import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from skimage.segmentation import slic
from dotenv import load_dotenv
import gc  # Biblioteca para forçar a limpeza da Memória RAM

def executar_maskslic_otimizado():
    load_dotenv()
    ROOT = os.getenv('PROJECT_ROOT')
    if not ROOT:
        print("❌ Erro: Variável 'PROJECT_ROOT' não encontrada no .env.")
        return

    dir_mapbiomas = os.path.join(ROOT, 'data', 'MapBiomas')
    dir_output = os.path.join(ROOT, 'data', 'Output')

    imagem_sat_path = os.path.join(dir_output, "02_SJC_Recortado_MapBiomas.tif")
    saida_segments = os.path.join(dir_output, "03_SJC_Superpixels_MaskSLIC.tif")

    if not os.path.exists(imagem_sat_path):
        print(f"❌ Erro: Arquivo {imagem_sat_path} não encontrado.")
        return

    mapbiomas_files = glob.glob(os.path.join(dir_mapbiomas, "*2021*coverage*10m*.tif"))
    if not mapbiomas_files:
        mapbiomas_files = glob.glob(os.path.join(dir_mapbiomas, "*coverage_10m*.tif"))
    mapbiomas_path = mapbiomas_files[0]

    # -------------------------------------------------------------
    # 1. LEITURA E OTIMIZAÇÃO DE MEMÓRIA (DOWNCAST)
    # -------------------------------------------------------------
    print("1/4 - Lendo e normalizando a imagem de Alta Resolução...")
    with rasterio.open(imagem_sat_path) as sat_src:
        meta_saida = sat_src.meta.copy()
        sat_data = sat_src.read()
        
        # Converte para float temporariamente para fazer a matemática
        sat_img_slic = np.moveaxis(sat_data, 0, -1).astype(np.float32)
        
        # Limpa o raster original da RAM
        del sat_data
        gc.collect() 

        # Normaliza as bandas para a escala 0-255 (Obrigatório para o SLIC não travar)
        for i in range(sat_img_slic.shape[2]):
            # Pega o valor máximo ignorando os 1% de pixels muito brilhantes (nuvens/ruído)
            band_max = np.percentile(sat_img_slic[:,:,i], 99) 
            if band_max > 0:
                sat_img_slic[:,:,i] = np.clip((sat_img_slic[:,:,i] / band_max) * 255.0, 0, 255)
        
        # Converte para uint8 (Reduz o peso da imagem drasticamente)
        sat_img_slic = sat_img_slic.astype(np.uint8)

    # -------------------------------------------------------------
    # 2. REPROJEÇÃO DA MÁSCARA MAPBIOMAS
    # -------------------------------------------------------------
    print("2/4 - Redimensionando a grade do MapBiomas...")
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

    agropecuaria = [14, 15, 18, 19, 39, 20, 40, 62, 41, 36, 46, 47, 35, 48, 9, 21]
    infra_urbana = [24]
    agua_rocha = [26, 33, 31, 29] 
    ruido_descartadas = [27]
    ids_mascara = agropecuaria + infra_urbana + agua_rocha + ruido_descartadas

    mask_boolean = np.isin(mb_resampled, ids_mascara)
    
    # Limpa a matriz do Mapbiomas reprojetada da RAM
    del mb_resampled
    gc.collect()

    # -------------------------------------------------------------
    # 3. EXECUÇÃO DO MASKSLIC TURBO
    # -------------------------------------------------------------
    print("3/4 - Executando a segmentação MaskSLIC (Isto pode levar 1 a 2 minutos)...")
    
    # convert2lab=False: Impede o uso abusivo de RAM.
    # max_num_iter=5: Limita o algoritmo a 5 passadas matemáticas em vez de 10.
    segments = slic(
        sat_img_slic, 
        n_segments=5000, 
        compactness=10.0, 
        mask=mask_boolean, 
        convert2lab=False, 
        max_num_iter=5,
        start_label=1
    )

    # Limpa a imagem processada da RAM (só precisamos dos segmentos agora)
    del sat_img_slic
    gc.collect()

    segments[~mask_boolean] = 0

    # -------------------------------------------------------------
    # 4. SALVANDO O RESULTADO
    # -------------------------------------------------------------
    print("4/4 - Gravando o arquivo de segmentos (.tif)...")
    meta_saida.update({
        "dtype": rasterio.int32,  
        "count": 1,
        "nodata": 0
    })

    with rasterio.open(saida_segments, "w", **meta_saida) as dst:
        dst.write(segments.astype(np.int32), 1)

    print(f"🎉 Sucesso! Superpixels salvos em:\n{saida_segments}")

if __name__ == "__main__":
    executar_maskslic_otimizado()