import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from skimage.segmentation import slic, find_boundaries
from dotenv import load_dotenv
import gc

def executar_mapa_final_tematico():
    load_dotenv()
    ROOT = os.getenv('PROJECT_ROOT')
    if not ROOT:
        print("❌ Erro: Variável 'PROJECT_ROOT' não encontrada no .env.")
        return

    dir_mapbiomas = os.path.join(ROOT, 'data', 'MapBiomas')
    dir_output = os.path.join(ROOT, 'data', 'Output')

    # Imagem CBERS (Apenas para base de cálculo da segmentação)
    imagem_sat_path = os.path.join(dir_output, "02_SJC_Recortado_MapBiomas.tif")
    
    # O arquivo ÚNICO que será gerado
    saida_visual = os.path.join(dir_output, "15_SJC_Mapa_Segmentado_Cores.tif")

    if not os.path.exists(imagem_sat_path):
        print(f"❌ Erro: Arquivo {imagem_sat_path} não encontrado.")
        return

    busca = glob.glob(os.path.join(dir_mapbiomas, "*2021*coverage*10m*.tif"))
    if not busca:
        busca = glob.glob(os.path.join(dir_mapbiomas, "*coverage_10m*.tif"))
    if not busca:
        print("❌ Erro: Arquivo do MapBiomas não encontrado.")
        return
    mapbiomas_path = busca[0]

    # -------------------------------------------------------------
    # 1. CARREGAMENTO E NORMALIZAÇÃO DA IMAGEM CBERS
    # -------------------------------------------------------------
    print("1/4 - Lendo imagem CBERS (Operação matemática invisível)...")
    with rasterio.open(imagem_sat_path) as sat_src:
        meta_sat = sat_src.meta.copy()
        height, width = sat_src.height, sat_src.width
        sat_transform, sat_crs = sat_src.transform, sat_src.crs
        
        # Lê apenas as 3 bandas RGB
        num_bands = min(3, sat_src.count)
        sat_data = sat_src.read(list(range(1, num_bands + 1)))
        
    # Converte e normaliza a imagem na RAM para não estourar o limite do Colab
    sat_data = np.moveaxis(sat_data, 0, -1).astype(np.float32)
    for i in range(sat_data.shape[2]):
        p99 = np.percentile(sat_data[:,:,i], 99)
        if p99 > 0:
            sat_data[:,:,i] = np.clip((sat_data[:,:,i] / p99) * 255.0, 0, 255.0)
    
    sat_img = sat_data.astype(np.uint8)
    del sat_data # Limpa o array pesado
    gc.collect()

    # -------------------------------------------------------------
    # 2. ALINHAMENTO DO MAPBIOMAS E DEFINIÇÃO DE CORES
    # -------------------------------------------------------------
    print("2/4 - Alinhando classes do MapBiomas...")
    with rasterio.open(mapbiomas_path) as mb_src:
        mb_aligned = np.zeros((height, width), dtype=np.uint8)
        reproject(
            source=rasterio.band(mb_src, 1),
            destination=mb_aligned,
            src_transform=mb_src.transform,
            src_crs=mb_src.crs,
            dst_transform=sat_transform,
            dst_crs=sat_crs,
            resampling=Resampling.nearest
        )

    # Separação das Classes
    ids_floresta = [1, 3, 4, 5, 6, 49]
    ids_nao_floresta = [10, 11, 12, 32, 50, 13]
    
    mask_floresta = np.isin(mb_aligned, ids_floresta)
    mask_nao_floresta = np.isin(mb_aligned, ids_nao_floresta)
    
    # Máscara de onde o SLIC é autorizado a funcionar
    mask_segmentacao = mask_floresta | mask_nao_floresta

    # Cria a tela RGB base (Tudo começa Preto)
    rgb_out = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Pinta as áreas da floresta de Verde e as não-florestais de Vermelho
    rgb_out[mask_floresta] = [0, 255, 0]
    rgb_out[mask_nao_floresta] = [255, 0, 0]
    
    del mb_aligned
    gc.collect()

    # -------------------------------------------------------------
    # 3. MASKSLIC (SEGMENTAÇÃO GLOBAL)
    # -------------------------------------------------------------
    print("3/4 - Calculando superpixels... (Isso pode levar alguns minutos, aguarde)")
    
    # Roda o algoritmo na imagem CBERS, mas restrito apenas às áreas Verde/Vermelha
    segments = slic(
        sat_img, 
        n_segments=15000, 
        compactness=10.0, 
        mask=mask_segmentacao, 
        convert2lab=False, 
        max_num_iter=5,
        start_label=1
    )
    
    # Zera qualquer superpixel que tenha invadido o fundo (A Máscara Preta)
    segments[~mask_segmentacao] = 0
    del sat_img
    gc.collect()

    # Extrai os contornos e pinta de AMARELO
    # O "background=0" impede que ele faça uma borda amarela ao redor de toda a área preta
    print("Desenhando os contornos da segmentação...")
    borders = find_boundaries(segments, mode='inner', background=0)
    rgb_out[borders] = [255, 255, 0]

    # -------------------------------------------------------------
    # 4. SALVAR O ARQUIVO TIF
    # -------------------------------------------------------------
    print("4/4 - Salvando o mapa final...")
    meta_vis = meta_sat.copy()
    meta_vis.update({
        "dtype": rasterio.uint8, 
        "count": 3, 
        "nodata": None, 
        "photometric": "RGB"
    })
    
    with rasterio.open(saida_visual, "w", **meta_vis) as dst_vis:
        for b in range(3):
            dst_vis.write(rgb_out[:, :, b], b+1)

    print(f"\n🎉 Sucesso! O seu mapa segmentado perfeito está em:\n{saida_visual}")

if __name__ == "__main__":
    executar_mapa_final_tematico()