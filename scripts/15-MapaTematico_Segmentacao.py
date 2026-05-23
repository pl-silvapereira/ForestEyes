import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from skimage.segmentation import slic, find_boundaries
from scipy.ndimage import label, find_objects
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

    # Imagem CBERS (Apenas para o cálculo invisível do SLIC)
    imagem_sat_path = os.path.join(dir_output, "02_SJC_Recortado_MapBiomas.tif")
    
    # Arquivo ÚNICO (Mapa Sólido com Contornos)
    saida_visual = os.path.join(dir_output, "15_SJC_Mapa_Segmentado_Cores.tif")

    if not os.path.exists(imagem_sat_path):
        print(f"❌ Erro: Arquivo {imagem_sat_path} não encontrado.")
        return

    busca = glob.glob(os.path.join(dir_mapbiomas, "*2021*coverage*10m*.tif"))
    if not busca:
        busca = glob.glob(os.path.join(dir_mapbiomas, "*coverage_10m*.tif"))
    mapbiomas_path = busca[0]

    # -------------------------------------------------------------
    # 1. CARREGAMENTO CBERS
    # -------------------------------------------------------------
    print("1/4 - Lendo imagem CBERS (Operação matemática invisível)...")
    with rasterio.open(imagem_sat_path) as sat_src:
        meta_sat = sat_src.meta.copy()
        height, width = sat_src.height, sat_src.width
        sat_transform, sat_crs = sat_src.transform, sat_src.crs
        
        num_bands = min(3, sat_src.count)
        sat_data = sat_src.read(list(range(1, num_bands + 1)))
        
    sat_img = np.moveaxis(sat_data, 0, -1).astype(np.float32)
    for i in range(sat_img.shape[2]):
        p99 = np.percentile(sat_img[:,:,i], 99)
        if p99 > 0:
            sat_img[:,:,i] = np.clip((sat_img[:,:,i] / p99) * 255.0, 0, 255.0)
    
    sat_img = sat_img.astype(np.uint8)
    del sat_data
    gc.collect()

    # -------------------------------------------------------------
    # 2. ALINHAMENTO DO MAPBIOMAS E CORES SÓLIDAS
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

    ids_floresta = [1, 3, 4, 5, 6, 49]
    ids_nao_floresta = [10, 11, 12, 32, 50, 13]
    
    mask_floresta = np.isin(mb_aligned, ids_floresta)
    mask_nao_floresta = np.isin(mb_aligned, ids_nao_floresta)
    mask_segmentacao = mask_floresta | mask_nao_floresta

    # Cria a tela RGB base: O fundo é Preto (0,0,0) absoluto para a Máscara
    rgb_out = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Pinta as áreas de interesse
    rgb_out[mask_floresta] = [0, 255, 0]      # Verde Sólido
    rgb_out[mask_nao_floresta] = [255, 0, 0]  # Vermelho Sólido
    
    del mb_aligned
    gc.collect()

    # -------------------------------------------------------------
    # 3. MASKSLIC FOCADO COM FIND_OBJECTS
    # -------------------------------------------------------------
    print("3/4 - Mapeando ilhas naturais para evitar travamento da RAM...")
    
    # Agrupa as áreas contínuas e encontra o Bounding Box de cada "Ilha"
    ilhas, num_ilhas = label(mask_segmentacao)
    caixas = find_objects(ilhas)
    
    print(f"-> Foram encontradas {num_ilhas} ilhas de vegetação separadas pela infraestrutura.")
    print("-> Iniciando a segmentação (Você acompanhará o progresso abaixo)...")

    total_pixels_alvo = np.sum(mask_segmentacao)
    global_segments = np.zeros((height, width), dtype=np.int32)
    global_id_offset = 0
    ilhas_processadas = 0

    for i, slc in enumerate(caixas):
        if slc is None:
            continue

        patch_mask = (ilhas[slc] == (i + 1))
        patch_area = np.sum(patch_mask)

        # Ignora ruídos minúsculos
        if patch_area < 50:
            continue

        ilhas_processadas += 1
        
        # Distribui os 15.000 superpixels de forma proporcional ao tamanho da ilha
        n_seg_patch = max(1, int(15000 * (patch_area / total_pixels_alvo)))

        # Roda o MaskSLIC exclusivamente na ilha atual
        patch_img = sat_img[slc]
        seg_patch = slic(
            patch_img, 
            n_segments=n_seg_patch, 
            compactness=10.0, 
            mask=patch_mask, 
            convert2lab=False, 
            enforce_connectivity=False, # Corta 90% do tempo de processamento
            max_num_iter=5,
            start_label=1
        )
        
        seg_patch[~patch_mask] = 0
        
        # Consolida os IDs no mapa geral sem sobrepor outras ilhas próximas
        max_id_local = seg_patch.max()
        if max_id_local > 0:
            seg_patch[seg_patch > 0] += global_id_offset
            global_id_offset += max_id_local
            
            global_patch = global_segments[slc]
            global_patch[patch_mask] = seg_patch[patch_mask]
            global_segments[slc] = global_patch

        # Exibe o progresso
        if ilhas_processadas % 20 == 0 or ilhas_processadas == num_ilhas:
            print(f"   Progresso: {ilhas_processadas} ilhas segmentadas...")

    del sat_img, ilhas
    gc.collect()

    print("-> Desenhando os contornos amarelos...")
    borders = find_boundaries(global_segments, mode='inner', background=0)
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

    print(f"\n🎉 Sucesso! Mapa temático sem CBERS e com contornos salvo em:\n{saida_visual}")

if __name__ == "__main__":
    executar_mapa_final_tematico()