import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from skimage.segmentation import slic, find_boundaries
from scipy.ndimage import label, find_objects
from dotenv import load_dotenv
import gc

def executar_mapa_tematico_verde_vermelho():
    load_dotenv()
    ROOT = os.getenv('PROJECT_ROOT')
    if not ROOT:
        print("❌ Erro: Variável 'PROJECT_ROOT' não encontrada no .env.")
        return

    dir_mapbiomas = os.path.join(ROOT, 'data', 'MapBiomas')
    dir_output = os.path.join(ROOT, 'data', 'Output')

    # Arquivos de Entrada e Saída
    imagem_sat_path = os.path.join(dir_output, "02_SJC_Recortado_MapBiomas.tif")
    saida_visual = os.path.join(dir_output, "15_SJC_Mapa_Verde_Vermelho_Segmentado.tif")

    if not os.path.exists(imagem_sat_path):
        print(f"❌ Erro: Arquivo {imagem_sat_path} não encontrado.")
        return

    busca = glob.glob(os.path.join(dir_mapbiomas, "*2021*coverage*10m*.tif"))
    if not busca:
        busca = glob.glob(os.path.join(dir_mapbiomas, "*coverage_10m*.tif"))
    mapbiomas_path = busca[0]

    # -------------------------------------------------------------
    # 1. LEITURA DE METADADOS E ALINHAMENTO DO MAPBIOMAS
    # -------------------------------------------------------------
    print("1/4 - Preparando metadados e alinhando classes do MapBiomas...")
    with rasterio.open(imagem_sat_path) as sat_src:
        meta_sat = sat_src.meta.copy()
        height, width = sat_src.height, sat_src.width
        sat_transform, sat_crs = sat_src.transform, sat_src.crs

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

    # -------------------------------------------------------------
    # 2. DEFINIÇÃO DAS CORES CHAPADAS (A BASE DO MAPA)
    # -------------------------------------------------------------
    print("2/4 - Criando o Fundo Sólido (Verde = Floresta, Vermelho = Não Floresta, Preto = Máscara)...")
    ids_floresta = [1, 3, 4, 5, 6, 49]
    ids_nao_floresta = [10, 11, 12, 32, 50, 13]
    
    mask_floresta = np.isin(mb_aligned, ids_floresta)
    mask_nao_floresta = np.isin(mb_aligned, ids_nao_floresta)
    mask_segmentacao = mask_floresta | mask_nao_floresta

    # Inicia a tela RGB que será exportada (Tudo que é 0 continua Preto Absoluto)
    rgb_out = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_out[mask_floresta] = [0, 255, 0]      # Pinta Floresta de Verde
    rgb_out[mask_nao_floresta] = [255, 0, 0]  # Pinta Formação Natural de Vermelho
    
    del mb_aligned
    gc.collect()

    # -------------------------------------------------------------
    # 3. SEGMENTAÇÃO DINÂMICA (SEM TRAVAR A MEMÓRIA)
    # -------------------------------------------------------------
    print("3/4 - Mapeando as ilhas de vegetação para cálculo dos superpixels...")
    
    ilhas, num_ilhas = label(mask_segmentacao)
    caixas = find_objects(ilhas)
    
    print(f"-> {num_ilhas} ilhas isoladas encontradas. Iniciando a segmentação...")

    total_pixels_alvo = np.sum(mask_segmentacao)
    ilhas_processadas = 0

    # Abre a imagem CBERS, mas lê APENAS pequenos blocos por vez
    with rasterio.open(imagem_sat_path) as sat_src:
        for i, slc in enumerate(caixas):
            if slc is None:
                continue

            patch_mask = (ilhas[slc] == (i + 1))
            patch_area = np.sum(patch_mask)

            # Ignora pontos minúsculos que não formam um superpixel real
            if patch_area < 50:
                continue

            ilhas_processadas += 1
            
            # Carrega da memória do HD apenas o retângulo exato da ilha atual
            window = Window.from_slices(slc[0], slc[1])
            patch_sat = sat_src.read((1,2,3), window=window)
            patch_sat = np.moveaxis(patch_sat, 0, -1).astype(np.float32)
            
            # Normaliza as cores do pequeno bloco para o algoritmo entender
            for b in range(patch_sat.shape[2]):
                p99 = np.percentile(patch_sat[:,:,b], 99)
                if p99 > 0:
                    patch_sat[:,:,b] = np.clip((patch_sat[:,:,b] / p99) * 255.0, 0, 255.0)
            patch_sat = patch_sat.astype(np.uint8)

            n_seg_patch = max(1, int(15000 * (patch_area / total_pixels_alvo)))

            # Executa o MaskSLIC restrito a esta ilha específica
            seg_patch = slic(
                patch_sat, 
                n_segments=n_seg_patch, 
                compactness=10.0, 
                mask=patch_mask, 
                convert2lab=False, 
                enforce_connectivity=False, 
                max_num_iter=5,
                start_label=1
            )
            
            # Procura os contornos dos superpixels gerados
            borders = find_boundaries(seg_patch, mode='inner', background=0)
            
            # Desenha os contornos (em Amarelo) DIRETAMENTE sobre a tela Verde/Vermelha
            # Utilizamos o Amarelo [255, 255, 0] para dar um alto contraste contra as cores base
            rgb_patch = rgb_out[slc]
            rgb_patch[borders] = [255, 255, 0]
            rgb_out[slc] = rgb_patch

            # Mostra o progresso no console para você saber que não travou
            if ilhas_processadas % 20 == 0 or ilhas_processadas == num_ilhas:
                print(f"   Progresso: {ilhas_processadas}/{num_ilhas} ilhas processadas...")

    del ilhas, caixas
    gc.collect()

    # -------------------------------------------------------------
    # 4. SALVANDO O MAPA FINAL
    # -------------------------------------------------------------
    print("4/4 - Salvando o mapa temático final...")
    meta_sat.update({
        "dtype": rasterio.uint8, 
        "count": 3, 
        "nodata": None, 
        "photometric": "RGB"
    })
    
    with rasterio.open(saida_visual, "w", **meta_sat) as dst_vis:
        for b in range(3):
            dst_vis.write(rgb_out[:, :, b], b+1)

    print(f"\n🎉 Sucesso! Mapa gerado com precisão e salvo em:\n{saida_visual}")

if __name__ == "__main__":
    executar_mapa_tematico_verde_vermelho()