import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from scipy.ndimage import find_objects
from skimage.segmentation import find_boundaries
from PIL import Image
from dotenv import load_dotenv

def gerar_crops_campanha():
    load_dotenv()
    ROOT = os.getenv('PROJECT_ROOT')
    if not ROOT:
        print("❌ Erro: Variável 'PROJECT_ROOT' não encontrada.")
        return

    dir_output = os.path.join(ROOT, 'data', 'Output')
    dir_mapbiomas = os.path.join(ROOT, 'data', 'MapBiomas')
    
    sat_path = os.path.join(dir_output, "02_SJC_Recortado_MapBiomas.tif")
    npy_path = os.path.join(dir_output, "17_SJC_Matriz_Segmentacao.npy")
    csv_alvos = os.path.join(dir_output, "19_SJC_Alvos_Campanha.csv")

    if not all(os.path.exists(p) for p in [sat_path, npy_path, csv_alvos]):
        print("❌ Erro: Faltam arquivos base (Execute os scripts 17 e 19).")
        return

    busca = glob.glob(os.path.join(dir_mapbiomas, "*2021*coverage*10m*.tif"))
    if not busca:
        busca = glob.glob(os.path.join(dir_mapbiomas, "*coverage_10m*.tif"))
    mapbiomas_path = busca[0]

    # Estrutura de subpastas exigida
    pastas = {
        'satelite': os.path.join(dir_output, 'crops', 'satelite'),
        'cinza': os.path.join(dir_output, 'crops', 'cinza'),
        'rgb': os.path.join(dir_output, 'crops', 'rgb')
    }
    for p in pastas.values():
        os.makedirs(p, exist_ok=True)

    print("1/3 - Carregando dados base...")
    df_alvos = pd.read_csv(csv_alvos, sep=';')
    matriz_seg = np.load(npy_path)
    
    # Prepara o alinhamento do MapBiomas para a geração da imagem RGB Thematica
    with rasterio.open(sat_path) as sat_src:
        height, width = sat_src.height, sat_src.width
        sat_transform, sat_crs = sat_src.transform, sat_src.crs
        sat_height, sat_width = height, width

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

    # Identificadores das classes
    ids_floresta = [1, 3, 4, 5, 6, 49]
    ids_nao_floresta = [9, 10, 11, 12, 13, 29, 32, 50] 

    print("2/3 - Calculando fatias (Bounding Boxes)...")
    caixas = find_objects(matriz_seg)

    print("3/3 - Extraindo as 100 imagens de contexto (20% de borda)...")
    with rasterio.open(sat_path) as sat_src:
        for idx, row in df_alvos.iterrows():
            sp_id = int(row['ID_Segmento'])
            classe = row['Classe_Majoritaria']
            tipo = row['Tipo_Selecao']
            
            fatia = caixas[sp_id - 1] # Indexação de IDs
            if fatia is None:
                continue
                
            # Extrai os limites geográficos iniciais do segmento
            min_r, max_r = fatia[0].start, fatia[0].stop
            min_c, max_c = fatia[1].start, fatia[1].stop
            
            # Calcula o tamanho do superpixel (Bounding Box)
            h_sp = max_r - min_r
            w_sp = max_c - min_c
            
            # Adiciona 20% de margem de contexto
            pad_h = int(h_sp * 0.20)
            pad_w = int(w_sp * 0.20)
            
            r_ini = max(0, min_r - pad_h)
            r_fim = min(sat_height, max_r + pad_h)
            c_ini = max(0, min_c - pad_w)
            c_fim = min(sat_width, max_c + pad_w)
            
            window = Window.from_slices((r_ini, r_fim), (c_ini, c_fim))
            
            # -------------------------------------------
            # LEITURA DE DADOS (SATÉLITE E MAPBIOMAS)
            # -------------------------------------------
            sat_crop = sat_src.read((1,2,3), window=window).astype(np.float32)
            sat_crop = np.moveaxis(sat_crop, 0, -1)
            mb_crop = mb_aligned[r_ini:r_fim, c_ini:c_fim]
            seg_crop = matriz_seg[r_ini:r_fim, c_ini:c_fim]
            
            # --- IMAGEM 1: SATÉLITE RGB ---
            sat_rgb = np.zeros_like(sat_crop, dtype=np.uint8)
            for b in range(3):
                p2, p98 = np.percentile(sat_crop[:,:,b], (2, 98))
                if p98 > p2:
                    sat_rgb[:,:,b] = np.clip((sat_crop[:,:,b] - p2) / (p98 - p2) * 255.0, 0, 255.0)

            # --- IMAGEM 2: SATÉLITE CINZA ---
            # Fórmula de luminância (R*0.299 + G*0.587 + B*0.114)
            cinza_1ch = np.dot(sat_rgb, [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            sat_cinza = np.stack((cinza_1ch, cinza_1ch, cinza_1ch), axis=-1)

            # --- IMAGEM 3: RGB THEMATICO ---
            tema_rgb = np.zeros_like(sat_rgb, dtype=np.uint8)
            # Fundo mais escuro (0, 100, 0) para o verde claro (0, 255, 0) da borda aparecer bem
            tema_rgb[np.isin(mb_crop, ids_floresta)] = [0, 100, 0] 
            tema_rgb[np.isin(mb_crop, ids_nao_floresta)] = [255, 0, 0]

            # -------------------------------------------
            # MARCAÇÃO: APENAS O SEGMENTO EM VERDE
            # -------------------------------------------
            mascara_alvo = (seg_crop == sp_id)
            borda_alvo = find_boundaries(mascara_alvo, mode='thick')

            # O contorno amarelo foi removido. Pinta só o contorno principal de Verde
            sat_rgb[borda_alvo] = [0, 255, 0]
            sat_cinza[borda_alvo] = [0, 255, 0]
            tema_rgb[borda_alvo] = [0, 255, 0]

            # -------------------------------------------
            # SALVAR IMAGENS
            # -------------------------------------------
            # Padrão: {Classe}_{Perfeito/Imperfeito}_ID.jpg
            nome_arq = f"{classe}_{tipo}_ID{sp_id}.jpg"
            
            Image.fromarray(sat_rgb).save(os.path.join(pastas['satelite'], nome_arq), quality=95)
            Image.fromarray(sat_cinza).save(os.path.join(pastas['cinza'], nome_arq), quality=95)
            Image.fromarray(tema_rgb).save(os.path.join(pastas['rgb'], nome_arq), quality=95)

    print("🎉 Sucesso! As 100 imagens contextuais foram recortadas nas pastas:")
    for v in pastas.values():
        print(f"📁 {v}")

if __name__ == "__main__":
    gerar_crops_campanha()