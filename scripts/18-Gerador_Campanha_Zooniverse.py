import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import center_of_mass
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv

def gerar_imagens_zooniverse():
    load_dotenv()
    ROOT = os.getenv('PROJECT_ROOT')
    if not ROOT:
        print("❌ Erro: Variável 'PROJECT_ROOT' não encontrada no .env.")
        return

    dir_output = os.path.join(ROOT, 'data', 'Output')
    
    # Arquivos base gerados nos scripts anteriores
    sat_path = os.path.join(dir_output, "02_SJC_Recortado_MapBiomas.tif")
    mapa_path = os.path.join(dir_output, "17_SJC_Mapa_Segmentado.tif")
    npy_path = os.path.join(dir_output, "17_SJC_Matriz_Segmentacao.npy")
    csv_path = os.path.join(dir_output, "17_SJC_Estatisticas_Superpixels.csv")

    if not all(os.path.exists(p) for p in [sat_path, mapa_path, npy_path, csv_path]):
        print("❌ Erro: Ficheiros do Script 17 (NPY, CSV, TIF) ou CBERS não encontrados.")
        return

    # Pastas de Saída para a Campanha
    pastas_campanha = {
        'RGB': os.path.join(dir_output, 'Zooniverse_Crops', '1_RGB'),
        'Falsa_Cor': os.path.join(dir_output, 'Zooniverse_Crops', '2_FalsaCor'),
        'NDVI': os.path.join(dir_output, 'Zooniverse_Crops', '3_NDVI'),
        'Tematico': os.path.join(dir_output, 'Zooniverse_Crops', '4_Tematico')
    }
    
    for pasta in pastas_campanha.values():
        os.makedirs(pasta, exist_ok=True)

    # -------------------------------------------------------------
    # 1. PREPARAÇÃO DOS DADOS
    # -------------------------------------------------------------
    print("1/4 - Carregando Matriz de Segmentação e Estatísticas...")
    matriz_segmentos = np.load(npy_path)
    df_stats = pd.read_csv(csv_path, sep=';')
    
    # Filtra apenas os segmentos de Floresta (O alvo da campanha)
    df_floresta = df_stats[df_stats['Classe_Majoritaria'] == 'Floresta'].copy()
    ids_alvo = df_floresta['ID_Segmento'].values
    print(f"-> {len(ids_alvo)} segmentos de floresta selecionados para a campanha.")

    print("2/4 - Calculando centróides exatos dos segmentos...")
    centroides = center_of_mass(matriz_segmentos, matriz_segmentos, ids_alvo)

    # -------------------------------------------------------------
    # 2. FUNÇÕES AUXILIARES DE IMAGEM
    # -------------------------------------------------------------
    def normalizar_banda(banda):
        """Aplica contraste dinâmico ignorando nuvens/sombras extremas"""
        p2, p98 = np.percentile(banda, (2, 98))
        if p98 > p2:
            banda_norm = np.clip((banda - p2) / (p98 - p2) * 255.0, 0, 255.0)
        else:
            banda_norm = np.zeros_like(banda)
        return banda_norm.astype(np.uint8)

    cmap_ndvi = plt.get_cmap('RdYlGn')
    TAMANHO = 256
    MEIO = TAMANHO // 2

    # -------------------------------------------------------------
    # 3. EXTRAÇÃO DAS IMAGENS (CROP LOOP)
    # -------------------------------------------------------------
    print("3/4 - Gerando fotografias contextuais (Isto pode levar alguns minutos)...")
    
    with rasterio.open(sat_path) as sat_src, rasterio.open(mapa_path) as map_src:
        tem_nir = sat_src.count >= 4
        if not tem_nir:
            print("⚠️ Aviso: A imagem CBERS possui menos de 4 bandas. Falsa Cor e NDVI não serão precisos.")

        processados = 0
        
        for idx, sp_id in enumerate(ids_alvo):
            centro_y, centro_x = centroides[idx]
            
            if np.isnan(centro_y) or np.isnan(centro_x):
                continue
                
            cy, cx = int(centro_y), int(centro_x)
            
            r_ini, r_fim = cy - MEIO, cy + MEIO
            c_ini, c_fim = cx - MEIO, cx + MEIO
            
            pad_top = max(0, -r_ini)
            pad_bottom = max(0, r_fim - sat_src.height)
            pad_left = max(0, -c_ini)
            pad_right = max(0, c_fim - sat_src.width)
            
            win_r_ini = max(0, r_ini)
            win_r_fim = min(sat_src.height, r_fim)
            win_c_ini = max(0, c_ini)
            win_c_fim = min(sat_src.width, c_fim)
            
            window = Window.from_slices((win_r_ini, win_r_fim), (win_c_ini, win_c_fim))
            
            sat_crop = sat_src.read(window=window).astype(np.float32)
            map_crop = map_src.read(window=window)
            seg_crop = matriz_segmentos[win_r_ini:win_r_fim, win_c_ini:win_c_fim]
            
            if any([pad_top, pad_bottom, pad_left, pad_right]):
                sat_crop = np.pad(sat_crop, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
                map_crop = np.pad(map_crop, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
                seg_crop = np.pad(seg_crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

            # --- PREPARANDO AS COMPOSIÇÕES ---
            r = normalizar_banda(sat_crop[0])
            g = normalizar_banda(sat_crop[1])
            b = normalizar_banda(sat_crop[2])
            img_rgb = np.dstack((r, g, b))

            if tem_nir:
                nir = sat_crop[3]
                nir_norm = normalizar_banda(nir)
                img_falsa_cor = np.dstack((nir_norm, r, g))
                
                numerador = (nir - sat_crop[0])
                denominador = (nir + sat_crop[0])
                ndvi_raw = np.divide(numerador, denominador, out=np.zeros_like(numerador), where=denominador!=0)
                
                ndvi_colorido = cmap_ndvi((ndvi_raw + 1) / 2.0) 
                img_ndvi = (ndvi_colorido[:, :, :3] * 255).astype(np.uint8)
            else:
                img_falsa_cor = img_rgb.copy()
                img_ndvi = img_rgb.copy()

            img_tematica = np.moveaxis(map_crop, 0, -1).astype(np.uint8)

            # --- DESENHANDO OS CONTORNOS ---
            bordas_gerais = find_boundaries(seg_crop, mode='inner', background=0)
            mascara_alvo = (seg_crop == sp_id)
            borda_alvo = find_boundaries(mascara_alvo, mode='thick')

            for img in [img_rgb, img_falsa_cor, img_ndvi, img_tematica]:
                img[bordas_gerais] = [255, 255, 0]
                img[borda_alvo] = [0, 255, 255]

            # --- SALVANDO AS FOTOS ---
            nome_arq = f"SJC_Sp_{sp_id}.jpg"
            
            Image.fromarray(img_rgb).save(os.path.join(pastas_campanha['RGB'], nome_arq), quality=95)
            Image.fromarray(img_falsa_cor).save(os.path.join(pastas_campanha['Falsa_Cor'], nome_arq), quality=95)
            Image.fromarray(img_ndvi).save(os.path.join(pastas_campanha['NDVI'], nome_arq), quality=95)
            Image.fromarray(img_tematica).save(os.path.join(pastas_campanha['Tematico'], nome_arq), quality=95)

            processados += 1
            if processados % 500 == 0:
                print(f"   [{processados}/{len(ids_alvo)}] pacotes de imagens gerados...")

    print("\n4/4 - 🎉 Sucesso! Imagens da campanha exportadas e prontas para upload.")
    for nome, caminho in pastas_campanha.items():
        print(f"📁 {nome}: {caminho}")

if __name__ == "__main__":
    gerar_imagens_zooniverse()