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

def gerar_crops_campanha_alta_resolucao():
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

    ids_floresta = [1, 3, 4, 5, 6, 49]
    ids_nao_floresta = [9, 10, 11, 12, 13, 29, 32, 50] 

    print("2/3 - Calculando fatias geográficas...")
    caixas = find_objects(matriz_seg)

    print("3/3 - Processando as 100 imagens com motor Fotográfico de Alta Resolução...")
    with rasterio.open(sat_path) as sat_src:
        for idx, row in df_alvos.iterrows():
            sp_id = int(row['ID_Segmento'])
            classe = row['Classe_Majoritaria']
            tipo = row['Tipo_Selecao']
            
            fatia = caixas[sp_id - 1] 
            if fatia is None:
                continue
                
            min_r, max_r = fatia[0].start, fatia[0].stop
            min_c, max_c = fatia[1].start, fatia[1].stop
            
            h_sp = max_r - min_r
            w_sp = max_c - min_c
            
            cy = (min_r + max_r) // 2
            cx = (min_c + max_c) // 2
            
            # Margem de 20%
            pad_h = int(h_sp * 0.20)
            pad_w = int(w_sp * 0.20)
            
            # Calcula o tamanho alvo nativo (Mínimo de 512px reais de satélite para dar contexto HD)
            target_h = max(h_sp + 2 * pad_h, 512)
            target_w = max(w_sp + 2 * pad_w, 512)
            
            # Força a janela a ser perfeitamente quadrada para não achatar no upscale
            target_size = max(target_h, target_w)
            
            r_ini = cy - target_size // 2
            r_fim = cy + target_size // 2
            c_ini = cx - target_size // 2
            c_fim = cx + target_size // 2
            
            # Ajuste de limites do rasterio para leitura
            read_r_ini = max(0, r_ini)
            read_r_fim = min(sat_height, r_fim)
            read_c_ini = max(0, c_ini)
            read_c_fim = min(sat_width, c_fim)
            
            window = Window.from_slices((read_r_ini, read_r_fim), (read_c_ini, read_c_fim))
            
            sat_crop = sat_src.read((1,2,3), window=window).astype(np.float32)
            sat_crop = np.moveaxis(sat_crop, 0, -1)
            mb_crop = mb_aligned[read_r_ini:read_r_fim, read_c_ini:read_c_fim]
            seg_crop = matriz_seg[read_r_ini:read_r_fim, read_c_ini:read_c_fim]
            
            # Preenchimento preto caso o contexto saia das bordas da cidade
            pad_top = max(0, -r_ini)
            pad_bottom = max(0, r_fim - sat_height)
            pad_left = max(0, -c_ini)
            pad_right = max(0, c_fim - sat_width)
            
            if any([pad_top, pad_bottom, pad_left, pad_right]):
                sat_crop = np.pad(sat_crop, ((pad_top, pad_bottom), (pad_left, pad_right), (0,0)), mode='constant')
                mb_crop = np.pad(mb_crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
                seg_crop = np.pad(seg_crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

            # --- PROCESSAMENTO RGB BASE ---
            sat_rgb = np.zeros_like(sat_crop, dtype=np.uint8)
            for b in range(3):
                p2, p98 = np.percentile(sat_crop[:,:,b], (2, 98))
                if p98 > p2:
                    sat_rgb[:,:,b] = np.clip((sat_crop[:,:,b] - p2) / (p98 - p2) * 255.0, 0, 255.0)

            cinza_1ch = np.dot(sat_rgb, [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            sat_cinza = np.stack((cinza_1ch, cinza_1ch, cinza_1ch), axis=-1)

            # -------------------------------------------------------------
            # UPSCALING FOTOGRÁFICO AVANÇADO (LANCZOS)
            # -------------------------------------------------------------
            ALVO_PX = (1024, 1024)
            
            # Interpolação fotográfica (LANCZOS) suaviza e preserva a resolução CBERS
            img_sat = Image.fromarray(sat_rgb).resize(ALVO_PX, Image.LANCZOS)
            img_cinza = Image.fromarray(sat_cinza).resize(ALVO_PX, Image.LANCZOS)
            
            # Interpolação Nearest para máscaras matemáticas (Impede que as IDs se misturem)
            seg_1024 = np.array(Image.fromarray(seg_crop, mode='I').resize(ALVO_PX, Image.NEAREST))
            mb_1024 = np.array(Image.fromarray(mb_crop).resize(ALVO_PX, Image.NEAREST))

            # Converte de volta para Arrays Numpy para pintar
            sat_rgb_1024 = np.array(img_sat)
            sat_cinza_1024 = np.array(img_cinza)

            # Constrói o RGB Temático já na Alta Resolução
            tema_rgb_1024 = np.zeros_like(sat_rgb_1024, dtype=np.uint8)
            tema_rgb_1024[np.isin(mb_1024, ids_floresta)] = [0, 100, 0] 
            tema_rgb_1024[np.isin(mb_1024, ids_nao_floresta)] = [255, 0, 0]

            # -------------------------------------------------------------
            # MARCAÇÃO DE BORDA VETORIAL (NÍTIDA)
            # -------------------------------------------------------------
            # Identifica as bordas do alvo DIRETAMENTE na matriz 1024x1024. 
            # Isso garante que a linha será afiada e terá 1 pixel exato de espessura HD.
            mascara_alvo_1024 = (seg_1024 == sp_id)
            borda_alvo_1024 = find_boundaries(mascara_alvo_1024, mode='thick')

            # O contorno amarelo foi removido. Pinta só o contorno principal de Verde Puro
            sat_rgb_1024[borda_alvo_1024] = [0, 255, 0]
            sat_cinza_1024[borda_alvo_1024] = [0, 255, 0]
            tema_rgb_1024[borda_alvo_1024] = [0, 255, 0]

            # -------------------------------------------
            # SALVAR IMAGENS 
            # -------------------------------------------
            nome_arq = f"{classe}_{tipo}_ID{sp_id}.jpg"
            
            Image.fromarray(sat_rgb_1024).save(os.path.join(pastas['satelite'], nome_arq), quality=95)
            Image.fromarray(sat_cinza_1024).save(os.path.join(pastas['cinza'], nome_arq), quality=95)
            Image.fromarray(tema_rgb_1024).save(os.path.join(pastas['rgb'], nome_arq), quality=95)

            if (idx + 1) % 20 == 0:
                print(f"   [{idx + 1}/{len(df_alvos)}] imagens em Alta Resolução exportadas...")

    print("\n🎉 Sucesso! As 100 imagens de campanha (1024x1024 px) estão salvas com máxima qualidade.")

if __name__ == "__main__":
    gerar_crops_campanha_alta_resolucao()