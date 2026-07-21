import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from dotenv import load_dotenv
import time

def executar_pan_sharpening():
    load_dotenv()
    ROOT = os.getenv('PROJECT_ROOT')
    if not ROOT:
        print("❌ Erro: Variável 'PROJECT_ROOT' não encontrada.")
        return

    dir_input = os.path.join(ROOT, 'data', 'Input')
    dir_output = os.path.join(ROOT, 'data', 'Output')

    # =========================================================================
    # ATENÇÃO: Ajuste os nomes dos arquivos conforme baixados do catálogo INPE
    # O arquivo MS deve conter as 4 bandas (Azul, Verde, Vermelho, NIR) em 8m
    # O arquivo PAN deve conter a banda Pancromática em 2m
    # =========================================================================
    ms_path = os.path.join(dir_input, "CBERS4A_WPM_Multiespectral_8m.tif")
    pan_path = os.path.join(dir_input, "CBERS4A_WPM_Pancromatica_2m.tif")
    
    saida_pansharp = os.path.join(dir_output, "21_SJC_CBERS4A_PanSharpened_2m.tif")

    if not os.path.exists(ms_path) or not os.path.exists(pan_path):
        print(f"❌ Erro: Arquivos base não encontrados. Verifique se as imagens MS (8m) e PAN (2m) estão na pasta {dir_input}.")
        return

    print("1/3 - Lendo metadados e configurando o Raster Virtual (VRT)...")
    
    with rasterio.open(pan_path) as pan_src, rasterio.open(ms_path) as ms_src:
        
        # Copia os metadados da imagem Pancromática (Alta Resolução - 2m)
        meta_pansharp = pan_src.meta.copy()
        # Atualizamos para 4 bandas (RGB + NIR) mantendo a resolução da PAN
        meta_pansharp.update({
            "count": ms_src.count,  # 4 bandas
            "dtype": ms_src.dtypes[0] # Mantém o tipo de dado original (ex: uint16 ou uint8)
        })

        # Configura o Raster Virtual para expandir a imagem de 8m para 2m "no ar"
        vrt_options = {
            'resampling': Resampling.bilinear,
            'crs': pan_src.crs,
            'transform': pan_src.transform,
            'height': pan_src.height,
            'width': pan_src.width,
        }

        print("2/3 - Iniciando a fusão de imagens (Pan-Sharpening - Método Brovey)...")
        print(f"-> Resolução alvo: {pan_src.width} x {pan_src.height} pixels")
        
        TILE_SIZE = 2000
        n_rows = int(np.ceil(pan_src.height / TILE_SIZE))
        n_cols = int(np.ceil(pan_src.width / TILE_SIZE))
        total_blocos = n_rows * n_cols
        bloco_atual = 0
        start_time = time.time()

        # Abre o arquivo de saída e aplica o VRT
        with rasterio.open(saida_pansharp, "w", **meta_pansharp) as dst_vis:
            with WarpedVRT(ms_src, **vrt_options) as vrt_ms:
                
                # Processamento em Blocos para evitar estouro de memória (RAM)
                for row in range(0, pan_src.height, TILE_SIZE):
                    for col in range(0, pan_src.width, TILE_SIZE):
                        bloco_atual += 1
                        
                        win_h = min(TILE_SIZE, pan_src.height - row)
                        win_w = min(TILE_SIZE, pan_src.width - col)
                        window = Window(col, row, win_w, win_h)
                        
                        # Lê os dados do bloco atual
                        # A imagem MS é lida já redimensionada para 2m através do VRT
                        pan_data = pan_src.read(1, window=window).astype(np.float32)
                        ms_data = vrt_ms.read(window=window).astype(np.float32)
                        
                        # --- ALGORITMO BROVEY PARA PAN-SHARPENING ---
                        # Evita divisão por zero somando uma constante minúscula
                        ms_sum = np.sum(ms_data, axis=0)
                        ms_sum[ms_sum == 0] = 1e-5 
                        
                        # Calcula a razão entre a alta resolução e a soma da baixa resolução
                        ratio = pan_data / ms_sum
                        
                        # Aplica a razão a cada uma das 4 bandas (R, G, B, NIR)
                        pan_sharpened_data = np.zeros_like(ms_data)
                        for i in range(ms_src.count):
                            banda_fundida = ms_data[i] * ratio
                            # Clip para evitar estourar o limite de cor do formato original
                            max_val = np.iinfo(meta_pansharp['dtype']).max if np.issubdtype(meta_pansharp['dtype'], np.integer) else 1.0
                            pan_sharpened_data[i] = np.clip(banda_fundida, 0, max_val)
                        
                        # Converte de volta para o tipo de dado original e salva o bloco
                        pan_sharpened_data = pan_sharpened_data.astype(meta_pansharp['dtype'])
                        dst_vis.write(pan_sharpened_data, window=window)
                        
                        if bloco_atual % 10 == 0 or bloco_atual == total_blocos:
                            print(f"   [{bloco_atual}/{total_blocos}] Blocos processados...")

    tempo_total = round(time.time() - start_time, 1)
    
    print("\n" + "="*50)
    print(f"✅ SCRIPT FINALIZADO EM {tempo_total} SEGUNDOS!")
    print("="*50)
    print(f"🎉 A imagem final com Pan-Sharpening (2m, 4 Bandas) foi salva em:\n{saida_pansharp}")

if __name__ == "__main__":
    executar_pan_sharpening()