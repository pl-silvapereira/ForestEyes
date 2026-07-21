import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
import time

# Se estiver a usar o Google Colab, descomente as duas linhas abaixo para montar o Drive automaticamente:
# from google.colab import drive
# drive.mount('/content/drive')

def executar_pan_sharpening():
    # ---------------------------------------------------------------------
    # CONFIGURAÇÃO DE CAMINHOS (Adaptado para o seu Google Drive)
    # ---------------------------------------------------------------------
    ROOT = "/content/drive/MyDrive/Mestrado/04-Projeto ForestEyes/ForestEyes"
    
    dir_input = os.path.join(ROOT, 'data', 'Input')
    dir_output = os.path.join(ROOT, 'data', 'Output')

    # Nomes dos ficheiros (Altere aqui se os seus ficheiros tiverem um nome diferente no Drive)
    NOME_ARQUIVO_MS = "CBERS4A_WPM_Multiespectral_8m.tif"
    NOME_ARQUIVO_PAN = "CBERS4A_WPM_Pancromatica_2m.tif"
    
    ms_path = os.path.join(dir_input, NOME_ARQUIVO_MS)
    pan_path = os.path.join(dir_input, NOME_ARQUIVO_PAN)
    
    saida_pansharp = os.path.join(dir_output, "21_SJC_CBERS4A_PanSharpened_2m.tif")

    # Verifica se a pasta existe, senão cria-a
    os.makedirs(dir_output, exist_ok=True)

    # Verifica se os ficheiros de entrada realmente existem no caminho especificado
    if not os.path.exists(ms_path):
        print(f"❌ Erro: Imagem Multiespectral (MS) não encontrada em:\n{ms_path}")
        print("Por favor, verifique o nome do ficheiro e faça o upload para a pasta Input.")
        return
        
    if not os.path.exists(pan_path):
        print(f"❌ Erro: Imagem Pancromática (PAN) não encontrada em:\n{pan_path}")
        print("Por favor, verifique o nome do ficheiro e faça o upload para a pasta Input.")
        return

    # ---------------------------------------------------------------------
    # EXECUÇÃO DO PAN-SHARPENING
    # ---------------------------------------------------------------------
    print("1/3 - A ler metadados e a configurar o Raster Virtual (VRT)...")
    
    with rasterio.open(pan_path) as pan_src, rasterio.open(ms_path) as ms_src:
        
        # Copia os metadados da imagem Pancromática (Alta Resolução - 2m)
        meta_pansharp = pan_src.meta.copy()
        # Atualizamos para 4 bandas (RGB + NIR) mantendo a resolução da PAN
        meta_pansharp.update({
            "count": ms_src.count,  # 4 bandas
            "dtype": ms_src.dtypes[0] # Mantém o tipo de dado original
        })

        # Configura o Raster Virtual para expandir a imagem de 8m para 2m "on-the-fly"
        vrt_options = {
            'resampling': Resampling.bilinear,
            'crs': pan_src.crs,
            'transform': pan_src.transform,
            'height': pan_src.height,
            'width': pan_src.width,
        }

        print("2/3 - A iniciar a fusão de imagens (Método Brovey)...")
        print(f"-> Resolução alvo: {pan_src.width} x {pan_src.height} píxeis")
        
        TILE_SIZE = 2000
        n_rows = int(np.ceil(pan_src.height / TILE_SIZE))
        n_cols = int(np.ceil(pan_src.width / TILE_SIZE))
        total_blocos = n_rows * n_cols
        bloco_atual = 0
        start_time = time.time()

        # Abre o ficheiro de saída e aplica o VRT
        with rasterio.open(saida_pansharp, "w", **meta_pansharp) as dst_vis:
            with WarpedVRT(ms_src, **vrt_options) as vrt_ms:
                
                # Processamento em Blocos para evitar estouro de memória (RAM) no Colab
                for row in range(0, pan_src.height, TILE_SIZE):
                    for col in range(0, pan_src.width, TILE_SIZE):
                        bloco_atual += 1
                        
                        win_h = min(TILE_SIZE, pan_src.height - row)
                        win_w = min(TILE_SIZE, pan_src.width - col)
                        window = Window(col, row, win_w, win_h)
                        
                        # Lê os dados do bloco atual
                        pan_data = pan_src.read(1, window=window).astype(np.float32)
                        ms_data = vrt_ms.read(window=window).astype(np.float32)
                        
                        # --- ALGORITMO BROVEY ---
                        # Evita divisão por zero
                        ms_sum = np.sum(ms_data, axis=0)
                        ms_sum[ms_sum == 0] = 1e-5 
                        
                        # Calcula o rácio
                        ratio = pan_data / ms_sum
                        
                        # Aplica o rácio a cada uma das bandas
                        pan_sharpened_data = np.zeros_like(ms_data)
                        for i in range(ms_src.count):
                            banda_fundida = ms_data[i] * ratio
                            max_val = np.iinfo(meta_pansharp['dtype']).max if np.issubdtype(meta_pansharp['dtype'], np.integer) else 1.0
                            pan_sharpened_data[i] = np.clip(banda_fundida, 0, max_val)
                        
                        # Converte e guarda o bloco
                        pan_sharpened_data = pan_sharpened_data.astype(meta_pansharp['dtype'])
                        dst_vis.write(pan_sharpened_data, window=window)
                        
                        if bloco_atual % 10 == 0 or bloco_atual == total_blocos:
                            print(f"   [{bloco_atual}/{total_blocos}] Blocos processados...")

    tempo_total = round(time.time() - start_time, 1)
    
    print("\n" + "="*50)
    print(f"✅ SCRIPT FINALIZADO EM {tempo_total} SEGUNDOS!")
    print("="*50)
    print(f"🎉 A imagem final com Pan-Sharpening (2m, 4 Bandas) foi guardada em:\n{saida_pansharp}")

if __name__ == "__main__":
    executar_pan_sharpening()