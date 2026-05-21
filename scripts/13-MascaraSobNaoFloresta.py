import os
import glob
import rasterio
import numpy as np
from PIL import Image
from dotenv import load_dotenv

def gerar_mascara_png(input_tif, output_png):
    print(f"Lendo o arquivo: {input_tif}...")
    
    # -------------------------------------------------------------
    # 1. DEFINIÇÃO DAS CLASSES 
    # -------------------------------------------------------------
    agropecuaria = [14, 15, 18, 19, 39, 20, 40, 62, 41, 36, 46, 47, 35, 48, 9, 21]
    infra_urbana = [24]
    agua_rocha = [26, 33, 31, 29] 
    ruido_descartadas = [27]
    
    ids_mascara = agropecuaria + infra_urbana + agua_rocha + ruido_descartadas
    
    try:
        with rasterio.open(input_tif) as src:
            mapbiomas_data = src.read(1)
            
            # Cria uma matriz 3D para a imagem (Altura, Largura, 4 canais RGBA)
            # O padrão é iniciar tudo com 0, ou seja: (0,0,0,0) = Preto 100% Transparente
            rgba_image = np.zeros((mapbiomas_data.shape[0], mapbiomas_data.shape[1], 4), dtype=np.uint8)
            
            # Encontra onde estão os pixels da máscara
            mask_condition = np.isin(mapbiomas_data, ids_mascara)
            
            # ONDE FOR MÁSCARA: Pintamos de preto opaco (R=0, G=0, B=0, Alpha=255)
            rgba_image[mask_condition] = [0, 0, 0, 255]
            
            # -------------------------------------------------------------
            # 2. SALVAR A IMAGEM PNG COM A BIBLIOTECA PILLOW (PIL)
            # -------------------------------------------------------------
            img = Image.fromarray(rgba_image, 'RGBA')
            img.save(output_png)
            
            # -------------------------------------------------------------
            # 3. CRIAR O ARQUIVO .PGW (Georreferenciamento para o QGIS)
            # -------------------------------------------------------------
            pgw_path = output_png.replace('.png', '.pgw')
            transform = src.transform
            
            # O World File exige o centro do pixel superior esquerdo
            x_center = transform.c + (transform.a / 2)
            y_center = transform.f + (transform.e / 2)
            
            with open(pgw_path, 'w') as f:
                f.write(f"{transform.a}\n")  # Tamanho do pixel (X)
                f.write(f"{transform.d}\n")  # Rotação (Y)
                f.write(f"{transform.b}\n")  # Rotação (X)
                f.write(f"{transform.e}\n")  # Tamanho do pixel (Y - negativo)
                f.write(f"{x_center}\n")     # Coordenada X
                f.write(f"{y_center}\n")     # Coordenada Y
                
        print(f"✅ Sucesso! Máscara visual salva em:\n{output_png}")
        print(f"✅ Georreferenciamento salvo em:\n{pgw_path}")
        
    except Exception as e:
        print(f"❌ Erro durante o processamento da imagem: {e}")

# ==========================================
# EXECUÇÃO E BUSCA DINÂMICA
# ==========================================
if __name__ == "__main__":
    load_dotenv()
    ROOT = os.getenv('PROJECT_ROOT')
    
    if not ROOT:
        print("❌ Erro: Variável 'PROJECT_ROOT' não encontrada no .env.")
        exit()

    mapbiomas_dir = os.path.join(ROOT, 'data', 'MapBiomas')
    pasta_saida = os.path.join(ROOT, 'data', 'Output')
    
    print("=== Buscando Arquivo MapBiomas de 2021 ===")
    busca = glob.glob(os.path.join(mapbiomas_dir, "*2021*coverage*10m*.tif"))
    
    if not busca:
        busca = glob.glob(os.path.join(mapbiomas_dir, "*coverage_10m*.tif"))
        
    if not busca:
        print(f"❌ Erro: Nenhum arquivo '*coverage_10m*.tif' encontrado em {mapbiomas_dir}")
        exit()
        
    ARQUIVO_ENTRADA = busca[0]
    print(f"🔍 Arquivo encontrado: {os.path.basename(ARQUIVO_ENTRADA)}")
    
    # Arquivo agora salvo como .png
    ARQUIVO_SAIDA = os.path.join(pasta_saida, "mascara_analise_2021_final.png")
    os.makedirs(pasta_saida, exist_ok=True)
    
    gerar_mascara_png(ARQUIVO_ENTRADA, ARQUIVO_SAIDA)