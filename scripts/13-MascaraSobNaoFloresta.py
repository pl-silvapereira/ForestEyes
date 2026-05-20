import os
import rasterio
import numpy as np
from dotenv import load_dotenv

def gerar_mascara_mapbiomas(input_tif, output_tif):
    """
    Lê um raster do MapBiomas e gera um novo TIF onde:
    - Agropecuária, Urbano, Água/Rocha e Ruído = Preto (Máscara)
    - Florestas e Vegetação Herbácea = Transparente
    """
    print(f"Lendo o arquivo: {input_tif}...")
    
    # -------------------------------------------------------------
    # 1. DEFINIÇÃO DAS CLASSES (IDs baseados na legenda do MapBiomas)
    # -------------------------------------------------------------
    
    # Classes que SERÃO a máscara (Cor Preta)
    agropecuaria = [14, 15, 18, 19, 39, 20, 40, 62, 41, 36, 46, 47, 35, 48, 9, 21]
    infra_urbana = [24]
    agua_rocha = [26, 33, 31, 29] 
    ruido_descartadas = [27]
    
    # Classes que ficarão TRANSPARENTES
    florestas = [1, 3, 4, 5, 6, 49]
    veg_herbacea = [10, 11, 12, 32, 50, 13]
    
    # Junta todos os IDs que vão compor a máscara num único vetor
    ids_mascara = agropecuaria + infra_urbana + agua_rocha + ruido_descartadas
    
    # -------------------------------------------------------------
    # 2. PROCESSAMENTO DA IMAGEM
    # -------------------------------------------------------------
    with rasterio.open(input_tif) as src:
        mapbiomas_data = src.read(1)
        
        # Cria as 4 matrizes (RGBA) começando com valor 0 (Preto e 100% Transparente)
        out_r = np.zeros_like(mapbiomas_data, dtype=np.uint8)
        out_g = np.zeros_like(mapbiomas_data, dtype=np.uint8)
        out_b = np.zeros_like(mapbiomas_data, dtype=np.uint8)
        out_a = np.zeros_like(mapbiomas_data, dtype=np.uint8)
        
        # Aplica a máscara: onde for da classe de interesse, muda o Alpha para 255 (Opaco)
        mask_condition = np.isin(mapbiomas_data, ids_mascara)
        out_a[mask_condition] = 255
        
        # Prepara os metadados para salvar em RGBA
        profile = src.profile
        profile.update(
            dtype=rasterio.uint8,
            count=4, 
            nodata=None, 
            photometric='RGB' 
        )
        
        # -------------------------------------------------------------
        # 3. SALVAR O RESULTADO
        # -------------------------------------------------------------
        with rasterio.open(output_tif, 'w', **profile) as dst:
            dst.write(out_r, 1) # Red (Preto)
            dst.write(out_g, 2) # Green (Preto)
            dst.write(out_b, 3) # Blue (Preto)
            dst.write(out_a, 4) # Alpha (Transparência)
            
    print(f"Sucesso! Máscara salva em: {output_tif}")

# ==========================================
# EXECUÇÃO E VARIÁVEIS DE AMBIENTE
# ==========================================
if __name__ == "__main__":
    # 1. Carrega as variáveis do arquivo .env
    load_dotenv()
    
    # 2. Busca o caminho do diretório de dados (ajuste o nome da variável se necessário)
    # Se a variável 'DATA_DIR' não existir no .env, ele usa a pasta atual ('./data') como padrão seguro
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    
    # 3. Monta os caminhos completos e dinâmicos de entrada e saída
    ARQUIVO_ENTRADA = os.path.join(DATA_DIR, "mapbiomas_2021.tif")
    
    # Cria uma pasta 'Output' dentro da sua pasta de dados do Drive/Local
    PASTA_SAIDA = os.path.join(DATA_DIR, "Output")
    ARQUIVO_SAIDA = os.path.join(PASTA_SAIDA, "mascara_analise_2021.tif")
    
    # Certifica-se de que a pasta de saída existe antes de salvar
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    
    # 4. Executa a função
    gerar_mascara_mapbiomas(ARQUIVO_ENTRADA, ARQUIVO_SAIDA)