import rasterio
import numpy as np
import os

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
    agua_rocha = [26, 33, 31, 29] # 29 é Afloramento Rochoso, 33/31 Água
    ruido_descartadas = [27]
    
    # Classes que ficarão TRANSPARENTES (Você não precisa delas na máscara)
    florestas = [1, 3, 4, 5, 6, 49]
    veg_herbacea = [10, 11, 12, 32, 50, 13]
    
    # Junta todos os IDs que vão compor a máscara num único vetor
    ids_mascara = agropecuaria + infra_urbana + agua_rocha + ruido_descartadas
    
    # -------------------------------------------------------------
    # 2. PROCESSAMENTO DA IMAGEM
    # -------------------------------------------------------------
    with rasterio.open(input_tif) as src:
        # Lê a banda 1 (dados com os IDs do MapBiomas)
        mapbiomas_data = src.read(1)
        
        # Cria as 4 matrizes para a imagem final (R, G, B, Alpha)
        # Ao usar np.zeros, todos já começam com valor 0.
        # Ou seja: R=0, G=0, B=0 (Preto) e A=0 (100% Transparente)
        out_r = np.zeros_like(mapbiomas_data, dtype=np.uint8)
        out_g = np.zeros_like(mapbiomas_data, dtype=np.uint8)
        out_b = np.zeros_like(mapbiomas_data, dtype=np.uint8)
        out_a = np.zeros_like(mapbiomas_data, dtype=np.uint8)
        
        # Encontra todos os pixels que têm valor igual aos IDs da máscara
        # Onde for verdadeiro, nós mudamos o Alpha para 255 (Fica Opaco)
        mask_condition = np.isin(mapbiomas_data, ids_mascara)
        out_a[mask_condition] = 255
        
        # Prepara os metadados para salvar o novo TIF em formato RGBA
        profile = src.profile
        profile.update(
            dtype=rasterio.uint8,
            count=4, # 4 bandas ao invés de 1
            nodata=None, # O canal Alpha assume o papel do "nodata"
            photometric='RGB' # QGIS/Softwares vão entender como imagem colorida c/ transparência
        )
        
        # -------------------------------------------------------------
        # 3. SALVAR O RESULTADO
        # -------------------------------------------------------------
        with rasterio.open(output_tif, 'w', **profile) as dst:
            dst.write(out_r, 1) # Red (Preto)
            dst.write(out_g, 2) # Green (Preto)
            dst.write(out_b, 3) # Blue (Preto)
            dst.write(out_a, 4) # Transparência (0 ou 255)
            
    print(f"Sucesso! Máscara salva em: {output_tif}")

# ==========================================
# COMO EXECUTAR O SCRIPT
# ==========================================
if __name__ == "__main__":
    # Substitua pelos caminhos reais do seu Google Drive/Computador
    ARQUIVO_ENTRADA = "data/mapbiomas_2021.tif" 
    ARQUIVO_SAIDA = "Output/mascara_analise_2021.tif"
    
    # Certifica-se de que a pasta de saída existe (caso não, o Python cria)
    os.makedirs(os.path.dirname(ARQUIVO_SAIDA), exist_ok=True)
    
    gerar_mascara_mapbiomas(ARQUIVO_ENTRADA, ARQUIVO_SAIDA)