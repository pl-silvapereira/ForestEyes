import os
import glob
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
    
    ids_mascara = agropecuaria + infra_urbana + agua_rocha + ruido_descartadas
    
    # -------------------------------------------------------------
    # 2. PROCESSAMENTO DA IMAGEM
    # -------------------------------------------------------------
    try:
        with rasterio.open(input_tif) as src:
            mapbiomas_data = src.read(1)
            
            # Matrizes RGBA iniciando com valor 0 (Preto e 100% Transparente)
            out_r = np.zeros_like(mapbiomas_data, dtype=np.uint8)
            out_g = np.zeros_like(mapbiomas_data, dtype=np.uint8)
            out_b = np.zeros_like(mapbiomas_data, dtype=np.uint8)
            out_a = np.zeros_like(mapbiomas_data, dtype=np.uint8)
            
            # Aplica a máscara (Muda o Alpha para 255/Opaco nas classes alvo)
            mask_condition = np.isin(mapbiomas_data, ids_mascara)
            out_a[mask_condition] = 255
            
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
                dst.write(out_r, 1) # R
                dst.write(out_g, 2) # G
                dst.write(out_b, 3) # B
                dst.write(out_a, 4) # Alpha
                
        print(f"✅ Sucesso! Máscara salva em:\n{output_tif}")
    except Exception as e:
        print(f"❌ Erro durante o processamento da imagem: {e}")

# ==========================================
# EXECUÇÃO E BUSCA DINÂMICA
# ==========================================
if __name__ == "__main__":
    # 1. Carrega as configurações (Igual ao Script 02)
    load_dotenv()
    ROOT = os.getenv('PROJECT_ROOT')
    
    if not ROOT:
        print("❌ Erro: Variável 'PROJECT_ROOT' não encontrada no .env.")
        exit()

    # 2. Define os diretórios base
    mapbiomas_dir = os.path.join(ROOT, 'data', 'MapBiomas')
    pasta_saida = os.path.join(ROOT, 'data', 'Output')
    
    # 3. Busca dinâmica pelo arquivo TIF (procurando por arquivos do MapBiomas contendo 2021)
    print("=== Buscando Arquivo MapBiomas de 2021 ===")
    busca = glob.glob(os.path.join(mapbiomas_dir, "*2021*coverage*10m*.tif"))
    
    # Fallback: Se não achar com "2021" no nome, tenta a busca genérica do Script 02
    if not busca:
        busca = glob.glob(os.path.join(mapbiomas_dir, "*coverage_10m*.tif"))
        
    if not busca:
        print(f"❌ Erro: Nenhum arquivo '*coverage_10m*.tif' encontrado em {mapbiomas_dir}")
        exit()
        
    # Pega o primeiro arquivo retornado pela busca
    ARQUIVO_ENTRADA = busca[0]
    print(f"🔍 Arquivo encontrado: {os.path.basename(ARQUIVO_ENTRADA)}")
    
    # 4. Configura a saída e executa
    ARQUIVO_SAIDA = os.path.join(pasta_saida, "mascara_analise_2021.tif")
    os.makedirs(pasta_saida, exist_ok=True)
    
    gerar_mascara_mapbiomas(ARQUIVO_ENTRADA, ARQUIVO_SAIDA)