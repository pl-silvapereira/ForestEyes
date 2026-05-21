import os
import glob
import rasterio
import numpy as np
from dotenv import load_dotenv

def gerar_mascara_mapbiomas(input_tif, output_tif):
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
            
            # -------------------------------------------------------------
            # 2. PROCESSAMENTO RGBA (Força a cor preta automática no QGIS)
            # -------------------------------------------------------------
            
            # Canais R, G e B começam totalmente zerados (Preto absoluto)
            out_r = np.zeros_like(mapbiomas_data, dtype=np.uint8)
            out_g = np.zeros_like(mapbiomas_data, dtype=np.uint8)
            out_b = np.zeros_like(mapbiomas_data, dtype=np.uint8)
            
            # Canal Alpha (Transparência) começa zerado (100% invisível)
            out_a = np.zeros_like(mapbiomas_data, dtype=np.uint8)
            
            # Aplica a máscara: Apenas onde for área de interesse, o Alpha vai para 255 (Opaco)
            mask_condition = np.isin(mapbiomas_data, ids_mascara)
            out_a[mask_condition] = 255
            
            # Prepara os metadados para salvar como RGBA
            profile = src.profile
            profile.update(
                dtype=rasterio.uint8,
                count=4,  # Mudamos para 4 bandas
                nodata=None, # O QGIS usará o canal Alpha como transparência nativa
                photometric='RGB' 
            )
            
            # -------------------------------------------------------------
            # 3. SALVAR O RESULTADO
            # -------------------------------------------------------------
            with rasterio.open(output_tif, 'w', **profile) as dst:
                dst.write(out_r, 1) # R (Sempre 0)
                dst.write(out_g, 2) # G (Sempre 0)
                dst.write(out_b, 3) # B (Sempre 0)
                dst.write(out_a, 4) # Alpha (0 para fundo, 255 para a máscara)
                
        print(f"✅ Sucesso! Máscara preta sólida automática salva em:\n{output_tif}")
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
    
    ARQUIVO_SAIDA = os.path.join(pasta_saida, "mascara_analise_2021_final.tif")
    os.makedirs(pasta_saida, exist_ok=True)
    
    gerar_mascara_mapbiomas(ARQUIVO_ENTRADA, ARQUIVO_SAIDA)