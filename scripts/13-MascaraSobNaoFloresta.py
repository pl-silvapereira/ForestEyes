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
            
            # Cria a matriz base preenchida com 0 (0 será o fundo transparente/NODATA)
            out_data = np.zeros_like(mapbiomas_data, dtype=np.uint8)
            
            # ONDE FOR MÁSCARA: Padronizamos tudo para o valor 1
            mask_condition = np.isin(mapbiomas_data, ids_mascara)
            out_data[mask_condition] = 1
            
            # Prepara os metadados
            profile = src.profile
            profile.update(
                dtype=rasterio.uint8,
                count=1, 
                nodata=0  # Define 0 como invisível no QGIS
            )
            
            # -------------------------------------------------------------
            # 2. SALVAR COM PALETA PRETA CUSTOMIZADA
            # -------------------------------------------------------------
            with rasterio.open(output_tif, 'w', **profile) as dst:
                dst.write(out_data, 1)
                
                # Criamos um "Mapa de Cores" manual.
                # Formato: {valor_do_pixel: (Red, Green, Blue, Alpha)}
                # 0 = Preto (R=0, G=0, B=0) com Opacidade Máxima (Alpha=255)
                colormap = {
                    1: (0, 0, 0, 255)
                }
                
                dst.write_colormap(1, colormap)
                
        print(f"✅ Sucesso! Máscara preta sólida salva em:\n{output_tif}")
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
    
    # Atualizado o nome do arquivo de saída para refletir a máscara final
    ARQUIVO_SAIDA = os.path.join(pasta_saida, "mascara_analise_2021_preta.tif")
    os.makedirs(pasta_saida, exist_ok=True)
    
    gerar_mascara_mapbiomas(ARQUIVO_ENTRADA, ARQUIVO_SAIDA)