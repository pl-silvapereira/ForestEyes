import os
import glob
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.features import shapes
from PIL import Image
from dotenv import load_dotenv

def gerar_mascara_completa(input_tif, output_png, output_shp):
    print(f"A ler o ficheiro: {input_tif}...")
    
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
            crs = src.crs
            transform = src.transform
            
            print("A calcular a localização da máscara...")
            # Encontra onde estão os píxeis de interesse apenas uma vez
            mask_condition = np.isin(mapbiomas_data, ids_mascara)
            
            # =========================================================
            # PARTE A: GERAR O PNG E O GEORREFERENCIAMENTO (.PGW)
            # =========================================================
            print("A gerar a imagem PNG e o ficheiro World File (.pgw)...")
            
            # Cria a matriz RGBA iniciando a 0 (Transparente)
            rgba_image = np.zeros((mapbiomas_data.shape[0], mapbiomas_data.shape[1], 4), dtype=np.uint8)
            
            # Pinta a máscara de Preto Opaco
            rgba_image[mask_condition] = [0, 0, 0, 255]
            
            # Guarda o PNG
            img = Image.fromarray(rgba_image, 'RGBA')
            img.save(output_png)
            
            # Guarda o ficheiro de coordenadas (PGW)
            pgw_path = output_png.replace('.png', '.pgw')
            x_center = transform.c + (transform.a / 2)
            y_center = transform.f + (transform.e / 2)
            
            with open(pgw_path, 'w') as f:
                f.write(f"{transform.a}\n")  
                f.write(f"{transform.d}\n")  
                f.write(f"{transform.b}\n")  
                f.write(f"{transform.e}\n")  
                f.write(f"{x_center}\n")     
                f.write(f"{y_center}\n")     
                
            print(f"✅ Ficheiros PNG e PGW guardados com sucesso!")

            # =========================================================
            # PARTE B: GERAR O SHAPEFILE VETORIAL (.SHP)
            # =========================================================
            print("A extrair polígonos para o Shapefile (isto pode demorar alguns segundos)...")
            
            # Cria a matriz binária (1 para máscara, 0 para o resto) necessária para a vetorização
            mask_bin = np.zeros_like(mapbiomas_data, dtype=np.uint8)
            mask_bin[mask_condition] = 1
            
            geometrias = []
            for geom, value in shapes(mask_bin, mask=(mask_bin == 1), transform=transform):
                geometrias.append({
                    'geometry': geom,
                    'properties': {'classe': 'mascara'}
                })
        
            if not geometrias:
                print("❌ Nenhuma área de máscara encontrada para gerar o Shapefile.")
            else:
                print("A compilar e a guardar o ficheiro Shapefile...")
                gdf = gpd.GeoDataFrame.from_features(geometrias)
                gdf.set_crs(crs, inplace=True) 
                
                # Guarda no formato Shapefile do ESRI
                gdf.to_file(output_shp, driver='ESRI Shapefile')
                print(f"✅ Shapefile (.shp) guardado com sucesso!")
                
            print(f"\n🎉 Processo concluído! Os ficheiros estão na pasta: {os.path.dirname(output_png)}")
            
    except Exception as e:
        print(f"❌ Erro durante o processamento: {e}")

# ==========================================
# EXECUÇÃO E BUSCA DINÂMICA
# ==========================================
if __name__ == "__main__":
    load_dotenv()
    ROOT = os.getenv('PROJECT_ROOT')
    
    if not ROOT:
        print("❌ Erro: Variável 'PROJECT_ROOT' não encontrada no ficheiro .env.")
        exit()

    mapbiomas_dir = os.path.join(ROOT, 'data', 'MapBiomas')
    pasta_saida = os.path.join(ROOT, 'data', 'Output')
    
    print("=== A buscar o ficheiro MapBiomas de 2021 ===")
    busca = glob.glob(os.path.join(mapbiomas_dir, "*2021*coverage*10m*.tif"))
    
    if not busca:
        busca = glob.glob(os.path.join(mapbiomas_dir, "*coverage_10m*.tif"))
        
    if not busca:
        print(f"❌ Erro: Nenhum ficheiro '*coverage_10m*.tif' encontrado na pasta {mapbiomas_dir}")
        exit()
        
    ARQUIVO_ENTRADA = busca[0]
    print(f"🔍 Ficheiro encontrado: {os.path.basename(ARQUIVO_ENTRADA)}")
    
    # Define os caminhos de saída para ambos os formatos
    ARQUIVO_SAIDA_PNG = os.path.join(pasta_saida, "mascara_analise_2021.png")
    ARQUIVO_SAIDA_SHP = os.path.join(pasta_saida, "mascara_analise_2021.shp")
    
    os.makedirs(pasta_saida, exist_ok=True)
    
    gerar_mascara_completa(ARQUIVO_ENTRADA, ARQUIVO_SAIDA_PNG, ARQUIVO_SAIDA_SHP)