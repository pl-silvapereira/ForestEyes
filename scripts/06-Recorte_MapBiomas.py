import os
import sys
import glob
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import shape

# ==========================================
# 1. CONFIGURAÇÃO DE CAMINHOS
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

dir_mapbiomas = os.path.join(ROOT, 'data', 'MapBiomas')
dir_output = os.path.join(ROOT, 'data', 'Output')

# Arquivos de entrada e saída
pan_exato = os.path.join(dir_output, "01_SJC_Pansharpened_Exato.tif")
saida_final = os.path.join(dir_output, "02_SJC_Recortado_MapBiomas.tif")

def executar_recorte():
    print(f"Diretório Raiz: {ROOT}")
    
    # Busca dinâmica pelo arquivo do MapBiomas
    mapbiomas_files = glob.glob(os.path.join(dir_mapbiomas, "*coverage_10m*.tif"))
    if not mapbiomas_files:
        print("❌ Erro: O arquivo '.tif' do MapBiomas não foi encontrado na pasta.")
        sys.exit()
    
    mapbiomas_path = mapbiomas_files[0]
    print(f"📄 Arquivo MapBiomas base: {os.path.basename(mapbiomas_path)}")
    print("📄 Arquivo de Alta Resolução: 01_SJC_Pansharpened_Exato.tif\n")

    # ==========================================
    # PASSO 1: EXTRAIR O FORMATO DO MUNICÍPIO
    # ==========================================
    print("1/3 - Extraindo o formato geopolítico do MapBiomas...")
    with rasterio.open(mapbiomas_path) as mb_src:
        mb_crs = mb_src.crs
        mb_data = mb_src.read(1)
        
        # Identifica qual é o valor do "fundo vazio" no MapBiomas (geralmente 0)
        mb_nodata = mb_src.nodata if mb_src.nodata is not None else 0
        
        # Cria uma máscara onde o MapBiomas tem dados reais (A cidade)
        mascara_cidade = (mb_data != mb_nodata) & (mb_data != 0)
        
        # Transforma os pixels agrupados em geometria (Vetorização On-The-Fly)
        gerador_shapes = shapes(mascara_cidade.astype('uint8'), mask=mascara_cidade, transform=mb_src.transform)
        poligonos = [shape(geom) for geom, valor in gerador_shapes if valor == 1]
        
        if not poligonos:
            print("❌ Erro: Não foi possível identificar a área válida no MapBiomas.")
            sys.exit()
            
        # Agrupa tudo num único objeto geométrico (útil se a cidade tiver ilhas)
        gdf_mb = gpd.GeoDataFrame({'geometry': poligonos}, crs=mb_crs)
        poligono_cidade = gdf_mb.geometry.unary_union
        
        # DataFrame final com o formato da cidade no sistema de coordenadas original
        gdf_base = gpd.GeoDataFrame({'geometry': [poligono_cidade]}, crs=mb_crs)

    # ==========================================
    # PASSO 2: REPROJETAR E RECORTAR
    # ==========================================
    print("2/3 - Alinhando coordenadas e recortando a imagem de Alta Resolução...")
    
    # Abre a imagem Pansharpened gerada no script anterior
    with rasterio.open(pan_exato) as pan_src:
        pan_crs = pan_src.crs
        
        # Força o formato da cidade a se converter para a projeção do satélite (ex: UTM)
        gdf_base_alinhado = gdf_base.to_crs(pan_crs)
        geometria_corte = [gdf_base_alinhado.geometry.iloc[0]]
        
        # Executa o recorte ("molde de biscoito")
        out_img, out_transform = mask(pan_src, geometria_corte, crop=True, filled=True, nodata=0)
        
        # Atualiza as propriedades para a nova imagem cortada
        out_meta = pan_src.meta.copy()
        out_meta.update({
            "height": out_img.shape[1],
            "width": out_img.shape[2],
            "transform": out_transform
        })

        # ==========================================
        # PASSO 3: SALVAR A IMAGEM FINAL
        # ==========================================
        print("3/3 - Salvando o arquivo final recortado...")
        with rasterio.open(saida_final, "w", **out_meta) as dest:
            dest.write(out_img)
            # Conserva a tabela de cores (RGB) original do Pansharpening
            dest.colorinterp = pan_src.colorinterp

    print(f"\n✅ Sucesso absoluto! A imagem recortada com a máscara do MapBiomas foi salva em:\n{saida_final}")

if __name__ == '__main__':
    executar_recorte()