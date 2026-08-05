import os
import sys
import json
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import shape
from dotenv import load_dotenv

def executar_recorte_por_vetorizacao(caminho_mapbiomas, caminho_cbers, caminho_saida):
    print("1/3 - Extraindo formato geopolítico do MapBiomas...")
    with rasterio.open(caminho_mapbiomas) as mb_src:
        mb_crs = mb_src.crs
        mb_data = mb_src.read(1)
        
        mb_nodata = mb_src.nodata if mb_src.nodata is not None else 0
        mascara_cidade = (mb_data != mb_nodata) & (mb_data != 0)
        
        # Cria vetores on-the-fly para definir exatamente o formato da cidade
        gerador_shapes = shapes(mascara_cidade.astype('uint8'), mask=mascara_cidade, transform=mb_src.transform)
        poligonos = [shape(geom) for geom, valor in gerador_shapes if valor == 1]
        
        if not poligonos:
            print("❌ Erro: Máscara não identificada no arquivo do MapBiomas.")
            sys.exit(1)
            
        gdf_mb = gpd.GeoDataFrame({'geometry': poligonos}, crs=mb_crs)
        poligono_cidade = gdf_mb.geometry.unary_union
        gdf_base = gpd.GeoDataFrame({'geometry': [poligono_cidade]}, crs=mb_crs)

    print("2/3 - Alinhando projeções e recortando a imagem CBERS-4A...")
    with rasterio.open(caminho_cbers) as cbers_src:
        cbers_crs = cbers_src.crs
        
        gdf_base_alinhado = gdf_base.to_crs(cbers_crs)
        geometria_corte = [gdf_base_alinhado.geometry.iloc[0]]
        
        out_img, out_transform = mask(cbers_src, geometria_corte, crop=True, filled=True, nodata=0)
        
        out_meta = cbers_src.meta.copy()
        out_meta.update({
            "height": out_img.shape[1],
            "width": out_img.shape[2],
            "transform": out_transform,
            "nodata": 0
        })

        print("3/3 - Salvando o arquivo multiespectral recortado...")
        os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
        
        with rasterio.open(caminho_saida, "w", **out_meta) as dest:
            dest.write(out_img)
            dest.colorinterp = cbers_src.colorinterp

    print(f"\n✅ Imagem alinhada e salva em:\n-> {caminho_saida}")
    return caminho_saida

if __name__ == "__main__":
    load_dotenv()
    
    if len(sys.argv) >= 3:
        CODE_MUNI = int(sys.argv[1])
        ANO = str(sys.argv[2])
    else:
        CODE_MUNI = int(os.environ.get('CODE_MUNI', 0))
        ANO = os.environ.get('ANO', '')

    if not CODE_MUNI or not ANO:
        print("Erro: Parâmetros CODE_MUNI e ANO obrigatórios.")
        sys.exit(1)

    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root: sys.exit(1)

    diretorio_reports = os.path.join(project_root, "reports")
    json_mapbiomas = os.path.join(diretorio_reports, f"{CODE_MUNI}_{ANO}.json")
    json_cbers = os.path.join(diretorio_reports, f"{CODE_MUNI}_{ANO}_cbers_results.json")

    if not os.path.exists(json_mapbiomas) or not os.path.exists(json_cbers):
        print(f"[ERRO] JSONs da etapa 01 ou 02 não encontrados para o ano {ANO}.")
        sys.exit(1)

    with open(json_mapbiomas, 'r', encoding='utf-8') as f:
        caminho_mapbiomas = json.load(f).get("arquivo_mapbiomas")

    with open(json_cbers, 'r', encoding='utf-8') as f:
        caminho_cbers = json.load(f).get("composicao_true_color")

    nome_saida = f"{CODE_MUNI}_{ANO}_CBERS_TRUE_COLOR_CLIPPED.tif"
    pasta_saida = os.path.join(project_root, "data", "output", "pansharpening", "geopolitic-RGBN", ANO)
    caminho_saida = os.path.join(pasta_saida, nome_saida)

    print(f"=== INICIANDO RECORTE: MUNICÍPIO {CODE_MUNI} | ANO {ANO} ===")
    arquivo_final = executar_recorte_por_vetorizacao(caminho_mapbiomas, caminho_cbers, caminho_saida)

    dados_finais = {
        "code_muni": CODE_MUNI,
        "ano": ANO,
        "arquivo_cbers_recortado": arquivo_final
    }
    json_final = os.path.join(diretorio_reports, f"{CODE_MUNI}_{ANO}_clipped_results.json")
    
    with open(json_final, 'w', encoding='utf-8') as f:
        json.dump(dados_finais, f, indent=4, ensure_ascii=False)