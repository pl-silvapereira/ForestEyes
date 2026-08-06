import os
import sys
import json
import rasterio
from rasterio.features import shapes
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import shape
from dotenv import load_dotenv

def executar_recorte(caminho_mapbiomas, caminho_cbers, caminho_saida):
    with rasterio.open(caminho_mapbiomas) as mb_src:
        mb_crs = mb_src.crs
        mb_data = mb_src.read(1)
        mb_nodata = mb_src.nodata if mb_src.nodata is not None else 0
        mascara_cidade = (mb_data != mb_nodata) & (mb_data != 0)
        
        gerador_shapes = shapes(mascara_cidade.astype('uint8'), mask=mascara_cidade, transform=mb_src.transform)
        poligonos = [shape(geom) for geom, valor in gerador_shapes if valor == 1]
        
        if not poligonos:
            sys.exit(1)
            
        gdf_mb = gpd.GeoDataFrame({'geometry': poligonos}, crs=mb_crs)
        poligono_cidade = gdf_mb.geometry.union_all()
        gdf_base = gpd.GeoDataFrame({'geometry': [poligono_cidade]}, crs=mb_crs)

    with rasterio.open(caminho_cbers) as cbers_src:
        cbers_crs = cbers_src.crs
        gdf_base_alinhado = gdf_base.to_crs(cbers_crs)
        geometria_corte = [gdf_base_alinhado.geometry.iloc[0]]
        
        out_img, out_transform = mask(cbers_src, geometria_corte, crop=True, filled=True, nodata=0)
        
        out_meta = cbers_src.meta.copy()
        out_meta.update({
            "height": out_img.shape[1], "width": out_img.shape[2],
            "transform": out_transform, "nodata": 0
        })

        os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
        with rasterio.open(caminho_saida, "w", **out_meta) as dest:
            dest.write(out_img)
            dest.colorinterp = cbers_src.colorinterp

    return caminho_saida

if __name__ == "__main__":
    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        raise ValueError("A variável PROJECT_ROOT não foi encontrada.")

    if len(sys.argv) >= 3:
        CODE_MUNI = int(sys.argv[1])
        ANO = str(sys.argv[2])
    else:
        CODE_MUNI = int(os.environ.get('CODE_MUNI', 0))
        ANO = str(os.environ.get('ANO', ''))

    if not CODE_MUNI or not ANO:
        sys.exit(1)

    diretorio_reports = os.path.join(project_root, "reports")
    json_mapbiomas = os.path.join(diretorio_reports, f"{CODE_MUNI}_{ANO}.json")
    json_cbers = os.path.join(diretorio_reports, f"{CODE_MUNI}_{ANO}_cbers_results.json")

    with open(json_mapbiomas, 'r', encoding='utf-8') as f:
        caminho_mapbiomas = json.load(f).get("arquivo_mapbiomas")

    with open(json_cbers, 'r', encoding='utf-8') as f:
        caminho_cbers = json.load(f).get("composicao_true_color")

    pasta_saida = os.path.join(project_root, "data", "output", "pansharpening", "geopolitic-RGBN", ANO)
    os.makedirs(pasta_saida, exist_ok=True)
    os.makedirs(diretorio_reports, exist_ok=True)

    nome_saida = f"{CODE_MUNI}_{ANO}_CBERS_TRUE_COLOR_CLIPPED.tif"
    caminho_saida = os.path.join(pasta_saida, nome_saida)

    arquivo_final = executar_recorte(caminho_mapbiomas, caminho_cbers, caminho_saida)

    dados_finais = {"code_muni": CODE_MUNI, "ano": ANO, "arquivo_cbers_recortado": arquivo_final}
    json_final = os.path.join(diretorio_reports, f"{CODE_MUNI}_{ANO}_clipped_results.json")
    
    with open(json_final, 'w', encoding='utf-8') as f:
        json.dump(dados_finais, f, indent=4, ensure_ascii=False)