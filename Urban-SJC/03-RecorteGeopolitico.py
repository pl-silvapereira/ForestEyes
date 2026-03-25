import geopandas as gpd
import rasterio
import rasterio.mask
from rasterio.vrt import WarpedVRT
from pathlib import Path

def recorte_geopolitico_final():
    base_path = Path(r"C:\Users\PedroLuizdaSilvaPere\development\Projects\ForestEyes\Urban-SJC")
    img_input = base_path / "CBERS-4A" / "SJC_Master_Cientifico_4Bandas_2m.tif"
    # Use o caminho exato do seu arquivo de municípios que aparece no QGIS
    shape_path = base_path / "vetores" / "SP_Municipios_2024.shp" 
    img_output = base_path / "SJC_AREA_URBANA_FINAL.tif"

    print("Carregando limites e preparando rebatimento espacial...")
    
    # Lendo o shapefile e filtrando apenas SJC
    mundo = gpd.read_file(shape_path)
    sjc = mundo[mundo['NM_MUN'] == 'São José dos Campos'].to_crs("EPSG:31983")

    with rasterio.open(str(img_input)) as src:
        # O WarpedVRT é quem vai 'puxar' a imagem para cima do polígono
        with WarpedVRT(src, crs="EPSG:31983") as vrt:
            print("Executando recorte por máscara...")
            out_image, out_transform = rasterio.mask.mask(vrt, sjc.geometry, crop=True)
            
            out_meta = vrt.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw",
                "BIGTIFF": "YES"
            })

            with rasterio.open(str(img_output), "w", **out_meta) as dest:
                dest.write(out_image)

    print(f"PROCESSO CONCLUÍDO! O arquivo {img_output.name} é o seu mapa de análise.")

if __name__ == "__main__":
    recorte_geopolitico_final()