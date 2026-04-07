import rasterio
from rasterio import features
import geopandas as gpd
import numpy as np
import os
import glob

# --- CONFIGURAÇÃO ---
input_dir = '../data/MapBiomas/' 
output_base = '../data/Output/SJC_Classificado_Rapido'
output_shp = f"{output_base}.shp"
output_qml = f"{output_base}.qml"

arquivos = glob.glob(os.path.join(input_dir, "*coverage*.tif"))
if not arquivos:
    print("❌ Arquivo não encontrado!")
    exit()
input_path = arquivos[0]

# --- DICIONÁRIO SEM O 0 (Para evitar o processamento do retângulo) ---
class_metadata = {
    3:  {'name': 'FORMACAO_FLORESTAL', 'color': '#006400', 'iso': '37120-Env'}, 
    15: {'name': 'PASTAGEM',           'color': '#edde8e', 'iso': '37120-Rec'}, 
    21: {'name': 'MOSAICO_USOS',       'color': '#ffefc3', 'iso': '37123-Res'}, 
    24: {'name': 'AREA_URBANIZADA',    'color': '#d4271e', 'iso': '37122-Urb'}, 
    25: {'name': 'INFRAESTRUTURA',     'color': '#dbd83d', 'iso': '37122-Inf'}, 
    33: {'name': 'RIO_LAGO_OCEANO',    'color': '#0000ff', 'iso': '37120-Wat'}  
}

def criar_arquivo_estilo_qml(caminho_qml, metadata):
    """Gera um arquivo de estilo QML para o QGIS aplicar as cores automaticamente."""
    print(f"🎨 Gerando arquivo de estilo: {os.path.basename(caminho_qml)}")
    
    categorias = ""
    simbolos = ""
    
    for cid, info in metadata.items():
        h = info['color'].lstrip('#')
        r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        
        # XML para cada categoria baseada na coluna 'class_name'
        categorias += f'<category render="true" symbol="{cid}" value="{info["name"]}" label="{info["name"]}"/>\n'
        
        # XML para a definição visual (cor e borda)
        simbolos += f"""
          <symbol alpha="1" clip_to_extent="1" type="fill" name="{cid}">
            <layer pass="0" class="SimpleFill" locked="0">
              <prop k="color" v="{r},{g},{b},255"/>
              <prop k="outline_color" v="35,35,35,255"/>
              <prop k="outline_style" v="solid"/>
              <prop k="outline_width" v="0.1"/>
            </layer>
          </symbol>"""

    conteudo_qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="AllStyleCategories" version="3.28.0">
  <renderer-v2 attr="class_name" type="categorizedSymbol" enableorderby="0">
    <categories>
      {categorias}
    </categories>
    <symbols>
      {simbolos}
    </symbols>
  </renderer-v2>
</qgis>"""
    
    with open(caminho_qml, 'w', encoding='utf-8') as f:
        f.write(conteudo_qml)

def processar_veloz():
    print(f"📖 Lendo arquivo: {os.path.basename(input_path)}")
    
    with rasterio.open(input_path) as src:
        data = src.read(1)
        transform = src.transform
        crs_orig = src.crs
        
        print("📐 Vetorizando apenas as classes de interesse...")
        shapes = features.shapes(data.astype(np.int32), transform=transform)
        
        polygons = []
        for s, v in shapes:
            v_int = int(v)
            if v_int in class_metadata:
                polygons.append({
                    'properties': {
                        'class_id': v_int,
                        'class_name': class_metadata[v_int]['name'],
                        'hex_color': class_metadata[v_int]['color'],
                        'iso_ref': class_metadata[v_int]['iso']
                    },
                    'geometry': s
                })

    if not polygons:
        print("⚠️ Nada encontrado.")
        return

    gdf = gpd.GeoDataFrame.from_features(polygons, crs=crs_orig)

    print("🧹 Filtrando ruídos pequenos...")
    gdf_metric = gdf.to_crs(gdf.estimate_utm_crs())
    gdf_metric = gdf_metric[gdf_metric.geometry.area > 10]
    gdf = gdf_metric.to_crs(crs_orig)
    
    # Salva o SHP
    gdf.to_file(output_shp)
    
    # Salva o QML com o mesmo nome do SHP
    criar_arquivo_estilo_qml(output_qml, class_metadata)
    
    print(f"✅ Concluído! O arquivo {os.path.basename(output_shp)} agora tem estilo automático.")

if __name__ == "__main__":
    processar_veloz()