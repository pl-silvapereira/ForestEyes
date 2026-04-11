import rasterio
from rasterio import features
import geopandas as gpd
import numpy as np
import os
import glob
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from dotenv import load_dotenv

# --- CONFIGURAÇÃO VIA .ENV ---
load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')

input_dir = os.path.join(ROOT, 'data', 'MapBiomas')
output_folder = os.path.join(ROOT, 'data', 'Output')
output_base = os.path.join(output_folder, '04_SJC_Mudancas_21_23_Classificadas')
output_shp = f"{output_base}.shp"
output_qml = f"{output_base}.qml"
output_tif = os.path.join(output_folder, "04_SJC_Mapa_Mudancas_21_23.tif")

# --- DICIONÁRIO INTELIGENTE MAPBIOMAS + ISO ---
class_metadata = {}

def registar_categoria(ids, name, color, iso120, iso122, iso123):
    """Função herdada do script 04 para padronizar metadados."""
    for i in ids:
        class_metadata[i] = {
            'name': name,
            'color': color,
            'iso_37120': iso120,
            'iso_37122': iso122,
            'iso_37123': iso123
        }

# 1. FLORESTAS
registar_categoria([3], 'Floresta', '#006400', 'Area verde (ha) por 100.000 hab.', 'Monitorizacao IoT vegetal', 'Mitigacao de UHI.')
registar_categoria([9], 'Floresta Antropica', '#93c47d', 'Area verde (ha) por 100.000 hab.', 'Monitorizacao IoT vegetal', 'Mitigacao de UHI.')

# 2. VEGETAÇÃO HERBÁCEA E ARBUSTIVA
registar_categoria([11, 12, 36], 'Vegetacao Herbacea', '#a8c04d', 'Biodiversidade local', 'Risco de queimadas', 'Permeabilidade solo.')

# 3. AGROPECUÁRIA E LAVOURAS
registar_categoria([15, 19, 20, 21, 39, 41, 46, 48], 'Agropecuaria', '#edde8e', 'Protecao terras araveis', 'Agrotech local', 'Seguranca alimentar.')

# 4. INFRAESTRUTURA URBANA
registar_categoria([24, 25], 'Infraestrutura Urbana', '#d4271e', 'Densidade populacional', 'Smart Grids', 'Vulnerabilidade infra.')

# 5. CORPOS HÍDRICOS (ÁGUA/ROCHA)
registar_categoria([29, 31, 33], 'Nao Observado (Agua)', '#0000ff', 'Qualidade hidrica', 'Telemetria e sensores', 'Prevencao inundacoes.')

# 6. DESCARTADAS (RUÍDO)
registar_categoria([4, 5, 6, 23, 27, 30, 32, 35, 40, 47, 49, 50, 62, 75], 'Descartadas', '#A9A9A9', 'N/A - Filtro precisao', 'N/A - Otimizacao', 'N/A - Remocao anomalias')

def localizar_raster(ano):
    busca = glob.glob(os.path.join(input_dir, f"{ano}_coverage_*.tif"))
    if not busca:
        raise FileNotFoundError(f"❌ Arquivo de {ano} não encontrado em {input_dir}")
    return busca[0]

def gerar_estilo_qml_automatico(caminho_qml, metadata):
    """Gera o arquivo de estilo para o QGIS baseado nas cores do dicionário."""
    categorias = ""
    simbolos = ""
    classes_unicas = {}
    for cid, info in metadata.items():
        if info['name'] not in classes_unicas:
            classes_unicas[info['name']] = info['color']
    
    for i, (nome, cor) in enumerate(classes_unicas.items()):
        h = cor.lstrip('#')
        r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        symbol_name = str(i)
        categorias += f'<category render="true" symbol="{symbol_name}" value="{nome}" label="{nome}"/>\n'
        simbolos += f"""
      <symbol alpha="1" type="fill" name="{symbol_name}">
        <layer pass="0" class="SimpleFill" locked="0">
          <prop k="color" v="{r},{g},{b},255"/>
          <prop k="outline_color" v="35,35,35,255"/>
          <prop k="outline_style" v="solid"/>
          <prop k="outline_width" v="0.1"/>
        </layer>
      </symbol>"""

    conteudo_qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="AllStyleCategories" version="3.28.0">
  <renderer-v2 attr="class_name" type="categorizedSymbol">
    <categories>{categorias}</categories>
    <symbols>{simbolos}</symbols>
  </renderer-v2>
</qgis>"""
    with open(caminho_qml, 'w', encoding='utf-8') as f:
        f.write(conteudo_qml)
    print(f"🎨 Estilo QML gerado: {os.path.basename(caminho_qml)}")

def processar_deteccao_mudancas():
    path_21 = localizar_raster("2021")
    path_23 = localizar_raster("2023")
    print(f"🚀 Iniciando Detecção de Mudanças: 2021 vs 2023")

    with rasterio.open(path_23) as src23:
        meta_tif = src23.meta.copy()
        with rasterio.open(path_21) as src21:
            # Alinhamento espacial
            with WarpedVRT(src21, crs=src23.crs, transform=src23.transform, 
                           width=src23.width, height=src23.height, 
                           resampling=Resampling.nearest) as vrt21:
                
                img23 = src23.read(1)
                img21 = vrt21.read(1)
                
                # Mapa de mudança (Pixels diferentes que não são zero)
                mudanca_mask = (img23 != img21) & (img23 != 0) & (img21 != 0)
                mapa_final = np.where(mudanca_mask, img23, 0).astype(np.int32)
                
                # 1. Salvar Raster TIF das mudanças
                meta_tif.update(dtype=rasterio.int32, nodata=0, count=1)
                with rasterio.open(output_tif, 'w', **meta_tif) as dst:
                    dst.write(mapa_final.astype(np.int32), 1)

                # 2. Vetorização e Enriquecimento ISO
                shapes = features.shapes(mapa_final, transform=src23.transform)
                polygons = []
                for s, v in shapes:
                    v_int = int(v)
                    if v_int == 0: continue
                    
                    if v_int in class_metadata:
                        info = class_metadata[v_int]
                        nome_classe = info['name']
                        i_120, i_122, i_123 = info['iso_37120'], info['iso_37122'], info['iso_37123']
                    else:
                        nome_classe = f"ID_{v_int}_Nao_Mapeado"
                        i_120 = i_122 = i_123 = "Analise manual necessária."

                    polygons.append({
                        'properties': {
                            'class_id': v_int,
                            'class_name': nome_classe,
                            'iso_37120': i_120,
                            'iso_37122': i_122,
                            'iso_37123': i_123
                        },
                        'geometry': s
                    })

    if polygons:
        gdf = gpd.GeoDataFrame.from_features(polygons, crs=src23.crs)
        gdf.to_file(output_shp)
        print(f"✅ Shapefile de mudanças (ISO) gerado: {os.path.basename(output_shp)}")
        gerar_estilo_qml_automatico(output_qml, class_metadata)
    else:
        print("⚠️ Nenhuma mudança detectada para vetorizar.")

if __name__ == "__main__":
    processar_deteccao_mudancas()