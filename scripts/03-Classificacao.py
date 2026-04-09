import rasterio
from rasterio import features
import geopandas as gpd
import numpy as np
import os
import glob
from dotenv import load_dotenv

# --- CONFIGURAÇÃO VIA .ENV ---
load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')

input_dir = os.path.join(ROOT, 'data', 'MapBiomas')
output_folder = os.path.join(ROOT, 'data', 'Output')
output_base = os.path.join(output_folder, 'SJC_Classificado_ForestEyes')
output_shp = f"{output_base}.shp"
output_qml = f"{output_base}.qml"

# --- DICIONÁRIO INTELIGENTE MAPBIOMAS + ISO ---
class_metadata = {}

def registar_categoria(ids, name, color, iso120, iso122, iso123):
    """Função auxiliar para preencher os metadados de forma limpa e organizada."""
    for i in ids:
        class_metadata[i] = {
            'name': name,
            'color': color,
            'iso_37120': iso120,
            'iso_37122': iso122,
            'iso_37123': iso123
        }

# 1. FLORESTAS
registar_categoria([3], 'Floresta', '#006400', 
    'Area verde (ha) por 100.000 hab. e indices de qualidade do ar.', 
    'Monitorizacao IoT de desmatamento e saude vegetal.', 
    'Mitigacao de Ilhas de Calor Urbanas (UHI).')

registar_categoria([9], 'Floresta Antrópica', '#93c47d', 
    'Area verde (ha) por 100.000 hab. e indices de qualidade do ar.', 
    'Monitorizacao IoT de desmatamento e saude vegetal.', 
    'Mitigacao de Ilhas de Calor Urbanas (UHI).')

# 2. VEGETAÇÃO HERBÁCEA E ARBUSTIVA
registar_categoria([11, 12, 36], 'Vegetacao Herbacea e Arbustiva', '#a8c04d', 
    'Manutencao da biodiversidade local e % de areas nao pavimentadas.', 
    'Monitorizacao preditiva de risco de queimadas.', 
    'Areas de amortecimento (Buffer Zones) e permeabilidade do solo.')

# 3. AGROPECUÁRIA E LAVOURAS
registar_categoria([15, 19, 20, 21, 39, 41, 46, 48], 'Agropecuaria (Campos, Lavouras)', '#edde8e', 
    'Protecao de terras araveis contra o espraiamento urbano (Urban Sprawl).', 
    'Agrotech, agricultura de precisao e rastreabilidade local.', 
    'Garantia de seguranca alimentar e autoabastecimento.')

# 4. INFRAESTRUTURA URBANA
registar_categoria([24, 25], 'Infraestrutura Urbana', '#d4271e', 
    'Densidade populacional, acesso a moradia e crescimento da mancha urbana.', 
    'Infraestrutura conectada (Smart Grids) e mobilidade inteligente.', 
    'Vulnerabilidade da infraestrutura critica frente a desastres.')

# 5. CORPOS HÍDRICOS (ÁGUA/ROCHA)
registar_categoria([29, 31, 33], 'Nao Observado (Agua, Rocha)', '#0000ff', 
    'Disponibilidade, acesso e qualidade das reservas hidricas superficiais.', 
    'Telemetria e sensores para controlo de qualidade e nivel.', 
    'Prevencao de inundacoes e respeito as Areas de Preservacao Permanente (APP).')

# 6. DESCARTADAS (RUÍDO)
registar_categoria([4, 5, 6, 23, 27, 30, 32, 35, 40, 47, 49, 50, 62, 75], 'Descartadas', '#A9A9A9', 
    'N/A - Filtragem de dados para precisao dos calculos.', 
    'N/A - Otimizacao de processamento analitico.', 
    'N/A - Remocao de anomalias da modelagem de risco.')

# (Nota: As descrições ISO foram ligeiramente abreviadas e sem acentos para garantir 
# total compatibilidade com o formato clássico .shp, que tem limites de caracteres nas colunas).

def gerar_estilo_qml_automatico(caminho_qml, metadata):
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
    print(f"🎨 Estilo QML gerado com sucesso: {os.path.basename(caminho_qml)}")

def processar_vetorizacao():
    busca = glob.glob(os.path.join(input_dir, "*coverage_10m*.tif"))
    if not busca:
        print(f"❌ Erro: Ficheiro do MapBiomas não encontrado em {input_dir}")
        return
    
    input_path = busca[0]
    print(f"📖 A processar e enriquecer dados ISO: {os.path.basename(input_path)}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with rasterio.open(input_path) as src:
        data = src.read(1)
        nodata_val = src.nodata if src.nodata is not None else 0
        shapes = features.shapes(data.astype(np.int32), transform=src.transform)
        
        polygons = []
        for s, v in shapes:
            v_int = int(v)
            
            if v_int == nodata_val or v_int == 0:
                continue
                
            if v_int in class_metadata:
                info = class_metadata[v_int]
                nome_classe = info['name']
                i_120 = info['iso_37120']
                i_122 = info['iso_37122']
                i_123 = info['iso_37123']
            else:
                nome_classe = f"Categoria_Nao_Mapeada_ID_{v_int}"
                class_metadata[v_int] = {'name': nome_classe, 'color': '#FF00FF'}
                i_120 = i_122 = i_123 = "Requer analise manual (Categoria nova)."
                
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

    if not polygons:
        print("⚠️ Nenhuma classe mapeada foi encontrada no ficheiro.")
        return

    gdf = gpd.GeoDataFrame.from_features(polygons, crs=src.crs)
    gdf.to_file(output_shp)
    print(f"✅ Shapefile enriquecido com normas ISO gerado: {os.path.basename(output_shp)}")

    gerar_estilo_qml_automatico(output_qml, class_metadata)

if __name__ == "__main__":
    processar_vetorizacao()