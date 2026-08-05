import sys
import rasterio
from rasterio import features
import geopandas as gpd
import numpy as np
import os
import json
from dotenv import load_dotenv

class_metadata = {}

def registar_categoria(ids, name, color, iso120, iso122, iso123):
    for i in ids:
        class_metadata[i] = {
            'name': name, 'color': color,
            'iso_37120': iso120, 'iso_37122': iso122, 'iso_37123': iso123
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

def gerar_estilo_qml_automatico(caminho_qml, metadata):
    categorias, simbolos = "", ""
    classes_unicas = {}
    for info in metadata.values():
        if info['name'] not in classes_unicas: classes_unicas[info['name']] = info['color']
    
    for i, (nome, cor) in enumerate(classes_unicas.items()):
        h = cor.lstrip('#')
        r, g, b = tuple(int(h[j:j+2], 16) for j in (0, 2, 4))
        categorias += f'<category render="true" symbol="{i}" value="{nome}" label="{nome}"/>\n'
        simbolos += f'''
      <symbol alpha="1" type="fill" name="{i}">
        <layer pass="0" class="SimpleFill" locked="0">
          <prop k="color" v="{r},{g},{b},255"/>
          <prop k="outline_color" v="0,0,0,0"/>
          <prop k="outline_style" v="no"/>
          <prop k="outline_width" v="0"/>
        </layer>
      </symbol>'''

    conteudo_qml = f"<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'><qgis styleCategories='AllStyleCategories' version='3.28.0'><renderer-v2 attr='class_name' type='categorizedSymbol'><categories>{categorias}</categories><symbols>{simbolos}</symbols></renderer-v2></qgis>"
    with open(caminho_qml, 'w', encoding='utf-8') as f: f.write(conteudo_qml)

def processar_vetorizacao(input_path, output_shp, output_qml):
    with rasterio.open(input_path) as src:
        data = src.read(1)
        nodata_val = src.nodata if src.nodata is not None else 0
        shapes_gen = features.shapes(data.astype(np.int32), transform=src.transform)
        
        polygons = []
        for s, v in shapes_gen:
            v_int = int(v)
            if v_int == nodata_val or v_int == 0: continue
            
            info = class_metadata.get(v_int, {'name': f"ID_{v_int}", 'iso_37120': '', 'iso_37122': '', 'iso_37123': ''})
            polygons.append({
                'properties': {
                    'class_id': v_int, 'class_name': info['name'],
                    'iso_37120': info['iso_37120'], 'iso_37122': info['iso_37122'], 'iso_37123': info['iso_37123']
                },
                'geometry': s
            })

    gdf = gpd.GeoDataFrame.from_features(polygons, crs=src.crs)
    gdf.to_file(output_shp)
    gerar_estilo_qml_automatico(output_qml, class_metadata)
    return output_shp, output_qml

if __name__ == "__main__":
    load_dotenv()
    
    if len(sys.argv) >= 3:
        CODE_MUNI = int(sys.argv[1])
        ANO = str(sys.argv[2])
    else:
        CODE_MUNI = int(os.environ.get('CODE_MUNI', 0))
        ANO = os.environ.get('ANO', '')

    if not CODE_MUNI or not ANO: sys.exit(1)

    project_root = os.environ.get("PROJECT_ROOT")
    diretorio_reports = os.path.join(project_root, "reports")
    json_mapbiomas = os.path.join(diretorio_reports, f"{CODE_MUNI}_{ANO}.json")

    with open(json_mapbiomas, 'r', encoding='utf-8') as f:
        caminho_mapbiomas = json.load(f).get("arquivo_mapbiomas")

    pasta_saida = os.path.join(project_root, "data", "output", "classification", ANO)
    os.makedirs(pasta_saida, exist_ok=True)
    
    prefixo = f"{CODE_MUNI}_Classificado_ForestEyes_{ANO}"
    caminho_shp = os.path.join(pasta_saida, f"{prefixo}.shp")
    caminho_qml = os.path.join(pasta_saida, f"{prefixo}.qml")

    print(f"=== CLASSIFICAÇÃO VETORIAL: {CODE_MUNI} | ANO: {ANO} ===")
    shp_gerado, qml_gerado = processar_vetorizacao(caminho_mapbiomas, caminho_shp, caminho_qml)

    json_final = os.path.join(diretorio_reports, f"{CODE_MUNI}_{ANO}_classification_results.json")
    with open(json_final, 'w', encoding='utf-8') as f:
        json.dump({"code_muni": CODE_MUNI, "ano": ANO, "arquivo_shp": shp_gerado}, f, indent=4)