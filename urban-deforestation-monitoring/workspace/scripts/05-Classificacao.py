import sys
import os
import json
import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
from dotenv import load_dotenv

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

def gerar_estilo_qml_automatico(caminho_qml, metadata):
    """Gera um arquivo de estilo do QGIS (.qml) com as cores corretas e sem bordas."""
    categorias = ""
    simbolos = ""
    
    classes_unicas = {}
    for cid, info in metadata.items():
        if info['name'] not in classes_unicas:
            classes_unicas[info['name']] = info['color']
    
    for i, (nome, cor) in enumerate(classes_unicas.items()):
        h = cor.lstrip('#')
        r, g, b = tuple(int(h[j:j+2], 16) for j in (0, 2, 4))
        symbol_name = str(i)
        
        categorias += f'<category render="true" symbol="{symbol_name}" value="{nome}" label="{nome}"/>\n'
        
        # outline_style="no" garante o contorno transparente que você solicitou
        simbolos += f"""
      <symbol alpha="1" type="fill" name="{symbol_name}">
        <layer pass="0" class="SimpleFill" locked="0">
          <prop k="color" v="{r},{g},{b},255"/>
          <prop k="outline_color" v="0,0,0,0"/>
          <prop k="outline_style" v="no"/>
          <prop k="outline_width" v="0"/>
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
    print(f"🎨 Estilo QML (contorno transparente) gerado com sucesso:\n-> {caminho_qml}")

def processar_vetorizacao(input_path, output_shp, output_qml):
    """Converte o raster do MapBiomas em polígonos inteligentes e enriquece com ISO."""
    print(f"📖 Lendo o raster do MapBiomas e iniciando vetorização:\n-> {input_path}")

    with rasterio.open(input_path) as src:
        data = src.read(1)
        nodata_val = src.nodata if src.nodata is not None else 0
        
        # Extrai os shapes (polígonos) diretamente da matriz de pixels
        shapes_gen = features.shapes(data.astype(np.int32), transform=src.transform)
        
        polygons = []
        for s, v in shapes_gen:
            v_int = int(v)
            
            # Ignora pixels de fundo (nodata ou zero)
            if v_int == nodata_val or v_int == 0:
                continue
                
            # Mapeia as informações caso o ID exista no nosso dicionário
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
        print("⚠️ Nenhuma classe mapeada foi encontrada no arquivo.")
        sys.exit(1)

    print("🧩 Criando GeoDataFrame e salvando Shapefile (isso pode levar alguns segundos)...")
    gdf = gpd.GeoDataFrame.from_features(polygons, crs=src.crs)
    gdf.to_file(output_shp)
    
    print(f"✅ Shapefile enriquecido com normas ISO gerado:\n-> {output_shp}")
    gerar_estilo_qml_automatico(output_qml, class_metadata)
    
    return output_shp, output_qml

if __name__ == "__main__":
    # 1. Capturar argumento
    if len(sys.argv) < 2:
        print("Erro: Parâmetros insuficientes.")
        print("Uso correto: python 05-gerarClassificacaoMapBiomas.py <code_muni>")
        sys.exit(1)

    CODE_MUNI = int(sys.argv[1])

    # 2. Carregar ambiente
    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        raise ValueError("A variável PROJECT_ROOT não foi encontrada no arquivo .env.")

    diretorio_reports = os.path.join(project_root, "reports")
    json_mapbiomas = os.path.join(diretorio_reports, f"{CODE_MUNI}.json")

    # 3. Validar a existência do relatório da Etapa 01
    if not os.path.exists(json_mapbiomas):
        print(f"[ERRO CRÍTICO] Arquivo JSON não encontrado: {json_mapbiomas}")
        print("Certifique-se de executar o Script 01 antes de rodar a classificação.")
        sys.exit(1)

    # 4. Ler JSON para encontrar a imagem bruta do MapBiomas
    with open(json_mapbiomas, 'r', encoding='utf-8') as f:
        dados_mb = json.load(f)
        
    caminho_mapbiomas = dados_mb.get("arquivo_mapbiomas")
    if not caminho_mapbiomas or not os.path.exists(caminho_mapbiomas):
        print("[ERRO] O caminho do arquivo MapBiomas original não foi encontrado no JSON ou no disco.")
        sys.exit(1)

    # 5. Configurar diretórios de saída
    pasta_saida = os.path.join(project_root, "data", "output", "classification")
    os.makedirs(pasta_saida, exist_ok=True)
    
    prefixo = f"{CODE_MUNI}_Classificado_ForestEyes"
    caminho_shp = os.path.join(pasta_saida, f"{prefixo}.shp")
    caminho_qml = os.path.join(pasta_saida, f"{prefixo}.qml")

    print(f"==================================================")
    print(f" INICIANDO CLASSIFICAÇÃO VETORIAL: MUNICÍPIO {CODE_MUNI}")
    print(f"==================================================")

    # 6. Executar vetorização e estilização
    shp_gerado, qml_gerado = processar_vetorizacao(caminho_mapbiomas, caminho_shp, caminho_qml)

    # 7. Salvar relatório JSON com o resultado
    dados_finais = {
        "code_muni": CODE_MUNI,
        "arquivo_origem_mapbiomas": caminho_mapbiomas,
        "arquivos_gerados": {
            "shapefile": shp_gerado,
            "qml_style": qml_gerado
        }
    }
    
    json_final = os.path.join(diretorio_reports, f"{CODE_MUNI}_classification_results.json")
    with open(json_final, 'w', encoding='utf-8') as f:
        json.dump(dados_finais, f, indent=4, ensure_ascii=False)

    print(f"\n[RELATÓRIO] JSON com mapeamento dos arquivos de classificação salvo em:\n-> {json_final}")