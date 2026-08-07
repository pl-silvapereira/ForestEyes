import os
import sys
import geopandas as gpd
import pandas as pd
import geobr
from dotenv import load_dotenv

# Dicionário de metadados e cores oficiais das categorias
class_metadata = {}

def registar_categoria(ids, name, color, iso120, iso122, iso123):
    for i in ids:
        class_metadata[i] = {
            'name': name,
            'color': color,
            'iso_37120': iso120,
            'iso_37122': iso122,
            'iso_37123': iso123
        }

registar_categoria([3], 'Floresta', '#006400', 'Area verde (ha) por 100.000 hab.', 'Monitorizacao IoT', 'Mitigacao de Ilhas de Calor.')
registar_categoria([9], 'Floresta Antrópica', '#93c47d', 'Area verde (ha) por 100.000 hab.', 'Monitorizacao IoT', 'Mitigacao de Ilhas de Calor.')
registar_categoria([11, 12, 36], 'Vegetacao Herbacea e Arbustiva', '#a8c04d', 'Biodiversidade local', 'Risco de queimadas', 'Buffer Zones.')
registar_categoria([15, 19, 20, 21, 39, 41, 46, 48], 'Agropecuaria (Campos, Lavouras)', '#edde8e', 'Protecao de terras', 'Agrotech', 'Seguranca alimentar.')
registar_categoria([24, 25], 'Infraestrutura Urbana', '#d4271e', 'Densidade populacional', 'Smart Grids', 'Vulnerabilidade a desastres.')
registar_categoria([29, 31, 33], 'Nao Observado (Agua, Rocha)', '#0000ff', 'Disponibilidade hidrica', 'Telemetria', 'Prevencao de inundacoes.')
registar_categoria([4, 5, 6, 23, 27, 30, 32, 35, 40, 47, 49, 50, 62, 75], 'Descartadas', '#A9A9A9', 'N/A', 'N/A', 'N/A')

def gerar_estilo_qml_automatico(caminho_qml, metadata):
    categorias, simbolos = "", ""
    classes_unicas = {info['name']: info['color'] for info in metadata.values()}
    
    for i, (nome, cor) in enumerate(classes_unicas.items()):
        h = cor.lstrip('#')
        r, g, b = tuple(int(h[j:j+2], 16) for j in (0, 2, 4))
        symbol_name = str(i)
        categorias += f'<category render="true" symbol="{symbol_name}" value="{nome}" label="{nome}"/>\n'
        simbolos += f"""
        <symbol alpha="1" type="fill" name="{symbol_name}">
            <layer pass="0" class="SimpleFill" locked="0">
                <prop k="color" v="{r},{g},{b},255"/>
                <prop k="outline_style" v="no"/>
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

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 08-gerarImagemDelta.py <code_muni> <ano_inicio> <ano_fim>")
        sys.exit(1)

    code_muni = int(sys.argv[1])
    ano_inicio = str(sys.argv[2])
    ano_fim = str(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    mask_dir = os.path.join(project_root, "data", "output", "mask", f"{ano_inicio}_vs_{ano_fim}")
    os.makedirs(mask_dir, exist_ok=True)

    print(f"Buscando informações para o código de município: {code_muni}...")
    try:
        gdf_info = geobr.read_municipality(code_muni=code_muni, year=2022)
        nome_cidade = gdf_info['name_muni'].values[0]
        uf = gdf_info['abbrev_state'].values[0]
    except Exception as e:
        nome_cidade = "Município"
        uf = "SP"

    path_shp_inicio = os.path.join(project_root, "data", "output", "classification", ano_inicio, f"{code_muni}_Classificado_ForestEyes_{ano_inicio}.shp")
    path_shp_fim = os.path.join(project_root, "data", "output", "classification", ano_fim, f"{code_muni}_Classificado_ForestEyes_{ano_fim}.shp")

    if not os.path.exists(path_shp_inicio) or not os.path.exists(path_shp_fim):
        print(f"[ERRO CRÍTICO] Shapefiles não encontrados para {ano_inicio} e/ou {ano_fim}.")
        sys.exit(1)

    print("=" * 115)
    print(f"🗺️ GERANDO DELTA VETORIAL COM ESTILO AUTOMÁTICO (QML)")
    print(f"📍 MUNICÍPIO: {nome_cidade} - {uf} | PERÍODO: {ano_inicio} vs {ano_fim}")
    print("=" * 115)

    gdf1 = gpd.read_file(path_shp_inicio)
    gdf2 = gpd.read_file(path_shp_fim)

    utm_crs = gdf1.estimate_utm_crs()
    gdf1 = gdf1.to_crs(utm_crs)
    gdf2 = gdf2.to_crs(utm_crs)

    def obter_coluna_classe(gdf):
        for c in ['class_code', 'gridcode', 'id', 'value', 'class_id', 'DN']:
            if c in gdf.columns:
                return c
        for col in gdf.columns:
            if col != 'geometry' and pd.api.types.is_numeric_dtype(gdf[col]):
                return col
        return None

    col1 = obter_coluna_classe(gdf1)
    col2 = obter_coluna_classe(gdf2)

    gdf1['cat_ini'] = gdf1[col1] if col1 else 1
    gdf2['cat_fim'] = gdf2[col2] if col2 else 1

    print("Executando overlay espacial para isolar o Delta...")
    overlap = gpd.overlay(gdf1[['cat_ini', 'geometry']], gdf2[['cat_fim', 'geometry']], how='intersection', keep_geom_type=True)
    delta_gdf = overlap[overlap['cat_ini'] != overlap['cat_fim']].copy()
    
    delta_gdf['class_id'] = delta_gdf['cat_fim']
    
    # Atribuir o nome correto da classe baseado no dicionário
    delta_gdf['class_name'] = delta_gdf['class_id'].apply(
        lambda x: class_metadata[x]['name'] if x in class_metadata else f"ID_{x}"
    )

    out_prefix = f"{code_muni}_delta_vector_{ano_inicio}_vs_{ano_fim}"
    out_shp_path = os.path.join(mask_dir, f"{out_prefix}.shp")
    out_qml_path = os.path.join(mask_dir, f"{out_prefix}.qml")

    # Salvar shapefile com as colunas de ID e Nome da Classe
    delta_gdf[['class_id', 'class_name', 'geometry']].to_file(out_shp_path)
    
    # Gerar arquivo de estilo QML correspondente
    gerar_estilo_qml_automatico(out_qml_path, class_metadata)

    print(f"✓ Shapefile delta salvo em:\n-> {out_shp_path}")
    print(f"✓ Estilo automático QML gerado em:\n-> {out_qml_path}")
    print("\n[SUCESSO] Ao abrir o .shp no QGIS, as cores serão aplicadas automaticamente!")

if __name__ == "__main__":
    main()