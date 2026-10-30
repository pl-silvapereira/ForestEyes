import sys
import os
import duckdb

# Correção de compatibilidade entre DuckDB e geobr no Python 3.13
if not hasattr(duckdb.DuckDBPyRelation, "fetchone"):
    duckdb.DuckDBPyRelation.fetchone = lambda self: self.fetchall()[0]

import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
import geobr
from dotenv import load_dotenv

def gerar_estilo_qml_garantido(caminho_qml):
    """Gera o estilo QML com 4 classes de cores distintas:
       - Floresta: Verde (#33a02c)
       - Nao_Floresta: Vermelho (#e31a1c)
       - Nao_Segmentar: Amarelo (#ffe600)
       - Extra_Geopolitico: Preto (#000000)
    """
    conteudo_qml = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="AllStyleCategories" version="3.28.0">
  <renderer-v2 attr="class_name" type="categorizedSymbol">
    <categories>
      <category render="true" symbol="0" value="Floresta" label="Floresta (Verde)"/>
      <category render="true" symbol="1" value="Nao_Floresta" label="Nao Floresta (Vermelho)"/>
      <category render="true" symbol="2" value="Nao_Segmentar" label="Nao Segmentar (Amarelo)"/>
      <category render="true" symbol="3" value="Extra_Geopolitico" label="Extra Geopolitico (Preto)"/>
    </categories>
    <symbols>
      <symbol alpha="1" type="fill" name="0">
        <layer pass="0" class="SimpleFill" locked="0">
          <prop k="color" v="51,160,44,255"/>
          <prop k="outline_style" v="no"/>
        </layer>
      </symbol>
      <symbol alpha="1" type="fill" name="1">
        <layer pass="0" class="SimpleFill" locked="0">
          <prop k="color" v="227,26,28,255"/>
          <prop k="outline_style" v="no"/>
        </layer>
      </symbol>
      <symbol alpha="1" type="fill" name="2">
        <layer pass="0" class="SimpleFill" locked="0">
          <prop k="color" v="255,230,0,255"/>
          <prop k="outline_style" v="no"/>
        </layer>
      </symbol>
      <symbol alpha="1" type="fill" name="3">
        <layer pass="0" class="SimpleFill" locked="0">
          <prop k="color" v="0,0,0,255"/>
          <prop k="outline_style" v="no"/>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
</qgis>"""
    
    with open(caminho_qml, 'w', encoding='utf-8') as f:
        f.write(conteudo_qml)
        f.flush()
        os.fsync(f.fileno())

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 05-segmentacaoZooniverse.py <code_muni> <ano_inicio> <ano_fim>")
        sys.exit(1)

    code_muni = str(sys.argv[1])
    ano_inicio = str(sys.argv[2])
    ano_fim = str(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    output_seg_dir = os.path.join(project_root, "data", "output", "segmentation", ano_fim)
    os.makedirs(output_seg_dir, exist_ok=True)

    print("=" * 80)
    print(f"🔬 SCRIPT 05 - GERAÇÃO DO MAPA DE SEGMENTAÇÃO EM 4 CLASSES ({ano_fim})")
    print("=" * 80)

    path_shp_class = os.path.join(project_root, "data", "output", "classification", ano_fim, f"{code_muni}_Classificado_ForestEyes_{ano_fim}.shp")
    path_shp_mudancas = os.path.join(project_root, "data", "output", "analysis", "mudancas", f"{code_muni}_Mudancas_{ano_inicio}_vs_{ano_fim}.shp")

    if not os.path.exists(path_shp_class):
        print(f"[ERRO CRÍTICO] Shapefile de classificação não encontrado: {path_shp_class}")
        sys.exit(1)

    # 1. Classe Floresta (Shapefile de Classificação)
    print("1. Lendo classe Floresta do shapefile de classificação...")
    gdf_class = gpd.read_file(path_shp_class)
    gdf_floresta = gdf_class[gdf_class['class_name'].isin(['Floresta'])].copy()
    geom_floresta_raw = unary_union(gdf_floresta.geometry.buffer(0)) if not gdf_floresta.empty else Polygon()

    # 2. Classe Nao_Floresta (Shapefile de Mudanças)
    print("2. Lendo áreas de mudanças (Nao_Floresta)...")
    geom_nao_floresta_raw = Polygon()
    if os.path.exists(path_shp_mudancas):
        gdf_mudancas = gpd.read_file(path_shp_mudancas)
        transicoes = [
            'Floresta -> Agropecuaria (Campos, Lavouras)',
            'Floresta -> Floresta Antrópica',
            'Floresta -> Infraestrutura Urbana',
            'Floresta -> Nao Observado (Agua, Rocha)',
            'Floresta -> Vegetacao Herbacea e Arbustiva'
        ]
        if 'transicao' in gdf_mudancas.columns:
            gdf_filt = gdf_mudancas[gdf_mudancas['transicao'].isin(transicoes)].copy()
            if not gdf_filt.empty:
                geom_nao_floresta_raw = unary_union(gdf_filt.geometry.buffer(0))

    # 3. Limite Geopolítico e Retângulo Envolvente (Bounding Box)
    print("3. Obtendo contorno geopolítico e retângulo envolvente...")
    gdf_muni = geobr.read_municipality(code_muni=int(code_muni), year=2022)
    gdf_muni = gdf_muni.to_crs(gdf_class.crs)
    limite = gdf_muni.geometry.union_all()

    bounds = limite.bounds
    retangulo = box(bounds[0], bounds[1], bounds[2], bounds[3])

    # 4. Processamento de Geometrias e Diferenças Espaciais
    print("4. Executando overlay espacial das 4 classes...")
    geom_floresta = geom_floresta_raw.intersection(limite)
    geom_nao_floresta = geom_nao_floresta_raw.difference(geom_floresta).intersection(limite)

    geom_segmentar_total = geom_floresta.union(geom_nao_floresta)
    geom_nao_segmentar = limite.difference(geom_segmentar_total)

    geom_extra_geopolitico = retangulo.difference(limite)

    # 5. Montagem do GeoDataFrame Final
    gdf_4classes = gpd.GeoDataFrame({
        'class_name': ['Floresta', 'Nao_Floresta', 'Nao_Segmentar', 'Extra_Geopolitico'],
        'geometry': [geom_floresta, geom_nao_floresta, geom_nao_segmentar, geom_extra_geopolitico]
    }, crs=gdf_class.crs)

    gdf_4classes = gdf_4classes[~gdf_4classes.geometry.is_empty]
    gdf_4classes = gdf_4classes.explode(index_parts=False).reset_index(drop=True)

    nome_base = f"{code_muni}_Segmentacao_4Classes_{ano_fim}"
    path_shp = os.path.join(output_seg_dir, f"{nome_base}.shp")
    path_qml = os.path.join(output_seg_dir, f"{nome_base}.qml")

    print("5. Salvando Shapefile...")
    gdf_4classes.to_file(path_shp, encoding='utf-8')
    
    print("6. Gerando arquivo de estilo QML...")
    gerar_estilo_qml_garantido(path_qml)

    tamanho_qml = os.path.getsize(path_qml)
    print(f"\n✅ SUCESSO! Arquivo QML gravado corretamente ({tamanho_qml} bytes).")
    print(f"✅ Shapefile salvo em: {path_shp}")

if __name__ == "__main__":
    main()