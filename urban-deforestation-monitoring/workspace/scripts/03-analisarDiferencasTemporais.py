import sys
import os
import geopandas as gpd
from dotenv import load_dotenv

# Paleta de cores distintas de alta visibilidade para as transições
CORES_DISTINTAS = [
    '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', 
    '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', 
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9',
    '#ff4500', '#2e8b57', '#da70d6', '#8b4513', '#483d8b'
]

def gerar_estilo_qml_mudancas(caminho_qml, map_cores):
    categorias = ""
    simbolos = ""
    
    for i, (transicao, cor) in enumerate(map_cores.items()):
        h = cor.lstrip('#')
        # Fallback caso a cor não seja hex válida
        try:
            r, g, b = tuple(int(h[j:j+2], 16) for j in (0, 2, 4))
        except:
            r, g, b = (128, 128, 128)
            
        symbol_name = str(i)
        categorias += f'<category render="true" symbol="{symbol_name}" value="{transicao}" label="{transicao}"/>\n'
        simbolos += f"""
      <symbol alpha="1" type="fill" name="{symbol_name}">
        <layer pass="0" class="SimpleFill" locked="0">
          <prop k="color" v="{r},{g},{b},255"/>
          <prop k="outline_style" v="no"/>
        </layer>
      </symbol>"""

    conteudo_qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="AllStyleCategories" version="3.28.0">
  <renderer-v2 attr="transicao" type="categorizedSymbol">
    <categories>{categorias}</categories>
    <symbols>{simbolos}</symbols>
  </renderer-v2>
</qgis>"""
    
    with open(caminho_qml, 'w', encoding='utf-8') as f:
        f.write(conteudo_qml)

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 04-gerarDiferencaClassificacao.py <code_muni> <ano_inicio> <ano_fim>")
        print("Exemplo: python 04-gerarDiferencaClassificacao.py 3549904 2023 2024")
        sys.exit(1)

    code_muni = str(sys.argv[1])
    ano_inicio = str(sys.argv[2])
    ano_fim = str(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    # Definir caminhos de entrada
    path_shp_inicio = os.path.join(project_root, "data", "output", "classification", ano_inicio, f"{code_muni}_Classificado_ForestEyes_{ano_inicio}.shp")
    path_shp_fim = os.path.join(project_root, "data", "output", "classification", ano_fim, f"{code_muni}_Classificado_ForestEyes_{ano_fim}.shp")

    if not os.path.exists(path_shp_inicio) or not os.path.exists(path_shp_fim):
        print(f"[ERRO CRÍTICO] Shapefiles não encontrados para os anos {ano_inicio} e/ou {ano_fim}.")
        print(f"Verifique se o script 03 foi executado com sucesso para esses anos.")
        sys.exit(1)

    # Criar diretório de saída
    pasta_saida = os.path.join(project_root, "data", "output", "analysis", "mudancas")
    os.makedirs(pasta_saida, exist_ok=True)
    
    prefixo_saida = f"{code_muni}_Mudancas_{ano_inicio}_vs_{ano_fim}"
    caminho_shp_saida = os.path.join(pasta_saida, f"{prefixo_saida}.shp")
    caminho_qml_saida = os.path.join(pasta_saida, f"{prefixo_saida}.qml")

    print("=" * 70)
    print(f"🗺️ MAPEAMENTO DE MUDANÇAS: {ano_inicio} -> {ano_fim} (IBGE: {code_muni})")
    print("=" * 70)

    print(f"Carregando Shapefile de {ano_inicio}...")
    gdf_inicio = gpd.read_file(path_shp_inicio)
    gdf_inicio = gdf_inicio[['class_name', 'geometry']].rename(columns={'class_name': 'classe_ini'})

    print(f"Carregando Shapefile de {ano_fim}...")
    gdf_fim = gpd.read_file(path_shp_fim)
    gdf_fim = gdf_fim[['class_name', 'geometry']].rename(columns={'class_name': 'classe_fim'})

    print("Alinhando sistemas de coordenadas (CRS)...")
    if gdf_inicio.crs != gdf_fim.crs:
        gdf_inicio = gdf_inicio.to_crs(gdf_fim.crs)

    print("Executando cruzamento espacial (Intersection)... Isso pode levar alguns minutos.")
    # Realiza a interseção geométrica para encontrar as sobreposições exatas
    gdf_diff = gpd.overlay(gdf_inicio, gdf_fim, how='intersection', keep_geom_type=True)

    print("Filtrando apenas as áreas que sofreram mudança...")
    # Mantém apenas os polígonos onde a classe inicial é diferente da final
    gdf_mudancas = gdf_diff[gdf_diff['classe_ini'] != gdf_diff['classe_fim']].copy()

    if gdf_mudancas.empty:
        print("\n[AVISO] Nenhuma mudança de uso do solo foi detectada entre os anos selecionados.")
        sys.exit(0)

    # Criação do rótulo de transição (ex: "Floresta -> Infraestrutura Urbana")
    gdf_mudancas['transicao'] = gdf_mudancas['classe_ini'] + " -> " + gdf_mudancas['classe_fim']

    # Gerar mapeamento de cores únicas para cada transição encontrada
    transicoes_unicas = gdf_mudancas['transicao'].unique()
    print(f"\nForam encontradas {len(transicoes_unicas)} combinações de mudanças diferentes:")
    
    mapa_cores = {}
    for i, transicao in enumerate(transicoes_unicas):
        cor = CORES_DISTINTAS[i % len(CORES_DISTINTAS)]
        mapa_cores[transicao] = cor
        print(f" - {transicao}")

    print("\nSalvando Shapefile de Mudanças...")
    gdf_mudancas.to_file(caminho_shp_saida)

    print("Gerando arquivo de estilo QGIS (.qml)...")
    gerar_estilo_qml_mudancas(caminho_qml_saida, mapa_cores)

    print(f"\n[SUCESSO] Arquivos gerados com sucesso em:")
    print(f" -> SHP: {caminho_shp_saida}")
    print(f" -> QML: {caminho_qml_saida}")

if __name__ == "__main__":
    main()