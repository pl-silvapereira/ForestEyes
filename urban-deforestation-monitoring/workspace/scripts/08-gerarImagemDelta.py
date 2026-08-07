import os
import sys
import geopandas as gpd
import pandas as pd
import geobr
from dotenv import load_dotenv

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
    print(f"🗺️ GERANDO SHAPEFILE DELTA (VETORIAL DE MUDANÇAS)")
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

    # Padronizar nome da coluna de classe para o cruzamento
    gdf1['cat_ini'] = gdf1[col1] if col1 else 1
    gdf2['cat_fim'] = gdf2[col2] if col2 else 1

    print("Executando overlay espacial para isolar apenas o que mudou...")
    # Realiza a intersecção geográfica entre os dois anos
    overlap = gpd.overlay(gdf1[['cat_ini', 'geometry']], gdf2[['cat_fim', 'geometry']], how='intersection', keep_geom_type=True)

    # Filtra mantendo apenas onde a categoria do ano inicio é DIFERENTE da categoria do ano fim (o Delta real)
    delta_gdf = overlap[overlap['cat_ini'] != overlap['cat_fim']].copy()

    # A classe final do delta passa a ser a categoria do ano fim (para refletir a nova classe)
    delta_gdf['class_id'] = delta_gdf['cat_fim']

    # Dicionário de cores HEX que você especificou para associar ao atributo de estilo se necessário
    cores_map = {
        3: '#006400', 9: '#93c47d', 
        11: '#a8c04d', 12: '#a8c04d', 36: '#a8c04d',
        15: '#edde8e', 19: '#edde8e', 20: '#edde8e', 21: '#edde8e', 39: '#edde8e', 41: '#edde8e', 46: '#edde8e', 48: '#edde8e',
        24: '#d4271e', 25: '#d4271e',
        29: '#0000ff', 31: '#0000ff', 33: '#0000ff'
    }
    
    delta_gdf['color'] = delta_gdf['class_id'].map(cores_map).fillna('#A9A9A9')

    out_shp_name = f"{code_muni}_delta_vector_{ano_inicio}_vs_{ano_fim}.shp"
    out_shp_path = os.path.join(mask_dir, out_shp_name)

    # Salvar o shapefile resultante
    delta_gdf[['class_id', 'color', 'geometry']].to_file(out_shp_path)

    print(f"✓ Shapefile delta gerado com sucesso em:\n-> {out_shp_path}")

if __name__ == "__main__":
    main()