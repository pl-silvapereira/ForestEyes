import sys
import os
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
import geobr
from dotenv import load_dotenv

def gerar_estilo_qml_automatico(caminho_qml, metadata):
    categorias, simbolos = "", ""
    for i, (nome, cor) in enumerate(metadata.items()):
        h = cor.lstrip('#')
        r, g, b = tuple(int(h[j:j+2], 16) for j in (0, 2, 4))
        symbol_name = str(i)
        categorias += f'\n'
        simbolos += f"""
    
      
        
        
      
    """

    conteudo_qml = f"""

  
    {categorias}
    {simbolos}
  
"""
    
    with open(caminho_qml, 'w', encoding='utf-8') as f:
        f.write(conteudo_qml)

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 05-segmentacaoZooniverse.py   ")
        print("Exemplo: python 05-segmentacaoZooniverse.py 3549904 2020 2024")
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
    print(f"🔬 SCRIPT 05 - GERAÇÃO DO MAPA BINÁRIO DE SEGMENTAÇÃO ({ano_fim})")
    print("=" * 80)

    path_shp_class = os.path.join(project_root, "data", "output", "classification", ano_fim, f"{code_muni}_Classificado_ForestEyes_{ano_fim}.shp")
    path_shp_mudancas = os.path.join(project_root, "data", "output", "analysis", "mudancas", f"{code_muni}_Mudancas_{ano_inicio}_vs_{ano_fim}.shp")

    if not os.path.exists(path_shp_class):
        print(f"[ERRO CRÍTICO] Shapefile de classificação não encontrado: {path_shp_class}")
        sys.exit(1)

    print("1. Lendo shapefile de classificação e filtrando 'Floresta' e 'Floresta Antrópica'...")
    gdf_class = gpd.read_file(path_shp_class)
    gdf_floresta_base = gdf_class[gdf_class['class_name'].isin(['Floresta', 'Floresta Antrópica'])].copy()
    gdf_floresta_base['class_name'] = 'Segmentar'

    lista_segmentar = [gdf_floresta_base[['geometry', 'class_name']]]

    if os.path.exists(path_shp_mudancas):
        print("2. Lendo shapefile de mudanças e filtrando as categorias de transição...")
        gdf_mudancas = gpd.read_file(path_shp_mudancas)
        transicoes_desejadas = [
            'Floresta -> Agropecuaria (Campos, Lavouras)',
            'Floresta -> Floresta Antrópica',
            'Floresta -> Infraestrutura Urbana',
            'Floresta -> Nao Observado (Agua, Rocha)',
            'Floresta -> Vegetacao Herbacea e Arbustiva'
        ]
        if 'transicao' in gdf_mudancas.columns:
            gdf_mud_filt = gdf_mudancas[gdf_mudancas['transicao'].isin(transicoes_desejadas)].copy()
            if not gdf_mud_filt.empty:
                gdf_mud_filt['class_name'] = 'Segmentar'
                lista_segmentar.append(gdf_mud_filt[['geometry', 'class_name']])

    gdf_segmentar_total = gpd.GeoDataFrame(pd.concat(lista_segmentar, ignore_index=True), crs=gdf_class.crs)

    print("3. Obtendo contorno geopolítico municipal via geobr...")
    gdf_muni = geobr.read_municipality(code_muni=int(code_muni), year=2022)
    gdf_muni = gdf_muni.to_crs(gdf_segmentar_total.crs)
    limite_municipal = gdf_muni.geometry.unary_union

    print("4. Processando áreas de Segmentar e preenchendo o Nao_Segmentar...")
    geometria_segmentar_unificada = unary_union(gdf_segmentar_total.geometry.buffer(0))
    geom_nao_segmentar = limite_municipal.difference(geometria_segmentar_unificada)

    gdf_binario = gpd.GeoDataFrame({
        'class_name': ['Segmentar', 'Nao_Segmentar'],
        'geometry': [geometria_segmentar_unificada, geom_nao_segmentar]
    }, crs=gdf_segmentar_total.crs)

    gdf_binario = gdf_binario[~gdf_binario.geometry.is_empty]
    gdf_binario = gdf_binario.explode(index_parts=False).reset_index(drop=True)

    nome_base = f"{code_muni}_Segmentar_vs_NaoSegmentar_{ano_fim}"
    path_shp_out = os.path.join(output_seg_dir, f"{nome_base}.shp")
    path_qml_out = os.path.join(output_seg_dir, f"{nome_base}.qml")

    gdf_binario.to_file(path_shp_out, encoding='utf-8')
    
    metadata_segmentacao = {
        'Segmentar': '#33a02c',     # Verde
        'Nao_Segmentar': '#ffe600'  # Amarelo
    }
    gerar_estilo_qml_automatico(path_qml_out, metadata_segmentacao)

    print(f"✅ Shapefile salvo em:\n   -> {path_shp_out}")
    print(f"✅ Arquivo QML gerado automaticamente em:\n   -> {path_qml_out}")
    print("🎉 Processo concluído com êxito!")

if __name__ == "__main__":
    main()