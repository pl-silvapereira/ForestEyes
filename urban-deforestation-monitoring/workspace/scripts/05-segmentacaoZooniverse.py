import os
import sys
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
import geobr
from dotenv import load_dotenv

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

    # Caminhos dos arquivos de entrada
    path_shp_class = os.path.join(project_root, "data", "output", "classification", ano_fim, f"{code_muni}_Classificado_ForestEyes_{ano_fim}.shp")
    path_shp_mudancas = os.path.join(project_root, "data", "output", "analysis", "mudancas", f"{code_muni}_Mudancas_{ano_inicio}_vs_{ano_fim}.shp")

    if not os.path.exists(path_shp_class):
        print(f"[ERRO CRÍTICO] Shapefile de classificação não encontrado: {path_shp_class}")
        sys.exit(1)

    print("1. Lendo shapefile de classificação e filtrando APENAS a categoria 'Floresta'...")
    gdf_class = gpd.read_file(path_shp_class)
    gdf_floresta = gdf_class[gdf_class['class_name'] == 'Floresta'].copy()
    gdf_floresta['tipo_seg'] = 'Segmentar'

    lista_segmentar = [gdf_floresta[['geometry', 'tipo_seg']]]

    if os.path.exists(path_shp_mudancas):
        print("2. Lendo shapefile de mudanças e filtrando as categorias de transição especificadas...")
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
                gdf_mud_filt['tipo_seg'] = 'Segmentar'
                lista_segmentar.append(gdf_mud_filt[['geometry', 'tipo_seg']])

    # Unir todas as geometrias de segmentação
    gdf_segmentar_total = gpd.GeoDataFrame(pd.concat(lista_segmentar, ignore_index=True), crs=gdf_class.crs)

    print("3. Obtendo contorno geopolítico municipal via geobr...")
    gdf_muni = geobr.read_municipality(code_muni=int(code_muni), year=2022)
    gdf_muni = gdf_muni.to_crs(gdf_segmentar_total.crs)
    limite_municipal = gdf_muni.geometry.unary_union

    print("4. Processando áreas de Segmentar e preenchendo o restante como Não Segmentar...")
    geometria_segmentar_unificada = unary_union(gdf_segmentar_total.geometry.buffer(0))
    
    # Preenche o contorno geopolítico sem sobrepor o que foi segmentado
    geom_nao_segmentar = limite_municipal.difference(geometria_segmentar_unificada)

    # Montar GeoDataFrame final com as duas categorias (Não Segmentar na base, Segmentar por cima)
    gdf_binario = gpd.GeoDataFrame({
        'Categoria': ['Não Segmentar', 'Segmentar'],
        'geometry': [geom_nao_segmentar, geometria_segmentar_unificada]
    }, crs=gdf_segmentar_total.crs)

    gdf_binario = gdf_binario[~gdf_binario.geometry.is_empty]
    gdf_binario = gdf_binario.explode(index_parts=False).reset_index(drop=True)

    # Salvar Shapefile
    path_shp_out = os.path.join(output_seg_dir, f"{code_muni}_Segmentar_vs_NaoSegmentar_{ano_fim}.shp")
    gdf_binario.to_file(path_shp_out)
    print(f"✅ Shapefile salvo com sucesso em:\n   -> {path_shp_out}")

    # Salvar QML (Não Segmentar = Amarelo, Segmentar = Verde)
    path_qml_out = os.path.join(output_seg_dir, f"{code_muni}_Segmentar_vs_NaoSegmentar_{ano_fim}.qml")
    qml_content = """

  
    
      
      
    
    
      
        
          
          
        
      
      
        
          
          
        
      
    
  
"""
    with open(path_qml_out, 'w', encoding='utf-8') as f:
        f.write(qml_content)
    print(f"✅ Arquivo QML salvo com sucesso em:\n   -> {path_qml_out}")
    print("🎉 Processo concluído com êxito!")

if __name__ == "__main__":
    main()