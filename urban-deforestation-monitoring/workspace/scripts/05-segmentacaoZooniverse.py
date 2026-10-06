import sys
import os
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
import geobr
from dotenv import load_dotenv

def gerar_estilo_qml_garantido(caminho_qml):
    # Usando lista de strings rígida para evitar QUALQUER erro de formatação/corte no Colab
    linhas_qml = [
        "",
        "",
        "  ",
        "    ",
        "      ",
        "      ",
        "    ",
        "    ",
        "      ",
        "        ",
        "          ",
        "          ",
        "        ",
        "      ",
        "      ",
        "        ",
        "          ",
        "          ",
        "        ",
        "      ",
        "    ",
        "  ",
        ""
    ]
    
    with open(caminho_qml, 'w', encoding='utf-8') as f:
        f.write("\n".join(linhas_qml))
        f.flush()
        os.fsync(f.fileno())

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 05-segmentacaoZooniverse.py   ")
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
        print(f"[ERRO CRÍTICO] Shapefile não encontrado: {path_shp_class}")
        sys.exit(1)

    print("1. Lendo shapefile de classificação...")
    gdf_class = gpd.read_file(path_shp_class)
    gdf_floresta = gdf_class[gdf_class['class_name'].isin(['Floresta', 'Floresta Antrópica'])].copy()
    
    # IMPORTANTE: Coluna curta 'classe' para evitar truncamento no Shapefile (.dbf)
    gdf_floresta['classe'] = 'Segmentar'
    lista_segmentar = [gdf_floresta[['geometry', 'classe']]]

    if os.path.exists(path_shp_mudancas):
        print("2. Lendo shapefile de mudanças...")
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
                gdf_filt['classe'] = 'Segmentar'
                lista_segmentar.append(gdf_filt[['geometry', 'classe']])

    gdf_seg_total = gpd.GeoDataFrame(pd.concat(lista_segmentar, ignore_index=True), crs=gdf_class.crs)

    print("3. Obtendo contorno geopolítico via geobr...")
    gdf_muni = geobr.read_municipality(code_muni=int(code_muni), year=2022)
    gdf_muni = gdf_muni.to_crs(gdf_seg_total.crs)
    
    # Atualizado para union_all() para evitar o DeprecationWarning no console
    limite = gdf_muni.geometry.union_all()

    print("4. Processando áreas...")
    geom_seg = unary_union(gdf_seg_total.geometry.buffer(0))
    geom_nao_seg = limite.difference(geom_seg)

    gdf_binario = gpd.GeoDataFrame({
        'classe': ['Segmentar', 'Nao_Segmentar'],
        'geometry': [geom_seg, geom_nao_seg]
    }, crs=gdf_seg_total.crs)

    gdf_binario = gdf_binario[~gdf_binario.geometry.is_empty]
    gdf_binario = gdf_binario.explode(index_parts=False).reset_index(drop=True)

    nome_base = f"{code_muni}_Segmentar_vs_NaoSegmentar_{ano_fim}"
    path_shp = os.path.join(output_seg_dir, f"{nome_base}.shp")
    path_qml = os.path.join(output_seg_dir, f"{nome_base}.qml")

    print("5. Salvando Shapefile...")
    gdf_binario.to_file(path_shp, encoding='utf-8')
    
    print("6. Gerando arquivo de estilo QML...")
    gerar_estilo_qml_garantido(path_qml)

    # Validador de gravação real
    tamanho_qml = os.path.getsize(path_qml)
    if tamanho_qml < 500:
        print(f"\n❌ ERRO GRAVE: O arquivo QML foi criado com {tamanho_qml} bytes! Isso não vai funcionar no QGIS.")
    else:
        print(f"\n✅ SUCESSO ABSOLUTO! Arquivo QML gravado corretamente (Tamanho real: {tamanho_qml} bytes).")
        print(f"✅ Shapefile pronto em: {path_shp}")

if __name__ == "__main__":
    main()