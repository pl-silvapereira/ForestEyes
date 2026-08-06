import os
import sys
import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv

def main():
    # 1. Validar argumentos da linha de comando
    # Uso correto: python 07-analisarDiferencasTemporais.py <code_muni> <ano_1> <ano_2>
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 07-analisarDiferencasTemporais.py <code_muni> <ano_1> <ano_2>")
        print("Exemplo: python 07-analisarDiferencasTemporais.py 3549904 2022 2023")
        sys.exit(1)

    code_muni = int(sys.argv[1])
    ano_1 = str(sys.argv[2])
    ano_2 = str(sys.argv[3])

    # 2. Carregar variáveis de ambiente e diretórios
    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    reports_dir = os.path.join(project_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # 3. Caminhos dos shapefiles gerados pelo Script 05
    path_shp_1 = os.path.join(project_root, "data", "output", "classification", ano_1, f"{code_muni}_Classificado_ForestEyes_{ano_1}.shp")
    path_shp_2 = os.path.join(project_root, "data", "output", "classification", ano_2, f"{code_muni}_Classificado_ForestEyes_{ano_2}.shp")

    if not os.path.exists(path_shp_1) or not os.path.exists(path_shp_2):
        print(f"[ERRO CRÍTICO] Shapefiles de classificação não encontrados para os anos {ano_1} e/ou {ano_2}.")
        print(f" -> Esperado [{ano_1}]: {path_shp_1}")
        print(f" -> Esperado [{ano_2}]: {path_shp_2}")
        sys.exit(1)

    print("=" * 70)
    print(f"📊 INICIANDO ANÁLISE DE MUDANÇA TEMPORAL DE USO DO SOLO")
    print(f"📍 MUNICÍPIO: {code_muni} | PERÍODO: {ano_1} ➔ {ano_2}")
    print("=" * 70)

    print(f"Carregando shapefile do ano {ano_1}...")
    df1 = gpd.read_file(path_shp_1)
    
    print(f"Carregando shapefile do ano {ano_2}...")
    df2 = gpd.read_file(path_shp_2)

    # 4. Reprojetar para UTM (métrico) para garantir cálculo de área exato em hectares
    print("Reprojetando bases para sistema métrico (UTM)...")
    utm_crs = df1.estimate_utm_crs()
    df1 = df1.to_crs(utm_crs)
    df2 = df2.to_crs(utm_crs)

    # 5. Calcular área total por categoria em cada ano (Convertendo para hectares: m² / 10000)
    df1['area_ha'] = df1.geometry.area / 10000.0
    df2['area_ha'] = df2.geometry.area / 10000.0

    totais_1 = df1.groupby('class_name')['area_ha'].sum()
    totais_2 = df2.groupby('class_name')['area_ha'].sum()

    todas_categorias = sorted(list(set(totais_1.index).union(set(totais_2.index))))

    print("Executando cruzamento espacial (Overlay) para detectar transições...")
    df1_sub = df1[['class_name', 'geometry']].rename(columns={'class_name': 'cat_ano1'})
    df2_sub = df2[['class_name', 'geometry']].rename(columns={'class_name': 'cat_ano2'})

    # Overlay de interseção para identificar conversões espaciais de uso
    overlap = gpd.overlay(df1_sub, df2_sub, how='intersection', keep_geom_type=True)
    overlap['area_ha'] = overlap.geometry.area / 10000.0

    transicoes = overlap.groupby(['cat_ano1', 'cat_ano2'])['area_ha'].sum().reset_index()

    # 6. Montar o relatório em texto estruturado
    linhas_relatorio = []
    linhas_relatorio.append("=" * 105)
    linhas_relatorio.append(f" RELATÓRIO DE ANÁLISE DE MUDANÇA TEMPORAL DE USO DO SOLO")
    linhas_relatorio.append(f" MUNICÍPIO: {code_muni} | PERÍODO ANALISADO: {ano_1} vs {ano_2}")
    linhas_relatorio.append("=" * 105)
    linhas_relatorio.append(f"{'CATEGORIA':<35} | {ano_1} (ha):<10} | {ano_2} (ha):<10} | DIFERENÇA | DESTINO DA MUDANÇA (NOVA CATEGORIA) | ÁREA (ha)")
    linhas_relatorio.append("-" * 105)

    for cat in todas_categorias:
        val_1 = totais_1.get(cat, 0.0)
        val_2 = totais_2.get(cat, 0.0)
        dif = val_2 - val_1

        trans_cat = transicoes[transicoes['cat_ano1'] == cat]

        primeira_linha = True
        if trans_cat.empty:
            linha = f"{cat:<35} | {val_1:>10.2f} | {val_2:>10.2f} | {dif:>+9.2f} | {'Nenhuma alteração / Mantido':<35} | {'-':>10}"
            linhas_relatorio.append(linha)
        else:
            for _, row in trans_cat.iterrows():
                destino = row['cat_ano2']
                qte_area = row['area_ha']
                if primeira_linha:
                    linha = f"{cat:<35} | {val_1:>10.2f} | {val_2:>10.2f} | {dif:>+9.2f} | {destino:<35} | {qte_area:>10.2f}"
                    linhas_relatorio.append(linha)
                    primeira_linha = False
                else:
                    linha = f"{'':<35} | {'':>10} | {'':>10} | {'':>9} | {destino:<35} | {qte_area:>10.2f}"
                    linhas_relatorio.append(linha)

    linhas_relatorio.append("=" * 105)

    # 7. Salvar o arquivo de relatório em .txt na pasta reports
    relatorio_nome = f"{code_muni}_change_report_{ano_1}_vs_{ano_2}.txt"
    relatorio_path = os.path.join(reports_dir, relatorio_nome)

    with open(relatorio_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(linhas_relatorio))

    print(f"\n[SUCESSO] Relatório gerado e salvo com sucesso em:\n-> {relatorio_path}\n")

if __name__ == "__main__":
    main()