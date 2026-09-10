import os
import sys
import geopandas as gpd
import pandas as pd
import geobr
from dotenv import load_dotenv

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 07-analisarDiferencasTemporais.py <code_muni> <ano_1> <ano_2>")
        print("Exemplo: python 07-analisarDiferencasTemporais.py 3549904 2023 2024")
        sys.exit(1)

    code_muni = int(sys.argv[1])
    ano_1 = str(sys.argv[2])
    ano_2 = str(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    reports_dir = os.path.join(project_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    print(f"Buscando informações para o código de município: {code_muni}...")
    try:
        gdf_info = geobr.read_municipality(code_muni=code_muni, year=2022)
        nome_cidade = gdf_info['name_muni'].values[0]
        uf = gdf_info['abbrev_state'].values[0]
    except Exception as e:
        nome_cidade = "Município Desconhecido"
        uf = "XX"
        print(f"⚠️ Aviso ao buscar nome da cidade: {e}")

    path_shp_1 = os.path.join(project_root, "data", "output", "classification", ano_1, f"{code_muni}_Classificado_ForestEyes_{ano_1}.shp")
    path_shp_2 = os.path.join(project_root, "data", "output", "classification", ano_2, f"{code_muni}_Classificado_ForestEyes_{ano_2}.shp")

    if not os.path.exists(path_shp_1) or not os.path.exists(path_shp_2):
        print(f"[ERRO CRÍTICO] Shapefiles não encontrados para {ano_1} e/ou {ano_2}.")
        sys.exit(1)

    print("=" * 135)
    print(f"📊 GERANDO RELATÓRIO DETALHADO DE TRANSIÇÕES (SALDO LÍQUIDO POR CLASSE)")
    print(f"📍 MUNICÍPIO: {nome_cidade} - {uf} (IBGE: {code_muni}) | PERÍODO: {ano_1} vs {ano_2}")
    print("=" * 135)

    df1 = gpd.read_file(path_shp_1)
    df2 = gpd.read_file(path_shp_2)

    print("Reprojetando bases para sistema métrico (UTM)...")
    utm_crs = df1.estimate_utm_crs()
    df1 = df1.to_crs(utm_crs)
    df2 = df2.to_crs(utm_crs)

    df1['area_ha'] = df1.geometry.area / 10000.0
    df2['area_ha'] = df2.geometry.area / 10000.0

    totais_1 = df1.groupby('class_name')['area_ha'].sum()
    totais_2 = df2.groupby('class_name')['area_ha'].sum()

    todas_categorias = sorted(list(set(totais_1.index).union(set(totais_2.index))))

    print("Executando cruzamento espacial (Overlay) para rastrear todas as transições...")
    df1_sub = df1[['class_name', 'geometry']].rename(columns={'class_name': 'cat_ano1'})
    df2_sub = df2[['class_name', 'geometry']].rename(columns={'class_name': 'cat_ano2'})
    
    overlap = gpd.overlay(df1_sub, df2_sub, how='intersection', keep_geom_type=True)
    overlap['area_ha'] = overlap.geometry.area / 10000.0
    transicoes = overlap.groupby(['cat_ano1', 'cat_ano2'])['area_ha'].sum().reset_index()

    linhas_relatorio = []
    linhas_relatorio.append("=" * 135)
    linhas_relatorio.append(f" RELATÓRIO DETALHADO DE TRANSIÇÕES DE USO DO SOLO (SALDO LÍQUIDO POR CLASSE)")
    linhas_relatorio.append(f" MUNICÍPIO: {nome_cidade} - {uf} (IBGE: {code_muni}) | PERÍODO: {ano_1} vs {ano_2}")
    linhas_relatorio.append("=" * 135)
    
    header = f"{'CATEGORIA':<32} | {ano_1 + ' (ha)':<12} | {ano_2 + ' (ha)':<12} | {'DIFERENÇA':<10} | {'FLUXO (De: Ganho / Para: Perda)':<35} | {'ÁREA (ha)':<10}"
    linhas_relatorio.append(header)
    linhas_relatorio.append("-" * 135)

    for cat in todas_categorias:
        val_1 = totais_1.get(cat, 0.0)
        val_2 = totais_2.get(cat, 0.0)
        dif = val_2 - val_1

        detalhes_list = []
        
        # Para cada categoria, calcular o saldo líquido com cada uma das outras classes
        for other_cat in todas_categorias:
            if other_cat == cat:
                continue
            
            # Ganhos vindos de other_cat para cat
            ganho_row = transicoes[(transicoes['cat_ano1'] == other_cat) & (transicoes['cat_ano2'] == cat)]
            area_ganho = ganho_row['area_ha'].values[0] if not ganho_row.empty else 0.0

            # Perdas de cat indo para other_cat
            perda_row = transicoes[(transicoes['cat_ano1'] == cat) & (transicoes['cat_ano2'] == other_cat)]
            area_perda = perda_row['area_ha'].values[0] if not perda_row.empty else 0.0

            net_area = area_ganho - area_perda

            if net_area > 0.001:
                detalhes_list.append((f"De: {other_cat}", net_area))
            elif net_area < -0.001:
                detalhes_list.append((f"Para: {other_cat}", net_area))
            else:
                detalhes_list.append((f"De/Para: {other_cat}", 0.00))

        # Ordenar os fluxos pelo valor absoluto (maiores saldos primeiro)
        detalhes_list = sorted(detalhes_list, key=lambda x: (abs(x[1]) > 0.001, abs(x[1])), reverse=True)

        if not detalhes_list:
            linha = f"{cat:<32} | {val_1:>12.2f} | {val_2:>12.2f} | {dif:>+10.2f} | {'-':<35} | {'-':>10}"
            linhas_relatorio.append(linha)
        else:
            primeira_linha = True
            for item_cat, item_area in detalhes_list:
                str_area = f"{item_area:>+10.2f}"
                if primeira_linha:
                    linha = f"{cat:<32} | {val_1:>12.2f} | {val_2:>12.2f} | {dif:>+10.2f} | {item_cat:<35} | {str_area}"
                    primeira_linha = False
                else:
                    linha = f"{'':<32} | {'':<12} | {'':<12} | {'':<10} | {item_cat:<35} | {str_area}"
                linhas_relatorio.append(linha)

    linhas_relatorio.append("-" * 135)
    linhas_relatorio.append("=" * 135)

    relatorio_nome = f"{code_muni}_change_report_losses_{ano_1}_vs_{ano_2}.txt"
    relatorio_path = os.path.join(reports_dir, relatorio_nome)

    with open(relatorio_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(linhas_relatorio))

    print(f"\n[SUCESSO] Relatório de saldo líquido gerado com sucesso em:\n-> {relatorio_path}\n")

if __name__ == "__main__":
    main()