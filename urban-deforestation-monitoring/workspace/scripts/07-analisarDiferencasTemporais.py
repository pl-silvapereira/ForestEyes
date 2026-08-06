import os
import sys
import geopandas as gpd
import pandas as pd
import numpy as np
import geobr
from sklearn.metrics import cohen_kappa_score, accuracy_score
from dotenv import load_dotenv

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 07-analisarDiferencasTemporais.py <code_muni> <ano_base> <ano_comparacao_1> [ano_comparacao_2 ...]")
        print("Exemplo: python 07-analisarDiferencasTemporais.py 3549904 2020 2021 2022 2023 2024")
        sys.exit(1)

    code_muni = int(sys.argv[1])
    ano_base = str(sys.argv[2])
    anos_comparacao = [str(a) for a in sys.argv[3:]]

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

    path_shp_base = os.path.join(project_root, "data", "output", "classification", ano_base, f"{code_muni}_Classificado_ForestEyes_{ano_base}.shp")
    if not os.path.exists(path_shp_base):
        print(f"[ERRO CRÍTICO] Shapefile base não encontrado para o ano {ano_base}: {path_shp_base}")
        sys.exit(1)

    print(f"Carregando shapefile base do ano {ano_base}...")
    df_base = gpd.read_file(path_shp_base)
    utm_crs = df_base.estimate_utm_crs()
    df_base = df_base.to_crs(utm_crs)
    df_base['area_ha'] = df_base.geometry.area / 10000.0
    totais_base = df_base.groupby('class_name')['area_ha'].sum()

    relatorio_geral_linhas = []

    for ano_comp in anos_comparacao:
        path_shp_comp = os.path.join(project_root, "data", "output", "classification", ano_comp, f"{code_muni}_Classificado_ForestEyes_{ano_comp}.shp")
        
        if not os.path.exists(path_shp_comp):
            print(f"⚠️ [AVISO] Shapefile não encontrado para o ano {ano_comp}. Pulando este período.")
            continue

        print(f"Processando comparativo real: {ano_base} vs {ano_comp}...")
        df_comp = gpd.read_file(path_shp_comp).to_crs(utm_crs)
        df_comp['area_ha'] = df_comp.geometry.area / 10000.0
        totais_comp = df_comp.groupby('class_name')['area_ha'].sum()

        todas_categorias = sorted(list(set(totais_base.index).union(set(totais_comp.index))))
        dif_dict = {cat: totais_comp.get(cat, 0.0) - totais_base.get(cat, 0.0) for cat in todas_categorias}

        # Cruzamento espacial real (Overlay) para pegar o destino exato da perda de cada classe
        df1_sub = df_base[['class_name', 'geometry']].rename(columns={'class_name': 'cat_ano1'})
        df2_sub = df_comp[['class_name', 'geometry']].rename(columns={'class_name': 'cat_ano2'})
        overlap = gpd.overlay(df1_sub, df2_sub, how='intersection', keep_geom_type=True)
        overlap['area_ha'] = overlap.geometry.area / 10000.0

        # Agrupar transições reais por categoria de origem e destino
        transicoes = overlap.groupby(['cat_ano1', 'cat_ano2'])['area_ha'].sum().reset_index()

        y_true = overlap['cat_ano1']
        y_pred = overlap['cat_ano2']
        sample_weight = overlap['area_ha'].values
        acc_global = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
        kappa_idx = cohen_kappa_score(y_true, y_pred, sample_weight=sample_weight)

        relatorio_geral_linhas.append("=" * 125)
        relatorio_geral_linhas.append(f" RELATÓRIO DE PERDAS LÍQUIDAS, DESTINOS E VALIDAÇÃO ESTATÍSTICA")
        relatorio_geral_linhas.append(f" MUNICÍPIO: {nome_cidade} - {uf} (IBGE: {code_muni}) | PERÍODO: {ano_base} (Ref) vs {ano_comp}")
        relatorio_geral_linhas.append("=" * 125)
        
        header = f"{'CATEGORIA':<32} | {ano_base + ' (ha)':<12} | {ano_comp + ' (ha)':<12} | {'DIFERENÇA':<10} | {'NOVA CATEGORIA (Destino)':<32} | {'ÁREA (ha)':<10}"
        relatorio_geral_linhas.append(header)
        relatorio_geral_linhas.append("-" * 125)

        for cat in todas_categorias:
            val_1 = totais_base.get(cat, 0.0)
            val_2 = totais_comp.get(cat, 0.0)
            dif = dif_dict[cat]

            # Filtrar apenas transições onde a categoria mudou (destino diferente da origem) e houve perda líquida (dif < 0)
            if dif < 0:
                df_perdas_cat = transicoes[(transicoes['cat_ano1'] == cat) & (transicoes['cat_ano2'] != cat) & (transicoes['area_ha'] > 0.01)]
                destinos_list = list(zip(df_perdas_cat['cat_ano2'], df_perdas_cat['area_ha'])) if not df_perdas_cat.empty else []
            else:
                destinos_list = []

            if not destinos_list:
                linha = f"{cat:<32} | {val_1:>12.2f} | {val_2:>12.2f} | {dif:>+10.2f} | {'-':<32} | {'-':>10}"
                relatorio_geral_linhas.append(linha)
            else:
                primeira_linha = True
                for dest_cat, dest_area in destinos_list:
                    str_area = f"{dest_area:>+10.2f}"
                    if primeira_linha:
                        linha = f"{cat:<32} | {val_1:>12.2f} | {val_2:>12.2f} | {dif:>+10.2f} | {dest_cat:<32} | {str_area}"
                        primeira_linha = False
                    else:
                        linha = f"{'':<32} | {'':<12} | {'':<12} | {'':<10} | {dest_cat:<32} | {str_area}"
                    relatorio_geral_linhas.append(linha)

            relatorio_geral_linhas.append("-" * 125)

        relatorio_geral_linhas.append("\n" + "=" * 60)
        relatorio_geral_linhas.append(f" VALIDAÇÃO ESTATÍSTICA ({ano_base} vs {ano_comp})")
        relatorio_geral_linhas.append("=" * 60)
        relatorio_geral_linhas.append(f" • Acurácia Global (Accuracy): {acc_global * 100:.2f}%")
        relatorio_geral_linhas.append(f" • Coeficiente Kappa (Kappa Index): {kappa_idx:.4f}")
        relatorio_geral_linhas.append("=" * 60 + "\n\n")

    relatorio_nome = f"{code_muni}_change_report_multi_corrected_{ano_base}_ate_{anos_comparacao[-1]}.txt"
    relatorio_path = os.path.join(reports_dir, relatorio_nome)

    with open(relatorio_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(relatorio_geral_linhas))

    print(f"\n[SUCESSO] Relatório corrigido gerado em:\n-> {relatorio_path}\n")

if __name__ == "__main__":
    main()