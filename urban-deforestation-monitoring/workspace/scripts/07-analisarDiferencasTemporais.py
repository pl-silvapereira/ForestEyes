import os
import sys
import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv

def main():
    # Uso correto: python 07-analisarDiferencasTemporais.py <code_muni> <ano_1> <ano_2>
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

    path_shp_1 = os.path.join(project_root, "data", "output", "classification", ano_1, f"{code_muni}_Classificado_ForestEyes_{ano_1}.shp")
    path_shp_2 = os.path.join(project_root, "data", "output", "classification", ano_2, f"{code_muni}_Classificado_ForestEyes_{ano_2}.shp")

    if not os.path.exists(path_shp_1) or not os.path.exists(path_shp_2):
        print(f"[ERRO CRÍTICO] Shapefiles não encontrados para {ano_1} e/ou {ano_2}.")
        print(f" -> [{ano_1}]: {path_shp_1}")
        print(f" -> [{ano_2}]: {path_shp_2}")
        sys.exit(1)

    print("=" * 125)
    print(f"📊 GERANDO RELATÓRIO DE PERDAS LÍQUIDAS E DESTINOS DE USO DO SOLO")
    print(f"📍 MUNICÍPIO: {code_muni} | PERÍODO: {ano_1} vs {ano_2}")
    print("=" * 125)

    df1 = gpd.read_file(path_shp_1)
    df2 = gpd.read_file(path_shp_2)

    print("Reprojetando bases para sistema métrico (UTM) para cálculo preciso em hectares...")
    utm_crs = df1.estimate_utm_crs()
    df1 = df1.to_crs(utm_crs)
    df2 = df2.to_crs(utm_crs)

    df1['area_ha'] = df1.geometry.area / 10000.0
    df2['area_ha'] = df2.geometry.area / 10000.0

    totais_1 = df1.groupby('class_name')['area_ha'].sum()
    totais_2 = df2.groupby('class_name')['area_ha'].sum()

    todas_categorias = sorted(list(set(totais_1.index).union(set(totais_2.index))))

    # Calcular diferenças para cada categoria
    dif_dict = {cat: totais_2.get(cat, 0.0) - totais_1.get(cat, 0.0) for cat in todas_categorias}

    # Identificar categorias que ganharam área (dif > 0) para usar como destino das perdas
    categorias_ganho = {cat: d for cat, d in dif_dict.items() if d > 0}

    linhas_relatorio = []
    linhas_relatorio.append("=" * 125)
    linhas_relatorio.append(f" RELATÓRIO DE PERDAS LÍQUIDAS E DESTINOS DE USO DO SOLO")
    linhas_relatorio.append(f" MUNICÍPIO: {code_muni} | PERÍODO: {ano_1} vs {ano_2}")
    linhas_relatorio.append("=" * 125)
    
    header = f"{'CATEGORIA':<32} | {ano_1 + ' (ha)':<12} | {ano_2 + ' (ha)':<12} | {'DIFERENÇA':<10} | {'NOVA CATEGORIA (Destino)':<32} | {'ÁREA (ha)':<10}"
    linhas_relatorio.append(header)
    linhas_relatorio.append("-" * 125)

    for cat in todas_categorias:
        val_1 = totais_1.get(cat, 0.0)
        val_2 = totais_2.get(cat, 0.0)
        dif = dif_dict[cat]

        # Se a categoria perdeu área (dif < 0), detalhar os destinos com base nas classes que ganharam
        if dif < 0 and categorias_ganho:
            destinos_list = list(categorias_ganho.items())
        else:
            destinos_list = []

        if not destinos_list:
            linha = f"{cat:<32} | {val_1:>12.2f} | {val_2:>12.2f} | {dif:>+10.2f} | {'-':<32} | {'-':>10}"
            linhas_relatorio.append(linha)
        else:
            primeira_linha = True
            for dest_cat, dest_dif in destinos_list:
                str_area = f"{dest_dif:>+10.2f}"
                if primeira_linha:
                    linha = f"{cat:<32} | {val_1:>12.2f} | {val_2:>12.2f} | {dif:>+10.2f} | {dest_cat:<32} | {str_area}"
                    primeira_linha = False
                else:
                    linha = f"{'':<32} | {'':<12} | {'':<12} | {'':<10} | {dest_cat:<32} | {str_area}"
                linhas_relatorio.append(linha)

        linhas_relatorio.append("-" * 125)

    linhas_relatorio.append("=" * 125)

    relatorio_nome = f"{code_muni}_change_report_losses_{ano_1}_vs_{ano_2}.txt"
    relatorio_path = os.path.join(reports_dir, relatorio_nome)

    with open(relatorio_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(linhas_relatorio))

    print(f"\n[SUCESSO] Relatório de perdas ajustado gerado em:\n-> {relatorio_path}\n")

if __name__ == "__main__":
    main()