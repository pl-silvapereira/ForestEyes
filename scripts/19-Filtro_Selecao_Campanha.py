import os
import pandas as pd
from dotenv import load_dotenv

def executar_selecao_e_relatorio():
    load_dotenv()
    ROOT = os.getenv('PROJECT_ROOT')
    if not ROOT:
        print("❌ Erro: Variável 'PROJECT_ROOT' não encontrada.")
        return

    dir_output = os.path.join(ROOT, 'data', 'Output')
    csv_path = os.path.join(dir_output, "19_SJC_Estatisticas_Superpixels.csv")
    csv_saida = os.path.join(dir_output, "19_SJC_Alvos_Campanha.csv")

    if not os.path.exists(csv_path):
        print("❌ Erro: Arquivo de estatísticas não encontrado.")
        return

    # -------------------------------------------------------------
    # 1. FILTRAGEM (Tamanho Mínimo 850px e Máximo 4000px)
    # -------------------------------------------------------------
    df = pd.read_csv(csv_path, sep=';')
    total_original = len(df)
    
    df_filtrado = df[(df['Quantidade_Pixels'] >= 850) & (df['Quantidade_Pixels'] <= 4000)].copy()
    total_filtrado = len(df_filtrado)

    df_floresta = df_filtrado[df_filtrado['Classe_Majoritaria'] == 'Floresta']
    df_nao_floresta = df_filtrado[df_filtrado['Classe_Majoritaria'] == 'Nao_Floresta']

    # -------------------------------------------------------------
    # 2. SELEÇÃO DOS 100 ALVOS PARA A CAMPANHA
    # -------------------------------------------------------------
    # A) 25 Perfeitos (Floresta) -> HoR = 100%, maiores tamanhos
    perf_f = df_floresta[df_floresta['Taxa_HoR'] == 100.0].nlargest(25, 'Quantidade_Pixels')
    perf_f['Tipo_Selecao'] = 'Perfeito'
    
    # B) 25 Imperfeitos (Floresta) -> HoR >= 70% e < 100%, menores HoR e maiores tamanhos
    imp_f = df_floresta[(df_floresta['Taxa_HoR'] >= 70.0) & (df_floresta['Taxa_HoR'] < 100.0)]
    imp_f = imp_f.sort_values(by=['Taxa_HoR', 'Quantidade_Pixels'], ascending=[True, False]).head(25)
    imp_f['Tipo_Selecao'] = 'Imperfeito'

    # C) 25 Perfeitos (Não Floresta)
    perf_nf = df_nao_floresta[df_nao_floresta['Taxa_HoR'] == 100.0].nlargest(25, 'Quantidade_Pixels')
    perf_nf['Tipo_Selecao'] = 'Perfeito'
    
    # D) 25 Imperfeitos (Não Floresta)
    imp_nf = df_nao_floresta[(df_nao_floresta['Taxa_HoR'] >= 70.0) & (df_nao_floresta['Taxa_HoR'] < 100.0)]
    imp_nf = imp_nf.sort_values(by=['Taxa_HoR', 'Quantidade_Pixels'], ascending=[True, False]).head(25)
    imp_nf['Tipo_Selecao'] = 'Imperfeito'

    # Junta as seleções e salva o novo CSV que será lido pelo Script 20
    df_campanha = pd.concat([perf_f, imp_f, perf_nf, imp_nf])
    df_campanha.to_csv(csv_saida, sep=';', index=False)

    # -------------------------------------------------------------
    # 3. IMPRESSÃO DO RELATÓRIO ESTATÍSTICO DETALHADO POR CLASSE
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("✅ SCRIPT DE FILTRAGEM FINALIZADO!")
    print("="*50)
    print("RELATÓRIO ESTATÍSTICO DE SUPERPIXELS FILTRADOS (850 - 4000 px):")
    print("-" * 50)
    print(f"🌲 Segmentos de Floresta: {len(df_floresta)}")
    print(f"🍂 Segmentos Não Floresta: {len(df_nao_floresta)}")
    print(f"   (Removidos {total_original - total_filtrado} segmentos fora do limite de tamanho)")
    
    for nome_classe, df_classe, icone in [("FLORESTA", df_floresta, "🌲"), ("NÃO FLORESTA", df_nao_floresta, "🍂")]:
        print("-" * 50)
        print(f"{icone} CLASSE {nome_classe}:")
        if len(df_classe) > 0:
            print("  📏 TAMANHO (PIXELS):")
            print(f"     Média:  {df_classe['Quantidade_Pixels'].mean():.2f}")
            print(f"     Mediana:{df_classe['Quantidade_Pixels'].median():.2f}")
            print(f"     Desvio: {df_classe['Quantidade_Pixels'].std():.2f}")
            print(f"     Maior:  {df_classe['Quantidade_Pixels'].max()}")
            print(f"     Menor:  {df_classe['Quantidade_Pixels'].min()}")
            print("  🎯 TAXA DE HOMOGENEIDADE (HoR %):")
            print(f"     Média:  {df_classe['Taxa_HoR'].mean():.2f}%")
            print(f"     Mediana:{df_classe['Taxa_HoR'].median():.2f}%")
            print(f"     Desvio: {df_classe['Taxa_HoR'].std():.2f}%")
        else:
            print("     Nenhum segmento passou nos filtros para esta classe.")
            
    print("="*50)
    print(f"100 alvos para a Campanha foram selecionados e salvos em:\n{csv_saida}")

if __name__ == "__main__":
    executar_selecao_e_relatorio()