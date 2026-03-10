import pandas as pd
import glob
import os

# Configurações de caminho
input_path = './Files/data' 
output_file = 'saopaulo_city_all_unido.csv'

# Busca todos os arquivos CSV na pasta
all_files = glob.glob(os.path.join(input_path, "*.csv"))

# Evita que o script leia o arquivo de saída caso ele já exista na pasta de entrada
all_files = [f for f in all_files if os.path.basename(f) != output_file]

if not all_files:
    print(f"Nenhum arquivo CSV encontrado em: {os.path.abspath(input_path)}")
else:
    print(f"Lendo {len(all_files)} arquivos...")
    
    df_list = []
    
    for filename in all_files:
        try:
            # Lendo com sep=';' porque seus arquivos usam ponto e vírgula
            # encoding='utf-8' ou 'latin1' dependendo de como os arquivos foram salvos
            df = pd.read_csv(filename, sep=';', encoding='utf-8-sig')
            df_list.append(df)
        except Exception as e:
            print(f"Erro ao ler {filename}: {e}")

    if df_list:
        # Concatena todos os DataFrames da lista
        combined_df = pd.concat(df_list, axis=0, ignore_index=True)

        # Salva o arquivo final mantendo o padrão de ponto e vírgula
        # O utf-8-sig garante que o Excel abra o arquivo corretamente no Windows
        combined_df.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')

        print(f"\nSucesso!")
        print(f"Total de linhas combinadas: {len(combined_df)}")
        print(f"Arquivo final salvo como: {output_file}")