import pandas as pd
import glob
import os

# Configurações de caminho
input_path = './Files/data' 
output_file = 'saopaulo_city_all.csv'

# Busca todos os arquivos CSV na pasta
all_files = glob.glob(os.path.join(input_path, "*.csv"))

# Remove o arquivo de saída da lista, caso ele já exista na pasta, 
# para evitar que o script tente ler o que ele mesmo está criando
all_files = [f for f in all_files if output_file not in f]

if not all_files:
    print("Nenhum arquivo CSV encontrado na pasta especificada.")
else:
    # 1. Cria uma lista de DataFrames
    # O Pandas lê o cabeçalho (linha 1) automaticamente em cada arquivo
    df_list = [pd.read_csv(filename) for filename in all_files]

    # 2. Concatena tudo
    # O Pandas alinha as colunas e ignora os cabeçalhos repetidos internamente
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)

    # 3. Salva o arquivo final
    # index=False impede que o Python crie uma coluna extra de números à esquerda
    combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"Processo concluído! {len(all_files)} arquivos unidos em '{output_file}'.")