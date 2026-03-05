# Guideline: Consolidação de Dados de Arborização (UrbanForest)

Este documento descreve as etapas lógicas e técnicas para a unificação dos arquivos CSV contendo dados sobre a floresta urbana da cidade de São Paulo. O objetivo é gerar um arquivo único chamado `saopaulo_city_all.csv`.

---

## 1. Mapeamento e Filtragem de Arquivos
O script inicia varrendo o diretório de dados para identificar as fontes de entrada.
* **Ferramenta:** Biblioteca `glob`.
* **Ação:** Localiza todos os arquivos com a extensão `.csv` dentro da pasta `./data` (origem: https://github.com/giva-lab/UrbanForest/tree/main/data).
* **Segurança:** O script filtra a lista para ignorar o próprio arquivo de saída (`saopaulo_city_all.csv`), caso ele já exista, evitando um loop de leitura infinito ou corrupção de dados.

## 2. Leitura com Identificação de Cabeçalho (Header)
Cada arquivo é processado individualmente pelo motor do Pandas.
* **Lógica de Header:** Por padrão, a função `pd.read_csv()` interpreta a **linha 1** de cada arquivo como o cabeçalho.
* **Carregamento:** Os dados são armazenados temporariamente em uma lista de objetos DataFrame, mantendo a estrutura de colunas separada do corpo de dados.

## 3. Empilhamento Lógico (Concatenação)
A unificação ocorre através da função `pd.concat()`, que segue estas regras:
* **Unicidade de Colunas:** O Pandas alinha as colunas pelo nome. Mesmo que a ordem das colunas mude entre os arquivos, ele garante que os dados caiam no lugar certo.
* **Descarte de Headers Repetidos:** Como os arquivos são tratados como objetos de dados, os cabeçalhos dos arquivos subsequentes não são inseridos como "linhas de texto", garantindo que o arquivo final tenha o título apenas no topo.
* **Reindexação:** O parâmetro `ignore_index=True` reconstrói a contagem de linhas (índice) de forma contínua para o novo conjunto de dados.



## 4. Escrita e Codificação Final
O resultado consolidado é exportado para o disco.
* **Parâmetro `index=False`:** Evita que o Pandas crie uma coluna extra de numeração à esquerda.
* **Codificação `utf-8-sig`:** Essencial para garantir que caracteres especiais (como "Ipê", "Jatobá" ou "Praça") sejam exibidos corretamente tanto no Excel quanto em sistemas Windows/Linux.

---

## 5. Solução de Problemas (Troubleshooting)

### 5.1. Colunas Diferentes entre Arquivos
Se houver colunas extras em alguns arquivos, o Pandas criará colunas vazias (`NaN`) para os arquivos que não as possuem.
* **Solução (Inner Join):** Para manter apenas as colunas comuns a todos, utilize:
  `pd.concat(df_list, axis=0, join='inner')`



### 5.2. Erros de Codificação (UnicodeDecodeError)
Arquivos de fontes governamentais antigas podem usar `latin1` em vez de `utf-8`.
* **Sintoma:** O script trava ao tentar ler o arquivo.
* **Solução:** Altere a leitura para `pd.read_csv(filename, encoding='latin1')`.

### 5.3. Linhas Duplicadas
Caso existam registros idênticos entre diferentes arquivos da pasta:
* **Solução:** Adicione `df.drop_duplicates(inplace=True)` antes de salvar o arquivo final.

---

## 6. Resumo Técnico do Fluxo

| Etapa | Ferramenta | Objetivo |
| :--- | :--- | :--- |
| **Varredura** | `glob` | Localizar múltiplos CSVs. |
| **Parsing** | `pd.read_csv` | Isolar dados e identificar cabeçalhos. |
| **Merge** | `pd.concat` | Unificar linhas sem repetir títulos. |
| **Export** | `to_csv` | Gerar o arquivo final limpo. |