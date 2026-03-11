# Metodologia de Distribuição e Análise Espacial

Após a classificação e a detecção de mudanças, os dados do *ForestEyes* são organizados para suportar análises territoriais. A distribuição desses dados é feita garantindo a integridade geográfica e a interoperabilidade com sistemas de informação.

## 1. Agregação por Unidades Territoriais
Para transformar dados de "pixels" em "informação de gestão", agregamos os resultados (floresta e desmatamento) por unidades administrativas.
* **Geoprocessamento:** Utilizamos a operação `reduceRegions` no Google Earth Engine. 
* **Lógica:** Cruzamos a camada de classificação (Raster) com a base vetorial de distritos ou subprefeituras de São Paulo (Shapefile).
* **Resultado:** Cada unidade administrativa recebe um valor calculado de área florestada (em hectares) e taxa de supressão no período, permitindo comparar quais regiões de São Paulo perderam mais cobertura vegetal.



## 2. Estrutura dos Dados Exportados
Os dados são exportados em dois formatos complementares:
1. **Dados Vetoriais (SHP/GeoJSON):** Contêm a geometria das manchas de desmatamento, permitindo a visualização de "onde" ocorreu o evento no QGIS.
2. **Dados Tabulares (CSV/Tabela):** Estruturados para facilitar o cruzamento com dados socioeconômicos (ex: densidade demográfica, índice de desenvolvimento humano por distrito).

## 3. Normalização
Para garantir uma comparação justa entre distritos de tamanhos diferentes, aplicamos a normalização por área:
$$\text{Taxa de Perda (\%)} = \left( \frac{\text{Área Desmatada (ha)}}{\text{Área Total do Distrito (ha)}} \right) \times 100$$
Isso permite identificar "pontos quentes" (hotspots) de desmatamento, independentemente do tamanho territorial do distrito.