# Metodologia de Recorte Geopolítico (ROI)

O processo de delimitação espacial, conhecido tecnicamente como **ROI (Region of Interest)**, é a etapa que garante que o processamento do *ForestEyes* ocorra exclusivamente dentro dos limites administrativos de São Paulo.

## 1. Fonte dos Limites Administrativos
Utilizamos o dataset **FAO/GAUL (Global Administrative Unit Layers)**, acessível via catálogo do GEE. Este banco de dados é um padrão internacional para a definição de limites de divisões administrativas (níveis 0, 1 e 2).
* **Nível Utilizado:** ADM2 (Municípios).
* **Filtro:** `filter(ee.Filter.eq('ADM2_NAME', 'Sao Paulo'))`.

## 2. Técnica de Recorte (Clipping)
O corte dos dados de satélite (Raster) é realizado através da função `.clip(roi)`. Tecnicamente, este processo envolve:
1. **Definição da Geometria:** A feição vetorial extraída do GAUL é convertida em um objeto `ee.Geometry` ou `ee.Feature`.
2. **Máscara de Recorte:** O GEE aplica uma operação de intersecção espacial, onde todos os pixels que estão fora da borda (fronteira do município) recebem um valor de `no-data` ou são mascarados.
3. **Alinhamento:** A grade de pixels (Landsat/ESA) é mantida inalterada, garantindo que o corte siga exatamente a fronteira vetorial sem distorcer as propriedades espectrais dos pixels de borda.



## 3. Importância da Precisão
Esta etapa é crucial para o cálculo de métricas de impacto (como hectares de floresta desmatada), pois:
* Evita a inclusão de áreas de municípios vizinhos na contabilidade do projeto.
* Reduz o volume de dados processados pelo *Random Forest*, otimizando o uso de memória (RAM) no servidor do GEE.
* Garante a consistência espacial entre as camadas de 2018 e 2024, permitindo que a análise de mudanças seja comparável ano a ano.