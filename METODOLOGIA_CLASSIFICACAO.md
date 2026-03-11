# Metodologia de Classificação de Imagens

A classificação no *ForestEyes* transforma dados espectrais brutos (reflectância) em um mapa categórico de uso e cobertura da terra. Este processo segue uma abordagem de aprendizado supervisionado.

## 1. O Fluxo de Processamento
O pipeline processa a imagem através de três etapas principais:

1. **Extração de Features:** A partir das bandas do Landsat (SR_B2, B3, B4, B5), calculamos índices auxiliares, sendo o **NDVI (Normalized Difference Vegetation Index)** o principal, devido à sua alta correlação com a densidade foliar.
   $$NDVI = \frac{NIR - Red}{NIR + Red}$$
2. **Treinamento Supervisionado:** O classificador *Random Forest* é treinado com uma matriz contendo as bandas multiespectrais e as classes definidas pelo *Ground Truth* (ESA WorldCover).
3. **Classificação Pixel-a-Pixel:** O modelo aplica as regras aprendidas em toda a cena, atribuindo cada pixel a uma das classes: **Floresta** ou **Não-Floresta**.



## 2. Refinamento (Pós-Classificação)
Após a classificação bruta, aplicamos filtros espaciais para garantir a qualidade do dado:
* **Filtro de Ruído:** Utilizamos operações morfológicas para eliminar "pixels isolados" (ruído sal-e-pimenta) que frequentemente ocorrem em áreas urbanas devido à sombra de edifícios.
* **Máscara de Água:** Opcionalmente, áreas de corpos hídricos são mascaradas para evitar falsos positivos na detecção de vegetação.

## 3. Matriz de Confusão e Precisão
A eficácia do classificador é medida comparando o mapa gerado com o conjunto de teste (30% dos dados). O resultado é sintetizado na Matriz de Confusão:

| Previsto \ Real | Floresta | Não-Floresta |
| :--- | :--- | :--- |
| **Floresta** | Verdadeiro Positivo | Falso Positivo |
| **Não-Floresta** | Falso Negativo | Verdadeiro Negativo |

> **Resultado Técnico:** A acurácia final é consolidada pelo Índice Kappa, que normaliza a precisão pela probabilidade de acerto ao acaso.