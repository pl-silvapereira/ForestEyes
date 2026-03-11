# Algoritmo de Classificação: Random Forest

Para a detecção de cobertura florestal em São Paulo, utilizamos o algoritmo **Random Forest** (Floresta Aleatória), um método de aprendizado de máquina supervisionado de alta performance, executado via Google Earth Engine.

## Fundamentos Técnicos
O Random Forest é um modelo de *Ensemble Learning* (Aprendizado em Conjunto) baseado na construção de múltiplas árvores de decisão durante o treinamento.

1. **Estrutura de Decisão:** Cada árvore é construída utilizando um subconjunto aleatório dos dados de treinamento e das bandas espectrais disponíveis (B2, B3, B4, B5 e NDVI).
2. **Votação Majoritária:** Na fase de inferência, cada pixel da imagem original é submetido a todas as árvores criadas. A classe final atribuída ao pixel (Floresta ou Não Floresta) é definida pela votação majoritária entre as árvores.



[Image of random forest classification decision tree diagram]


## Parametrização no ForestEyes
* **Número de Árvores:** Configuramos `100` árvores de decisão. Este valor provê um balanço ideal entre precisão e custo computacional.
* **Input Features:** O modelo é alimentado com um vetor de características espectrais por pixel:
    * **Bandas Landsat:** Blue (SR_B2), Green (SR_B3), Red (SR_B4), NIR (SR_B5).
    * **Índice Espectral:** NDVI (Normalized Difference Vegetation Index), essencial para separar a vegetação de superfícies impermeabilizadas.
* **Treinamento Supervisionado:** O modelo utiliza os pontos amostrais extraídos do *Ground Truth* (ESA WorldCover) para aprender os padrões que distinguem a reflectância da biomassa florestal urbana do restante da malha urbana.

## Vantagens da Escolha
* **Robustez a Outliers:** Por usar uma média de múltiplas árvores, o modelo é altamente resistente a ruídos atmosféricos residuais nas imagens Landsat.
* **Não linearidade:** Diferente de métodos lineares, o Random Forest consegue capturar a relação complexa entre o NDVI e a densidade da vegetação em ambientes densamente edificados como São Paulo.
* **Feature Importance:** O modelo pondera automaticamente quais bandas são mais relevantes para a classificação, permitindo uma separação eficaz entre floresta e zonas de sombra projetadas por edifícios altos.