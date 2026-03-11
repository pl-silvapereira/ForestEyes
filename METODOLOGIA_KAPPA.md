# Validação de Classificação: Índice Kappa

O Índice de concordância Kappa ($\kappa$) é a métrica estatística utilizada para avaliar a qualidade da classificação do modelo *Random Forest* no projeto *ForestEyes*. Diferente da acurácia simples, o Kappa ajusta o resultado considerando a probabilidade de concordância por acaso.

## O Conceito Técnico
O coeficiente é calculado a partir da Matriz de Confusão, comparando os valores previstos pelo modelo contra os dados de validação (*Ground Truth* - ESA WorldCover). A fórmula é:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

Onde:
* $p_o$ (Concordância Observada): A proporção de pixels corretamente classificados pelo modelo em relação à verdade de campo.
* $p_e$ (Concordância Esperada): A probabilidade de concordância que ocorreria puramente por acaso, baseada nas frequências marginais da matriz de erro.

## Interpretação dos Resultados
O valor de $\kappa$ varia de -1 a 1, sendo tipicamente interpretado conforme a escala de Landis e Koch:

| Valor de Kappa | Nível de Concordância |
| :--- | :--- |
| < 0.00 | Pobre |
| 0.01 – 0.20 | Leve |
| 0.21 – 0.40 | Razoável |
| 0.41 – 0.60 | Moderada |
| 0.61 – 0.80 | Substancial |
| 0.81 – 1.00 | Quase Perfeita |

## Aplicação no ForestEyes
No nosso pipeline, o cálculo é realizado via `errorMatrix('class', 'classification')` dentro do Google Earth Engine, utilizando 30% dos dados que não foram apresentados ao classificador durante o treino (*Hold-out validation*). 

> **Nota:** Um Kappa elevado no nosso projeto indica que o modelo foi capaz de distinguir floresta de áreas construídas com sucesso, independentemente da distribuição de classes na cidade de São Paulo.