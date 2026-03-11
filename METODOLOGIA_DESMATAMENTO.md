# Metodologia de Monitoramento do Desmatamento

O monitoramento do desmatamento no *ForestEyes* é baseado na **Análise de Mudança Multi-temporal (Change Detection)** entre duas datas de referência (2018 e 2024).

## 1. Lógica Algorítmica
O desmatamento é definido pela transição de estado de um pixel entre os dois períodos. Tecnicamente, aplicamos uma operação lógica booleana sobre os resultados da classificação:

* **Estado 2018 ($S_1$):** Floresta (Valor = 1)
* **Estado 2024 ($S_2$):** Não-Floresta (Valor = 0)
* **Mudança Detectada ($D$):** $D = (S_1 == 1) \land (S_2 == 0)$



## 2. Quantificação de Área (Hectares)
Uma vez isolada a camada de desmatamento, realizamos a conversão de área:
1. **Contagem de Pixels:** Somamos todos os pixels classificados como "desmatamento" dentro da ROI (cidade de São Paulo).
2. **Cálculo de Área:** Multiplicamos o número de pixels pela resolução espacial do sensor (Landsat = 30m x 30m = 900m²).
3. **Conversão de Unidade:** Convertemos de metros quadrados para hectares ($1 \text{ hectare} = 10.000 \text{ m}^2$).
   $$\text{Área (ha)} = \frac{\sum (\text{pixels desmatados} \times 900)}{10.000}$$

## 3. Filtragem de Ruído (Pós-Processamento)
Para evitar "falsos positivos" (como variações sazonais de umidade que o modelo pode confundir com desmatamento), aplicamos dois filtros:
* **Filtro de Área Mínima:** Removemos manchas menores que 0.5 hectares para eliminar ruídos de sensor ou erros de registro posicional.
* **Sobreposição com Malha Urbana:** Cruzamos os dados com o mapa de expansão urbana para garantir que a mudança detectada não seja apenas uma alteração de reflectância em área já construída.

