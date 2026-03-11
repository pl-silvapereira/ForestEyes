# Metodologia de Segmentação de Imagem (SNIC)

No projeto *ForestEyes*, a segmentação é o processo de agrupar pixels adjacentes com características espectrais semelhantes em "objetos" ou "superpixels". Isso reduz o ruído da classificação e permite uma análise espacial mais próxima da realidade visual do terreno.

## 1. O Algoritmo SNIC
Utilizamos o **SNIC (Simple Non-Iterative Clustering)**, que é a implementação padrão no Google Earth Engine para segmentação orientada a objetos (GEOBIA).

* **Funcionamento Técnico:** Diferente de algoritmos tradicionais que exigem várias iterações (como o SLIC), o SNIC é não-iterativo. Ele expande sementes de superpixels de forma paralela, garantindo eficiência computacional mesmo em áreas urbanas extensas como São Paulo.
* **Critério de Agrupamento:** O algoritmo minimiza a distância entre pixels com base em dois fatores:
    1. **Similaridade Espectral:** A diferença nos valores das bandas (ex: diferença entre o verde e o infravermelho).
    2. **Proximidade Espacial:** A distância física entre os pixels na malha da imagem.



## 2. Parâmetros Configurados
Para o *ForestEyes*, utilizamos a seguinte parametrização:
* **`size` (Tamanho):** Define o tamanho inicial da semente do superpixel. Um valor maior resulta em objetos mais abstratos e grandes; um valor menor captura detalhes mais finos da vegetação.
* **`compactness` (Compacidade):** Controla o equilíbrio entre a forma do superpixel (o quanto ele tende a ser um quadrado ou círculo) e a fidelidade aos limites espectrais da floresta.

## 3. Vantagens na Classificação Urbana
Ao realizar a segmentação antes da classificação (Random Forest):
1. **Redução de Ruído:** Remove o efeito "sal-e-pimenta", onde pixels isolados de floresta seriam classificados erroneamente dentro de áreas urbanas densas.
2. **Contextualização:** O modelo passa a classificar o "objeto" floresta como um todo, em vez de pixels isolados, o que é muito mais condizente com a estrutura de parques e fragmentos vegetais urbanos.
3. **Eficiência:** Reduz drasticamente o número de unidades a serem processadas pelo algoritmo de *Machine Learning*.