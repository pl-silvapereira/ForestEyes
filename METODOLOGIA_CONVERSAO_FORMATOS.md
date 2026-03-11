# Metodologia de Exportação e Conversão de Formatos

Para garantir que os resultados do *ForestEyes* sejam acessíveis tanto para sistemas de geoprocessamento quanto para apresentações visuais, realizamos a conversão dos dados de vetores (`.shp`) para imagens (`.png` e `.jpg`).

## 1. Do Shapefile (.shp) para Imagem (.png)
A conversão visa transformar a geometria vetorial (polígonos) em uma camada visual raster com fundo transparente.
* **Processamento:** Utilizamos a função `geemap.export_image` ou a exportação nativa via `Map.toImage()` no GEE.
* **Transparência:** Aplicamos uma máscara alfa (*alpha channel*) onde as áreas fora do município de São Paulo são definidas como `0` (transparente), permitindo a sobreposição dessas imagens em mapas base ou relatórios.
* **Resolução:** A exportação é feita mantendo a escala nativa do Landsat (30m/pixel), garantindo que a fidelidade geométrica seja preservada visualmente.

## 2. Do Raster para Imagem com Fundo Branco (.jpg)
Para as imagens de satélite (compostas por bandas RGB), a exportação é realizada com preenchimento de fundo branco (RGB 255, 255, 255).
* **Finalidade:** Facilitar a interpretação visual e o uso em documentos impressos, onde o fundo branco melhora o contraste com os elementos da imagem.