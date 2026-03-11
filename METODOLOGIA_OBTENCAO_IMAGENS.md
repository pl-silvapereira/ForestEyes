# Metodologia de Obtenção e Processamento de Imagens (GEE)

O pipeline *ForestEyes* utiliza a infraestrutura de computação em nuvem do **Google Earth Engine (GEE)** para acessar e processar séries temporais de satélite.

## 1. Fonte de Dados (Landsat 8/9)
Utilizamos a coleção `LANDSAT/LC08/C02/T1_L2` e `LANDSAT/LC09/C02/T1_L2`. Estes dados são de **Nível 2**, o que significa que já passaram por:
* **Correção Atmosférica:** Remoção de efeitos de dispersão atmosférica e absorção.
* **Correção Radiométrica:** Conversão para refletância de superfície (*Surface Reflectance*).

## 2. Processamento em Nuvem
O script realiza a ingestão e limpeza através dos seguintes passos técnicos:

1. **Filtragem Espacial e Temporal:** Delimitamos a área de interesse (ROI) utilizando o banco de dados FAO/GAUL e filtramos as cenas por data (ex: janela de 2018 e 2024).
2. **Redução de Nuvens (Cloud Masking):** Utilizamos a banda de controle de qualidade (`QA_PIXEL`) para identificar e mascarar pixels contaminados por nuvens e sombras, garantindo que a análise considere apenas dados terrestres límpidos.
3. **Escalonamento (Scaling):** Os valores brutos do Landsat são convertidos para o intervalo real de reflectância (0 a 1) utilizando fatores de escala:
   $$Reflectância = (DigitalNumber \times 0.0000275) - 0.2$$



## 3. Composição e Filtragem
Para obter uma imagem "limpa" para cada ano, aplicamos uma função de **Mediana (Median Composite)** sobre a coleção filtrada. 
* **Por que Mediana?** A estatística de mediana é robusta contra valores extremos (ruídos residuais), garantindo uma representação espectral estável de cada pixel ao longo do ano, mesmo em áreas com alta nebulosidade.

## 4. Exportação
Após o processamento, as imagens são recortadas (`clip`) pela geometria da cidade de São Paulo e exportadas para o ambiente local em múltiplos formatos para garantir compatibilidade com ferramentas de SIG (raster para visualização e shapefiles para análise vetorial).

