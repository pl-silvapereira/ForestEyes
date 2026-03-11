# Metodologia de Ground Truth (Verdade de Campo)

Para treinar o classificador *Random Forest* do projeto *ForestEyes*, utilizamos o dataset **ESA WorldCover v200** como referência de Verdade de Campo (*Ground Truth*).

## Por que o ESA WorldCover?
O ESA WorldCover é um produto global de cobertura da terra com resolução de **10 metros**, gerado a partir de dados multiespectrais do Sentinel-2 e dados de radar do Sentinel-1. A escolha técnica se baseia em três pilares:
1. **Alta Resolução Espacial:** Supera a resolução do Landsat (30m), permitindo um treinamento mais preciso das bordas de vegetação urbana.
2. **Consistência Global:** Dados validados cientificamente que reduzem o viés de interpretação humana na criação de polígonos de treino.
3. **Multimodalidade:** Combina dados ópticos e de radar, o que torna a identificação de áreas florestadas muito mais robusta em ambientes urbanos complexos.

## Extração e Pré-processamento
O processo de transformação do dado bruto em *Ground Truth* para o modelo ocorre da seguinte forma:

1. **Filtragem de Classe:** O script isola especificamente a classe `10` (que corresponde a "Tree Cover" ou vegetação arbórea densa) do dataset original.
2. **Máscara Binária:** Criamos uma variável booleana onde `1` representa a presença de floresta e `0` representa áreas não florestadas (urbano, solo exposto, água, etc.).
3. **Amostragem Aleatória Estratificada:** - O algoritmo percorre a área da cidade de São Paulo e extrai pontos amostrais de forma distribuída.
   - Isso evita o "vício de amostragem", onde o modelo aprenderia apenas sobre áreas florestadas grandes e ignoraria os fragmentos menores típicos de florestas urbanas.



## Validação (Hold-out Method)
Para garantir que o modelo não esteja apenas "decorando" os dados (overfitting), aplicamos a divisão de dados:
- **Treino (70%):** Pontos usados pelo *Random Forest* para ajustar os pesos de cada banda espectral (B2, B3, B4, B5 e NDVI).
- **Validação (30%):** Pontos nunca vistos pelo modelo, utilizados exclusivamente para calcular a Matriz de Erro e o Índice Kappa.

> **Conclusão:** Esta abordagem garante que o *ForestEyes* possua um embasamento técnico sólido, utilizando dados de referência validados por satélites de observação da Terra de última geração.