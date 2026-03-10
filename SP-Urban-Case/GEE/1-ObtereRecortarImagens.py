import ee
import geemap
import os

# 1. INICIALIZAÇÃO
PROJECT_ID = 'foresteyes-regioes-urbanas'
ee.Initialize(project=PROJECT_ID)

sp_boundary = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq('ADM2_NAME', 'Sao Paulo'))
roi = sp_boundary.geometry()

# 2. FUNÇÕES DE PROCESSAMENTO
def get_landsat_scaled(year):
    col_id = "LANDSAT/LC08/C02/T1_L2" if year < 2022 else "LANDSAT/LC09/C02/T1_L2"
    col = ee.ImageCollection(col_id).filterBounds(roi).filterDate(f'{year}-01-01', f'{year}-12-31').filter(ee.Filter.lt('CLOUD_COVER', 25))
    
    def apply_scale(image):
        optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        return image.addBands(optical, None, True)

    img = col.median().clip(roi)
    img = apply_scale(img)
    ndvi = img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    return img.addBands(ndvi)

# 3. TREINO E CLASSIFICAÇÃO
print("Processando imagens...")
img_2018 = get_landsat_scaled(2018)
img_2024 = get_landsat_scaled(2024)

esa = ee.ImageCollection("ESA/WorldCover/v200").first().clip(roi)
ground_truth = esa.eq(10).rename('class')

points = ground_truth.sample(region=roi, scale=30, numPixels=1000, seed=42, geometries=True).randomColumn()
train_pts = points.filter(ee.Filter.lt('random', 0.7))
test_pts = points.filter(ee.Filter.gte('random', 0.7))

bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'NDVI']

def run_classification(image):
    trained = ee.Classifier.smileRandomForest(100).train(
        features=image.select(bands).sampleRegions(collection=train_pts, properties=['class'], scale=30),
        classProperty='class', inputProperties=bands
    )
    return image.select(bands).classify(trained)

class_2018 = run_classification(img_2018)
class_2024 = run_classification(img_2024)

# 4. MÉTRICAS
matrix = class_2024.sampleRegions(collection=test_pts, properties=['class'], scale=30).errorMatrix('class', 'classification')
print(f"Índice Kappa (2024): {matrix.kappa().getInfo():.4f}")

def get_area_ha(image):
    return ee.Number(image.eq(1).multiply(ee.Image.pixelArea()).reduceRegion(reducer=ee.Reducer.sum(), geometry=roi, scale=30, maxPixels=1e9).values().get(0)).divide(10000)

a18, a24 = get_area_ha(class_2018).getInfo(), get_area_ha(class_2024).getInfo()
deforestation = class_2018.eq(1).And(class_2024.eq(0)).selfMask()
print(f"Floresta 2018: {a18:.2f} ha | Floresta 2024: {a24:.2f} ha | Perda: {a18-a24:.2f} ha")

# =================================================================
# 5. EXPORTAÇÕES (AJUSTADAS: JPG BRANCO E PNG TRANSPARENTE)
# =================================================================
print("Gerando imagens...")

# Função para JPG com fundo BRANCO
def save_jpg_white_bg(ee_img, viz, name):
    # Criamos um fundo branco puro
    background = ee.Image(1).visualize(palette=['#FFFFFF'])
    # Mesclamos o fundo com a imagem (que tem máscara)
    final_img = background.blend(ee_img.visualize(**viz))
    
    geemap.get_image_thumbnail(
        final_img, 
        f"{name}.jpg", 
        {'min':0, 'max':255}, 
        dimensions=1024, 
        region=roi, 
        format='jpg'
    )
    print(f"JPG fundo branco salvo: {name}.jpg")

# Função para PNG TRANSPARENTE (para classificações)
def save_png_transparent(ee_obj, viz, name):
    geemap.get_image_thumbnail(
        ee_obj.selfMask().visualize(**viz), 
        f"{name}.png", 
        {'min':0, 'max':255}, 
        dimensions=1024, 
        region=roi, 
        format='png'
    )
    print(f"PNG transparente salvo: {name}.png")

# --- EXECUTANDO AS EXPORTAÇÕES ---

# Satélite em JPG Fundo Branco
viz_sat = {'bands':['SR_B4','SR_B3','SR_B2'], 'min':0, 'max':0.3}
save_jpg_white_bg(img_2018, viz_sat, "satelite_2018_fundo_branco")
save_jpg_white_bg(img_2024, viz_sat, "satelite_2024_fundo_branco")

# Floresta e Desmatamento em PNG Transparente
save_png_transparent(class_2018.eq(1), {'palette':['#228B22']}, "floresta_2018_transparente")
save_png_transparent(class_2024.eq(1), {'palette':['#228B22']}, "floresta_2024_transparente")
save_png_transparent(deforestation, {'palette':['#FF0000']}, "desmatamento_transparente")


# 6. EXPORTAÇÕES (SHP)
def export_shp(ee_img, name):
    vec = ee_img.eq(1).selfMask().reduceToVectors(geometry=roi, scale=100, geometryType='polygon', maxPixels=1e9)
    geemap.ee_export_vector(vec, filename=f"{name}.shp")

export_shp(class_2018, "floresta_2018")
export_shp(class_2024, "floresta_2024")
export_shp(deforestation, "desmatamento_detectado")

# ... [Defina aqui: get_landsat_scaled, run_classification, etc] ...

# 2. DEFINIÇÃO DOS 5 PROCESSOS (Criação das camadas)
# 1. Segmentação
snic = ee.Algorithms.Image.Segmentation.SNIC(image=img_2024.select(['SR_B2','SR_B3','SR_B4']), size=32, compactness=0).select(['clusters'])

# 2. Classificação (0: Não Floresta, 1: Floresta)
class_final = class_2024.rename('class')

# 3. Ground Truth (ESA)
ground_truth = esa.eq(10).rename('class')

# 4. Mapeamento da Floresta (2024)
floresta_2024 = class_2024.eq(1).rename('floresta')

# 5. Monitoramento do Desmatamento
desmatamento = class_2018.eq(1).And(class_2024.eq(0)).selfMask().rename('desmatamento')

# 3. FUNÇÃO DE EXPORTAÇÃO PARA VETOR (.shp)
def export_as_shp(ee_img, name, label_prop='class'):
    print(f"Convertendo {name} para vetores...")
    # O reduceToVectors converte pixels em polígonos
    vectors = ee_img.reduceToVectors(
        geometry=roi, scale=30, geometryType='polygon', 
        labelProperty=label_prop, maxPixels=1e9
    )
    geemap.ee_export_vector(vectors, filename=f"{name}.shp")
    print(f"Sucesso: {name}.shp exportado.")

# 4. EXPORTANDO OS 5 ARQUIVOS
print("Iniciando exportação dos 5 processos...")

export_as_shp(snic, "1_segmentacao_superpixel")
export_as_shp(class_final, "2_classificacao_geral")
export_as_shp(ground_truth, "3_ground_truth_esa")
export_as_shp(floresta_2024, "4_distribuicao_floresta_urbana", label_prop='floresta')
export_as_shp(desmatamento, "5_monitoramento_desmatamento", label_prop='desmatamento')

print("--- TODOS OS 5 SHP GERADOS COM SUCESSO ---")

print("--- PROCESSO CONCLUÍDO ---")