import sys
import os
import gc
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize, shapes
from skimage.segmentation import slic
from skimage.measure import regionprops_table
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from shapely.geometry import shape
from dotenv import load_dotenv

def encontrar_coluna_classe(gdf):
    """Procura a coluna que contém os nomes das classes no shapefile."""
    possiveis = ['class', 'classe', 'nome', 'name', 'legenda', 'class_name']
    for col in possiveis:
        if col in gdf.columns: return col
    # Se não achar, pega a primeira coluna de texto
    for col in gdf.columns:
        if gdf[col].dtype == 'O' and col != 'geometry': return col
    return gdf.columns[0]

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 07-preClassificacao.py <code_muni> <ano_inicio> <ano_fim>")
        sys.exit(1)

    code_muni = str(sys.argv[1])
    ano_inicio = str(sys.argv[2])
    ano_fim = str(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    # Diretórios
    output_seg_dir = os.path.join(project_root, "data", "output", "segmentation", ano_fim)
    output_class_dir = os.path.join(project_root, "data", "output", "classification", ano_fim)
    output_report_dir = os.path.join(project_root, "reports")
    os.makedirs(output_class_dir, exist_ok=True)
    os.makedirs(output_report_dir, exist_ok=True)

    # Entradas
    path_cbers = os.path.join(project_root, "data", "output", "pansharpening", ano_fim, f"{code_muni}_{ano_fim}_CBERS_TRUE_COLOR_2M.tif")
    path_shp_verdade = os.path.join(output_class_dir, f"{code_muni}_Classificado_ForestEyes_{ano_fim}.shp")

    if not os.path.exists(path_shp_verdade):
        print(f"❌ [ERRO] Shapefile da Verdade Terrestre não encontrado: {path_shp_verdade}")
        sys.exit(1)

    print("=" * 80, flush=True)
    print(f"🌲 SCRIPT 07 - MACHINE LEARNING OBIA (RF & SVM) - {ano_fim}", flush=True)
    print("=" * 80, flush=True)

    # 1. Preparar a Verdade Terrestre (Ground Truth)
    print("1. A ler e rasterizar a Verdade Terrestre (Shapefile Classificado)...", flush=True)
    with rasterio.open(path_cbers) as src:
        cbers_meta = src.meta.copy()
        cbers_crs = src.crs
        cbers_transform = src.transform
        cbers_shape = (src.height, src.width)
        pixel_area_ha = (abs(cbers_transform[0] * cbers_transform[4])) / 10000.0

    gdf_verdade = gpd.read_file(path_shp_verdade).to_crs(cbers_crs)
    col_classe = encontrar_coluna_classe(gdf_verdade)
    
    # Filtra o que é floresta (inclui 'floresta antropica', 'floresta', etc)
    is_floresta = gdf_verdade[col_classe].astype(str).str.lower().str.contains('florest')
    geom_floresta = gdf_verdade[is_floresta].geometry
    
    mask_verdade = rasterize([(geom, 1) for geom in geom_floresta], out_shape=cbers_shape, transform=cbers_transform, fill=0, dtype=np.uint8)

    # 2. SLIC Block-by-Block para Extrair Features
    print("\n2. A gerar Superpixels e Extrair Atributos da Imagem...", flush=True)
    temp_slic_ids = f"/content/temp_slic_ids_{code_muni}_{ano_fim}.tif"
    
    meta_ids = cbers_meta.copy()
    meta_ids.update(dtype=rasterio.uint32, count=1, nodata=0)

    tile_size = 1024
    h, w = cbers_shape
    total_blocos = int(np.ceil(h / tile_size)) * int(np.ceil(w / tile_size))
    
    df_list = []
    global_offset = 0
    bloco_atual = 0

    with rasterio.open(path_cbers) as src, rasterio.open(temp_slic_ids, 'w', **meta_ids) as dst_ids:
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                bloco_atual += 1
                w_janela, h_janela = min(tile_size, w - x), min(tile_size, h - y)
                window = Window(x, y, w_janela, h_janela)
                
                rgb_raw = src.read(window=window)[:3]
                
                # Para evitar SLIC no vazio
                if np.max(rgb_raw) > 0:
                    rgb_ch_last = np.moveaxis(rgb_raw, 0, -1).astype(np.float32)
                    max_val = rgb_ch_last.max()
                    if max_val > 0: rgb_ch_last /= max_val
                    
                    qts_tile = int((h_janela * w_janela) / 62.5)
                    segmentos_tile = slic(rgb_ch_last, n_segments=qts_tile, compactness=10, start_label=1, max_num_iter=5)
                    
                    validos = segmentos_tile > 0
                    segmentos_tile[validos] += global_offset
                    
                    # Extrair classe predominante do Ground Truth
                    mask_verdade_tile = mask_verdade[y:y+h_janela, x:x+w_janela]
                    
                    props = regionprops_table(segmentos_tile, intensity_image=np.moveaxis(rgb_raw, 0, -1), properties=('label', 'area', 'intensity_mean'))
                    props_gt = regionprops_table(segmentos_tile, intensity_image=mask_verdade_tile, properties=('label', 'intensity_mean'))
                    
                    df_tile = pd.DataFrame({
                        'ID_Segmento': props['label'],
                        'Area_px': props['area'],
                        'Mean_R': props['intensity_mean-0'],
                        'Mean_G': props['intensity_mean-1'],
                        'Mean_B': props['intensity_mean-2'],
                        'Prop_Floresta_Real': props_gt['intensity_mean']
                    })
                    df_list.append(df_tile)
                    
                    global_offset = df_tile['ID_Segmento'].max()
                    dst_ids.write(segmentos_tile.astype(np.uint32), 1, window=window)
                else:
                    dst_ids.write(np.zeros((h_janela, w_janela), dtype=np.uint32), 1, window=window)

                if bloco_atual % 20 == 0:
                    print(f"   -> [Bloco {bloco_atual}/{total_blocos}] mapeado e extraído.", flush=True)

    df_all = pd.concat(df_list, ignore_index=True)
    df_all['Classe_Real'] = np.where(df_all['Prop_Floresta_Real'] >= 0.5, 1, 0) # 1 = Floresta, 0 = Nao Floresta

    # 3. Machine Learning (Treino e Teste)
    print("\n3. A iniciar o Treinamento de Inteligência Artificial...", flush=True)
    X = df_all[['Mean_R', 'Mean_G', 'Mean_B']]
    y = df_all['Classe_Real']
    
    # Regra 1: 30% Treino, 70% Predição
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X, y, df_all['ID_Segmento'], test_size=0.70, random_state=42, stratify=y)
    
    print(f"   -> Conjunto de Treino (Amarelo/Não Analisado no mapa): {len(id_train):,} superpixels (30%)")
    print(f"   -> Conjunto de Predição (Verde/Vermelho no mapa): {len(id_test):,} superpixels (70%)")

    print("\n   [ Treinando Random Forest... ]")
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"   ✔️ Acurácia Random Forest: {rf_acc * 100:.2f}%")

    print("\n   [ Treinando Support Vector Machine (LinearSVC)... ]")
    svm = LinearSVC(random_state=42, dual=False) # Linear para suportar milhões de linhas em segundos
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_pred)
    print(f"   ✔️ Acurácia SVM: {svm_acc * 100:.2f}%")

    # Utilizaremos o Random Forest para construir o mapa por padrão (Costuma lidar melhor com florestas)
    df_test_results = pd.DataFrame({'ID_Segmento': id_test, 'Predicao': rf_pred})
    df_train_results = pd.DataFrame({'ID_Segmento': id_train, 'Predicao': 2}) # 2 = Treino (Amarelo)
    
    df_final = pd.concat([df_test_results, df_train_results])
    
    # Array de mapeamento ultrarrápido para a imagem
    # 0=Fundo, 1=Floresta (Verde), 2=Treino (Amarelo), 3=Nao Floresta (Vermelho)
    max_id = df_all['ID_Segmento'].max()
    mapa_cores_ids = np.zeros(max_id + 1, dtype=np.uint8)
    
    # Popula o array
    mapa_cores_ids[df_final[df_final['Predicao'] == 1]['ID_Segmento'].values] = 1 # Floresta (Verde)
    mapa_cores_ids[df_final[df_final['Predicao'] == 0]['ID_Segmento'].values] = 3 # Nao Floresta (Vermelho)
    mapa_cores_ids[df_final[df_final['Predicao'] == 2]['ID_Segmento'].values] = 2 # Treino (Amarelo)

    # 4. Geração do Raster TIF com as 3 Cores
    print("\n4. A gerar o Mapa Raster (Verde, Vermelho e Amarelo)...", flush=True)
    path_tif_class = os.path.join(output_class_dir, f"{code_muni}_Classificacao_OBIA_Raster_{ano_fim}.tif")
    
    meta_class = cbers_meta.copy()
    meta_class.update(dtype=rasterio.uint8, count=3, photometric='RGB')

    temp_class_1band = f"/content/temp_class_1band_{code_muni}.tif" # Útil para vetorizar depois
    meta_1band = cbers_meta.copy()
    meta_1band.update(dtype=rasterio.uint8, count=1, nodata=0)

    with rasterio.open(temp_slic_ids) as src_ids, \
         rasterio.open(path_tif_class, 'w', **meta_class) as dst_rgb, \
         rasterio.open(temp_class_1band, 'w', **meta_1band) as dst_1b:
        
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                w_janela, h_janela = min(tile_size, w - x), min(tile_size, h - y)
                window = Window(x, y, w_janela, h_janela)
                
                bloco_ids = src_ids.read(1, window=window)
                bloco_classes = mapa_cores_ids[bloco_ids]
                
                # Matrizes de cor (RGB)
                r = np.zeros_like(bloco_classes, dtype=np.uint8)
                g = np.zeros_like(bloco_classes, dtype=np.uint8)
                b = np.zeros_like(bloco_classes, dtype=np.uint8)
                
                # 1 = Verde (Floresta Predita)
                mask_verde = (bloco_classes == 1)
                r[mask_verde], g[mask_verde], b[mask_verde] = 0, 255, 0
                
                # 2 = Amarelo (Treino / Não Analisado)
                mask_amarelo = (bloco_classes == 2)
                r[mask_amarelo], g[mask_amarelo], b[mask_amarelo] = 255, 255, 0
                
                # 3 = Vermelho (Não Floresta Predita)
                mask_vermelho = (bloco_classes == 3)
                r[mask_vermelho], g[mask_vermelho], b[mask_vermelho] = 255, 0, 0
                
                dst_rgb.write(r, 1, window=window)
                dst_rgb.write(g, 2, window=window)
                dst_rgb.write(b, 3, window=window)
                
                # Salva a banda única para vetorizar (1=Floresta, 2=Treino, 3=Nao Floresta)
                dst_1b.write(bloco_classes, 1, window=window)

    print(f"✅ Raster de Classificação gerado: {path_tif_class}")

    # 5. Geração do Shapefile Contornado e Arquivo QML
    print("\n5. A vetorizar o resultado para o Shapefile QGIS...", flush=True)
    path_shp_final = os.path.join(output_class_dir, f"{code_muni}_Mapeamento_Final_OBIA_{ano_fim}.shp")
    
    shapes_list = []
    with rasterio.open(temp_class_1band) as src_1b:
        image = src_1b.read(1)
        results = (
            {'properties': {'Classe_ID': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(image, mask=(image > 0), transform=src_1b.transform))
        )
        shapes_list = list(results)

    gdf_final = gpd.GeoDataFrame.from_features(shapes_list, crs=cbers_crs)
    
    # Mapear os IDs para os nomes da classe para ficar amigável
    dict_classes = {1: 'Ainda Floresta', 2: 'Treino/Referencia', 3: 'Nao eh mais Floresta'}
    gdf_final['Classe'] = gdf_final['Classe_ID'].map(dict_classes)
    
    # Dissolver os polígonos para juntar áreas vizinhas e deixar o shapefile muito mais leve
    gdf_final = gdf_final.dissolve(by='Classe').reset_index()
    gdf_final.to_file(path_shp_final, index=False)
    
    # Criar o arquivo QML para estilizar o Shapefile automaticamente no QGIS (Contornos Amarelo/Vermelho)
    qml_content = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.0-Firenze" styleCategories="Symbology">
  <renderer-v2 type="categorizedSymbol" attr="Classe" symbollevels="0">
    <categories>
      <category symbol="0" value="Ainda Floresta" label="Ainda Floresta (Amarelo)"/>
      <category symbol="1" value="Nao eh mais Floresta" label="Nao eh mais Floresta (Vermelho)"/>
      <category symbol="2" value="Treino/Referencia" label="Area de Treino (Cinza tracejado)"/>
    </categories>
    <symbols>
      <symbol type="fill" name="0" clip_to_extent="1" force_rhr="0">
        <layer pass="0" class="SimpleFill" locked="0">
          <prop k="border_width_map_unit_scale" v="3x:0,0,0,0,0,0"/>
          <prop k="color" v="255,255,0,0"/> <!-- Fundo Transparente -->
          <prop k="outline_color" v="255,255,0,255"/> <!-- Borda Amarela -->
          <prop k="outline_style" v="solid"/>
          <prop k="outline_width" v="0.5"/>
        </layer>
      </symbol>
      <symbol type="fill" name="1" clip_to_extent="1" force_rhr="0">
        <layer pass="0" class="SimpleFill" locked="0">
          <prop k="border_width_map_unit_scale" v="3x:0,0,0,0,0,0"/>
          <prop k="color" v="255,0,0,0"/> <!-- Fundo Transparente -->
          <prop k="outline_color" v="255,0,0,255"/> <!-- Borda Vermelha -->
          <prop k="outline_style" v="solid"/>
          <prop k="outline_width" v="0.5"/>
        </layer>
      </symbol>
      <symbol type="fill" name="2" clip_to_extent="1" force_rhr="0">
        <layer pass="0" class="SimpleFill" locked="0">
          <prop k="border_width_map_unit_scale" v="3x:0,0,0,0,0,0"/>
          <prop k="color" v="150,150,150,0"/>
          <prop k="outline_color" v="150,150,150,255"/>
          <prop k="outline_style" v="dash"/>
          <prop k="outline_width" v="0.3"/>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
</qgis>"""
    
    path_qml = path_shp_final.replace(".shp", ".qml")
    with open(path_qml, "w") as f:
        f.write(qml_content)
        
    print(f"✅ Shapefile salvo com estilo de contornos automáticos em: {path_shp_final}")

    # 6. Relatório Final de Desmatamento
    print("\n6. A gerar Relatório de Desmatamento...", flush=True)
    df_test = df_all.loc[df_all['ID_Segmento'].isin(id_test)]
    
    area_total_ha = df_test['Area_px'].sum() * pixel_area_ha
    area_mantida_ha = df_test[df_test['Classe_Real'] == 1]['Area_px'].sum() * pixel_area_ha
    
    # Desmatamento real (aquilo que a IA e os dados dizem que não é mais floresta no 70%)
    area_nao_floresta_ha = df_test[df_test['Classe_Real'] == 0]['Area_px'].sum() * pixel_area_ha
    
    path_txt_relatorio = os.path.join(output_report_dir, f"{code_muni}_Relatorio_IA_Desmatamento_{ano_fim}.txt")
    with open(path_txt_relatorio, 'w', encoding='utf-8') as f:
        f.write("=========================================================\n")
        f.write(f" RELATÓRIO DE INTELIGÊNCIA ARTIFICIAL OBIA - {ano_fim}\n")
        f.write("=========================================================\n")
        f.write(f"Algoritmos de Aprendizado de Máquina (70% Teste / 30% Treino)\n\n")
        
        f.write("1. PERFORMANCE DOS MODELOS:\n")
        f.write(f" - Random Forest (Acurácia Global): {rf_acc * 100:.2f}%\n")
        f.write(f" - Support Vector Machine (Acurácia Global): {svm_acc * 100:.2f}%\n\n")
        
        f.write("2. RESULTADOS DO DESMATAMENTO (Baseado no Conjunto de Predição):\n")
        f.write(f" - Área Analisada (70%): {area_total_ha:.2f} hectares\n")
        f.write(f" - Área que AINDA É FLORESTA: {area_mantida_ha:.2f} hectares\n")
        f.write(f" - ÁREA NÃO FLORESTA (Possível Desmatamento): {area_nao_floresta_ha:.2f} hectares\n")
        f.write("=========================================================\n")

    print(f"✅ Relatório de IA finalizado em: {path_txt_relatorio}")

    # Limpeza final
    os.remove(temp_slic_ids)
    os.remove(temp_class_1band)
    
    print("\n🎉 O SCRIPT 07 FOI CONCLUÍDO COM SUCESSO! SEU MODELO DE IA ESTÁ TREINADO E OS MAPAS GERADOS.")

if __name__ == "__main__":
    main()