import sys
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

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

    # Diretórios e Caminhos
    output_seg_dir = os.path.join(project_root, "data", "output", "segmentation", ano_fim)
    output_class_dir = os.path.join(project_root, "data", "output", "classification", ano_fim)
    output_report_dir = os.path.join(project_root, "reports")
    dir_forest = os.path.join(output_report_dir, "dataframes_csv", "forest")
    dir_non_forest = os.path.join(output_report_dir, "dataframes_csv", "non-forest")
    
    os.makedirs(output_class_dir, exist_ok=True)
    os.makedirs(output_report_dir, exist_ok=True)

    # Ficheiros de Entrada
    path_cbers = os.path.join(project_root, "data", "output", "pansharpening", ano_fim, f"{code_muni}_{ano_fim}_CBERS_TRUE_COLOR_2M.tif")
    path_ids_tif = os.path.join(output_seg_dir, f"{code_muni}_IDs_Superpixels_{ano_fim}.tif")
    path_csv_forest = os.path.join(dir_forest, f"df_forest_sort_{code_muni}_{ano_fim}.csv")
    path_csv_non_forest = os.path.join(dir_non_forest, f"df_non-forest_sort_{code_muni}_{ano_fim}.csv")

    if not os.path.exists(path_ids_tif) or not os.path.exists(path_csv_forest):
        print(f"❌ [ERRO] Ficheiros do Script 06 não encontrados. Certifique-se de executar o SCRIPT 06 primeiro.")
        sys.exit(1)

    print("=" * 80, flush=True)
    print(f"🌲 SCRIPT 07 - MACHINE LEARNING OBIA RÁPIDO (RF & SVM) - {ano_fim}", flush=True)
    print("=" * 80, flush=True)

    # 1. Carregar CSVs e Preparar Dados
    print("1. A carregar os atributos extraídos pelo Script 06...", flush=True)
    df_f = pd.read_csv(path_csv_forest)
    df_nf = pd.read_csv(path_csv_non_forest)
    
    # 1 = Floresta Real, 0 = Não Floresta Real
    df_f['Classe_Real'] = 1
    df_nf['Classe_Real'] = 0

    # Metodologia ForestEyes: Amostragem Fixa Balanceada (50 Perfeitos, 50 Úteis por Classe)
    def amostrar_segmentos(df, n_perf=50, n_util=50):
        df_perf = df[df['HoR'] == 1.0]
        df_util = df[(df['HoR'] >= 0.7) & (df['HoR'] < 1.0)]
        
        # O min() evita erros caso haja menos de 50 segmentos úteis em alguma classe
        amostra_perf = df_perf.sample(n=min(n_perf, len(df_perf)), random_state=42)
        amostra_util = df_util.sample(n=min(n_util, len(df_util)), random_state=42)
        
        return pd.concat([amostra_perf, amostra_util])

    df_train_f = amostrar_segmentos(df_f)
    df_train_nf = amostrar_segmentos(df_nf)
    
    # Conjunto de Treinamento e Base Total
    df_train = pd.concat([df_train_f, df_train_nf], ignore_index=True)
    df_all = pd.concat([df_f, df_nf], ignore_index=True)
    
    # Conjunto de Teste/Predição (Tudo que não foi sorteado para treino)
    train_ids = df_train['Segment_label'].values
    df_test = df_all[~df_all['Segment_label'].isin(train_ids)].copy()

    # Selecionar apenas features numéricas
    features_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
    features_cols = [c for c in features_cols if c not in ['Segment_label', 'Classe_Real', 'HoR', 'Size', 'Size_m2']]
    
    X_train = df_train[features_cols]
    y_train = df_train['Classe_Real']
    id_train = df_train['Segment_label']
    
    X_test = df_test[features_cols]
    y_test = df_test['Classe_Real']
    id_test = df_test['Segment_label']

    print(f" -> Conjunto de Treino: {len(id_train)} superpixels metodológicos (Balanceados)")
    print(f" -> Conjunto de Predição: {len(id_test):,} superpixels")

    # 2. Treino de Machine Learning
    print("\n2. A iniciar o Treinamento de Inteligência Artificial...", flush=True)
    
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f" ✔️ Acurácia Random Forest: {rf_acc * 100:.2f}%")

    svm = LinearSVC(random_state=42, dual=False)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_pred)
    print(f" ✔️ Acurácia SVM: {svm_acc * 100:.2f}%")

    # Mapeamento Rápido Numpy (Utilizando RF)
    max_id = df_all['Segment_label'].max()
    mapa_cores_ids = np.zeros(max_id + 1, dtype=np.uint8)
    
    # Classes para o Raster:
    # 0 = Fundo, 1 = Floresta (Verde), 2 = Não Floresta (Vermelho), 3 = Treino (Amarelo)
    mapa_cores_ids[id_test[rf_pred == 1].values] = 1  # Floresta
    mapa_cores_ids[id_test[rf_pred == 0].values] = 2  # Desmatamento
    mapa_cores_ids[id_train.values] = 3  # Treino

    # 3. Geração Espacial Otimizada (Raster)
    print("\n3. A gerar o Mapa Raster Espacial bloco a bloco...", flush=True)
    path_tif_class = os.path.join(output_class_dir, f"{code_muni}_Classificacao_OBIA_Raster_{ano_fim}.tif")
    temp_class_1band = f"/content/temp_class_1band_{code_muni}.tif"
    
    with rasterio.open(path_cbers) as src_cbers:
        cbers_meta = src_cbers.meta.copy()
        cbers_crs = src_cbers.crs
        h, w = src_cbers.shape

    meta_class = cbers_meta.copy()
    meta_class.update(dtype=rasterio.uint8, count=3, photometric='RGB', nodata=0)
    
    meta_1band = cbers_meta.copy()
    meta_1band.update(dtype=rasterio.uint8, count=1, nodata=0)

    tile_size = 1024
    with rasterio.open(path_ids_tif) as src_ids, \
         rasterio.open(path_tif_class, 'w', **meta_class) as dst_rgb, \
         rasterio.open(temp_class_1band, 'w', **meta_1band) as dst_1b:
        
        for y_idx in range(0, h, tile_size):
            for x_idx in range(0, w, tile_size):
                w_janela, h_janela = min(tile_size, w - x_idx), min(tile_size, h - y_idx)
                window = Window(x_idx, y_idx, w_janela, h_janela)
                
                bloco_ids = src_ids.read(1, window=window)
                # Filtra valores fora do limite para evitar erros no Numpy
                bloco_ids_validos = np.clip(bloco_ids, 0, max_id)
                bloco_classes = mapa_cores_ids[bloco_ids_validos]
                # Preserva o nodata original
                bloco_classes[bloco_ids == 0] = 0
                
                r = np.zeros_like(bloco_classes, dtype=np.uint8)
                g = np.zeros_like(bloco_classes, dtype=np.uint8)
                b = np.zeros_like(bloco_classes, dtype=np.uint8)
                
                r[bloco_classes == 1], g[bloco_classes == 1], b[bloco_classes == 1] = 0, 255, 0    # Verde
                r[bloco_classes == 2], g[bloco_classes == 2], b[bloco_classes == 2] = 255, 0, 0    # Vermelho
                r[bloco_classes == 3], g[bloco_classes == 3], b[bloco_classes == 3] = 255, 255, 0  # Amarelo
                
                dst_rgb.write(r, 1, window=window)
                dst_rgb.write(g, 2, window=window)
                dst_rgb.write(b, 3, window=window)
                dst_1b.write(bloco_classes, 1, window=window)

    # 4. Vetorização (Shapefile e QML)
    print("\n4. A vetorizar o resultado para o Shapefile QGIS...", flush=True)
    path_shp_final = os.path.join(output_class_dir, f"{code_muni}_Mapeamento_Final_OBIA_{ano_fim}.shp")
    
    with rasterio.open(temp_class_1band) as src_1b:
        image = src_1b.read(1)
        results = (
            {'properties': {'Classe_ID': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(image, mask=(image > 0), transform=src_1b.transform))
        )
        gdf_final = gpd.GeoDataFrame.from_features(list(results), crs=cbers_crs)

    # Mapeamento exato para o estilo QML (1=Floresta, 2=Não Floresta, 3=Treino)
    dict_classes = {1: 'Ainda Floresta', 2: 'Nao eh mais Floresta', 3: 'Treino/Referencia'}
    gdf_final['Classe'] = gdf_final['Classe_ID'].map(dict_classes)
    gdf_final = gdf_final.dissolve(by='Classe').reset_index()
    gdf_final.to_file(path_shp_final, index=False)

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
        
    # 5. Gerar Imagem TIF do Desmatamento (Isolada)
    print("\n5. A gerar a máscara TIF isolando as áreas de desmatamento...", flush=True)
    path_tif_desmatamento = os.path.join(output_class_dir, f"{code_muni}_Mascara_Desmatamento_CBERS_{ano_fim}.tif")
    
    with rasterio.open(path_cbers) as src_rgb, \
         rasterio.open(temp_class_1band) as src_1b, \
         rasterio.open(path_tif_desmatamento, 'w', **cbers_meta) as dst_desm:
         
        for y_idx in range(0, h, tile_size):
            for x_idx in range(0, w, tile_size):
                w_janela, h_janela = min(tile_size, w - x_idx), min(tile_size, h - y_idx)
                window = Window(x_idx, y_idx, w_janela, h_janela)
                
                rgb_raw = src_rgb.read(window=window)
                bloco_class = src_1b.read(1, window=window)
                
                # Zera tudo o que NÃO for desmatamento (classe 2)
                mask_apagar = (bloco_class != 2)
                for band_idx in range(rgb_raw.shape[0]):
                    rgb_raw[band_idx][mask_apagar] = 0
                    
                dst_desm.write(rgb_raw, window=window)

    # 6. Relatório Final
    print("\n6. A calcular áreas e gerar o Relatório Textual...", flush=True)
    
    # Cálculo em Hectares utilizando Size_m2 da base original para manter precisão geométrica da segmentação MaskSLIC
    df_test_final = df_all.loc[df_all['Segment_label'].isin(id_test)]
    area_total_ha = df_test_final['Size'].sum() * 4.0 / 10000.0
    area_mantida_ha = df_test_final.loc[rf_pred == 1, 'Size'].sum() * 4.0 / 10000.0
    area_nao_floresta_ha = df_test_final.loc[rf_pred == 0, 'Size'].sum() * 4.0 / 10000.0
    
    path_txt_relatorio = os.path.join(output_report_dir, f"{code_muni}_Relatorio_IA_Desmatamento_{ano_fim}.txt")
    with open(path_txt_relatorio, 'w', encoding='utf-8') as f:
        f.write("=========================================================\n")
        f.write(f" RELATÓRIO DE INTELIGÊNCIA ARTIFICIAL OBIA - {ano_fim}\n")
        f.write("=========================================================\n")
        f.write(f"Algoritmos de Aprendizado de Máquina (Amostragem Fixa Balanceada Metodológica)\n\n")
        f.write("1. PERFORMANCE DOS MODELOS (Validação contra o Conjunto de Predição):\n")
        f.write(f"  - Random Forest (Acurácia Global): {rf_acc * 100:.2f}%\n")
        f.write(f"  - Support Vector Machine (Acurácia Global): {svm_acc * 100:.2f}%\n\n")
        f.write("2. RESULTADOS DO DESMATAMENTO (Baseado no Conjunto de Predição):\n")
        f.write(f"  - Área Analisada (Predição): {area_total_ha:,.2f} hectares\n")
        f.write(f"  - Área que AINDA É FLORESTA: {area_mantida_ha:,.2f} hectares\n")
        f.write(f"  - ÁREA NÃO FLORESTA (Possível Desmatamento): {area_nao_floresta_ha:,.2f} hectares\n")
        f.write("=========================================================\n")

    os.remove(temp_class_1band)
    print(f"\n🎉 SCRIPT 07 CONCLUÍDO COM SUCESSO! Modelos treinados e mapas espaciais otimizados gerados.")

if __name__ == "__main__":
    main()