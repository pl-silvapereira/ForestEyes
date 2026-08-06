import os
import sys
import rasterio
import numpy as np
import geobr
from dotenv import load_dotenv

def main():
    # Uso correto: python 08-gerarImagemDelta.py <code_muni> <ano_inicio> <ano_fim>
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 08-gerarImagemDelta.py <code_muni> <ano_inicio> <ano_fim>")
        print("Exemplo: python 08-gerarImagemDelta.py 3549904 2023 2024")
        sys.exit(1)

    code_muni = int(sys.argv[1])
    ano_inicio = str(sys.argv[2])
    ano_fim = str(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    # Diretório de saída para as máscaras delta
    mask_dir = os.path.join(project_root, "data", "output", "mask", f"{ano_inicio}_vs_{ano_fim}")
    os.makedirs(mask_dir, exist_ok=True)

    print(f"Buscando informações para o código de município: {code_muni}...")
    try:
        gdf_info = geobr.read_municipality(code_muni=code_muni, year=2022)
        nome_cidade = gdf_info['name_muni'].values[0]
        cidade_limpa = nome_cidade.lower().replace(" ", "_").replace("ã", "a").replace("ç", "c").replace("é", "e").replace("ó", "o")
    except Exception as e:
        cidade_limpa = "municipio"
        print(f"⚠️ Aviso ao buscar nome da cidade: {e}")

    # Caminhos dos rasters de classificação gerados pelas etapas anteriores
    path_class_inicio = os.path.join(project_root, "data", "output", "classification", ano_inicio, f"mapbiomas_lulc_10m_{cidade_limpa}_{ano_inicio}.tif")
    path_class_fim = os.path.join(project_root, "data", "output", "classification", ano_fim, f"mapbiomas_lulc_10m_{cidade_limpa}_{ano_fim}.tif")

    # Caminho do mosaico multiespectral/RGB do satélite correspondente ao ano fim (ou ano inicio)
    path_sat = os.path.join(project_root, "data", "output", "pansharpening", "multispectral-RGBN-bands", ano_fim, f"{code_muni}_multispectral_RGBN_{ano_fim}.tif")

    if not os.path.exists(path_class_inicio) or not os.path.exists(path_class_fim):
        print(f"[ERRO CRÍTICO] Rasters de classificação não encontrados para {ano_inicio} e/ou {ano_fim}.")
        print(f" -> [{ano_inicio}]: {path_class_inicio}")
        print(f" -> [{ano_fim}]: {path_class_fim}")
        sys.exit(1)

    print("=" * 115)
    print(f"🖼️ GERANDO IMAGENS DELTA (MÁSCARA DE MUDANÇAS)")
    print(f"📍 MUNICÍPIO: {code_muni} | PERÍODO: {ano_inicio} vs {ano_fim}")
    print("=" * 115)

    # 1. Processar a Imagem Delta da Classificação
    print("Processando raster de classificação delta...")
    with rasterio.open(path_class_inicio) as src1, rasterio.open(path_class_fim) as src2:
        meta = src1.meta.copy()
        arr1 = src1.read(1)
        arr2 = src2.read(1)

        # Onde houve mudança (arr1 != arr2), mantemos a classe do ano fim; onde não houve, colocamos 0 (preto)
        delta_class = np.where(arr1 != arr2, arr2, 0).astype(meta['dtype'])

        out_class_name = f"{code_muni}_delta_classification_{ano_inicio}_vs_{ano_fim}.tif"
        out_class_path = os.path.join(mask_dir, out_class_name)

        meta.update(dtype=rasterio.uint16, count=1, nodata=0)
        with rasterio.open(out_class_path, 'w', **meta) as dst:
            dst.write(delta_class, 1)

    print(f"✓ Imagem delta da classificação salva em:\n-> {out_class_path}")

    # 2. Processar a Imagem Delta sobre o Satélite (se o raster de satélite existir)
    if os.path.exists(path_sat):
        print("Processando raster de satélite com máscara delta aplicada...")
        with rasterio.open(path_sat) as src_sat:
            meta_sat = src_sat.meta.copy()
            sat_img = src_sat.read() # Lê todas as bandas (ex: RGBN)

            # Criar máscara booleana onde houve mudança (delta_class != 0)
            mascara_mudanca = (delta_class != 0)

            # Aplicar máscara preta (0) nas bandas onde não houve mudança
            delta_sat = np.zeros_like(sat_img)
            for i in range(sat_img.shape[0]):
                delta_sat[i] = np.where(mascara_mudanca, sat_img[i], 0)

            out_sat_name = f"{code_muni}_delta_satellite_{ano_inicio}_vs_{ano_fim}.tif"
            out_sat_path = os.path.join(mask_dir, out_sat_name)

            meta_sat.update(nodata=0)
            with rasterio.open(out_sat_path, 'w', **meta_sat) as dst:
                dst.write(delta_sat)

        print(f"✓ Imagem delta de satélite salva em:\n-> {out_sat_path}")
    else:
        print(f"⚠️ [AVISO] Raster de satélite não encontrado em '{path_sat}'. Apenas a imagem delta de classificação foi gerada.")

    print("\n[SUCESSO] Processo de geração de imagens delta concluído com sucesso!")

if __name__ == "__main__":
    main()