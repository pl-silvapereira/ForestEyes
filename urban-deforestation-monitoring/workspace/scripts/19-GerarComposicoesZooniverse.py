import os
import sys
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from dotenv import load_dotenv

def normalize_band(band_data):
    """Normaliza uma banda para 8-bits (0-255) aplicando corte de percentil (2% - 98%) para melhor contraste visual."""
    band_data = band_data.astype(np.float32)
    p2, p98 = np.percentile(band_data[band_data > 0], (2, 98)) if np.any(band_data > 0) else (0, 1)
    normalized = np.clip((band_data - p2) / (p98 - p2) * 255.0, 0, 255)
    return normalized.astype(np.uint8)

def main():
    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    # =========================================================================
    # CONFIGURAÇÕES E DIRETÓRIOS
    # =========================================================================
    # Ajuste os anos conforme a recomendação do Prof. Álvaro (MapBiomas 10m de 2022, CBERS 2024)
    ano_base = "2022" 
    ano_alvo = "2024"
    code_muni = "3549904"

    output_dir = os.path.join(project_root, "data", "output", "zooniverse", "composicoes")
    os.makedirs(output_dir, exist_ok=True)

    # Caminhos (ajuste os nomes dos arquivos conforme sua estrutura real)
    path_mapbiomas = os.path.join(project_root, "data", "input", "MapBiomas", ano_base, f"mapbiomas_lulc_10m_sao_jose_dos_campos_{ano_base}.tif")
    path_cbers = os.path.join(project_root, "data", "output", "pansharpening", "geopolitic-RGBN", ano_alvo, f"{code_muni}_{ano_alvo}_CBERS_TRUE_COLOR_CLIPPED.tif")

    if not os.path.exists(path_mapbiomas) or not os.path.exists(path_cbers):
        print(f"[ERRO CRÍTICO] Arquivos de entrada não encontrados.")
        print(f"MapBiomas: {path_mapbiomas}")
        print(f"CBERS: {path_cbers}")
        sys.exit(1)

    print("=" * 115)
    print("🌍 GERANDO COMPOSIÇÕES VISUAIS PARA ZOONIVERSE (DETECÇÃO DE MUDANÇAS)")
    print("=" * 115)

    # =========================================================================
    # PASSO 1: MÁSCARA MAPBIOMAS 10M (ANO BASE)
    # =========================================================================
    print(f"Processando raster do MapBiomas 10m ({ano_base}) para isolar classe de Floresta...")
    with rasterio.open(path_mapbiomas) as src_mb:
        mb_meta = src_mb.meta.copy()
        mb_data = src_mb.read(1)
        mb_transform = src_mb.transform
        mb_crs = src_mb.crs

    # Categorias Florestais do MapBiomas (3=Formação Florestal, 4=Formação Savânica, etc.)
    mask_floresta_mb = np.isin(mb_data, [3, 4, 5, 6])

    # =========================================================================
    # PASSO 2: LEITURA DO CBERS-4A E REPROJEÇÃO DA MÁSCARA
    # =========================================================================
    print(f"Lendo CBERS-4A ({ano_alvo}) e reprojetando máscara...")
    with rasterio.open(path_cbers) as src_cbers:
        cbers_meta = src_cbers.meta.copy()
        cbers_img = src_cbers.read()
        cbers_transform = src_cbers.transform
        cbers_crs = src_cbers.crs
        height, width = src_cbers.height, src_cbers.width

    # ÍNDICES DAS BANDAS (Ajuste se o seu TIF estiver ordenado de forma diferente)
    # Assumindo ordem padrão: 0=Blue, 1=Green, 2=Red, 3=NIR
    B_BLUE, B_GREEN, B_RED, B_NIR = 0, 1, 2, 3

    # Reprojetar máscara do MapBiomas (10m) para o grid do CBERS (2m)
    mask_reprojected = np.zeros((height, width), dtype=np.uint8)
    reproject(
        source=mask_floresta_mb.astype(np.uint8),
        destination=mask_reprojected,
        src_transform=mb_transform,
        src_crs=mb_crs,
        dst_transform=cbers_transform,
        dst_crs=cbers_crs,
        resampling=Resampling.nearest
    )
    is_forest_base = (mask_reprojected == 1)

    # Normalizar as 4 bandas individualmente para 8-bits
    print("Normalizando bandas do CBERS-4A (Realce de Contraste)...")
    blue_norm = normalize_band(cbers_img[B_BLUE])
    green_norm = normalize_band(cbers_img[B_GREEN])
    red_norm = normalize_band(cbers_img[B_RED])
    nir_norm = normalize_band(cbers_img[B_NIR])

    # =========================================================================
    # PASSO 3: GERAR AS 4 COMPOSIÇÕES RECOMENDADAS PELO PROF. ÁLVARO
    # =========================================================================
    composicoes = {
        "1_RGB_Natural": [red_norm, green_norm, blue_norm],              # Cor Natural
        "2_NIR_R_G_FalsaCor": [nir_norm, red_norm, green_norm],          # Falsa Cor Principal (⭐⭐⭐⭐⭐)
        "3_NIR_G_B_FalsaCorAlt": [nir_norm, green_norm, blue_norm]       # Falsa Cor Alternativa
    }

    cbers_meta.update({"count": 4, "dtype": rasterio.uint8, "nodata": 0, "photometric": "RGB"})

    # Salvar as 3 composições RGB
    for nome, bandas in composicoes.items():
        print(f"Gerando composição: {nome}...")
        
        # Aplicar a máscara: mantém a imagem onde era floresta, fundo preto (ou transparente) no resto
        out_bands = np.zeros((4, height, width), dtype=np.uint8)
        out_bands[0] = np.where(is_forest_base, bandas[0], 0) # Canal R do PNG
        out_bands[1] = np.where(is_forest_base, bandas[1], 0) # Canal G do PNG
        out_bands[2] = np.where(is_forest_base, bandas[2], 0) # Canal B do PNG
        out_bands[3] = np.where(is_forest_base, 255, 0)       # Canal Alpha (Transparência)

        out_path = os.path.join(output_dir, f"{code_muni}_{nome}_{ano_alvo}.tif")
        with rasterio.open(out_path, "w", **cbers_meta) as dst:
            dst.write(out_bands)
    
    # Gerar NDVI (Imagem 4 separada)
    print("Gerando composição: 4_NDVI...")
    # Converter para float para cálculo do NDVI
    red_f = cbers_img[B_RED].astype(np.float32)
    nir_f = cbers_img[B_NIR].astype(np.float32)
    
    # Evitar divisão por zero
    denominator = (nir_f + red_f)
    ndvi = np.zeros_like(red_f)
    valid_mask = denominator != 0
    ndvi[valid_mask] = (nir_f[valid_mask] - red_f[valid_mask]) / denominator[valid_mask]
    
    # Escalar NDVI de [-1, 1] para [0, 255]
    ndvi_scaled = np.clip((ndvi + 1) / 2 * 255, 0, 255).astype(np.uint8)
    
    # Preparar saída NDVI (1 banda + Alpha)
    ndvi_meta = cbers_meta.copy()
    ndvi_meta.update({"count": 2, "photometric": "MINISBLACK"})
    
    out_ndvi_bands = np.zeros((2, height, width), dtype=np.uint8)
    out_ndvi_bands[0] = np.where(is_forest_base, ndvi_scaled, 0) # Banda NDVI
    out_ndvi_bands[1] = np.where(is_forest_base, 255, 0)         # Alpha

    out_ndvi_path = os.path.join(output_dir, f"{code_muni}_4_NDVI_{ano_alvo}.tif")
    with rasterio.open(out_ndvi_path, "w", **ndvi_meta) as dst:
        dst.write(out_ndvi_bands)

    print(f"\n[SUCESSO] Todas as 4 composições foram geradas e salvas em:\n-> {output_dir}")

if __name__ == "__main__":
    main()