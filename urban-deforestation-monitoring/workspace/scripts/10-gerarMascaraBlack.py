import os
import sys
import rasterio
from rasterio.enums import ColorInterp
import numpy as np
from dotenv import load_dotenv

def main():
    # Uso correto: python 10-gerarMascaraBlack.py <code_muni> <ano_inicio> <ano_fim>
    # Exemplo: python 10-gerarMascaraBlack.py 3549904 2023 2024
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 10-gerarMascaraBlack.py <code_muni> <ano_inicio> <ano_fim>")
        print("Exemplo: python 10-gerarMascaraBlack.py 3549904 2023 2024")
        sys.exit(1)

    code_muni = int(sys.argv[1])
    ano_inicio = str(sys.argv[2])
    ano_fim = str(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    # Subpasta solicitada: data/output/mask/black/
    mask_black_dir = os.path.join(project_root, "data", "output", "mask", "black")
    os.makedirs(mask_black_dir, exist_ok=True)

    # Caminho da imagem de satélite mascarada gerada pelo Script 09
    path_rgbn_mask = os.path.join(
        project_root, "data", "output", "mask", "rgbn", 
        f"{code_muni}_masked_rgbn_delta_{ano_inicio}_vs_{ano_fim}.tif"
    )

    if not os.path.exists(path_rgbn_mask):
        print(f"[ERRO CRÍTICO] Imagem RGBN do Script 09 não encontrada em:\n-> {path_rgbn_mask}")
        print("Execute o Script 09 primeiro para gerar a máscara RGBN base.")
        sys.exit(1)

    print("=" * 115)
    print(f"⬛ GERANDO MÁSCARA COM FUNDO PRETO E PONTOS TRANSPARENTES (ANO: {ano_fim})")
    print(f"📍 MUNICÍPIO: {code_muni} | PERÍODO: {ano_inicio} vs {ano_fim}")
    print("=" * 115)

    print("Lendo raster base do Script 09...")
    with rasterio.open(path_rgbn_mask) as src:
        meta = src.meta.copy()
        bands = src.read() # Lê todas as bandas disponíveis
        height, width = src.height, src.width

    # Garantir que temos pelo menos 3 bandas (RGB)
    if bands.shape[0] < 3:
        print("[ERRO CRÍTICO] A imagem de entrada não possui bandas RGB suficientes.")
        sys.exit(1)

    r = bands[0]
    g = bands[1]
    b = bands[2]

    # Identificar onde há dados válidos (pontos de diferença do Script 08, que não são pretos/0)
    is_valid_point = (r > 0) | (g > 0) | (b > 0)

    print("Construindo canais com fundo preto opaco e pontos transparentes...")
    
    # Canais RGB de saída:
    # - Onde é ponto válido, mantém a cor original do satélite.
    # - Onde é fundo, define cor preta (0).
    out_r = np.where(is_valid_point, r, 0)
    out_g = np.where(is_valid_point, g, 0)
    out_b = np.where(is_valid_point, b, 0)

    # Canal Alfa (Transparência):
    # - 0 nos locais que contêm os pontos (transparente)
    # - 255 no fundo preto (opaco)
    max_val_alpha = 255 if meta['dtype'] in ['uint8', 'int8'] else 65535
    out_alpha = np.where(is_valid_point, 0, max_val_alpha).astype(meta['dtype'])

    # Empacotar as 4 bandas (R, G, B, Alpha)
    out_data = np.array([out_r, out_g, out_b, out_alpha], dtype=meta['dtype'])

    # Atualizar metadados para 4 bandas com suporte a Alfa
    meta.update({
        'count': 4,
        'nodata': None
    })

    out_tif_name = f"{code_muni}_masked_black_transparent_{ano_inicio}_vs_{ano_fim}.tif"
    out_tif_path = os.path.join(mask_black_dir, out_tif_name)

    print("Salvando raster resultante na pasta mask/black...")
    with rasterio.open(out_tif_path, 'w', **meta) as dst:
        dst.write(out_data)
        # Definir a interpretação de cores das 4 bandas de uma só vez (tupla)
        dst.colorinterp = (
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha
        )

    print(f"\n[SUCESSO] Máscara com fundo preto e pontos transparentes gerada em:\n-> {out_tif_path}")

if __name__ == "__main__":
    main()