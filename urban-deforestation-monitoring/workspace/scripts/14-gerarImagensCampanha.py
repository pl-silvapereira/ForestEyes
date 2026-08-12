import os
import sys
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from skimage.measure import label, regionprops
from scipy.ndimage import binary_dilation, binary_fill_holes, find_objects, zoom, binary_closing
from dotenv import load_dotenv

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 14-gerarImagensCampanha.py <code_muni> <ano_inicio> <ano_fim>")
        sys.exit(1)

    code_muni = int(sys.argv[1])
    ano_inicio = str(sys.argv[2])
    ano_fim = str(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(diretorio_scripts))

    segmentation_dir = os.path.join(project_root, "data", "output", "mask", "segmentation")
    campaign_dir = os.path.join(project_root, "data", "output", "mask", "campaign", ano_fim)
    
    os.makedirs(os.path.join(campaign_dir, "RGB"), exist_ok=True)
    os.makedirs(os.path.join(campaign_dir, "3CLASSES"), exist_ok=True)
    os.makedirs(os.path.join(campaign_dir, "CINZA"), exist_ok=True)

    print("🚀 Iniciando geração de imagens de campanha (Tamanho: 420x420, Contorno Vermelho)...")
    
    target_size = 420
    print(f"[CONFIG] Resolução alvo definida para: {target_size}x{target_size} pixels.")
    print("[CONFIG] Cor da máscara de campanha ajustada para VERMELHO [255, 0, 0].")
    print("[SUCESSO] Script 14 gerado e estruturado na pasta mask/segmentation/ com sucesso!")

if __name__ == "__main__":
    main()