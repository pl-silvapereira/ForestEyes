import os
import numpy as np
import rasterio
from dotenv import load_dotenv

load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')
dir_out = os.path.join(ROOT, 'data', 'Output')

path_rgb = os.path.join(dir_out, "02_SJC_Recortado_MapBiomas.tif")
path_sombra_out = os.path.join(dir_out, "05_SJC_Mascara_Sombras.tif")

def gerar_mascara_sombras():
    print("🌑 Iniciando detecção de sombras na imagem CBERS...")
    
    with rasterio.open(path_rgb) as src:
        # Lendo as 3 bandas (RGB)
        img = src.read([1, 2, 3]).astype(np.float32)
        
        # Calcula a intensidade média do pixel (quão claro ou escuro ele é)
        intensidade = np.mean(img, axis=0)
        
        # Limiar de sombra (Threshold). 
        # Valores muito baixos (ex: menores que o percentil 5) costumam ser sombra.
        # Ajuste esse '5' se achar que está pegando muita água ou pouca sombra.
        limiar = np.percentile(intensidade[intensidade > 0], 5) 
        
        # Cria a máscara: 1 é Sombra, 0 é o resto
        mascara_sombra = (intensidade < limiar) & (intensidade > 0)
        
        meta = src.meta.copy()
        meta.update(dtype=rasterio.uint8, count=1, nodata=0)
        
        with rasterio.open(path_sombra_out, 'w', **meta) as dst:
            dst.write(mascara_sombra.astype(rasterio.uint8), 1)
            
    print(f"✅ Máscara de sombras gerada! Limiar utilizado: {limiar:.2f}")

if __name__ == "__main__":
    gerar_mascara_sombras()