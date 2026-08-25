import os
import numpy as np
import rasterio
from rasterio.windows import Window
from skimage.segmentation import find_boundaries
from scipy.ndimage import find_objects

# Exemplo de adaptação para colorir a imagem inteira com base nas classes:
# - Floresta -> Contorno / Tonalidade Vermelha [255, 0, 0]
# - Não-Floresta -> Contorno / Tonalidade Azul [0, 0, 255]

def gerar_visualizazione_global_colorida(path_sat, path_labels, path_saida, df_metricas):
    """
    Lê a imagem de satélite completa e o raster de labels, desenhando
    os contornos dos superpixels na cor correspondente à sua classe.
    """
    with rasterio.open(path_sat) as src_sat:
        meta = src_sat.meta.copy()
        sat_data = src_sat.read()
        height, width = src_sat.height, src_sat.width

    with rasterio.open(path_labels) as src_lab:
        labels = src_lab.read(1)

    # Dicionário de classes por ID de superpixel (obtido do seu DataFrame de métricas)
    # Exemplo: {sp_id: 'Floresta', ...}
    mapa_classes = dict(zip(df_metricas['ID_Segmento'], df_metricas['Classe_Majoritaria']))

    # Normalização de contraste da imagem inteira
    rgb_full = np.zeros((3, height, width), dtype=np.uint8)
    for b in range(min(3, sat_data.shape[0])):
        band = sat_data[b].astype(np.float32)
        p2, p98 = np.percentile(band[band > 0], (2, 98)) if np.any(band > 0) else (0, 1)
        rgb_full[b] = np.clip((band - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)

    # Transpor para formato (H, W, 3) para manipulação visual
    img_visual = np.moveaxis(rgb_full, 0, -1).copy()

    print("Desenhando superpixels coloridos na imagem inteira...")
    ids_unicos = np.unique(labels)
    ids_unicos = ids_unicos[ids_unicos > 0]
    slices = find_objects(labels)

    for sp_id in ids_unicos:
        slc = slices[sp_id - 1]
        if slc is None: continue
        
        # Isola o superpixel atual
        mask_sp = (labels[slc] == sp_id)
        if not np.any(mask_sp): continue

        # Identifica a classe majoritária do superpixel
        classe = mapa_classes.get(sp_id, 'Floresta')
        
        # Define a cor do contorno conforme solicitado:
        # Floresta = Vermelho [255, 0, 0] | Não-Floresta = Azul [0, 0, 255]
        cor_borda = [255, 0, 0] if classe == 'Floresta' else [0, 0, 255]

        # Extrai os limites da borda apenas para este superpixel
        borda_sp = find_boundaries(mask_sp, mode='outer')
        
        # Aplica a cor diretamente na fatia correspondente da imagem global
        sub_img = img_visual[slc[0], slc[1]]
        sub_img[borda_sp] = cor_borda

    # Salva o resultado final em TIF georreferenciado
    meta.update({"dtype": rasterio.uint8, "count": 3, "photometric": "RGB"})
    with rasterio.open(path_saida, "w", **meta) as dst:
        for b in range(3):
            dst.write(img_visual[..., b], b + 1)

    print(f"✅ Imagem global com superpixels coloridos salva em:\n-> {path_saida}")