import sys
import os
import gc
import shutil
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from skimage.segmentation import slic, find_boundaries
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from dotenv import load_dotenv

def main():
    if len(sys.argv) < 4:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 06-relatorioSegmentacao.py <code_muni> <ano_inicio> <ano_fim>")
        sys.exit(1)

    code_muni = str(sys.argv[1])
    ano_inicio = str(sys.argv[2])
    ano_fim = str(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(diretorio_scripts)

    output_seg_dir = os.path.join(project_root, "data", "output", "segmentation", ano_fim)
    output_report_dir = os.path.join(project_root, "reports")
    os.makedirs(output_report_dir, exist_ok=True)

    path_shp_binario = os.path.join(output_seg_dir, f"{code_muni}_Segmentar_vs_NaoSegmentar_{ano_fim}.shp")
    path_cbers = os.path.join(project_root, "data", "output", "pansharpening", ano_fim, f"{code_muni}_{ano_fim}_CBERS_TRUE_COLOR_2M.tif")

    if not os.path.exists(path_shp_binario):
        print(f"[ERRO CRÍTICO] Shapefile não encontrado.", flush=True)
        sys.exit(1)
    if not os.path.exists(path_cbers):
        print(f"[ERRO CRÍTICO] Imagem CBERS não encontrada.", flush=True)
        sys.exit(1)

    print("=" * 80, flush=True)
    print(f"📄 SCRIPT 06 - RELATÓRIO DE SEGMENTAÇÃO E MASK-SLIC ({ano_fim})", flush=True)
    print("=" * 80, flush=True)

    # 1. Obter metadados SEM carregar a imagem na RAM
    print("1. A ler metadados e calcular contraste global de forma otimizada...", flush=True)
    with rasterio.open(path_cbers) as src:
        cbers_meta = src.meta.copy()
        cbers_crs = src.crs
        cbers_transform = src.transform
        cbers_shape = (src.height, src.width)
        
        pixel_res_x = abs(cbers_transform[0])
        pixel_res_y = abs(cbers_transform[4])
        pixel_area_ha = (pixel_res_x * pixel_res_y) / 10000.0

        escala = 10
        small_shape = (src.count, src.height // escala, src.width // escala)
        img_miniatura = src.read(out_shape=small_shape)
        
        p2 = np.percentile(img_miniatura, 2, axis=(1, 2))
        p98 = np.percentile(img_miniatura, 98, axis=(1, 2))
        p98 = np.maximum(p98, p2 + 1)
        
        del img_miniatura
        gc.collect()

    print("2. A rasterizar áreas do Shapefile...", flush=True)
    gdf_binario = gpd.read_file(path_shp_binario).to_crs(cbers_crs)
    
    geom_seg = gdf_binario[gdf_binario['class_name'] == 'Segmentar'].geometry
    geom_nao_seg = gdf_binario[gdf_binario['class_name'] == 'Nao_Segmentar'].geometry

    mask_seg = rasterize([(geom, 1) for geom in geom_seg], out_shape=cbers_shape, transform=cbers_transform, fill=0, dtype=np.uint8)
    mask_nao_seg = rasterize([(geom, 1) for geom in geom_nao_seg], out_shape=cbers_shape, transform=cbers_transform, fill=0, dtype=np.uint8)

    qtp = int(np.sum(mask_seg == 1))
    qps = 62.5
    qts = int(qtp / qps) if qps > 0 else 0
    qtp_ha = qtp * pixel_area_ha

    nao_seg_pixels = int(np.sum(mask_nao_seg == 1))
    nao_seg_ha = nao_seg_pixels * pixel_area_ha

    if qts <= 0:
        print("❌ ERRO: A classe 'Segmentar' tem 0 pixéis.", flush=True)
        sys.exit(1)

    relatorio_txt_path = os.path.join(output_report_dir, f"{code_muni}_Relatorio_MaskSLIC_{ano_fim}.txt")
    with open(relatorio_txt_path, 'w', encoding='utf-8') as f:
        f.write("=========================================================\n")
        f.write(f" RELATÓRIO TÉCNICO DE SEGMENTAÇÃO (MASK-SLIC) - {ano_fim}\n")
        f.write("=========================================================\n")
        f.write(f"Município / Código: {code_muni}\n\n")
        f.write("1. PARÂMETROS BASE:\n")
        f.write(f" - Resolução do Pixel: {pixel_res_x:.1f}m x {pixel_res_y:.1f}m\n")
        f.write(f" - QPS (Qtd. de Pixels por Segmento): {qps:.1f} px\n\n")
        f.write("2. CLASSE: SEGMENTAR (Área de Interesse)\n")
        f.write(f" - QTP (Qtd. Total de Pixels): {qtp:,}\n")
        f.write(f" - Área Total: {qtp_ha:,.4f} hectares\n")
        f.write(f" - QTS (Qtd. Total de Segmentos/Superpixels): {qts:,}\n\n")
        f.write("3. CLASSE: NÃO SEGMENTAR (Restante do Município)\n")
        f.write(f" - Quantidade Total de Pixels: {nao_seg_pixels:,}\n")
        f.write(f" - Área Total: {nao_seg_ha:,.4f} hectares\n")
        f.write("=========================================================\n")
    print(f"✅ Relatório TXT atualizado: {relatorio_txt_path}", flush=True)

    print(f"\n3. Iniciando MaskSLIC c/ Proteção Anti-Travamento (Engrossamento Ativado)...", flush=True)
    
    # TRUQUE DO DISCO LOCAL: Grava no SSD nativo do Colab para evitar timeout de rede no Google Drive
    temp_tif_local = f"/content/temp_slic_{code_muni}_{ano_fim}.tif"
    path_tif_slic = os.path.join(output_seg_dir, f"{code_muni}_MaskSLIC_ContornosAmarelos_{ano_fim}.tif")
    
    cbers_meta.update(dtype=rasterio.uint8, count=3, nodata=None)

    tile_size = 1024 # Reduzido para ser ainda mais rápido
    h, w = cbers_shape
    blocos_y = int(np.ceil(h / tile_size))
    blocos_x = int(np.ceil(w / tile_size))
    total_blocos = blocos_y * blocos_x
    
    print(f"   🗺️  A imagem tem {total_blocos} blocos. A gravar no SSD local para velocidade máxima...", flush=True)

    bloco_atual = 0

    with rasterio.open(path_cbers) as src, rasterio.open(temp_tif_local, 'w', **cbers_meta) as dst:
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                bloco_atual += 1
                
                h_janela = min(tile_size, h - y)
                w_janela = min(tile_size, w - x)
                window = Window(x, y, w_janela, h_janela)
                
                mask_tile = mask_seg[y:y+h_janela, x:x+w_janela]
                qtp_tile = np.sum(mask_tile)
                
                # Lê rgb
                rgb_raw = src.read(window=window).astype(np.float32)
                
                # Aplica contraste rápido
                r = np.clip((rgb_raw[0] - p2[0]) / (p98[0] - p2[0]) * 255, 0, 255).astype(np.uint8)
                g = np.clip((rgb_raw[1] - p2[1]) / (p98[1] - p2[1]) * 255, 0, 255).astype(np.uint8)
                b = np.clip((rgb_raw[2] - p2[2]) / (p98[2] - p2[2]) * 255, 0, 255).astype(np.uint8)
                rgb_tile = np.dstack([r, g, b])

                if qtp_tile > 0:
                    qts_tile = int(qtp_tile / qps)
                    if qts_tile > 0:
                        segmentos_tile = slic(rgb_tile, n_segments=qts_tile, compactness=10, mask=(mask_tile == 1), start_label=1, max_num_iter=5)
                        contornos_tile = find_boundaries(segmentos_tile, mode='thick')
                        
                        # Engrossa a Borda (Efeito "Marca-Texto")
                        contornos_tile = binary_dilation(contornos_tile, iterations=1) 
                        
                        # Pinta o contorno grosso de Amarelo
                        rgb_tile[contornos_tile] = [255, 255, 0]
                        print(f"   ✅ [Bloco {bloco_atual}/{total_blocos}] -> SLIC desenhado com bordas grossas.", flush=True)
                else:
                    # Dá um feedback mesmo em blocos vazios para você saber que não travou
                    if bloco_atual % 20 == 0:
                        print(f"   ⏩ [Bloco {bloco_atual}/{total_blocos}] -> Avançando áreas sem floresta...", flush=True)

                # Gravação imediata no SSD Local
                dst.write(rgb_tile[:, :, 0], 1, window=window)
                dst.write(rgb_tile[:, :, 1], 2, window=window)
                dst.write(rgb_tile[:, :, 2], 3, window=window)

                del rgb_raw, rgb_tile, r, g, b, mask_tile
                gc.collect()

    print("\n4. A transferir a imagem final do SSD Local para o seu Google Drive...", flush=True)
    shutil.move(temp_tif_local, path_tif_slic)
    print(f"✅ Imagem GeoTIFF guardada com contornos realçados em: {path_tif_slic}", flush=True)

    print("5. A gerar pré-visualização rápida (PNG)...", flush=True)
    path_png_slic = os.path.join(output_seg_dir, f"{code_muni}_MaskSLIC_Preview_{ano_fim}.png")
    
    with rasterio.open(path_tif_slic) as src:
        passo = max(1, src.width // 3000)
        out_shape = (src.count, src.height // passo, src.width // passo)
        preview_rgb = src.read(out_shape=out_shape)
        preview_rgb = np.moveaxis(preview_rgb, 0, -1) 

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(preview_rgb)
    ax.axis('off')
    plt.savefig(path_png_slic, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()

    print(f"✅ Pré-visualização PNG salva em: {path_png_slic}", flush=True)
    print("\n🎉 Processo 100% concluído! Verifique a pasta para ver o resultado.", flush=True)

if __name__ == "__main__":
    main()