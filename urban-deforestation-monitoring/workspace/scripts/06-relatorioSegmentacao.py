import sys
import os
import gc
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from skimage.segmentation import slic, find_boundaries
import matplotlib.pyplot as plt
from dotenv import load_dotenv

def esticar_contraste(banda):
    """Aplica contraste a uma banda (2% - 98%) para melhor visualização."""
    p2, p98 = np.percentile(banda[banda > 0], (2, 98))
    banda_eq = np.clip((banda - p2) / (p98 - p2), 0, 1)
    return banda_eq

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
        print(f"[ERRO CRÍTICO] Shapefile binário não encontrado: {path_shp_binario}")
        sys.exit(1)
    if not os.path.exists(path_cbers):
        print(f"[ERRO CRÍTICO] Imagem CBERS não encontrada: {path_cbers}")
        sys.exit(1)

    print("=" * 80, flush=True)
    print(f"📄 SCRIPT 06 - RELATÓRIO DE SEGMENTAÇÃO E MASK-SLIC ({ano_fim})", flush=True)
    print("=" * 80, flush=True)

    print("1. A ler imagem de satélite e Shapefile...", flush=True)
    with rasterio.open(path_cbers) as src:
        cbers_transform = src.transform
        cbers_shape = (src.height, src.width)
        cbers_meta = src.meta.copy()
        cbers_crs = src.crs
        
        pixel_res_x = abs(cbers_transform[0])
        pixel_res_y = abs(cbers_transform[4])
        pixel_area_ha = (pixel_res_x * pixel_res_y) / 10000.0
        
        band_r = src.read(1).astype(np.float32)
        band_g = src.read(2).astype(np.float32)
        band_b = src.read(3).astype(np.float32)

    gdf_binario = gpd.read_file(path_shp_binario)
    gdf_binario = gdf_binario.to_crs(cbers_crs) 
    
    print("2. A rasterizar máscaras (Segmentar e Não Segmentar)...", flush=True)
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
        print("❌ ERRO: A classe 'Segmentar' tem 0 pixéis. Verifique a sobreposição dos mapas.", flush=True)
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
    print(f"✅ Relatório TXT gerado em: {relatorio_txt_path}", flush=True)

    print(f"3. A processar o MaskSLIC ({qts:,} superpixels)...", flush=True)
    print("   ⏳ Otimização ativada: Processamento em blocos menores com feedback em tempo real (aguarde uns minutos)...", flush=True)
    
    r_eq = esticar_contraste(band_r)
    g_eq = esticar_contraste(band_g)
    b_eq = esticar_contraste(band_b)
    
    r_out = (r_eq * 255).astype(np.uint8)
    g_out = (g_eq * 255).astype(np.uint8)
    b_out = (b_eq * 255).astype(np.uint8)
    
    del band_r, band_g, band_b
    gc.collect()

    h, w = mask_seg.shape
    tile_size = 1024  # Bloco reduzido para acelerar o feedback visual
    
    total_linhas = len(range(0, h, tile_size))
    linha_atual = 0

    for y in range(0, h, tile_size):
        linha_atual += 1
        blocos_processados_nesta_linha = 0
        
        for x in range(0, w, tile_size):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            
            mask_tile = mask_seg[y:y_end, x:x_end]
            qtp_tile = np.sum(mask_tile)
            
            # Pula blocos vazios (sem floresta)
            if qtp_tile == 0:
                continue
                
            qts_tile = int(qtp_tile / qps)
            if qts_tile <= 0:
                continue
                
            rgb_tile = np.dstack([
                r_eq[y:y_end, x:x_end], 
                g_eq[y:y_end, x:x_end], 
                b_eq[y:y_end, x:x_end]
            ])
            
            # SLIC com metade das iterações (max_num_iter=5) para acelerar 2x
            segmentos_tile = slic(rgb_tile, n_segments=qts_tile, compactness=10, mask=(mask_tile == 1), start_label=1, max_num_iter=5)
            contornos_tile = find_boundaries(segmentos_tile, mode='inner')
            
            r_out[y:y_end, x:x_end][contornos_tile] = 255
            g_out[y:y_end, x:x_end][contornos_tile] = 255
            b_out[y:y_end, x:x_end][contornos_tile] = 0
            
            blocos_processados_nesta_linha += 1

        # Mostra o status a cada "linha" da grelha processada, forçando a tela a atualizar
        print(f"   -> Linha da grelha {linha_atual}/{total_linhas} concluída ({blocos_processados_nesta_linha} blocos de floresta desenhados).", flush=True)

    print("\n4. A guardar o GeoTIFF final (Alta Resolução)...", flush=True)
    path_tif_slic = os.path.join(output_seg_dir, f"{code_muni}_MaskSLIC_ContornosAmarelos_{ano_fim}.tif")
    cbers_meta.update(dtype=rasterio.uint8, count=3, nodata=None)

    with rasterio.open(path_tif_slic, 'w', **cbers_meta) as dst:
        dst.write(r_out, 1)
        dst.write(g_out, 2)
        dst.write(b_out, 3)
    
    print(f"✅ Imagem GeoTIFF salva em: {path_tif_slic}", flush=True)

    print("5. A gerar pré-visualização rápida (PNG)...", flush=True)
    path_png_slic = os.path.join(output_seg_dir, f"{code_muni}_MaskSLIC_Preview_{ano_fim}.png")
    
    passo = max(1, cbers_shape[1] // 3000) 
    rgb_preview = np.dstack([r_out[::passo, ::passo], g_out[::passo, ::passo], b_out[::passo, ::passo]])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb_preview)
    ax.axis('off')
    plt.savefig(path_png_slic, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()

    print(f"✅ Pré-visualização PNG salva em: {path_png_slic}", flush=True)
    print("\n🎉 Processo totalmente concluído!", flush=True)

if __name__ == "__main__":
    main()