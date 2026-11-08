#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Etapa 01: Verificação de Arquivos de Entrada, Pan-sharpening e Composições
==============================================================================
"""

import sys
import time
import shutil
import os
import unicodedata
import json
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
import geopandas as gpd
import geobr
from dotenv import load_dotenv
import duckdb

# Compatibilidade DuckDB e geobr no Python 3.13
if not hasattr(duckdb.DuckDBPyRelation, "fetchone"):
    duckdb.DuckDBPyRelation.fetchone = lambda self: self.fetchall()[0]

try:
    from cbers4asat import Cbers4aAPI
    from cbers4asat.tools import rgbn_composite
except ImportError:
    pass

def limpar_para_ee(texto):
    nfkd = unicodedata.normalize('NFKD', texto)
    return "".join([c for c in nfkd if not unicodedata.combining(c)]).replace(" ", "_")

def verificar_mapbiomas_local(code_muni, ano, nome_cidade, projeto_root):
    """Verifica se o arquivo do MapBiomas do ano especificado já existe localmente."""
    cidade_limpa = limpar_para_ee(nome_cidade)
    nome_arquivo = f'mapbiomas_lulc_10m_{cidade_limpa.lower()}_{ano}.tif'
    caminho_local = os.path.join(projeto_root, "data", "input", "MapBiomas", str(ano), nome_arquivo)
    if os.path.exists(caminho_local) and os.path.getsize(caminho_local) > 1000:
        return caminho_local
    return None

def baixar_mapbiomas(code_muni, ano, limite_geopolitico, nome_cidade, uf, projeto_root, gdf_muni):
    caminho_existente = verificar_mapbiomas_local(code_muni, ano, nome_cidade, projeto_root)
    if caminho_existente:
        print(f"✅ MAPBIOMAS ({ano}) encontrado localmente: {caminho_existente}")
        return caminho_existente

    print(f"\n--- MAPBIOMAS ({ano}) NÃO ENCONTRADO LOCALMENTE. INICIANDO DOWNLOAD VIA EARTH ENGINE ---")
    import google.auth
    import ee
    print("Inicializando Google Earth Engine...")
    credentials, _ = google.auth.default(
        scopes=['https://www.googleapis.com/auth/earthengine', 'https://www.googleapis.com/auth/cloud-platform']
    )
    ee.Initialize(credentials, project='foresteyes-regioes-urbanas')

    cidade_limpa = limpar_para_ee(nome_cidade)
    nome_arquivo_mapbiomas = f'mapbiomas_lulc_10m_{cidade_limpa.lower()}_{ano}.tif'
    diretorio_data_input = os.path.join(projeto_root, "data", "input", "MapBiomas", str(ano))
    os.makedirs(diretorio_data_input, exist_ok=True)
    caminho_mapbiomas_local = os.path.join(diretorio_data_input, nome_arquivo_mapbiomas)

    banda_ano = f'classification_{ano}'
    asset_mapbiomas_10m = 'projects/mapbiomas-public/assets/brazil/lulc_10m/collection3/mapbiomas_10m_collection3_integration_v1'
    mapbiomas_10m = ee.Image(asset_mapbiomas_10m).select(banda_ano)
    imagem_recortada = mapbiomas_10m.clip(limite_geopolitico).unmask(0).short()

    pasta_temporaria = f'MapBiomas_Temp_{ano}'
    tarefa = ee.batch.Export.image.toDrive(
        image=imagem_recortada,
        description=f'Export_mapbiomas_lulc_10m_{cidade_limpa.lower()}_{ano}',
        folder=pasta_temporaria,
        fileNamePrefix=f'mapbiomas_lulc_10m_{cidade_limpa.lower()}_{ano}',
        region=limite_geopolitico.bounds(),
        scale=10,
        maxPixels=1e9,
        fileFormat='GeoTIFF',
        formatOptions={'noData': 0}
    )
    tarefa.start()

    while True:
        status = ee.data.getTaskStatus(tarefa.id)[0]
        state = status.get('state')
        if state == 'COMPLETED':
            print(f"[SUCESSO] MapBiomas {ano} concluído no Earth Engine!")
            break
        elif state in ['FAILED', 'CANCELLED']:
            raise Exception(f"Exportação MapBiomas {ano} falhou: {status.get('error_message')}")
        time.sleep(15)

    raiz_drive = os.path.abspath(os.path.join(projeto_root, "../../../../../"))
    caminho_temp_local = os.path.join(raiz_drive, pasta_temporaria)
    arquivo_origem = os.path.join(caminho_temp_local, nome_arquivo_mapbiomas)

    tentativas = 0
    while not os.path.exists(arquivo_origem) and tentativas < 20:
        time.sleep(5)
        tentativas += 1

    if os.path.exists(arquivo_origem):
        shutil.move(arquivo_origem, caminho_mapbiomas_local)
        try: os.rmdir(caminho_temp_local)
        except OSError: pass
    else:
        raise Exception(f"[ERRO] {nome_arquivo_mapbiomas} não encontrado no Drive pós-exportação.")

    return caminho_mapbiomas_local

def recortar_banda(caminho_in, gdf_limite, caminho_out):
    with rasterio.open(caminho_in) as src:
        gdf_proj = gdf_limite.to_crs(src.crs)
        try:
            out_image, out_transform = mask(src, gdf_proj.geometry, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            with rasterio.open(caminho_out, "w", **out_meta) as dest:
                dest.write(out_image)
            return True
        except ValueError:
            print(f"Aviso: Não houve intersecção válida para {caminho_in}")
            return False

def aplicar_pansharpening_em_blocos(caminho_pan, caminho_multi, caminho_saida):
    if os.path.exists(caminho_saida) and os.path.getsize(caminho_saida) > 1000:
        print(f"✅ Pan-sharpening (2m) já existe: {caminho_saida}. Pulando esta etapa.")
        return

    print("🚀 Aplicando Pan-sharpening (Fusão para 2m) em blocos para economia de RAM...")
    with rasterio.open(caminho_pan) as pan_src, rasterio.open(caminho_multi) as multi_src:
        pan_meta = pan_src.meta.copy()
        num_bands = multi_src.count
        pan_meta.update(
            count=num_bands,
            dtype=rasterio.uint16,
            photometric='RGB',
            tiled=True,
            blockxsize=1024,
            blockysize=1024
        )

        with rasterio.open(caminho_saida, 'w', **pan_meta) as dst:
            block_size = 2048
            height, width = pan_meta['height'], pan_meta['width']

            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    window_width = min(block_size, width - x)
                    window_height = min(block_size, height - y)
                    window = rasterio.windows.Window(x, y, window_width, window_height)

                    pan_block = pan_src.read(1, window=window).astype(np.float32)
                    multi_block = np.empty((num_bands, window_height, window_width), dtype=np.float32)
                    win_transform = rasterio.windows.transform(window, pan_src.transform)

                    reproject(
                        source=rasterio.band(multi_src, tuple(range(1, num_bands + 1))),
                        destination=multi_block,
                        src_transform=multi_src.transform,
                        src_crs=multi_src.crs,
                        dst_transform=win_transform,
                        dst_crs=pan_meta['crs'],
                        resampling=Resampling.bilinear
                    )

                    media_multi = np.mean(multi_block, axis=0)
                    media_multi[media_multi == 0] = 1e-6

                    ratio = pan_block / media_multi
                    pan_sharpened = np.clip(multi_block * ratio, 0, 65535).astype(np.uint16)

                    dst.write(pan_sharpened, window=window)

            dst.set_band_description(1, "Red")
            dst.set_band_description(2, "Green")
            dst.set_band_description(3, "Blue")
            if num_bands >= 4:
                dst.set_band_description(4, "NIR")

def obter_estatisticas_banda(src, band_idx, decimation=10):
    out_shape = (1, max(1, src.height // decimation), max(1, src.width // decimation))
    amostra = src.read(band_idx, out_shape=out_shape).astype(np.float32)
    p2, p98 = np.percentile(amostra[amostra > 0], (2, 98)) if np.any(amostra > 0) else (0, 1)
    if p98 == p2:
        p98 = p2 + 1
    return p2, p98

def normalizar_array(data, p2, p98):
    return np.clip((data - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)

def gerar_composicoes_zooniverse(caminho_stack, pasta_saida, code_muni, ano):
    print("\n--- GERANDO COMPOSIÇÕES VISUAIS (ZOONIVERSE) EM BLOCOS ---")
    pasta_comp = os.path.join(pasta_saida, "composicoes")
    os.makedirs(pasta_comp, exist_ok=True)

    with rasterio.open(caminho_stack) as src:
        if src.count < 4:
            print("[AVISO] O stack não possui 4 bandas completas.")
            return

        meta = src.meta.copy()
        meta.update(tiled=True, blockxsize=1024, blockysize=1024)
        height, width = meta['height'], meta['width']

        p2_r, p98_r = obter_estatisticas_banda(src, 1)
        p2_g, p98_g = obter_estatisticas_banda(src, 2)
        p2_b, p98_b = obter_estatisticas_banda(src, 3)
        p2_nir, p98_nir = obter_estatisticas_banda(src, 4)

        meta_rgb = meta.copy()
        meta_rgb.update(count=3, dtype=rasterio.uint8, photometric='RGB')
        meta_ndvi = meta.copy()
        meta_ndvi.update(count=1, dtype=rasterio.uint8, photometric='MINISBLACK')

        path_rgb = os.path.join(pasta_comp, f"{code_muni}_{ano}_1_CorNatural_RGB.tif")
        path_nir_rg = os.path.join(pasta_comp, f"{code_muni}_{ano}_2_FalsaCor_NIR-R-G.tif")
        path_nir_gb = os.path.join(pasta_comp, f"{code_muni}_{ano}_3_FalsaCor_NIR-G-B.tif")
        path_ndvi = os.path.join(pasta_comp, f"{code_muni}_{ano}_4_NDVI_Cinza.tif")

        with rasterio.open(path_rgb, "w", **meta_rgb) as dst_rgb, \
             rasterio.open(path_nir_rg, "w", **meta_rgb) as dst_nir_rg, \
             rasterio.open(path_nir_gb, "w", **meta_rgb) as dst_nir_gb, \
             rasterio.open(path_ndvi, "w", **meta_ndvi) as dst_ndvi:

            dst_rgb.set_band_description(1, "Red"); dst_rgb.set_band_description(2, "Green"); dst_rgb.set_band_description(3, "Blue")
            dst_nir_rg.set_band_description(1, "NIR"); dst_nir_rg.set_band_description(2, "Red"); dst_nir_rg.set_band_description(3, "Green")
            dst_nir_gb.set_band_description(1, "NIR"); dst_nir_gb.set_band_description(2, "Green"); dst_nir_gb.set_band_description(3, "Blue")
            dst_ndvi.set_band_description(1, "NDVI")

            block_size = 2048
            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    window_width = min(block_size, width - x)
                    window_height = min(block_size, height - y)
                    window = rasterio.windows.Window(x, y, window_width, window_height)

                    r_data = src.read(1, window=window).astype(np.float32)
                    g_data = src.read(2, window=window).astype(np.float32)
                    b_data = src.read(3, window=window).astype(np.float32)
                    nir_data = src.read(4, window=window).astype(np.float32)

                    r_norm = normalizar_array(r_data, p2_r, p98_r)
                    g_norm = normalizar_array(g_data, p2_g, p98_g)
                    b_norm = normalizar_array(b_data, p2_b, p98_b)
                    nir_norm = normalizar_array(nir_data, p2_nir, p98_nir)

                    dst_rgb.write(r_norm, 1, window=window)
                    dst_rgb.write(g_norm, 2, window=window)
                    dst_rgb.write(b_norm, 3, window=window)

                    dst_nir_rg.write(nir_norm, 1, window=window)
                    dst_nir_rg.write(r_norm, 2, window=window)
                    dst_nir_rg.write(g_norm, 3, window=window)

                    dst_nir_gb.write(nir_norm, 1, window=window)
                    dst_nir_gb.write(g_norm, 2, window=window)
                    dst_nir_gb.write(b_norm, 3, window=window)

                    denominator = (nir_data + r_data)
                    ndvi_block = np.zeros_like(r_data)
                    valid_mask = denominator != 0
                    ndvi_block[valid_mask] = (nir_data[valid_mask] - r_data[valid_mask]) / denominator[valid_mask]
                    ndvi_scaled = np.clip((ndvi_block + 1) / 2 * 255, 0, 255).astype(np.uint8)

                    dst_ndvi.write(ndvi_scaled, 1, window=window)

    print(f" -> Composições geradas com sucesso na pasta:\n {pasta_comp}")

def localizar_bandas_locais(pasta_entrada_cbers):
    """
    Localiza dinamicamente as bandas CBERS no diretório local, considerando:
    - Variações de sufixo BAND0 / BANDO
    - Arquivos brutos ou já recortados (_recortado.tif)
    """
    bandas = ['BAND0', 'BAND1', 'BAND2', 'BAND3', 'BAND4']
    arquivos_locais = {}
    arquivos_recortados = {}

    if not os.path.exists(pasta_entrada_cbers):
        return arquivos_locais, arquivos_recortados

    todos_arquivos = os.listdir(pasta_entrada_cbers)

    for b in bandas:
        variacoes_b = [b, b.replace('BAND0', 'BANDO')]
        
        recortados = [f for f in todos_arquivos if any(v in f for v in variacoes_b) and f.endswith('_recortado.tif')]
        if recortados:
            arquivos_recortados[b] = os.path.join(pasta_entrada_cbers, recortados[0])

        brutos = [f for f in todos_arquivos if any(v in f for v in variacoes_b) and f.endswith('.tif') and not f.endswith('_recortado.tif')]
        if brutos:
            arquivos_locais[b] = os.path.join(pasta_entrada_cbers, brutos[0])

    return arquivos_locais, arquivos_recortados

def baixar_e_processar_cbers(code_muni, ano_fim, dados_json, projeto_root, gdf_muni):
    print(f"\n--- PROCESSAMENTO CBERS-4A ({ano_fim}) ---")
    pasta_entrada_cbers = os.path.join(projeto_root, "data", "input", "CBERS-4A-WPM", str(ano_fim))
    pasta_saida_pansharpening = os.path.join(projeto_root, "data", "output", "pansharpening", str(ano_fim))

    os.makedirs(pasta_entrada_cbers, exist_ok=True)
    os.makedirs(pasta_saida_pansharpening, exist_ok=True)

    bandas = ['BAND0', 'BAND1', 'BAND2', 'BAND3', 'BAND4']

    # 1. Checa a presença local das bandas CBERS
    arquivos_brutos, arquivos_recortados = localizar_bandas_locais(pasta_entrada_cbers)

    tem_todos_recortados = len(arquivos_recortados) == 5
    tem_todos_brutos = len(arquivos_brutos) == 5

    if tem_todos_recortados:
        print("✅ As 5 bandas CBERS-4A RECORTADAS já existem no Drive. Saltando download e recorte.")
        bandas_para_stack = arquivos_recortados
    elif tem_todos_brutos:
        print("✅ As 5 bandas CBERS-4A BRUTAS foram localizadas. Saltando requisição ao INPE.")
        print("Recortando bandas para os limites geopolíticos...")
        bandas_para_stack = {}
        for b in bandas:
            caminho_in = arquivos_brutos[b]
            caminho_out = os.path.join(pasta_entrada_cbers, f"{os.path.basename(caminho_in).replace('.tif', '')}_recortado.tif")
            if not os.path.exists(caminho_out):
                recortar_banda(caminho_in, gdf_muni, caminho_out)
            bandas_para_stack[b] = caminho_out
    else:
        print("⚠️ Bandas CBERS não localizadas por completo. Consultando API do INPE...")
        coords = dados_json['coordenadas_extremas']
        bbox_muni = [coords['oeste'], coords['sul'], coords['leste'], coords['norte']]
        email_inpe = 'pereira.pedro@unifesp.br'
        api = Cbers4aAPI(email_inpe)

        resultados = api.query(
            location=bbox_muni,
            initial_date=date(int(ano_fim), 1, 1),
            end_date=date(int(ano_fim), 12, 31),
            cloud=20, limit=100,
            collections=['CBERS4A_WPM_L4_DN']
        )
        cenas_encontradas = resultados.get('features', [])
        if not cenas_encontradas:
            print(f"❌ Nenhuma cena CBERS encontrada via API para {ano_fim}.")
            sys.exit(1)

        cena_selecionada = cenas_encontradas[1] if len(cenas_encontradas) > 1 else cenas_encontradas[0]
        id_cena = cena_selecionada.get('id', 'CBERS_4A_WPM')
        url_thumbnail = cena_selecionada['assets']['thumbnail']['href']
        url_base_tiff = url_thumbnail.replace("/datastore/BROWSE/", "/api/download/TIFF/").replace("/BROWSE/", "/api/download/TIFF/").replace(".jpg", "").replace(".png", "")
        if not url_base_tiff.endswith("_L4"):
            url_base_tiff += "_L4"

        bandas_para_stack = {}
        import requests
        for b in bandas:
            url_download = f"{url_base_tiff}_{b}.tif?email={email_inpe}"
            caminho_bruto = os.path.join(pasta_entrada_cbers, f"{id_cena}_{b}.tif")
            caminho_recortado = os.path.join(pasta_entrada_cbers, f"{id_cena}_{b}_recortado.tif")

            if not os.path.exists(caminho_bruto):
                print(f"Baixando {b} do INPE...")
                resp = requests.get(url_download, stream=True, timeout=120)
                resp.raise_for_status()
                with open(caminho_bruto, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk: f.write(chunk)

            if not os.path.exists(caminho_recortado):
                recortar_banda(caminho_bruto, gdf_muni, caminho_recortado)
            bandas_para_stack[b] = caminho_recortado

    # 2. Empilhamento 8m (RGBN)
    nome_arquivo_stack_8m = f"{code_muni}_{ano_fim}_CBERS_STACK_8m.tif"
    caminho_stack_8m = os.path.join(pasta_saida_pansharpening, nome_arquivo_stack_8m)

    if not os.path.exists(caminho_stack_8m):
        print("📦 Gerando Stack 8m (RGBN)...")
        rgbn_composite(
            red=bandas_para_stack['BAND3'],
            green=bandas_para_stack['BAND2'],
            blue=bandas_para_stack['BAND1'],
            nir=bandas_para_stack['BAND4'],
            filename=nome_arquivo_stack_8m,
            outdir=pasta_saida_pansharpening
        )
    else:
        print(f"✅ Stack 8m já existe: {caminho_stack_8m}")

    # 3. Pan-sharpening (fusão para 2m)
    nome_arquivo_stack_2m = f"{code_muni}_{ano_fim}_CBERS_TRUE_COLOR_2M.tif"
    caminho_stack_2m = os.path.join(pasta_saida_pansharpening, nome_arquivo_stack_2m)

    aplicar_pansharpening_em_blocos(bandas_para_stack['BAND0'], caminho_stack_8m, caminho_stack_2m)

    # 4. Geração das composições visuais
    gerar_composicoes_zooniverse(caminho_stack_2m, pasta_saida_pansharpening, code_muni, ano_fim)

    return caminho_stack_2m

def main():
    if len(sys.argv) < 4:
        print("Erro: Parâmetros insuficientes.")
        print("Uso correto: python 01-Downloads-MapBiomas-CBERS.py <code_muni> <ano_inicio> <ano_fim>")
        sys.exit(1)

    code_muni = int(sys.argv[1])
    ano_inicio = int(sys.argv[2])
    ano_fim = int(sys.argv[3])

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        raise ValueError("A variável PROJECT_ROOT não foi encontrada no arquivo .env.")

    diretorio_reports = os.path.join(project_root, "reports")
    os.makedirs(diretorio_reports, exist_ok=True)

    print(f"Buscando informações municipais no geobr (IBGE: {code_muni})...")
    gdf_muni = geobr.read_municipality(code_muni=code_muni, year=2022)
    nome_cidade = gdf_muni['name_muni'].values[0]
    uf = gdf_muni['abbrev_state'].values[0]

    print("=" * 60)
    print(f" MUNICÍPIO: {nome_cidade} - {uf} (IBGE: {code_muni})")
    print(f" PERÍODO MAPBIOMAS: {ano_inicio} e {ano_fim} | CBERS: {ano_fim}")
    print("=" * 60)

    gdf_muni_4326 = gdf_muni.to_crs(epsg=4326)

    # download ou validação local do MapBiomas
    caminho_mb_inicio = baixar_mapbiomas(code_muni, ano_inicio, None, nome_cidade, uf, project_root, gdf_muni_4326)
    caminho_mb_fim = baixar_mapbiomas(code_muni, ano_fim, None, nome_cidade, uf, project_root, gdf_muni_4326)

    bounds = gdf_muni_4326.total_bounds
    dados_json = {
        "code_muni": code_muni,
        "nome_cidade": nome_cidade,
        "uf": uf,
        "ano_inicio": ano_inicio,
        "ano_fim": ano_fim,
        "coordenadas_extremas": {"norte": bounds[3], "sul": bounds[1], "leste": bounds[2], "oeste": bounds[0]},
        "arquivo_mapbiomas_inicio": caminho_mb_inicio,
        "arquivo_mapbiomas_fim": caminho_mb_fim
    }

    caminho_json = os.path.join(diretorio_reports, f"{code_muni}_{ano_inicio}_vs_{ano_fim}.json")
    with open(caminho_json, 'w', encoding='utf-8') as f_json:
        json.dump(dados_json, f_json, indent=4, ensure_ascii=False)

    # Processamento CBERS (Empilhamento 8m + Pan-sharpening 2m + Composições)
    caminho_cbers = baixar_e_processar_cbers(code_muni, ano_fim, dados_json, project_root, gdf_muni)

    print(f"\n✅ [SUCESSO] Processamento do Script 01 finalizado com êxito!")

if __name__ == "__main__":
    main()