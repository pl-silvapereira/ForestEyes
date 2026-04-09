import os
import glob
import rasterio
import geopandas as gpd
from rasterio.warp import transform_bounds
from dotenv import load_dotenv

# --- CONFIGURAÇÃO ---
load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')

ibge_shp = os.path.join(ROOT, 'data', 'IBGE', 'SJC_2025.shp')
mapbiomas_dir = os.path.join(ROOT, 'data', 'MapBiomas')

def formatar_coords(n, s, l, o, fonte):
    """Exibe as coordenadas formatadas para conferência."""
    print(f"\n📍 Limites extraídos via {fonte}:")
    print("-" * 35)
    print(f"Norte (Lat): {n:.6f}")
    print(f"Sul   (Lat): {s:.6f}")
    print(f"Leste (Lon): {l:.6f}")
    print(f"Oeste (Lon): {o:.6f}")
    print("-" * 35)
    print(f"Bounding Box (W, S, E, N): {o:.6f}, {s:.6f}, {l:.6f}, {n:.6f}\n")

def verificar_limites_ibge():
    """Lê o Shapefile do IBGE e extrai os limites."""
    print("=== Verificando Limites IBGE ===")
    if os.path.exists(ibge_shp):
        try:
            gdf = gpd.read_file(ibge_shp)
            print(f"🔍 Referência IBGE: {os.path.basename(ibge_shp)}")
            print(f"   - Sistema de Coordenadas (CRS): {gdf.crs}")
            
            # Converte para WGS84 (Lat/Lon) para garantir a leitura correta
            gdf_wgs84 = gdf.to_crs(epsg=4326)
            minx, miny, maxx, maxy = gdf_wgs84.total_bounds
            formatar_coords(maxy, miny, maxx, minx, "SHP do IBGE")
            
        except Exception as e:
            print(f"❌ Erro ao ler IBGE: {e}")
    else:
        print(f"⚠️ Arquivo IBGE não encontrado em {ibge_shp}\n")

def verificar_referencia_mapbiomas():
    """Lê o raster do MapBiomas, mostra metadados nativos e extrai limites."""
    print("=== Verificando Referência MapBiomas ===")
    busca = glob.glob(os.path.join(mapbiomas_dir, "*coverage_10m*.tif"))
    
    if not busca:
        print(f"❌ Erro: Nenhum arquivo '*coverage_10m*.tif' encontrado em {mapbiomas_dir}")
        return

    mapbiomas_path = busca[0]
    print(f"🔍 Referência MapBiomas: {os.path.basename(mapbiomas_path)}")

    try:
        with rasterio.open(mapbiomas_path) as src:
            print(f"   - Sistema de Coordenadas (CRS) Nativo: {src.crs}")
            print(f"   - Dimensões: {src.width}x{src.height}")
            print(f"   - Extensão (Bounds Nativos): {src.bounds}")
            
            bounds = src.bounds
            
            # Se o raster não estiver em Lat/Lon (EPSG:4326), reprojeta os limites apenas para exibição
            if src.crs.to_epsg() != 4326:
                o, s, l, n = transform_bounds(src.crs, 'EPSG:4326', *bounds)
            else:
                o, s, l, n = bounds
                
            formatar_coords(n, s, l, o, "Raster MapBiomas")
            
    except Exception as e:
        print(f"❌ Erro ao ler MapBiomas: {e}")

if __name__ == "__main__":
    print("Iniciando verificação espacial e metadados...\n")
    verificar_limites_ibge()
    verificar_referencia_mapbiomas()
    print("Verificação concluída.")