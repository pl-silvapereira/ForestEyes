import os
import glob
import rasterio
import geopandas as gpd
from rasterio.warp import transform_bounds
from dotenv import load_dotenv

# --- CONFIGURAÇÃO ---
load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')

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
    verificar_referencia_mapbiomas()
    print("Verificação concluída.")