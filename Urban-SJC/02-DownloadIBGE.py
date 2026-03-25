import geobr
import os
from pathlib import Path

def baixar_limite_sjc():
    home = Path.home()
    # Caminho onde o seu script de recorte espera o arquivo
    pasta_vetores = home / "development" / "Projects" / "ForestEyes" / "Urban-SJC" / "vetores"
    
    if not pasta_vetores.exists():
        os.makedirs(pasta_vetores)

    print("Conectando ao banco de dados do IBGE via geobr...")
    
    # Baixa o polígono do município (Código IBGE de SJC: 3549904)
    # O geobr permite baixar por nome ou código
    mun = geobr.read_municipality(code_muni=3549904, year=2022)

    # Caminho final do arquivo
    caminho_saida = pasta_vetores / "limite_sjc.shp"

    # Salva em formato Shapefile
    # O geobr retorna um objeto GeoDataFrame (pandas especializado em mapas)
    mun.to_file(str(caminho_saida))

    print(f"Sucesso! O limite de São José dos Campos foi salvo em: {caminho_saida}")

if __name__ == "__main__":
    baixar_limite_sjc()