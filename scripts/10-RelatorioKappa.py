import os
import glob
import pandas as pd
import numpy as np
import rasterio
from sklearn.metrics import cohen_kappa_score
from dotenv import load_dotenv

# --- CONFIGURAÇÃO ---
load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')
dir_mb = os.path.join(ROOT, 'data', 'MapBiomas')
dir_out = os.path.join(ROOT, 'data', 'Output')

# --- DICIONÁRIO DE CATEGORIZAÇÃO OFICIAL ---
class_info = {}

def registar_categoria(ids, name, color, iso120, iso122, iso123):
    for i in ids:
        class_info[i] = {'name': name}

# Mesma lógica de agrupamento do Script 09
registar_categoria([3], 'Floresta', '', '', '', '')
registar_categoria([9], 'Floresta Antropica', '', '', '', '')
registar_categoria([11, 12, 36], 'Vegetacao Herbacea', '', '', '', '')
registar_categoria([15, 19, 20, 21, 39, 41, 46, 48], 'Agropecuaria', '', '', '', '')
registar_categoria([24, 25], 'Infraestrutura Urbana', '', '', '', '')
registar_categoria([29, 31, 33], 'Nao Observado (Agua)', '', '', '', '')
registar_categoria([4, 5, 6, 23, 27, 30, 32, 35, 40, 47, 49, 50, 62, 75], 'Descartadas', '', '', '', '')

def localizar_raster(ano):
    busca = glob.glob(os.path.join(dir_mb, f"{ano}_coverage_*.tif"))
    return busca[0]

def calcular_area_pixel_ha(src):
    """Calcula a área do pixel convertendo diretamente para Hectares."""
    res_x, res_y = abs(src.res[0]), abs(src.res[1])
    if res_x < 1: # Graus (EPSG:4326)
        m_per_deg = 111320 * 0.92 
        area_m2 = (res_x * m_per_deg) * (res_y * 111320)
    else: # Metros (UTM)
        area_m2 = res_x * res_y
    return area_m2 / 10000.0 # Retorna em Hectares

def gerar_relatorio_padronizado():
    p21, p23 = localizar_raster("2021"), localizar_raster("2023")
    
    with rasterio.open(p23) as src23, rasterio.open(p21) as src21:
        pixel_ha = calcular_area_pixel_ha(src23)
        y23, y21 = src23.read(1), src21.read(1)
        
        mask = (y23 != 0) & (y21 != 0)
        y23_v, y21_v = y23[mask], y21[mask]
        
        kappa = cohen_kappa_score(y21_v, y23_v)
        
        # Agrupamento
        stats = {}
        ids_all = np.unique(np.concatenate([y21_v, y23_v]))

        for cid in ids_all:
            nome = class_info.get(cid, {'name': f'ID_{cid}'})['name']
            if nome not in stats: stats[nome] = {'px21': 0, 'px23': 0}
            stats[nome]['px21'] += np.sum(y21_v == cid)
            stats[nome]['px23'] += np.sum(y23_v == cid)

        resumo = []
        for nome, c in stats.items():
            ha21, ha23 = c['px21'] * pixel_ha, c['px23'] * pixel_ha
            var_ha = ha23 - ha21
            pct = (var_ha / ha21 * 100) if ha21 > 0 else 0
            
            resumo.append({
                'CATEGORIA': nome,
                'HECTARES_2021': round(ha21, 2),
                'HECTARES_2023': round(ha23, 2),
                'VARIACAO_ABS_HA': round(var_ha, 3),
                'VARIACAO_PCT': round(pct, 2)
            })

        df = pd.DataFrame(resumo).sort_values(by='HECTARES_2023', ascending=False)
        df.to_csv(os.path.join(dir_out, "10_Relatorio_Estatistico_SJC.csv"), index=False, sep=';')
        
        with open(os.path.join(dir_out, "10_Parecer_Tecnico_Kappa.txt"), 'w', encoding='utf-8') as f:
            f.write("====================================================\n")
            f.write("      PARECER TÉCNICO: MONITORAMENTO SJC 21-23      \n")
            f.write("====================================================\n\n")
            f.write(f"COEFICIENTE KAPPA: {kappa:.4f} | UNIDADE: HECTARES (ha)\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\n----------------------------------------------------\n")
            f.write("Gerado automaticamente pelo Pipeline ForestEyes\n")

    print(f"✅ Relatório em Hectares gerado! Kappa: {kappa:.4f}")

if __name__ == "__main__":
    gerar_relatorio_padronizado()