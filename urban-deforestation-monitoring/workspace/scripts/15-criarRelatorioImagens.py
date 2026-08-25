import os
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import textwrap

# 1. Configuração dos caminhos base
os.environ['PROJECT_ROOT'] = '/content/drive/MyDrive/Mestrado/04-Projeto ForestEyes/ForestEyes/urban-deforestation-monitoring/workspace/'
project_root = os.environ['PROJECT_ROOT']

campaign_dir = os.path.join(project_root, 'data', 'output', 'mask', 'campaign', '2024')
reports_dir = os.path.join(project_root, 'reports')
os.makedirs(reports_dir, exist_ok=True)
pdf_path = os.path.join(reports_dir, 'Catalogo_Composicoes_Dinamico.pdf')

# 2. Segmentos analíticos alvo
segmentos = {
    'Floresta': '11406',
    'Não-Floresta': '10969'
}

print(f"Lendo imagens do diretório: {campaign_dir}")
print("Mapeando composições dinamicamente...")

# Dicionário para armazenar as imagens encontradas para cada ID
imagens_encontradas = {seg_id: [] for seg_id in segmentos.values()}

# 3. Varredura Dinâmica das Subpastas
if os.path.exists(campaign_dir):
    # Ordena as pastas para manter um padrão consistente no PDF
    for nome_pasta in sorted(os.listdir(campaign_dir)):
        caminho_pasta = os.path.join(campaign_dir, nome_pasta)
        
        # Garante que estamos lendo apenas diretórios (subpastas)
        if os.path.isdir(caminho_pasta):
            for arquivo in os.listdir(caminho_pasta):
                # Percorre os IDs alvo
                for seg_id in segmentos.values():
                    # A busca por f"ID_{seg_id}" garante que não haja conflitos numéricos
                    if f"ID_{seg_id}" in arquivo and arquivo.endswith('.png'):
                        imagens_encontradas[seg_id].append({
                            'caminho': os.path.join(caminho_pasta, arquivo),
                            'legenda_pasta': nome_pasta
                        })
else:
    print(f"❌ Erro: Diretório da campanha não encontrado em {campaign_dir}")
    exit()

print("Iniciando geração do catálogo em PDF...")

with PdfPages(pdf_path) as pdf:
    for classe, seg_id in segmentos.items():
        imagens_do_id = imagens_encontradas[seg_id]
        
        if not imagens_do_id:
            print(f"⚠️ Aviso: Nenhuma imagem encontrada para o Segmento {classe} (ID: {seg_id}).")
            continue
            
        # 4. Paginação (6 imagens por folha A4 em formato 3 linhas x 2 colunas)
        imagens_por_pagina = 6
        total_paginas = math.ceil(len(imagens_do_id) / imagens_por_pagina)
        
        for num_pagina in range(total_paginas):
            fig = plt.figure(figsize=(8.27, 11.69)) # Dimensões padrão A4
            
            # Título do cabeçalho com indicador de página
            titulo_header = f'Catálogo de Composições - Projeto ForestEyes\nSegmento: {classe} (ID: {seg_id}) - Pág. {num_pagina + 1}/{total_paginas}'
            fig.suptitle(titulo_header, fontsize=16, fontweight='bold', y=0.95, color='#1b4d3e')
            
            # Fatiar a lista de imagens para a página atual
            inicio = num_pagina * imagens_por_pagina
            fim = min(inicio + imagens_por_pagina, len(imagens_do_id))
            imagens_pagina_atual = imagens_do_id[inicio:fim]
            
            # 5. Plotagem do Grid
            for i, item in enumerate(imagens_pagina_atual):
                ax = fig.add_subplot(3, 2, i + 1) # 3 linhas, 2 colunas
                
                # Leitura e injeção da imagem
                img_patch = Image.open(item['caminho'])
                ax.imshow(img_patch)
                ax.axis('off')
                
                # Formatação da legenda usando o nome da pasta
                texto_legenda = item['legenda_pasta']
                
                # Quebra de linha automática (textwrap) caso o nome da pasta seja muito longo
                texto_quebrado = "\n".join(textwrap.wrap(texto_legenda, width=35))
                
                # Adiciona o nome da subpasta exatamente como legenda (título do subplot)
                ax.set_title(f"Pasta: {texto_quebrado}", fontsize=9, pad=8, fontweight='medium', color='#2d3748')
                
            # Ajuste de layout para evitar sobreposições
            plt.tight_layout(rect=[0, 0.03, 1, 0.90])
            pdf.savefig(fig)
            plt.close()

print(f"\n[SUCESSO] Arquivo PDF gerado de forma dinâmica em:\n-> {pdf_path}")