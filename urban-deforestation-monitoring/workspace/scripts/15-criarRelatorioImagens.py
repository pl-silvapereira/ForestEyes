import os
import re
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

# 1. Configurações de caminhos base
os.environ['PROJECT_ROOT'] = '/content/drive/MyDrive/Mestrado/04-Projeto ForestEyes/ForestEyes/urban-deforestation-monitoring/workspace/'
project_root = os.environ['PROJECT_ROOT']

# Aponta para o diretório raiz das campanhas de 2024
campaign_dir = os.path.join(project_root, 'data', 'output', 'mask', 'campaign', '2024')
reports_dir = os.path.join(project_root, 'reports')
os.makedirs(reports_dir, exist_ok=True)
pdf_path = os.path.join(reports_dir, 'Catalogo_Composicoes_ForestEyes.pdf')

# 2. Definição dos IDs alvo
segmentos = {
    'Floresta': '11406',
    'Não-Floresta': '10969'
}

def formatar_legenda(nome_pasta):
    """
    Transforma: 'composicao_B1_Azul_B2_Verde_B3_Vermelho_2024'
    Em:         'Composição B1: Azul, B2: Verde, B3: Vermelho'
    """
    # Trata as pastas especiais geradas no script 13
    if nome_pasta.upper() == "3CLASSES":
        return "3 Classes (Verde=Floresta, Vermelho=Não-Floresta)"
    if nome_pasta.upper() == "CINZA":
        return "Escala de Cinza (Contorno Vermelho)"
        
    # 1. Remove o ano do final (ex: "_2024")
    nome = re.sub(r'_\d{4}$', '', nome_pasta)
    
    # 2. Substitui "composicao_" por "Composição "
    nome = re.sub(r'^composicao_', 'Composição ', nome, flags=re.IGNORECASE)
    
    # 3. Troca o "_" logo após o número da banda por ": " (Ex: "B1_Azul" -> "B1: Azul")
    nome = re.sub(r'(B\d)_', r'\1: ', nome)
    
    # 4. Troca o "_" antes da próxima banda por ", " (Ex: "Azul_B2" -> "Azul, B2")
    nome = nome.replace("_B", ", B")
    
    return nome

print(f"📂 Vasculhando o diretório de campanhas: {campaign_dir}")
print("🚀 Iniciando geração do catálogo dinâmico de composições...")

with PdfPages(pdf_path) as pdf:
    for classe, seg_id in segmentos.items():
        
        imagens_encontradas = []
        
        # 3. Percorrer as subpastas dinamicamente
        if os.path.exists(campaign_dir):
            for subpasta in sorted(os.listdir(campaign_dir)):
                caminho_subpasta = os.path.join(campaign_dir, subpasta)
                
                if os.path.isdir(caminho_subpasta):
                    for file in os.listdir(caminho_subpasta):
                        if f"ID_{seg_id}" in file and file.endswith('.png'):
                            imagens_encontradas.append((os.path.join(caminho_subpasta, file), subpasta))
                            break # Achou a imagem, pode pular para a próxima subpasta
        
        if not imagens_encontradas:
            print(f"⚠️ Nenhuma imagem encontrada para o ID {seg_id} na pasta {campaign_dir}")
            continue

        print(f"✅ {len(imagens_encontradas)} composições localizadas para {classe} (ID: {seg_id})")

        # 4. Configuração do Grid Dinâmico
        cols = 2  
        rows = math.ceil(len(imagens_encontradas) / cols)
        
        fig_height = max(11.69, rows * 4.5) 
        fig = plt.figure(figsize=(8.27, fig_height)) 
        
        fig.suptitle(f'Catálogo de Composições - Projeto ForestEyes\nClasse: {classe} (ID: {seg_id})\n', 
                     fontsize=16, fontweight='bold', color='#1b4d3e', y=0.98)
        
        for idx, (img_path, nome_subpasta) in enumerate(imagens_encontradas):
            ax = fig.add_subplot(rows, cols, idx + 1)
            
            img = Image.open(img_path)
            ax.imshow(img)
            
            # Aplicando a nova função de formatação para a legenda
            legenda_formatada = formatar_legenda(nome_subpasta)
            
            # Inserir a legenda no rodapé da imagem
            ax.set_title(legenda_formatada, fontsize=10, y=-0.15, wrap=True, color='#2d3748', fontweight='medium')
            
            ax.axis('off')
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.95], h_pad=3.0)
        pdf.savefig(fig)
        plt.close()

print(f"\n[SUCESSO] Catálogo gerado e salvo com sucesso em:\n-> {pdf_path}")