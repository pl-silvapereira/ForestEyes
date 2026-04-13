from google.colab import drive
import os

# 1. Conecta o seu Google Drive (vai pedir autorização)
drive.mount('/content/drive')

# 2. Define o caminho exato de destino no seu Drive
caminho_destino = "/content/drive/MyDrive/Mestrado/04-Projeto ForestEyes/ForestEyes/scripts"

# Garante que a pasta de destino exista
os.makedirs(caminho_destino, exist_ok=True)

print("Baixando o repositório...")

# 3. Comandos de terminal integrados para baixar e mover os arquivos
# Remove qualquer pasta temporária antiga
!rm -rf /tmp/ForestEyes_temp

# Clona APENAS a branch 'develop' (usando --single-branch fica muito mais rápido)
!git clone -b develop --single-branch https://github.com/pl-silvapereira/ForestEyes.git /tmp/ForestEyes_temp

print("Copiando a pasta 'scripts' para o Google Drive...")

# Copia os arquivos da pasta scripts para o seu Drive
# (As aspas em "{caminho_destino}" protegem o espaço no nome da sua pasta)
!cp -r /tmp/ForestEyes_temp/scripts/* "{caminho_destino}/"

# Limpa a lixeira (apaga o resto do repositório que não precisamos)
!rm -rf /tmp/ForestEyes_temp

print("✅ Download e cópia concluídos com sucesso!")