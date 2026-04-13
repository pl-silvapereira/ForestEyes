import os
import subprocess
import shutil
from google.colab import drive

# 1. Conecta o seu Google Drive
drive.mount('/content/drive')

# 2. Caminhos
caminho_destino = "/content/drive/MyDrive/Mestrado/04-Projeto ForestEyes/ForestEyes/scripts"
temp_dir = "/tmp/ForestEyes_temp"

# Garante que a pasta de destino exista
os.makedirs(caminho_destino, exist_ok=True)

print("🧹 Limpando pastas temporárias antigas...")
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)

print("📥 Baixando o repositório (branch develop)...")
# Usamos subprocess.run para executar comandos de sistema dentro do .py
subprocess.run([
    "git", "clone", "-b", "develop", "--single-branch", 
    "https://github.com/pl-silvapereira/ForestEyes.git", temp_dir
], check=True)

print("📂 Copiando scripts para o Google Drive...")
# Listamos os arquivos e copiamos um a um ou a pasta toda
origem_scripts = os.path.join(temp_dir, "scripts")
if os.path.exists(origem_scripts):
    # Copia o conteúdo da pasta scripts temporária para o destino no Drive
    for item in os.listdir(origem_scripts):
        s = os.path.join(origem_scripts, item)
        d = os.path.join(caminho_destino, item)
        if os.path.isdir(s):
            if os.path.exists(d): shutil.rmtree(d)
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)

# Limpeza final
print("🧼 Limpando arquivos temporários...")
shutil.rmtree(temp_dir)

print("✅ Sincronização concluída com sucesso!")