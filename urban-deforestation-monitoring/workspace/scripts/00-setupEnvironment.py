import os
import subprocess
import sys

# Define o diretório raiz do projeto
os.environ['PROJECT_ROOT'] = '/content/drive/MyDrive/Mestrado/04-Projeto ForestEyes/ForestEyes/urban-deforestation-monitoring/workspace/' # Corrigido para a raiz do repositório

# Define o caminho para o arquivo requirements.txt
requirements_path = os.path.join(os.environ['PROJECT_ROOT'], 'requirements.txt') # Usa PROJECT_ROOT corrigido

# Instala os pacotes do arquivo requirements.txt usando subprocess
try:
    print(f"Verificando e instalando dependências de {requirements_path}...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
    print("Dependências instaladas com sucesso (se ainda não estavam). ")
except subprocess.CalledProcessError as e:
    print(f"Erro ao instalar dependências: {e}")
    sys.exit(1)

print("Script 00-setupEnvironment.py finalizado com sucesso.")