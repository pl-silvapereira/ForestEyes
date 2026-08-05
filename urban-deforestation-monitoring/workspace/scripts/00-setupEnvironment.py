import os
import subprocess
import sys
import ee

# 1. Autentica interativamente (vai abrir o link para você logar com sua conta Google)
ee.Authenticate()

# 2. Inicializa o projeto
ee.Initialize(project='foresteyes-regioes-urbanas')
print("Autenticado e inicializado com sucesso!")

# Define o diretório raiz do projeto
os.environ['PROJECT_ROOT'] = '/content/drive/MyDrive/Mestrado/04-Projeto ForestEyes/ForestEyes/urban-deforestation-monitoring/workspace/' # Corrigido para a raiz do repositório

# Configura o código do município e o ano
os.environ['CODE_MUNI'] = '3549904'  # São José dos Campos
os.environ['ANO'] = '2023'

print("Variáveis de ambiente definidas:")
print(f"  PROJECT_ROOT: {os.environ['PROJECT_ROOT']}")
print(f"  CODE_MUNI: {os.environ['CODE_MUNI']}")
print(f"  ANO: {os.environ['ANO']}")

# Define o caminho para o arquivo requirements.txt
requirements_path = os.path.join(os.environ['PROJECT_ROOT'], '../requirements.txt') # Usa PROJECT_ROOT corrigido

# Instala os pacotes do arquivo requirements.txt usando subprocess
try:
    print(f"Verificando e instalando dependências de {requirements_path}...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
    print("Dependências instaladas com sucesso (se ainda não estavam). ")
except subprocess.CalledProcessError as e:
    print(f"Erro ao instalar dependências: {e}")
    sys.exit(1)

print("Script 00-setupEnvironment.py finalizado com sucesso.")