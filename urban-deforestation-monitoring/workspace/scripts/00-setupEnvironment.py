import os
import ee

# Set an environment variable
os.environ['PROJECT_ROOT'] = '/content/drive/MyDrive/Mestrado/04-Projeto ForestEyes/ForestEyes/urban-deforestation-monitoring/workspace/'
# configurando o código do IBGE da cidade São José dos Campos
os.environ['CODE_MUNI'] = '3549904'
# configurando o ano para busca do mapa
os.environ['ANO'] = '2023'

# Or directly specify the path to requirements.txt
requirements_path = '/content/drive/MyDrive/Mestrado/04-Projeto ForestEyes/ForestEyes/urban-deforestation-monitoring/requirements.txt'

# Install the packages from the requirements.txt file
!pip install -r "{requirements_path}"

# 1. Autentica interativamente (vai abrir o link para você logar com sua conta Google)
ee.Authenticate()

# 2. Inicializa o projeto
ee.Initialize(project='foresteyes-regioes-urbanas')
print("Autenticado e inicializado com sucesso!")