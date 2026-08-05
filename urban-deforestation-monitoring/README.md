Aqui está uma versão completa e estruturada do arquivo `README.md`. Ele consolida o contexto do projeto, as funcionalidades do script, as instruções de configuração de ambiente e as orientações de uso.

Você pode copiar o código abaixo e salvar diretamente no seu arquivo `README.md`.

```markdown
# Urban Deforestation Monitoring - Módulo de Pré-processamento

Este repositório contém os scripts de pré-processamento de imagens multiespectrais de sensoriamento remoto para o projeto de monitoramento de desmatamento, alinhado à metodologia do projeto **ForestEyes**. 

O objetivo deste módulo é processar recortes brutos de imagens de satélite (como do CBERS-4A), aplicar realces visuais e exportar composições em falsa-cor otimizadas para validação visual humana através da plataforma de ciência cidadã **Zooniverse**.

---

## 📋 Funcionalidades

O *script* principal deste repositório realiza as seguintes etapas operacionais:

*   **Extração de Bandas:** Leitura de arquivos `.tif` base e separação das 4 bandas fundamentais: Azul, Verde, Vermelho e Infravermelho Próximo (NIR).
*   **Realce de Contraste (Stretch):** Aplicação de *stretch* linear isolando os dados entre os percentis 2% e 98%, ignorando valores nulos da máscara, para maximizar o contraste do terreno.
*   **Geração de Falsa-Cor:** Cálculo de todas as 24 permutações possíveis agrupando as 4 bandas em trios para destacar feições não visíveis no espectro RGB padrão.
*   **Otimização e Redimensionamento:** Redimensionamento inteligente utilizando o filtro `LANCZOS` (largura máxima de 1920 pixels) mantendo a proporção geométrica original.
*   **Exportação:** Salvamento automático das imagens no formato PNG com compressão otimizada (arquivos < 1MB).

---

## 🛠️ Configuração do Ambiente de Desenvolvimento (Local)

As instruções abaixo descrevem como configurar o ambiente local de desenvolvimento, focado em sistemas operacionais **Windows**.

### 1. Criando e Ativando o Ambiente Virtual (venv)

Abra o terminal na raiz do projeto e crie o ambiente virtual executando:

```powershell
python -m venv venv

```

Para ativar o ambiente virtual:

```powershell
.\venv\Scripts\Activate.ps1

```

*(Nota: Caso o PowerShell bloqueie a execução de scripts, rode `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` uma única vez antes de ativar novamente).*

### 2. Instalando Dependências

Com o ambiente ativado (você verá `(venv)` no terminal), instale as bibliotecas necessárias:

```powershell
pip install -r requirements.txt

```

*(Para atualizar o arquivo com novos pacotes instalados durante o desenvolvimento, utilize `pip freeze > requirements.txt`).*

### 3. Configurando as Variáveis de Ambiente

Utilizamos a biblioteca `python-dotenv` para gerenciar caminhos de diretórios sem fixá-los (*hardcode*) no código-fonte.

Crie um arquivo chamado `.env` na raiz do projeto e defina a variável `PROJECT_ROOT` com o caminho absoluto do seu espaço de trabalho local:

```env
PROJECT_ROOT=C:\Users\NomeDoUsuario\caminho\para\urban-deforestation-monitoring\workspace

```

---

## ☁️ Execução no Google Colab

Este repositório foi projetado para facilitar a migração e execução na nuvem. Ao migrar para o Google Colab:

1. **Não** faça o upload do arquivo `.env`.
2. Faça o upload dos scripts `.py` e do arquivo `requirements.txt`.
3. Em uma célula inicial, instale as dependências:
```python
!pip install -r requirements.txt

```


4. Configure o caminho do seu Google Drive utilizando a ferramenta **Secrets** do Colab (ícone de chave no menu lateral). Crie um *secret* chamado `PROJECT_ROOT` com o caminho base (ex: `/content/drive/MyDrive/...`).
5. O acesso à variável no código continuará funcionando via `os.getenv("PROJECT_ROOT")`, garantindo compatibilidade entre execução local e em nuvem.

---

## 🚀 Como Executar

Com o ambiente devidamente configurado e o arquivo `.env` preenchido, execute o *script* principal de pré-processamento.

*(Substitua `main.py` pelo nome exato do seu script principal, caso seja diferente)*

```powershell
python main.py

```

Os arquivos processados serão gerados na pasta de saída (ex: `/output/`) configurada no *script*, prontos para serem submetidos à plataforma Zooniverse.

---

## 📂 Estrutura de Diretórios (Exemplo)

```text
urban-deforestation-monitoring/
│
├── .env                  # Variáveis de ambiente (NÃO commitar)
├── .gitignore            # Arquivos ignorados pelo Git
├── requirements.txt      # Dependências do projeto
├── README.md             # Documentação do projeto
├── main.py               # Script principal de pré-processamento
│
├── workspace/            # Definido pelo PROJECT_ROOT
│   ├── data/             # Imagens brutas (.tif)
│   └── output/           # Imagens processadas (.png) geradas pelo script

```

```

```