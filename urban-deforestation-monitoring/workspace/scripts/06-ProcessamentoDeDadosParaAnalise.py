import os
import sys
import subprocess
from dotenv import load_dotenv

def main():
    # 1. Validar se o município e o ano foram passados por argumento
    # Uso correto: python 06-ProcessamentoDeDadosParaAnalise.py <code_muni> <ano>
    if len(sys.argv) < 3:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 06-ProcessamentoDeDadosParaAnalise.py <code_muni> <ano>")
        print("Exemplo: python 06-ProcessamentoDeDadosParaAnalise.py 3549904 2023")
        sys.exit(1)

    code_muni = sys.argv[1]
    ano = sys.argv[2]

    # 2. Carregar variáveis de ambiente
    load_dotenv()
    project_root = os.getenv('PROJECT_ROOT')

    if not project_root:
        raise ValueError("A variável PROJECT_ROOT não foi encontrada no arquivo .env.")

    # 3. Criação automática de todos os diretórios e subpastas no Google Drive se não existirem
    reports_dir = os.path.join(project_root, "reports")
    data_input_mapbiomas = os.path.join(project_root, "data", "input", "MapBiomas", str(ano))
    data_input_cbers = os.path.join(project_root, "data", "input", "CBERS-4A-WPM", str(ano))
    data_output_cbers = os.path.join(project_root, "data", "output", "pansharpening", str(ano))
    geopolitic_rgbn = os.path.join(project_root, "data", "output", "pansharpening", "geopolitic-RGBN", str(ano))
    multibands_dir = os.path.join(project_root, "data", "output", "pansharpening", "multispectral-RGBN-bands", str(ano))
    classification_dir = os.path.join(project_root, "data", "output", "classification", str(ano))

    diretorios_necessarios = [
        reports_dir, 
        data_input_mapbiomas, 
        data_input_cbers, 
        data_output_cbers, 
        geopolitic_rgbn, 
        multibands_dir, 
        classification_dir
    ]

    print("Verificando e criando diretórios de trabalho automaticamente...")
    for diretorio in diretorios_necessarios:
        os.makedirs(diretorio, exist_ok=True)

    print("=" * 70)
    print(f"🚀 INICIANDO ORQUESTRAÇÃO DE DADOS URBANOS")
    print(f"📍 MUNICÍPIO: {code_muni} | 📅 ANO DE ANÁLISE: {ano}")
    print("=" * 70)

    # 4. Localizar o diretório onde os scripts estão salvos (mesma pasta do script 06)
    diretorio_scripts = os.path.dirname(os.path.abspath(__file__))

    # Lista dos 5 scripts na ordem correta de execução
    scripts = [
        "01-downloadMapaBiomas.py",
        "02-downloadProcessarCBERS.py",
        "03-recortarCBERS.py",
        "04-gerarMultibandas.py",
        "05-gerarClassificacaoMapBiomas.py"
    ]

    # Injetar o ano nas variáveis de ambiente locais para os scripts filhos
    env = os.environ.copy()
    env['ANO'] = str(ano)

    # 5. Execução sequencial robusta
    for script in scripts:
        caminho_script = os.path.join(diretorio_scripts, script)
        
        if not os.path.exists(caminho_script):
            print(f"❌ [ERRO CRÍTICO] Script não encontrado: {caminho_script}")
            sys.exit(1)

        print(f"\n---> Executando Etapa: {script}")
        
        # Passa explicitamente o code_muni e o ano como argumentos para cada script
        comando = [sys.executable, caminho_script, str(code_muni), str(ano)]
        
        resultado = subprocess.run(comando, env=env)

        if resultado.returncode != 0:
            print(f"\n[FALHA] A execução do {script} falhou e retornou código de erro {resultado.returncode}.")
            print(f"Interrompendo a esteira para o ano {ano} para evitar dados corrompidos.")
            sys.exit(1)

        print(f"✓ {script} concluído com sucesso.")

    print("\n" + "=" * 70)
    print(f"✅ CICLO COMPLETO DO ANO {ano} FINALIZADO COM SUCESSO!")
    print("=" * 70)

if __name__ == "__main__":
    main()