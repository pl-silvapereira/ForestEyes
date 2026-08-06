import os
import sys
import subprocess

def main():
    # 1. Validar se o município e o ano foram passados por argumento
    if len(sys.argv) < 3:
        print("❌ Erro: Parâmetros insuficientes.")
        print("Uso correto: python 06-ProcessamentoDeDadosParaAnalise.py <code_muni> <ano>")
        print("Exemplo: python 06-ProcessamentoDeDadosParaAnalise.py 3549904 2023")
        sys.exit(1)

    code_muni = sys.argv[1]
    ano = sys.argv[2]

    # 2. AUTO-DESCOBERTA DO PROJECT_ROOT (Bypass Seguro)
    # Descobre o caminho dinamicamente baseado na localização física deste arquivo.
    # Como o script está em ".../workspace/scripts/", ele sempre achará o "/workspace" correto.
    diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(diretorio_scripts) 

    # 3. Criação automática de todos os diretórios e subpastas no Google Drive
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

    print("=" * 70)
    print(f"🚀 INICIANDO ORQUESTRAÇÃO DE DADOS URBANOS")
    print(f"📍 MUNICÍPIO: {code_muni} | 📅 ANO DE ANÁLISE: {ano}")
    print(f"📁 WORKSPACE: {project_root}")
    print("=" * 70)

    print("Verificando e criando diretórios de trabalho automaticamente...")
    for diretorio in diretorios_necessarios:
        os.makedirs(diretorio, exist_ok=True)

    # Lista dos 5 scripts na ordem correta de execução
    scripts = [
        "01-downloadMapaBiomas.py",
        "02-downloadProcessarCBERS.py",
        "03-recortarCBERS.py",
        "04-gerarMultibandas.py",
        "05-gerarClassificacaoMapBiomas.py"
    ]

    # 4. Injetar o ano e o ROOT exato no ambiente virtual da execução
    env = os.environ.copy()
    env['ANO'] = str(ano)
    env['PROJECT_ROOT'] = project_root # <--- Força os scripts 01 a 05 a usarem a pasta correta

    # 5. Execução sequencial robusta
    for script in scripts:
        caminho_script = os.path.join(diretorio_scripts, script)
        
        if not os.path.exists(caminho_script):
            print(f"❌ [ERRO CRÍTICO] Script não encontrado: {caminho_script}")
            sys.exit(1)

        print(f"\n---> Executando Etapa: {script}")
        
        # Passa explicitamente o code_muni e o ano como argumentos para cada script
        comando = [sys.executable, caminho_script, str(code_muni), str(ano)]
        
        # O parâmetro env repassa o PROJECT_ROOT forçado para os subprocessos
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