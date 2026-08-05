import sys
import os
import subprocess
from dotenv import load_dotenv

def executar_pipeline():
    # Verifica se os argumentos mínimos foram passados (Script + Município + Pelo menos 1 Ano)
    if len(sys.argv) < 3:
        print("Erro: Parâmetros insuficientes.")
        print("Uso correto: python 06-ProcessamentoDeDadosParaAnalise.py <CODE_MUNI> <ANO_1> [ANO_2] [ANO_3] ...")
        print("Exemplo: python 06-ProcessamentoDeDadosParaAnalise.py 3549904 2023 2024")
        sys.exit(1)

    code_muni = sys.argv[1]
    anos_para_processar = sys.argv[2:]

    load_dotenv()
    project_root = os.environ.get("PROJECT_ROOT")
    
    if not project_root:
        # Tenta inferir o diretório caso não esteja no .env
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        print(f"Aviso: PROJECT_ROOT não encontrado. Inferindo como: {project_root}")

    # Força a buscar os scripts na MESMA pasta onde este orquestrador (Script 06) está salvo
    diretorio_scripts = os.path.dirname(os.path.abspath(__file__))

    # Lista exata dos scripts em ordem de execução
    scripts_da_esteira = [
        "01-downloadMapaBiomas.py",
        "02-downloadProcessarCBERS.py",
        "03-recortarCBERS.py",
        "04-gerarMultibandas.py",
        "05-gerarClassificacaoMapBiomas.py"
    ]

    print(f"\n{'='*70}")
    print(f"🚀 INICIANDO ORQUESTRAÇÃO DE DADOS URBANOS")
    print(f"📍 MUNICÍPIO: {code_muni}")
    print(f"📅 ANOS NA FILA: {', '.join(anos_para_processar)}")
    print(f"{'='*70}\n")

    for ano in anos_para_processar:
        print(f"\n{'#'*70}")
        print(f"▶ PROCESSANDO CICLO: ANO {ano}")
        print(f"{'#'*70}")

        for nome_script in scripts_da_esteira:
            caminho_script = os.path.join(diretorio_scripts, nome_script)
            
            if not os.path.exists(caminho_script):
                print(f"\n[ERRO CRÍTICO] Script não encontrado: {caminho_script}")
                sys.exit(1)

            print(f"\n---> Executando Etapa: {nome_script}")
            
            # Monta o comando usando sys.executable para garantir que use o mesmo Python do Colab/Venv
            comando = [sys.executable, caminho_script, str(code_muni), str(ano)]
            
            try:
                # O parâmetro check=True faz o Python levantar uma exceção se o script filho falhar
                subprocess.run(comando, check=True)
            except subprocess.CalledProcessError as e:
                print(f"\n[FALHA] A execução do {nome_script} falhou e retornou código de erro {e.returncode}.")
                print(f"Interrompendo a esteira para o ano {ano} para evitar dados corrompidos.")
                break # Quebra o loop dos scripts e vai para o próximo ano (se houver)
        else:
            # Esse else pertence ao for (só executa se o loop não for interrompido por um break)
            print(f"\n{'='*70}")
            print(f"✅ CICLO {ano} FINALIZADO COM SUCESSO EM TODAS AS 5 ETAPAS!")
            print(f"{'='*70}")

    print("\n🎉 ORQUESTRAÇÃO FINALIZADA. Todos os anos da fila foram processados!")

if __name__ == "__main__":
    executar_pipeline()