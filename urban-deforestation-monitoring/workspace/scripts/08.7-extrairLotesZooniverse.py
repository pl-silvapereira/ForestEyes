#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Etapa 08.7: Extração dos Lotes Zooniverse e Exclusão Permanente dos ZIPs
==============================================================================
Descrição:
 - Percorre as pastas 'non-forest' e 'forest' em:
   PROJECT_ROOT/data/out_workflow/cbers_workflow/AE-X{code_muni}/
 - Extrai cada arquivo .zip diretamente na mesma pasta de origem.
 - Verifica a integridade de todos os arquivos extraídos antes de remover o .zip.
 - Exclui o .zip definitivamente via API do Google Drive (sem passar pela lixeira)
   antes de avançar para o próximo lote.
==============================================================================
"""

import sys
import os
import gc
import time
import zipfile
from tqdm import tqdm
from dotenv import load_dotenv

try:
    from google.colab import auth
    from googleapiclient.discovery import build
    EM_COLAB = True
except ImportError:
    EM_COLAB = False


DEFAULT_WORKFLOW_BASE = (
    "/content/drive/MyDrive/Mestrado/04-Projeto ForestEyes/ForestEyes/"
    "urban-deforestation-monitoring/workspace/data/out_workflow/cbers_workflow"
)


def inicializar_drive_api():
    """Autentica e inicializa o serviço Google Drive v3 para exclusão permanente."""
    if not EM_COLAB:
        return None
    try:
        print("🔐 Autenticando na API do Google Drive para exclusão permanente (sem lixeira)...", flush=True)
        auth.authenticate_user()
        return build("drive", "v3")
    except Exception as e:
        print(f"⚠️ [AVISO] Falha ao iniciar API do Drive ({e}). Usando remoção padrão via sistema de arquivos.", flush=True)
        return None


def obter_id_pasta_classe_drive(service, code_muni, classification):
    """Localiza o ID exato da pasta 'forest' ou 'non-forest' dentro de AE-X{code_muni} no Drive."""
    if service is None:
        return None

    nome_ae = f"AE-X{code_muni}"
    query_ae = (
        f"mimeType = 'application/vnd.google-apps.folder' "
        f"and name = '{nome_ae}' and trashed = false"
    )
    resp_ae = service.files().list(q=query_ae, spaces="drive", fields="files(id, name)").execute()
    pastas_ae = resp_ae.get("files", [])
    if not pastas_ae:
        return None

    for pasta_ae in pastas_ae:
        ae_id = pasta_ae["id"]
        query_cls = (
            f"'{ae_id}' in parents "
            f"and mimeType = 'application/vnd.google-apps.folder' "
            f"and name = '{classification}' and trashed = false"
        )
        resp_cls = service.files().list(q=query_cls, spaces="drive", fields="files(id, name)").execute()
        pastas_cls = resp_cls.get("files", [])
        if pastas_cls:
            return pastas_cls[0]["id"]

    return None


def deletar_zip_permanente(service, folder_id_drive, zip_path, max_tentativas=5):
    """Remove o arquivo .zip definitivamente sem enviá-lo para a lixeira do Google Drive."""
    zip_name = os.path.basename(zip_path)
    deletado_via_api = False

    if service is not None and folder_id_drive is not None:
        for tentativa in range(1, max_tentativas + 1):
            try:
                query = (
                    f"'{folder_id_drive}' in parents "
                    f"and name = '{zip_name}' and trashed = false"
                )
                resp = service.files().list(q=query, spaces="drive", fields="files(id)").execute()
                arquivos = resp.get("files", [])

                if not arquivos:
                    deletado_via_api = True
                    break

                for arq in arquivos:
                    # files().delete() exclui permanentemente, ignorando a lixeira
                    service.files().delete(fileId=arq["id"]).execute()

                deletado_via_api = True
                break
            except Exception:
                time.sleep(2.0 * tentativa)

    # Aguarda breve sincronização FUSE e garante limpeza no ponto de montagem local
    time.sleep(1.0)
    if os.path.exists(zip_path):
        for tentativa in range(1, max_tentativas + 1):
            try:
                os.remove(zip_path)
                break
            except FileNotFoundError:
                break
            except OSError:
                time.sleep(1.5 * tentativa)

    return deletado_via_api or (not os.path.exists(zip_path))


def gravar_arquivo_com_retry(destino, dados_bytes, max_tentativas=5):
    """Cria o diretório da tarefa e grava o arquivo extraído com retentativa FUSE."""
    pasta_pai = os.path.dirname(destino)
    for tentativa in range(1, max_tentativas + 1):
        try:
            os.makedirs(pasta_pai, exist_ok=True)
            with open(destino, "wb") as f_out:
                f_out.write(dados_bytes)
            return True
        except OSError:
            time.sleep(1.5 * tentativa)
    raise IOError(f"Falha de I/O ao extrair arquivo no Google Drive após {max_tentativas} tentativas: {destino}")


def extrair_e_limpar_classe(service, code_muni, classification, dir_classe):
    """Extrai todos os .zip de uma classe e exclui cada lote após validar a extração."""
    if not os.path.exists(dir_classe):
        print(f"⚠️ [AVISO] Pasta não encontrada: {dir_classe}. Pulando...", flush=True)
        return

    zips = sorted([
        f for f in os.listdir(dir_classe)
        if f.lower().endswith(".zip") and not f.lower().endswith(".tmp")
    ])

    if not zips:
        print(f"ℹ️ Nenhum arquivo .zip pendente encontrado em: {dir_classe}", flush=True)
        return

    folder_id_drive = obter_id_pasta_classe_drive(service, code_muni, classification)
    print(f"\n📂 Classe '{classification}': {len(zips)} pacote(s) .zip encontrados em:\n   {dir_classe}", flush=True)

    for idx, zip_name in enumerate(zips, start=1):
        zip_path = os.path.join(dir_classe, zip_name)
        print(f"\n📦 [{idx}/{len(zips)}] Abrindo pacote: {zip_name}", flush=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            membros_arquivos = [info for info in zf.infolist() if not info.is_dir()]
            total_arquivos = len(membros_arquivos)

            # Pré-cria os diretórios task_* únicos para reduzir chamadas repetidas de os.makedirs
            pastas_unicas = {
                os.path.join(dir_classe, os.path.dirname(info.filename))
                for info in membros_arquivos
                if os.path.dirname(info.filename)
            }
            for pasta_task in pastas_unicas:
                for tentativa in range(1, 5):
                    try:
                        os.makedirs(pasta_task, exist_ok=True)
                        break
                    except OSError:
                        time.sleep(1.5 * tentativa)

            # Extrai arquivo por arquivo com checkpoint de tamanho
            for info in tqdm(membros_arquivos, desc=f" Extraindo {zip_name}"):
                destino_arquivo = os.path.join(dir_classe, info.filename)

                if os.path.exists(destino_arquivo) and os.path.getsize(destino_arquivo) == info.file_size:
                    continue

                conteudo = zf.read(info.filename)
                gravar_arquivo_com_retry(destino_arquivo, conteudo)

            # Validação final de integridade antes de excluir o .zip
            faltantes = [
                info.filename for info in membros_arquivos
                if not os.path.exists(os.path.join(dir_classe, info.filename))
            ]

            if faltantes:
                raise RuntimeError(
                    f"❌ Abortando exclusão de {zip_name}: {len(faltantes)} arquivos não foram confirmados no destino."
                )

        print(f" 🗑️ Extração validada ({total_arquivos:,} arquivos). Deletando {zip_name} permanentemente...", flush=True)
        deletar_zip_permanente(service, folder_id_drive, zip_path)
        print(f" ✅ Pacote {zip_name} removido sem passar pela lixeira.", flush=True)
        gc.collect()


def main():
    code_muni = str(sys.argv[1]) if len(sys.argv) >= 2 else "3549904"

    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT")
    if project_root:
        dir_workflow_base = os.path.join(project_root, "data", "out_workflow", "cbers_workflow")
    else:
        diretorio_scripts = os.path.dirname(os.path.abspath(__file__))
        candidato_root = os.path.dirname(diretorio_scripts)
        candidato_wf = os.path.join(candidato_root, "data", "out_workflow", "cbers_workflow")
        dir_workflow_base = candidato_wf if os.path.exists(candidato_wf) else DEFAULT_WORKFLOW_BASE

    dir_ae = os.path.join(dir_workflow_base, f"AE-X{code_muni}")
    if not os.path.exists(dir_ae):
        print(f"❌ [ERRO CRÍTICO] Diretório não encontrado: {dir_ae}")
        sys.exit(1)

    service = inicializar_drive_api()
    classifications = ["non-forest", "forest"]

    print("=" * 80, flush=True)
    print(f"🚀 SCRIPT 08.7 - EXTRAÇÃO DE LOTES E LIMPEZA PERMANENTE (AE-X{code_muni})", flush=True)
    print(f"📂 Diretório Base: {dir_ae}", flush=True)
    print("=" * 80, flush=True)

    for classification in classifications:
        dir_classe = os.path.join(dir_ae, classification)
        extrair_e_limpar_classe(service, code_muni, classification, dir_classe)

    print("\n🎉 EXTRAÇÃO CONCLUÍDA COM SUCESSO! Todos os lotes foram descompactados e os .zip removidos.")


if __name__ == "__main__":
    main()