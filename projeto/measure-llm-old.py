#!/usr/bin/env python3
import subprocess
import time
import os
import sys
import re

# --- CONFIGURAÇÃO DO AMBIENTE ---
LLAMA_CLI_PATH = "../build/bin/llama-cli"
OPTANE_DIR = "/mnt/nvram0" 
RESULT_CSV = "resultados.csv"

N_RUNS = 10
N_PREDICT = 128 # Atualizado para 128 (potência de 2, padrão em artigos)
N_CTX = 2048    # Subi para 2048 para caber o prompt com folga
TAMANHO_PROMPT = 512 # Palavras exatas para o Prefill

if not os.path.exists(OPTANE_DIR):
    try:
        os.makedirs(OPTANE_DIR)
    except Exception as e:
        print(f"Aviso: Não foi possível criar {OPTANE_DIR}. Erro: {e}")

# ==========================================
# MATRIZ FATORIAL DE EXPERIÊNCIAS (8 Variações)
# Cabeçalho: KV-Cache | Atenção | FFN/MLP
# ==========================================
EXPERIMENTS = [
    # BASELINE: Tudo na DRAM (Referência)
    {"name": "00_Baseline", "placement": "", "kv_optane": False},
    
    # A: Optane | DRAM | DRAM (Testa se KV-Cache é latency-sensitive)
    {"name": "01_Config_A_KV", "placement": "", "kv_optane": True},
    
    # B: DRAM | Optane | DRAM (Testa se a Atenção tolera a latência da Optane)
    {"name": "02_Config_B_Attn", "placement": "0-31.attn:optane", "kv_optane": False},
    
    # C: DRAM | DRAM | Optane (A nossa hipótese principal: FFN isolada na Optane)
    {"name": "03_Config_C_FFN", "placement": "0-31.ffn:optane", "kv_optane": False},
    
    # D: DRAM | Optane | Optane (Só o KV-Cache fica protegido na DRAM)
    # Colocamos ATTN e FFN na Optane combinando as strings
    {"name": "04_Config_D_Attn_FFN", "placement": "0-31.attn:optane,0-31.ffn:optane", "kv_optane": False},
    
    # E: Optane | Optane | Optane (Pior caso absoluto: modelo inteiro e contexto na Optane)
    {"name": "05_Config_E_Tudo", "placement": "0-31:optane,embed:optane,output:optane", "kv_optane": True},
    
    # F: Optane | Optane | DRAM (Controlo invertido: FFN gigante na DRAM, o resto na Optane)
    {"name": "06_Config_F_KV_Attn", "placement": "0-31.attn:optane", "kv_optane": True},
    
    # G: Optane | DRAM | Optane (Protege só a Atenção na DRAM, o resto na Optane)
    {"name": "07_Config_G_KV_FFN", "placement": "0-31.ffn:optane", "kv_optane": True}
]

def drop_linux_caches():
    try:
        subprocess.run(["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"], 
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        pass

def extract_metrics_from_file(log_path):
    if not os.path.exists(log_path):
        return {"prefill_tps": "Erro", "decode_tps": "Erro"}

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    metrics = {"prefill_tps": "N/A", "decode_tps": "N/A"}
    p_match = re.search(r"prompt eval time.*?=\s*.*?(\d+[.,]\d+)\s*tokens per second", content, re.DOTALL)
    if p_match: metrics["prefill_tps"] = p_match.group(1).replace(',', '.')

    d_match = re.search(r"(?<!prompt )eval time.*?=\s*.*?(\d+[.,]\d+)\s*tokens per second", content, re.DOTALL)
    if d_match: metrics["decode_tps"] = d_match.group(1).replace(',', '.')
        
    return metrics

def run_experiment(model_src_path, exp_config, prompt_file, run_number):
    mode_name = exp_config["name"]
    placement_rule = exp_config["placement"]
    kv_in_optane = exp_config["kv_optane"]
    
    cmd_args = [
        LLAMA_CLI_PATH,
        "-m", model_src_path,
        "-f", prompt_file,
        "-n", str(N_PREDICT),
        "-c", str(N_CTX),
        "--temp", "0",
        "--ignore-eos",
        "--batch-size", "512",
        "-no-cnv"
    ]

    # Configuração dos Pesos
    if placement_rule != "":
        optane_weights_file = os.path.join(OPTANE_DIR, f"weights_{mode_name}_run{run_number}.bin")
        cmd_args.extend([
            "--optane-url", optane_weights_file,
            "--optane-placement", placement_rule
        ])

    # Configuração do KV Cache
    if kv_in_optane:
        kv_file_path = os.path.join(OPTANE_DIR, f"kv_{mode_name}_run{run_number}.bin")
        if os.path.exists(kv_file_path):
            os.remove(kv_file_path)
        cmd_args.extend(["--kv-cache-mmap", kv_file_path])

    drop_linux_caches()
    
    log_filename = f"log_{mode_name}_run{run_number}.txt"
    print(f"  [{run_number}/{N_RUNS}] Executando: {mode_name}...")
    
    start_time = time.time()
    
    with open(log_filename, "w") as outfile:
        try:
            subprocess.run(cmd_args, stdout=outfile, stderr=outfile, text=True, check=False)
        except subprocess.CalledProcessError:
            print(f"!!! Erro na execução de {mode_name}. Verifique {log_filename}")
            return
        except KeyboardInterrupt:
            print("\n!!! Interrompido pelo usuário.")
            sys.exit(1)

    total_time = time.time() - start_time
    metrics = extract_metrics_from_file(log_filename)
    
    file_exists = os.path.isfile(RESULT_CSV)
    with open(RESULT_CSV, "a") as f:
        if not file_exists:
            f.write("Mode,Run,Total_Time_s,Prefill_TPS,Decode_TPS\n")
        f.write(f"{mode_name},{run_number},{total_time:.2f},{metrics['prefill_tps']},{metrics['decode_tps']}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: sudo ./measure-llm.py <caminho/do/modelo.gguf>")
        sys.exit(1)

    model = sys.argv[1]
    
    # Criando arquivo temporário de texto para o prompt ser fixo
    prompt_file = "prompt_padrao.txt"
    with open(prompt_file, "w") as f:
        f.write("sun " * TAMANHO_PROMPT)
    
    print(f"=== INICIANDO BATERIA NODE05 ({N_RUNS} REPETIÇÕES) ===")
    
    for run in range(1, N_RUNS + 1):
        print(f"\n--- INICIANDO RODADA {run} ---")
        for exp in EXPERIMENTS:
            run_experiment(model, exp, prompt_file, run)

    # Limpa o arquivo de prompt
    if os.path.exists(prompt_file):
        os.remove(prompt_file)

    print(f"\n=== FIM DOS TESTES. Dados salvos em {RESULT_CSV} ===")