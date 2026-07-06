#!/usr/bin/env python3
import subprocess
import time
import os
import sys
import re

# --- CONFIGURAÇÃO DO AMBIENTE ---
LLAMA_CLI_PATH = "../build/bin/llama-cli"
OPTANE_DIR = "/mnt/nvram0" 
RESULT_CSV = "resultados_llama3.csv"

N_RUNS = 10

if not os.path.exists(OPTANE_DIR):
    try:
        os.makedirs(OPTANE_DIR)
    except Exception as e:
        print(f"Aviso: Não foi possível criar {OPTANE_DIR}. Erro: {e}")

# ==========================================
# 1. OS 5 CENÁRIOS DE WORKLOAD
# ==========================================
SCENARIOS = [
    {"id": "C1_Sumarizacao", "n_prompt": 4096, "n_gen": 128, "n_ctx": 8192},
    {"id": "C2_Geracao",     "n_prompt": 128,  "n_gen": 4096, "n_ctx": 8192},
    {"id": "C3_Conversacao", "n_prompt": 1024, "n_gen": 1024, "n_ctx": 4096},
    {"id": "C4_Estresse",    "n_prompt": 7000, "n_gen": 1000, "n_ctx": 8192},
    {"id": "C5_Controle",    "n_prompt": 64,   "n_gen": 64,   "n_ctx": 512}
]

# ==========================================
# 2. AS 8 CONFIGURAÇÕES ARQUITETURAIS
# ==========================================
EXPERIMENTS = [
    {"name": "00_Baseline", "placement": "", "kv_optane": False},
    {"name": "01_KV", "placement": "", "kv_optane": True},
    {"name": "02_Attn", "placement": "0-31.attn:optane", "kv_optane": False},
    {"name": "03_FFN", "placement": "0-31.ffn:optane", "kv_optane": False},
    {"name": "04_KV_Attn", "placement": "0-31.attn:optane", "kv_optane": True},
    {"name": "05_KV_FFN", "placement": "0-31.ffn:optane", "kv_optane": True},
    {"name": "06_Attn_FFN", "placement": "0-31.attn:optane,0-31.ffn:optane", "kv_optane": False},
    {"name": "07_Tudo", "placement": "0-31:optane,embed:optane,output:optane", "kv_optane": True}
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

def run_experiment(model_src_path, exp_config, scenario, prompt_file, run_number):
    mode_name = exp_config["name"]
    placement_rule = exp_config["placement"]
    kv_in_optane = exp_config["kv_optane"]
    
    cmd_args = [
        "numactl", "--cpunodebind=0", "--membind=0", # Isolamento NUMA (CPU 0 e RAM 0)
        LLAMA_CLI_PATH,
        "-m", model_src_path,
        "-f", prompt_file,
        "-n", str(scenario["n_gen"]),
        "-c", str(scenario["n_ctx"]),
        "--temp", "0.0",
        "--ignore-eos",
        "--batch-size", "512",
        "-no-cnv",
        "-s", "0",              # Seed fixa
        "-t", "12",             # Threads restritas aos 12 cores físicos do Node 0 (Decode)
        "-tb", "12"             # Threads restritas aos 12 cores físicos do Node 0 (Prefill)
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
    
    log_filename = f"log_{scenario['id']}_{mode_name}_run{run_number}.txt"
    print(f"    Executando: {mode_name} | {scenario['id']} (Run {run_number}/{N_RUNS})")
    
    start_time = time.time()
    
    with open(log_filename, "w") as outfile:
        try:
            subprocess.run(cmd_args, stdout=outfile, stderr=outfile, text=True, check=False)
        except subprocess.CalledProcessError:
            print(f"!!! Erro na execução de {mode_name}. Verifique {log_filename}")
            return
        except KeyboardInterrupt:
            print("\n!!! Interrompido pelo utilizador.")
            sys.exit(1)

    total_time = time.time() - start_time
    metrics = extract_metrics_from_file(log_filename)
    
    # Gravando no CSV
    file_exists = os.path.isfile(RESULT_CSV)
    with open(RESULT_CSV, "a") as f:
        if not file_exists:
            f.write("Scenario,N_Prompt,N_Gen,Mode,Run,Total_Time_s,Prefill_TPS,Decode_TPS\n")
        f.write(f"{scenario['id']},{scenario['n_prompt']},{scenario['n_gen']},{mode_name},{run_number},{total_time:.2f},{metrics['prefill_tps']},{metrics['decode_tps']}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: sudo ./measure-llm.py <caminho/do/modelo.gguf>")
        sys.exit(1)

    model = sys.argv[1]
    prompt_file = "prompt_dinamico.txt"
    
    total_executions = len(SCENARIOS) * len(EXPERIMENTS) * N_RUNS
    print(f"=== INICIANDO BATERIA MASSIVA NODE05 ===")
    print(f"Total de execuções planejadas: {total_executions}")
    
    execution_count = 1
    
    # Loop 1: Cenários
    for scenario in SCENARIOS:
        print(f"\n==================================================")
        print(f"Iniciando Cenário: {scenario['id']} (Prompt: {scenario['n_prompt']}, Gen: {scenario['n_gen']})")
        print(f"==================================================")
        
        # Gerar o prompt específico para este cenário
        with open(prompt_file, "w") as f:
            f.write("sun " * scenario["n_prompt"])
            
        # Loop 2: Rodadas de repetibilidade
        for run in range(1, N_RUNS + 1):
            print(f"\n  --- Rodada {run}/{N_RUNS} ---")
            
            # Loop 3: Configurações de Tiering
            for exp in EXPERIMENTS:
                print(f"[{execution_count}/{total_executions}]", end=" ")
                run_experiment(model, exp, scenario, prompt_file, run)
                execution_count += 1

    if os.path.exists(prompt_file):
        os.remove(prompt_file)

    print(f"\n=== FIM DOS TESTES. Dados guardados em {RESULT_CSV} ===")