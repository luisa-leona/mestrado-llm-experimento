#!/usr/bin/env python3
import subprocess
import re
import csv
import os

# ==========================================
# CONFIGURAÇÕES DO EXPERIMENTO
# ==========================================
LLAMA_CLI_PATH = "./llama-cli" 
MODELO = "models/meta-llama-3-8b.Q4_K_M.gguf" # Usaremos o LLaMA 3
ARQUIVO_SAIDA = "resultado_contexto_DRAM.csv" # MUDE PARA _OPTANE na 2ª rodada!

# Os tamanhos de Prompt que vão encher o KV-Cache gradualmente
TAMANHOS_PROMPT = [512, 1024, 2048, 4096]
TOKENS_DECODE = 128  # Quantidade fixa de geração para o Decode ser justo

# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================
def criar_arquivo_prompt(qtd_tokens):
    """Cria um arquivo de texto temporário com o tamanho exato de palavras."""
    # A palavra "sun" com espaço costuma ser mapeada como 1 token exato.
    texto = "sun " * qtd_tokens
    with open("prompt_temp.txt", "w") as f:
        f.write(texto)

# ==========================================
# FUNÇÃO PRINCIPAL
# ==========================================
def main():
    if not os.path.exists(LLAMA_CLI_PATH):
        print(f"❌ Erro: {LLAMA_CLI_PATH} não encontrado.")
        return

    resultados = []
    print(f"🚀 Iniciando Mapeamento de Sensibilidade de Contexto...\n")
    print(f"Modelo: {os.path.basename(MODELO)}")
    print(f"Salvando em: {ARQUIVO_SAIDA}\n")

    for tamanho in TAMANHOS_PROMPT:
        print(f"⏳ Testando KV-Cache com {tamanho} tokens...", end="", flush=True)
        
        # Cria o prompt do tamanho específico da iteração
        criar_arquivo_prompt(tamanho)
        
        # Monta o comando. Usamos -c 8192 para garantir que o contexto nunca estoure
        comando = [
            LLAMA_CLI_PATH,
            "-m", MODELO,
            "-c", "8192",
            "-f", "prompt_temp.txt", # Lê o nosso arquivo gerado
            "-n", str(TOKENS_DECODE), # Gera 128 tokens
            "--temp", "0.0"          # Temperatura 0 para ser determinístico
        ]

        try:
            processo = subprocess.run(comando, capture_output=True, text=True, check=False)
            saida = processo.stdout + processo.stderr
            
            # Extrai o TPS (Tokens Por Segundo) do Prefill (prompt eval time)
            match_prefill = re.search(r"prompt eval time = .*? \(\s*[0-9.]+\s*ms per token,\s*([0-9.]+)\s*tokens per second\)", saida)
            # Extrai o TPS do Decode (eval time)
            match_decode = re.search(r"eval time = .*? \(\s*[0-9.]+\s*ms per token,\s*([0-9.]+)\s*tokens per second\)", saida)
            
            if match_prefill and match_decode:
                prefill_tps = float(match_prefill.group(1))
                decode_tps = float(match_decode.group(1))
                print(f" [OK] -> Prefill: {prefill_tps:.2f} T/s | Decode: {decode_tps:.2f} T/s")
                
                resultados.append({
                    "Tamanho_Contexto": tamanho,
                    "Prefill_TPS": prefill_tps,
                    "Decode_TPS": decode_tps
                })
            else:
                print(" [FALHOU] Não encontrou as métricas no log.")
                
        except Exception as e:
            print(f" [ERRO: {e}]")

    # Limpa o arquivo temporário
    if os.path.exists("prompt_temp.txt"):
        os.remove("prompt_temp.txt")

    # Salva no CSV
    if resultados:
        with open(ARQUIVO_SAIDA, mode='w', newline='') as csvfile:
            campos = ["Tamanho_Contexto", "Prefill_TPS", "Decode_TPS"]
            writer = csv.DictWriter(csvfile, fieldnames=campos)
            writer.writeheader()
            writer.writerows(resultados)
        print(f"\n✅ Concluído! Resultados salvos em {ARQUIVO_SAIDA}")

if __name__ == "__main__":
    main()