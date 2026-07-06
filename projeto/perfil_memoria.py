#!/usr/bin/env python3
import subprocess
import re
import csv
import os
import sys

# Tenta importar a biblioteca gguf
try:
    from gguf import GGUFReader
except ImportError:
    print("❌ Erro: A biblioteca 'gguf' não foi encontrada.")
    print("Execute no terminal: pip install gguf")
    sys.exit(1)

# ==========================================
# CONFIGURAÇÕES DO EXPERIMENTO
# ==========================================
LLAMA_CLI_PATH = "../build/bin/llama-cli" 

MODELOS = [
    "../models/llama-2-7b.Q4_K_M.gguf",
    "../models/meta-llama-3-8b.Q4_K_M.gguf"
]

CONTEXTOS = [512, 1024, 2048, 4096, 8192]
ARQUIVO_SAIDA = "tamanhos_memoria.csv"

# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================
def obter_tamanhos_estaticos(modelo_path):
    """Lê o arquivo .gguf e soma o tamanho exato em bytes das matrizes ATTN e FFN."""
    print("  -> Extraindo pesos (FFN e ATTN) do cabeçalho GGUF...", end="", flush=True)
    reader = GGUFReader(modelo_path)
    
    attn_bytes = 0
    ffn_bytes = 0
    total_bytes = 0
    
    for tensor in reader.tensors:
        # Pega o tamanho exato do tensor em bytes (funciona nativamente com quantização)
        t_size = tensor.data.nbytes
        total_bytes += t_size
        
        nome = tensor.name.lower()
        if "attn" in nome:
            attn_bytes += t_size
        elif "ffn" in nome or "mlp" in nome: # Modelos OPT/alguns LLaMAs usam 'mlp'
            ffn_bytes += t_size
            
    print(" [OK]")
    
    # Converte tudo para Megabytes (MB)
    return {
        "ATTN_MB": round(attn_bytes / (1024 * 1024), 2),
        "FFN_MB": round(ffn_bytes / (1024 * 1024), 2),
        "Total_Pesos_MB": round(total_bytes / (1024 * 1024), 2)
    }

# ==========================================
# FUNÇÃO PRINCIPAL
# ==========================================
def main():
    if not os.path.exists(LLAMA_CLI_PATH):
        print(f"❌ Erro: Executável não encontrado em {LLAMA_CLI_PATH}")
        return

    resultados = []
    print("🚀 Iniciando extração completa de arquitetura e memória...\n")

    for modelo in MODELOS:
        if not os.path.exists(modelo):
            print(f"⚠️ Aviso: Modelo não encontrado: {modelo}. Pulando...")
            continue
            
        nome_modelo = os.path.basename(modelo)
        print(f"Mapeando modelo: {nome_modelo}")

        # 1. Extrai os dados estáticos (só precisa ser feito uma vez por modelo)
        dados_estaticos = obter_tamanhos_estaticos(modelo)

        # 2. Extrai os dados dinâmicos (KV-Cache) para cada contexto
        for ctx in CONTEXTOS:
            print(f"  -> Testando KV-Cache (Contexto: {ctx})...", end="", flush=True)
            
            comando = [
                LLAMA_CLI_PATH,
                "-m", modelo,
                "-c", str(ctx),
                "-p", "a",
                "-n", "1"
            ]

            try:
                processo = subprocess.run(comando, capture_output=True, text=True, check=False)
                saida_completa = processo.stdout + processo.stderr
                
                # Regex atualizada: agora aceita números com vírgula ou ponto [0-9.,]+
                match = re.search(r"(?i)kv.*?size\s*=\s*([0-9.,]+)\s*(MB|MiB)", saida_completa)
                
                if match:
                    # Captura o texto (ex: "512,00"), troca a vírgula por ponto e converte para float
                    kv_str = match.group(1).replace(',', '.')
                    kv_size_mb = float(kv_str)
                    
                    print(f" [OK: {kv_size_mb} {match.group(2)}]")
                    
                    resultados.append({
                        "Modelo": nome_modelo,
                        "Contexto_Tokens": ctx,
                        "KV_Cache_MB": kv_size_mb,
                        "ATTN_MB": dados_estaticos["ATTN_MB"],
                        "FFN_MB": dados_estaticos["FFN_MB"],
                        "Total_Pesos_MB": dados_estaticos["Total_Pesos_MB"]
                    })
                else:
                    print(" [FALHOU]")
                    
            except Exception as e:
                print(f" [ERRO EXCEÇÃO: {e}]")
        print("-" * 40)

    # 3. Salva no CSV
    if resultados:
        print(f"\n💾 Salvando tabela final em: {ARQUIVO_SAIDA}")
        with open(ARQUIVO_SAIDA, mode='w', newline='') as csvfile:
            campos = ["Modelo", "Contexto_Tokens", "KV_Cache_MB", "ATTN_MB", "FFN_MB", "Total_Pesos_MB"]
            writer = csv.DictWriter(csvfile, fieldnames=campos)
            writer.writeheader()
            writer.writerows(resultados)
        print("✅ Automação concluída com sucesso! O CSV está pronto para a dissertação.")

if __name__ == "__main__":
    main()