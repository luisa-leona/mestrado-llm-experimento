#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Arquivos de entrada e saída
CSV_FILE = "tamanhos_memoria.csv"
OUTPUT_IMG = "./graficos/grafico_kv_cache_mha_vs_gqa.png"

def main():
    if not os.path.exists(CSV_FILE):
        print(f"Erro: Arquivo {CSV_FILE} não encontrado.")
        return

    # Lê os dados
    df = pd.read_csv(CSV_FILE)

    # Limpa os nomes dos modelos para a legenda ficar mais bonita
    df['Modelo'] = df['Modelo'].replace({
        'llama-2-7b.Q4_K_M.gguf': 'LLaMA 2 (7B) - MHA',
        'meta-llama-3-8b.Q4_K_M.gguf': 'LLaMA 3 (8B) - GQA'
    })

    # Configura o estilo do Seaborn (fundo branco com linhas de grade, ótimo para artigos)
    sns.set_theme(style="whitegrid", palette="tab10")
    
    # Cria a figura
    plt.figure(figsize=(10, 6))

    # Plota o gráfico de linhas
    ax = sns.lineplot(
        data=df, 
        x="Contexto_Tokens", 
        y="KV_Cache_MB", 
        hue="Modelo", 
        marker="o",       # Adiciona bolinhas nos pontos exatos de medição
        linewidth=2.5, 
        markersize=8
    )

    # Customizações de Título e Eixos
    plt.title("Crescimento do KV-Cache: LLaMA 2 vs LLaMA 3", fontsize=16, pad=20, fontweight='bold')
    plt.xlabel("Tamanho da Janela de Contexto (Tokens)", fontsize=13, labelpad=10)
    plt.ylabel("Consumo de Memória (MB)", fontsize=13, labelpad=10)

    # Força o eixo X a mostrar apenas os tamanhos de contexto que nós testamos
    plt.xticks(df['Contexto_Tokens'].unique())
    
    # Melhora a legenda
    plt.legend(title="Arquitetura de Atenção", fontsize=11, title_fontsize=12, loc='upper left')

    # Ajusta o layout para não cortar as bordas
    plt.tight_layout()

    # Salva em alta resolução (300 DPI é o padrão para dissertações e papers)
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"✅ Gráfico gerado com sucesso: {OUTPUT_IMG}")

if __name__ == "__main__":
    main()