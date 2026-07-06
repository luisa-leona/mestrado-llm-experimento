#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# CONFIGURAÇÕES
# ==========================================
ARQUIVO_CSV = "tabelas/resultados_node05_sensibilidade.csv"
IMAGEM_SAIDA = "graficos/grafico_analise_arquitetura.png"

def main():
    if not os.path.exists(ARQUIVO_CSV):
        print(f"❌ Erro: O ficheiro {ARQUIVO_CSV} não foi encontrado.")
        return

    # 1. Carregar os dados
    df = pd.read_csv(ARQUIVO_CSV)

    # 2. Mapeamento para nomes mais limpos
    mapa_nomes = {
        "01_Baseline_DRAM": "1. Baseline (100% DRAM)",
        "02_KV_Optane": "2. Apenas KV Cache na Optane",
        "03_Attn_Optane": "3. Apenas Atenção (ATTN) na Optane",
        "04_FFN_Optane": "4. Apenas FFN na Optane",
        "05_Primeira_Metade_Optane": "5. Metade Inicial [0-15] na Optane",
        "06_Segunda_Metade_Optane": "6. Metade Final [16-31] na Optane",
        "07_Extremos_Optane": "7. Camadas Extremas [0-3, 28-31] na Optane",
        "08_Miolo_Optane": "8. Miolo [4-27] na Optane",
        "09_All_Weights_Optane": "9. Pior Cenário (100% Optane)"
    }
    
    # Substituir os nomes na coluna 'Mode'
    df['Modo_Limpo'] = df['Mode'].map(mapa_nomes)

    # ==========================================
    # PLOTAGEM DO GRÁFICO (2 Painéis Lado a Lado)
    # ==========================================
    sns.set_theme(style="whitegrid")
    
    # Criar a figura com 1 linha e 2 colunas
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)

    # --- PAINEL 1: PREFILL (Compute-Bound) ---
    sns.barplot(
        data=df, 
        y='Modo_Limpo', 
        x='Prefill_TPS', 
        ax=axes[0], 
        color="#4C72B0", # Azul académico
        edgecolor='black',
        capsize=0.1,     # Adiciona as barras de erro (std dev / conf. interval)
        errorbar='sd'    # Mostra o Desvio Padrão das 5 runs
    )
    axes[0].set_title("Fase de Prefill (Compute-Bound)\nImpacto Menor da Latência", fontsize=14, pad=15, fontweight='bold')
    axes[0].set_xlabel("Vazão (Tokens por Segundo)", fontsize=12, labelpad=10)
    axes[0].set_ylabel("") # Ocultar label do eixo Y para não ser redundante

    # --- PAINEL 2: DECODE (Memory-Bound) ---
    sns.barplot(
        data=df, 
        y='Modo_Limpo', 
        x='Decode_TPS', 
        ax=axes[1], 
        color="#DD8452", # Laranja/Vermelho
        edgecolor='black',
        capsize=0.1,
        errorbar='sd'
    )
    axes[1].set_title("Fase de Decode (Memory-Bound)\nAlta Sensibilidade à Memória", fontsize=14, pad=15, fontweight='bold')
    axes[1].set_xlabel("Vazão (Tokens por Segundo)", fontsize=12, labelpad=10)
    axes[1].set_ylabel("")

    # Adicionar o valor exato no final de cada barra (para facilitar a leitura da banca)
    for ax in axes:
        for p in ax.patches:
            largura = p.get_width()
            if largura > 0:
                ax.annotate(f'{largura:.1f}', 
                            (largura, p.get_y() + p.get_height() / 2.), 
                            ha='left', va='center', 
                            fontsize=10, color='black', xytext=(5, 0), 
                            textcoords='offset points')

    # Título geral da figura
    plt.suptitle("Análise de Sensibilidade Arquitetural: PMem vs DRAM (LLaMA 3 - 8B)\n", fontsize=16, fontweight='bold')

    # Ajustar o layout para que os gráficos não se sobreponham
    plt.tight_layout()

    # Guardar a imagem em alta resolução (ideal para LaTeX ou Word)
    plt.savefig(IMAGEM_SAIDA, dpi=300, bbox_inches='tight')
    print(f"✅ Sucesso! O gráfico '{IMAGEM_SAIDA}' foi gerado e está pronto.")

if __name__ == "__main__":
    main()