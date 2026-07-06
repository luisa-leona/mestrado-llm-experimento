#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches  # <-- Adicione esta linha
import seaborn as sns
import os

# ==========================================
# CONFIGURAÇÕES
# ==========================================
ARQUIVO_DRAM = "tabelas/resultado_contexto_DRAM.csv"
ARQUIVO_OPTANE = "tabelas/resultado_contexto_OPTANE.csv"
IMAGEM_SAIDA = "graficos/grafico_sensibilidade_fases.png"

def main():
    # Verifica se os dois ficheiros existem antes de começar
    if not os.path.exists(ARQUIVO_DRAM) or not os.path.exists(ARQUIVO_OPTANE):
        print(f"❌ Erro: Certifique-se de que {ARQUIVO_DRAM} e {ARQUIVO_OPTANE} estão na mesma pasta.")
        return

    # 1. Carregar os dados
    df_dram = pd.read_csv(ARQUIVO_DRAM)
    df_optane = pd.read_csv(ARQUIVO_OPTANE)

    # 2. Mesclar as duas tabelas usando o Tamanho do Contexto como chave
    df_merged = pd.merge(df_dram, df_optane, on="Tamanho_Contexto", suffixes=('_DRAM', '_OPTANE'))

    # 3. Calcular a Degradação (%) para Prefill e Decode
    # Fórmula: ((DRAM - Optane) / DRAM) * 100
    df_merged['Degradacao_Prefill'] = ((df_merged['Prefill_TPS_DRAM'] - df_merged['Prefill_TPS_OPTANE']) / df_merged['Prefill_TPS_DRAM']) * 100
    df_merged['Degradacao_Decode'] = ((df_merged['Decode_TPS_DRAM'] - df_merged['Decode_TPS_OPTANE']) / df_merged['Decode_TPS_DRAM']) * 100

    # Se a Optane por algum motivo for mais rápida (ruído de medição), limitamos a 0% para não ter barras negativas confusas
    df_merged['Degradacao_Prefill'] = df_merged['Degradacao_Prefill'].clip(lower=0)
    df_merged['Degradacao_Decode'] = df_merged['Degradacao_Decode'].clip(lower=0)

    # 4. Preparar os dados para o gráfico de barras agrupadas (Formato 'Melted' do Seaborn)
    df_plot = pd.melt(df_merged, 
                      id_vars=['Tamanho_Contexto'], 
                      value_vars=['Degradacao_Prefill', 'Degradacao_Decode'],
                      var_name='Fase', 
                      value_name='Degradacao_Perc')

    # Limpar os nomes para a legenda ficar bonita
    df_plot['Fase'] = df_plot['Fase'].replace({
        'Degradacao_Prefill': 'Prefill (Compute-Bound)',
        'Degradacao_Decode': 'Decode (Memory-Bound)'
    })

    # ==========================================
    # PLOTAGEM DO GRÁFICO (Padrão Académico)
    # ==========================================
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Definir as cores (Azul para Prefill, Vermelho/Laranja para Decode mostrando o alerta de gargalo)
    cores = ["#4C72B0", "#DD8452"]

    ax = sns.barplot(
        data=df_plot, 
        x='Tamanho_Contexto', 
        y='Degradacao_Perc', 
        hue='Fase',
        palette=cores,
        edgecolor='black', # Borda preta nas barras fica ótimo em artigos
        linewidth=1
    )

    # Adicionar os números exatos em cima de cada barra
    for p in ax.patches:
        # Verifica se o patch é realmente uma barra (Rectangle) para o Pylance não reclamar
        if isinstance(p, patches.Rectangle):
            altura = p.get_height()
            if altura > 0: # Só coloca número se houver degradação
                ax.annotate(f'{altura:.1f}%', 
                            (p.get_x() + p.get_width() / 2., altura), 
                            ha='center', va='bottom', 
                            fontsize=10, color='black', xytext=(0, 4), 
                            textcoords='offset points')

    # Customizações de Título e Eixos
    plt.title("Sensibilidade de Desempenho na PMem (Optane vs DRAM)\nImpacto do Tamanho do Contexto", fontsize=15, pad=20, fontweight='bold')
    plt.xlabel("Tamanho da Janela de Contexto (Tokens)", fontsize=12, labelpad=10)
    plt.ylabel("Degradação de Desempenho (%)", fontsize=12, labelpad=10)
    
    # Limite do eixo Y (0 a 100%)
    plt.ylim(0, 100)

    # Posicionar a legenda no canto superior esquerdo para não cobrir as barras maiores
    plt.legend(title="Fase de Inferência", fontsize=11, title_fontsize=12, loc='upper left')

    plt.tight_layout()
    plt.savefig(IMAGEM_SAIDA, dpi=300)
    print(f"✅ Sucesso! O gráfico '{IMAGEM_SAIDA}' foi gerado e está pronto para a dissertação.")

if __name__ == "__main__":
    main()