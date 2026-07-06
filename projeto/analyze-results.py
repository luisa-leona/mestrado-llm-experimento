#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.container as mcontainer
import seaborn as sns
import numpy as np
import scipy.stats as st
import os

CSV_FILE = "projeto/tabelas/resultado.csv"
OUTPUT_DIR = "projeto/graficos"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_data():
    df = pd.read_csv(CSV_FILE)
    df['Decode_TPS'] = pd.to_numeric(df['Decode_TPS'], errors='coerce')
    df['Prefill_TPS'] = pd.to_numeric(df['Prefill_TPS'], errors='coerce')
    # Remove os prefixos numéricos (1_, 2_) usados só para ordenação cronológica no log
    df['Display_Mode'] = df['Mode'].apply(lambda x: x.split('_', 1)[1].replace('_', ' '))
    return df

def calculate_stats(df, metric):
    stats = []
    # Ordena pelo modo que teve a maior média, para a tabela ficar decrescente!
    modes = df.groupby('Display_Mode')[metric].mean().sort_values(ascending=False).index
    
    for mode in modes:
        data = df[df['Display_Mode'] == mode][metric].dropna()
        n = len(data)
        if n < 2: continue
        
        mean = np.mean(data)
        sem = st.sem(data)
        ci = sem * st.t.ppf((1 + 0.95) / 2., n-1)
        
        stats.append({
            'Cenário': mode,
            'Média (T/s)': round(mean, 3),
            'IC 95% (±)': round(ci, 3),
            'N': n
        })
    return pd.DataFrame(stats)

def plot_with_ci(df, metric_col, title, ylabel, filename):
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    # Ordena as barras da maior para a menor performance
    order = df.groupby('Display_Mode')[metric_col].mean().sort_values(ascending=False).index
    
    ax = sns.barplot(
        x="Display_Mode", y=metric_col, data=df, 
        palette="magma", errorbar=('ci', 95), 
        capsize=0.1, order=order, hue="Display_Mode", legend=False
    )
    
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("")
    plt.xticks(rotation=45, ha='right')
    
    # Usa a API moderna do Matplotlib para adicionar os números no topo das barras
    # Percorre todos os objetos desenhados no gráfico
    for container in ax.containers:
        # Verifica se o objeto é realmente uma Barra (ignora as linhas de erro)
        if isinstance(container, mcontainer.BarContainer):
            ax.bar_label(container, fmt='%.2f', padding=3, fontweight='bold')

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    df = load_data()
    
    print("\n=== ESTATÍSTICAS: DECODE (Memory-Bound) ===")
    stats_decode = calculate_stats(df, 'Decode_TPS')
    print(stats_decode.to_string(index=False))
    
    print("\n=== ESTATÍSTICAS: PREFILL (Compute-Bound) ===")
    stats_prefill = calculate_stats(df, 'Prefill_TPS')
    print(stats_prefill.to_string(index=False))

    plot_with_ci(df, 'Decode_TPS', 'Impacto da Optane no Decode (Geração de Texto)', 'Tokens/s', 'decode_ci95.png')
    plot_with_ci(df, 'Prefill_TPS', 'Impacto da Optane no Prefill (Processamento do Prompt)', 'Tokens/s', 'prefill_ci95.png')
    
    print("\n--> Gráficos gerados na pasta 'graficos'")