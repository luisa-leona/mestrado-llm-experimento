import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Inserindo os seus dados reais
dados = {
    'Cenário': ['Baseline', 'Config A (KV)', 'Config B (Attn)', 'Config F (KV+Attn)', 
                'Config C (FFN)', 'Config G (KV+FFN)', 'Config D (Attn+FFN)', 'Config E (Tudo)'],
    'Decode_TPS': [15.262, 12.573, 10.897, 9.854, 5.446, 5.118, 4.791, 4.278],
    'Decode_IC': [0.224, 0.426, 0.267, 0.072, 0.014, 0.019, 0.011, 0.013],
    'Prefill_TPS': [111.664, 117.955, 123.211, 108.326, 96.408, 88.770, 96.882, 84.824],
    'Prefill_IC': [1.704, 10.987, 11.011, 2.513, 5.364, 6.625, 6.387, 6.018]
}

df = pd.DataFrame(dados)

# Ordenar de forma lógica: do melhor Decode para o pior (já está, mas garantimos)
df = df.sort_values(by='Decode_TPS', ascending=False)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Cores: Azul suave para Prefill, Laranja queimado para Decode
cor_prefill = "#4C72B0"
cor_decode = "#DD8452"

# Painel 1: PREFILL
axes[0].barh(df['Cenário'], df['Prefill_TPS'], xerr=df['Prefill_IC'], color=cor_prefill, edgecolor='black', capsize=5)
axes[0].set_title('Fase de Prefill (Compute-Bound)\nAlta Resiliência à Memória Optane', fontweight='bold', pad=15)
axes[0].set_xlabel('Vazão (Tokens/s)', fontweight='bold')
axes[0].invert_yaxis() # Para o Baseline ficar no topo

# Painel 2: DECODE
axes[1].barh(df['Cenário'], df['Decode_TPS'], xerr=df['Decode_IC'], color=cor_decode, edgecolor='black', capsize=5)
axes[1].set_title('Fase de Decode (Memory-Bound)\nGargalo na Largura de Banda da Matriz FFN', fontweight='bold', pad=15)
axes[1].set_xlabel('Vazão (Tokens/s)', fontweight='bold')

# Adicionar os rótulos de dados
for i, v in enumerate(df['Prefill_TPS']):
    axes[0].text(v + df['Prefill_IC'].iloc[i] + 2, i, f"{v:.1f}", va='center', fontsize=10)
    
for i, v in enumerate(df['Decode_TPS']):
    axes[1].text(v + df['Decode_IC'].iloc[i] + 0.5, i, f"{v:.1f}", va='center', fontsize=10)

plt.suptitle("Impacto do Tiering de Memória (DRAM vs PMem) por Componente Arquitetural\nLLaMA 3 8B - Batch Size 1, Contexto 2048", y=1.05, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("grafico_fatorial_final.png", dpi=300, bbox_inches='tight')
print("✅ Gráfico guardado como 'grafico_fatorial_final.png'")