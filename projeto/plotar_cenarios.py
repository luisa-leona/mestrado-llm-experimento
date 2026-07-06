import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar os dados
df = pd.read_csv("tabelas/resultados_llama2_v2.csv")

# 2. Renomear e limpar as categorias no gráfico
mapa_cenarios = {
    'C1_Sumarizacao': 'C1: Sumarização\n(P: 4K, G: 128)',
    'C2_Geracao': 'C2: Geração\n(P: 128, G: 4K)',
    'C3_Conversacao': 'C3: Chat\n(P: 1K, G: 1K)',
    'C4_Estresse': 'C4: Stress\n(P: 7K, G: 1K)',
    'C5_Controle': 'C5: Controle\n(P: 64, G: 64)'
}

mapa_configs = {
    '00_Baseline': '1. Baseline (DRAM)',
    '01_KV': '2. Config A (KV)',
    '02_Attn': '3. Config B (Attn)',
    '04_KV_Attn': '4. Config F (KV+Attn)',
    '03_FFN': '5. Config C (FFN)',
    '05_KV_FFN': '6. Config G (KV+FFN)',
    '06_Attn_FFN': '7. Config D (Attn+FFN)',
    '07_Tudo': '8. Config E (Tudo)'
}

df['Scenario_Label'] = df['Scenario'].map(mapa_cenarios)
df['Mode_Label'] = df['Mode'].map(mapa_configs)

# 3. Calcular a média das 10 repetições (Runs)
df_mean = df.groupby(['Mode_Label', 'Scenario_Label'])[['Prefill_TPS', 'Decode_TPS']].mean().reset_index()

# 4. Criar as matrizes Pivot (Linhas = Configs, Colunas = Cenários)
pivot_prefill = df_mean.pivot(index='Mode_Label', columns='Scenario_Label', values='Prefill_TPS')
pivot_decode = df_mean.pivot(index='Mode_Label', columns='Scenario_Label', values='Decode_TPS')

# Ordenar colunas (C1 a C5) e linhas (1 a 8) para manter a lógica visual
ordem_cenarios = [mapa_cenarios[k] for k in ['C1_Sumarizacao', 'C2_Geracao', 'C3_Conversacao', 'C4_Estresse', 'C5_Controle']]
pivot_prefill = pivot_prefill[ordem_cenarios]
pivot_decode = pivot_decode[ordem_cenarios]

# 5. Configurar o estilo e tamanho da figura
sns.set_theme(style="white", font_scale=1.1)
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Paleta de cores: Vermelho (Lento) -> Amarelo -> Verde (Rápido)
cmap = "RdYlGn"

# Heatmap 1: PREFILL
sns.heatmap(pivot_prefill, annot=True, fmt=".1f", cmap=cmap, ax=axes[0], 
            cbar_kws={'label': 'Tokens / segundo'}, linewidths=.5, vmin=60, vmax=130)
axes[0].set_title('Fase de Prefill (Compute-Bound)\nMenor Degradação (Mascaramento de Latência)', fontsize=16, fontweight='bold', pad=15)
axes[0].set_ylabel('')
axes[0].set_xlabel('')
axes[0].tick_params(axis='x', rotation=0)

# Heatmap 2: DECODE
sns.heatmap(pivot_decode, annot=True, fmt=".1f", cmap=cmap, ax=axes[1], 
            cbar_kws={'label': 'Tokens / segundo'}, linewidths=.5, vmin=2.5, vmax=16)
axes[1].set_title('Fase de Decode (Memory-Bound)\nForte Gargalo com FFN na PMem', fontsize=16, fontweight='bold', pad=15)
axes[1].set_ylabel('')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=0)

# Ajuste estético final
plt.suptitle("Mapa de Calor da Vazão de Inferência por Cenário e Alocação (LLaMA 3 8B)", 
             fontsize=20, fontweight='bold', y=1.05)
plt.tight_layout()

# Guardar a imagem
plt.savefig("heatmap.png", dpi=300, bbox_inches='tight')
print("✅ Gráfico guardado com sucesso como 'heatmap.png'")