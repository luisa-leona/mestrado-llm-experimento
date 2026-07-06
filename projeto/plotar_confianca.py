import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar os dados
df = pd.read_csv("tabelas/resultados_llama2_v2.csv")

# 2. ESCOLHER O CENÁRIO PARA O "ZOOM-IN"
#CENARIO_ALVO = 'C1_Sumarizacao'
#CENARIO_ALVO = 'C2_Geracao'
#CENARIO_ALVO = 'C3_Conversacao'
#CENARIO_ALVO = 'C4_Estresse'
CENARIO_ALVO = 'C5_Controle'

# Filtrar apenas o cenário escolhido
df_alvo = df[df['Scenario'] == CENARIO_ALVO].copy()

# Mapear os nomes no gráfico
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
df_alvo['Mode_Label'] = df_alvo['Mode'].map(mapa_configs)

# Ordenar da melhor performance de Decode para a pior (baseado na média)
ordem = df_alvo.groupby('Mode_Label')['Decode_TPS'].mean().sort_values(ascending=False).index

# 3. Configuração do Gráfico
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

cor_prefill = "#4C72B0"
cor_decode = "#DD8452"

# Painel 1: PREFILL (O seaborn calcula o IC 95% automaticamente pelo parâmetro errorbar)
sns.barplot(
    data=df_alvo, x='Prefill_TPS', y='Mode_Label', order=ordem,
    ax=axes[0], color=cor_prefill, edgecolor='black', 
    capsize=0.1, err_kws={'linewidth': 2, 'color': 'black'}, errorbar=('ci', 95)
)
axes[0].set_title(f'Fase de Prefill ({CENARIO_ALVO})\nAlta Resiliência à PMem', fontweight='bold', pad=15)
axes[0].set_xlabel('Vazão (Tokens/s)', fontweight='bold')
axes[0].set_ylabel('')

# Painel 2: DECODE
sns.barplot(
    data=df_alvo, x='Decode_TPS', y='Mode_Label', order=ordem,
    ax=axes[1], color=cor_decode, edgecolor='black', 
    capsize=0.1, err_kws={'linewidth': 2, 'color': 'black'}, errorbar=('ci', 95)
)
axes[1].set_title(f'Fase de Decode ({CENARIO_ALVO})\nImpacto Estrutural da Matriz FFN', fontweight='bold', pad=15)
axes[1].set_xlabel('Vazão (Tokens/s)', fontweight='bold')
axes[1].set_ylabel('')

plt.suptitle(f"Estabilidade de Inferência e Intervalo de Confiança (95%)\nZoom no Cenário: {CENARIO_ALVO}", y=1.05, fontsize=16, fontweight='bold')
plt.tight_layout()

plt.savefig(f"graficos/grafico_confianca_{CENARIO_ALVO}.png", dpi=300, bbox_inches='tight')
print(f"✅ Gráfico guardado como 'grafico_confianca_{CENARIO_ALVO}.png'")