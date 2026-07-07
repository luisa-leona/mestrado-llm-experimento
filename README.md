# Estratificação Seletiva de Componentes de LLMs em Memória Heterogênea

Este repositório contém o arcabouço experimental e as modificações aplicadas ao motor de inferência `llama.cpp` para a realização dos experimentos de pós-graduação apresentados na dissertação. O objetivo deste trabalho é avaliar o impacto de desempenho (em TPS) ao isolar e alocar componentes estruturais do *Transformer* (matrizes FFN, pesos de Atenção e o KV-Cache) entre diferentes camadas de memória (*DRAM* e *Intel Optane PMem*).

---

## 🛠️ O que foi alterado?

A partir do código-fonte original do `llama.cpp` e de sua biblioteca tensorial subjacente (`GGML`), foram implementadas modificações estruturais para permitir o **tiering explícito e cirúrgico por componente**:

* **Isolamento de Tensores:** Modificação no carregador de modelos (*loader*) para identificar e separar as matrizes de pesos da Camada Feed-Forward (`FFN`), do bloco de Atenção (`Attn`) e do estado dinâmico da janela de contexto (`KV-Cache`).
* **Gerenciamento de Alocação Heterogênea:** Substituição da paginação opaca do Sistema Operacional por ponteiros de alocação explícitos, direcionando os componentes selecionados para nós NUMA específicos (Nó 0 para DRAM de alta largura de banda e Nó 1 para a memória persistente/lenta).
* **Exposição de Modos de Operação:** Adição de rotinas de controle para chaveamento entre as 8 configurações fatoriais de teste (`00_Baseline`, `01_KV`, `02_Attn`, `03_FFN`, `04_KV_Attn`, `05_KV_FFN`, `06_Attn_FFN` e `07_Tudo`).

---

## 🚀 Como Executar os Experimentos

### 1. Pré-requisitos e Ambiente
Os testes pressupõem um ambiente Linux configurado com arquitetura de memória heterogênea (ex: Intel Optane em modo *App Direct* exposta como nós NUMA limpos).

* **Compilador:** GCC/G++ 11+ ou Clang equivalente.
* **Dependências:** `numactl`, `libmemkind-dev` (se aplicável), Python 3.9+ (para automação dos cenários).
* **Modelos:** Arquivos quantizados em formato GGUF (utilizados: LLaMA 2 7B e LLaMA 3 8B em quantização `Q4_K_M`).

### 2. Compilação do Código Modificado
Clone o repositório e compile a versão customizada do motor de inferência:

```bash
git clone [https://github.com/luisa-leona/mestrado-llm-experimento.git](https://github.com/luisa-leona/mestrado-llm-experimento.git)
cd mestrado-llm-experimento
make -j$(nproc)
```

### 3. Execução dos Cenários de Carga
Os experimentos são automatizados através de scripts que varrem os cenários definidos na metodologia (C1 a C5), alternando o tamanho do prompt inicial (N_Prompt) e os passos de geração (N_Gen).

Para rodar a bateria completa de testes (10 repetições independentes por cenário) e gerar as planilhas de métricas brutas:

python3 projeto/measure_llm.py

### 4. Estrutura dos Dados Gerados
Os arquivos .csv resultantes serão estruturados com as seguintes colunas para posterior análise estatística e plotagem de gráficos:

* **Scenario:** Identificador do caso de uso (ex: C1_Sumarizacao, C5_Controle).
* **Mode:** Técnica de alocação ativa (do Baseline ao Tudo na PMem).
* **Run:** Índice da repetição (1 a N).
* **Prefill_TPS:** Vazão de processamento do prompt (Tokens/s).
* **Decode_TPS:** Vazão de geração autoregressiva (Tokens/s).






