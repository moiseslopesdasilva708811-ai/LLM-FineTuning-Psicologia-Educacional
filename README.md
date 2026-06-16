# Projeto de Fine-Tuning e Avaliação de Modelos LLM com LoRA especializado em Psicologia Educacional


<p align="center" style="position: -webkit-sticky; position: sticky; top: 0; background: #000; padding: 10px 0; border-bottom: 2px solid #333; z-index: 1000;">
  <a href="#objetivos"><img src="https://img.shields.io/badge/_Objetivos-2B2B2B?style=for-the-badge&logoColor=white&labelColor=000000" alt="Objetivos">
  <a href="#instalacao"><img src="https://img.shields.io/badge/_Instalação-2B2B2B?style=for-the-badge&logoColor=white&labelColor=000000" alt="Objetivos">
  </a>
  <a href="#estrutura">
    <img src="https://img.shields.io/badge/_Estrutura-2B2B2B?style=for-the-badge&logoColor=white&labelColor=000000" alt="Estrutura">
  </a>
  <a href="#demo">
    <img src="https://img.shields.io/badge/_Setup-2B2B2B?style=for-the-badge&logoColor=white&labelColor=000000" alt="Setup">
  </a>
  <a href="#benchmarks">
    <img src="https://img.shields.io/badge/_Benchmarks-2B2B2B?style=for-the-badge&logoColor=white&labelColor=000000" alt="Benchmarks">
  </a>
  <a href="#autor">
    <img src="https://img.shields.io/badge/_Autor-2B2B2B?style=for-the-badge&logoColor=white&labelColor=000000" alt="Autor">
  </a>
</p>
    
<div align="center">
  <h3 style="color: #4A90E2;"> Experimente o Agente Especialista</h3>
  <p>Interaja com o sistema de IA treinado em Psicologia Educacional em tempo real:</p>
  
  <a href="https://huggingface.co/spaces/josemoiseslsilva/agente-psicologia" target="_blank">
    <img src="https://img.shields.io/badge/DEMONSTRAÇÃO_ONLINE_AGORA-007BFF?style=for-the-badge&logo=huggingface&logoColor=white&link=https://huggingface.co/spaces/josemoiseslsilva/agente-psicologia" alt="Acessar Sistema">
  </a>
  <p><i>Acesse o link acima para interagir diretamente com a API e os modelos configurados.</i></p>
</div>


<div align="center">

<a href="https://youtu.be/ENGRShkkhuM" target="_blank">
  <img src="https://img.youtube.com/vi/ENGRShkkhuM/maxresdefault.jpg" alt="Demonstração do Agente" width="700">
</a>

</div>


<p align="center">
  <a href="https://medium.com/@moiseslopesdasilva708811/desenvolvimento-de-um-agente-especialista-em-psicologia-educacional-implementa%C3%A7%C3%A3o-de-rag-e-629914adae69">
    <img src="https://img.shields.io/badge/LEIA_O_ARTIGO_COMPLETO_NO_MEDIUM-000000?style=for-the-badge&logo=medium&logoColor=white" alt="Link para o Medium">
  </a>
</p>

> Para acessar cada parte nesse repositório, clique nos botões de acesso acima

> [!IMPORTANT]
> Este projeto tem como objetivo realizar o treinamento, ajuste fino (Fine-Tuning) e avaliação de modelos de linguagem utilizando a técnica **LoRA (Low-Rank Adaptation)**, permitindo adaptar modelos pré-treinados para tarefas específicas de forma eficiente. Neste contexto, este trabalho explora o ciclo de vida completo de um sistema de
RAG integrado ao ajuste fino eficiente via Low-Rank Adaptation (LoRA). A proposta visa ir além da arquitetura de busca semântica tradicional através da especialização
de pesos, permitindo que modelos de linguagem ajustem seu comportamento ao domínio em questão.

> [!TIP]
> **Precisa de ajuda com o código?** Acesse a pasta `./notebooks/` para encontrar um guia modular e detalhado sobre todo o fluxo de implementação deste projeto.

> [!NOTE]
> Ao final deste README, você estará apto a clonar, configurar e rodar todo o sistema. Gratidão por seu interesse neste projeto!

---

## 🚀 Tecnologias Utilizadas

### Inteligência Artificial & Machine Learning
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![LoRA](https://img.shields.io/badge/LoRA-Fine--Tuning-blue?style=for-the-badge)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface&logoColor=black)

### Ciência de Dados
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

### Recuperação de Informação (RAG)
![ChromaDB](https://img.shields.io/badge/ChromaDB-6E57E0?style=for-the-badge)
![RAG](https://img.shields.io/badge/RAG-Retrieval%20Augmented%20Generation-green?style=for-the-badge)

### Backend
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)

### Frontend
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

## Material usado para treinamento do projeto
![Referencial Teórico](images/Rerencia.png)


<a name="estrutura"></a>
## 📂 Estrutura do Projeto

```text
C:.
│   .gitignore
│   README.md
│   requirements.txt
│
├───api
│       main.py
│
├───chroma_db
│   │   chroma.sqlite3
│   │
│   └───2ba4a2c2-ed4e-4199-8335-eac837d36c91
│           data_level0.bin
│           header.bin
│           length.bin
│           link_lists.bin
│
├───dataset
│       dataset_gerado.jsonl
│       dataset_gerado_curado.jsonl
│
├───dataset
│       20260042690.pdf
├───images
│       Diagrama_de_sequência_da_Aplicação.png
│       fluxograma_vertical.png
│       grafico_bart_tiny.png
│       grafico_Flan_T5_Small.png
│       grafico_gpt_neo.png
│       grafico_Qwen.png
│       Gráfico_Radar.png
│       Referencia.png
│
├───notebooks
│       01_rag.ipynb
│       02_lora.ipynb
│       03_avaliacao_modelo_finetuned.ipynb
│
├───static
│       index.html
│
└───train
    ├───lora_seq2seq_model_1
    │       adapter_config.json
    │       adapter_model.safetensors
    │       tokenizer_config.json
    │       ...
    │
    ├───lora_tiny_bart_final
    │       adapter_config.json
    │       adapter_model.safetensors
    │       tokenizer_config.json
    │       ...
    │
    ├───modelo_final
    │       config.json
    │       model.safetensors
    │       tokenizer.json
    │       ...
    │
    └───modelo_final_lora
            adapter_config.json
            adapter_model.safetensors
            tokenizer_config.json
            ...
```
### Descrição das Pastas


| Pasta        | Descrição                                                                                                        |
| ------------ | ---------------------------------------------------------------------------------------------------------------- |
| `api/`       | Implementação da API FastAPI responsável por receber as perguntas e retornar as respostas geradas pelos modelos. |
| `chroma_db/` | Banco vetorial ChromaDB utilizado pelo sistema RAG para recuperação de contexto.                                 |
| `dataset/`   | Conjunto de dados utilizado para treinamento e ajuste fino dos modelos de linguagem.                             |
| `images/`    | Diagramas, fluxogramas e gráficos gerados durante os experimentos e avaliações dos modelos.                      |
| `notebooks/` | Notebooks contendo os experimentos de RAG, Fine-Tuning com LoRA e avaliação dos modelos treinados.               |
| `static/`    | Interface Web da aplicação.                                                                                      |
| `train/`    | Modelos treinados e seus formatos .JSONL                                                                          |
    
```
```

---
<a name="objetivos"></a>
## Objetivos

- Realizar Fine-Tuning de modelos LLM utilizando LoRA.
- Comparar diferentes arquiteturas:
  - Qwen2-0.5B
  - GPT-Neo-125M
  - BART
  - FLAN-T5-small
- Construir uma API para inferência utilizando FastAPI.
- Avaliar desempenho dos modelos ajustados.
- Integrar técnicas de RAG (Retrieval-Augmented Generation).


---

## Destaques do Projeto
- **Arquitetura Híbrida:** Integração de *Retrieval-Augmented Generation* (RAG) com modelos especialistas via LoRA.
- **Benchmark Rigoroso:** Análise quantitativa baseada em Perplexidade (PPL), ROUGE-L e BLEU.
- **Eficiência Computacional:** Uso de adaptadores LoRA para reduzir o custo de inferência e treinamento.
- **Modularidade:** API FastAPI pronta para produção com documentação Swagger integrada.

### Análise Crítica do Melhor Modelo

> Melhor qualitativamente: Qwen2-0.5B
> Melhor quantitativamente: Flan-T5-Small

<a name="demo"></a>

## Demo do Projeto - Análise dos Modelos Finetunados (Funcionalidade na aplicação mais abaixo)
Ao final, a tela desenvolvida ficou exatamente desta forma abaixo.
<img src="./Tela%20funcional%20completa.png" alt="Autor">
A próxima demonstração designa a funcionalidade a qual a aplicação conseguiu desenvolver no modelo Qwen2-0.5B, sem nenhuma alucinação, tudo com base no que estava documentado
no PDF disponível na pasta ./docs/

![Demo](./video%20Qwen%20model%20final.gif)

Mais abaixo, está presente o teste da aplicação agora empregado aos modelos BART, Flan-T5-small e GPT-Neo-125M. Notei que o BART alucinou e por isso decidi descartar a possibilidade de usar ele como modelo principal deste trabalho. O modelo Flan-T5-small por ter uma base menor, escreveu uma pequena resposta como é possível ver na demonstração a seguir e o GPT-neo-125M conseguiu trazer a resposta de forma adequada também ao projeto.

![Demo](./demonstracao%20teste%20Demo%20modelos.gif)

Um passo importante para um projeto envolvendo um chat IA inteligente é conectar API para que haja a comunicação na arquitetura cliente servidor, com isso, trago a demonstração da API funcionando mais abaixo.

![Demo](./FASTAPI%20funcionando.gif)

## Demo do Projeto - Análise dos Modelos Bases - NÃO Finetunados (Funcionalidade na aplicação mais abaixo)


## Gráficos de Convergência do treinamento dos modelos

### Gráfico do Modelo Qwen2-0.5B
![Gráfico Qwen](images/grafico_Qwen.png)

### Gráfico do Modelo GPT-Neo-125M
![Gráfico GPT-Neo](images/grafico_gpt_neo.png)

### Gráfico do Modelo BART-Tiny
![Gráfico Bart](images/grafico_bart_tiny.png)

### Gráfico do Modelo FLAN-T5-Small
![Gráfico T5](images/grafico_Flat_T5_Small.png)


## Diagrama de Sequência da aplicação desenvolvida em questão

<img src="images/Diagrama_de_sequência_da_Aplicação.png" width="700" alt="Gráfico de Loss do Qwen">

## 📊 Dataset

O conjunto de treinamento encontra-se em:

```bash
dataset/dataset_treino.jsonl     # depois troquei pelo dataset_curado para ver qual seria o melhor para responder as perguntas
```

Formato esperado:

```json
{
  "instruction": "Pergunta do usuário",
  "input": "",
  "output": "Resposta esperada"
}
```

---
<a name="benchmarks"></a>
## 🧠 Modelos Utilizados

| Modelo | Tipo | Finalidade |
|----------|----------|----------|
| Qwen2-0.5B | Causal LM | Chat e geração de texto |
| GPT-Neo-125M | Causal LM | Benchmark leve |
| BART | Seq2Seq | Resumo e geração condicional |
| FLAN-T5-small | Seq2Seq | Perguntas e respostas |


# Avaliação Comparativa de Modelos Base para Geração de Texto

## Visão Geral

> [!NOTE]
> Este experimento teve como objetivo avaliar o desempenho de diferentes modelos de linguagem em uma tarefa de geração textual, utilizando métricas amplamente empregadas na literatura de Processamento de Linguagem Natural (NLP).

Foram avaliados os seguintes modelos:

- Flan-T5-Small
- BART-Tiny
- Qwen2-0.5B
- EleutherAI

As métricas utilizadas foram:

- **BLEU (Bilingual Evaluation Understudy)**
- **ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)**
- **Perplexidade (Perplexity)**

---

# Resultados
Observei cada ponto gerado nesses notebooks em que as análises foram precisas e cheguei a um modelo campeão que nesse caso foi o Flan T5 Small, siga a leitura.

## Tabela Comparativa

| Modelo (Base) | BLEU | ROUGE-L | Perplexidade |
|---------------|------|----------|-------------:|
| **Flan-T5-Small** | 0.0000 | **0.0080** | **1.66** |
| **BART-Tiny** | 0.0000 | 0.0000 | 50253.99 |
| **Qwen2-0.5B** | 0.0000 | 0.0000 | 27.22 |
| **GPT-NEO-125M** | 0.0000 | 0.0034 | 2317.00 |

---

### Visualização de Performance (Gráfico Radar)

O gráfico abaixo ilustra o equilíbrio entre as métricas de performance (estabilidade, precisão estrutural e fluidez linguística) para cada modelo:

![Gráfico Radar de Desempenho](images/Gráfico_Radar.png)

*O gráfico reforça a superioridade do Flan-T5-Small, que apresenta um polígono com maior área de cobertura nas métricas de menor incerteza (menor PPL) e maior precisão.*

# Análise Técnica dos Resultados

## 1. BLEU Score

O BLEU Score apresentou valor **0.0000 para todos os modelos avaliados**.

O BLEU mede a similaridade entre a saída gerada pelo modelo e uma resposta de referência através da correspondência de n-gramas. A ausência total de correspondência sugere que:

- As respostas produzidas diferem significativamente das respostas de referência;
- O conjunto de avaliação pode permitir múltiplas respostas semanticamente corretas;
- A métrica não conseguiu capturar adequadamente a qualidade das gerações produzidas neste contexto específico.

### Interpretação

Embora o BLEU seja amplamente utilizado em tarefas de tradução automática, ele possui limitações em cenários de geração aberta, onde diferentes formulações podem representar respostas igualmente válidas.

Dessa forma, o BLEU não forneceu poder discriminativo suficiente para diferenciar os modelos avaliados.

---

## 2. ROUGE-L

O ROUGE-L mede a maior subsequência comum (Longest Common Subsequence) entre o texto gerado e a referência.

### Resultados

| Modelo | ROUGE-L |
|----------|---------:|
| Flan-T5-Small | **0.0080** |
| GPT-NEO-125M | 0.0034 |
| BART-Tiny | 0.0000 |
| Qwen2-0.5B | 0.0000 |

### Interpretação

Os valores obtidos foram extremamente baixos, indicando baixa sobreposição estrutural entre as respostas geradas e as respostas esperadas.

Mesmo assim, observa-se que:

- O **Flan-T5-Small** apresentou o melhor resultado;
- O **GPT-NEO-125M** demonstrou alguma correspondência residual;
- **Qwen2-0.5B** e **BART-Tiny** não apresentaram alinhamento estrutural mensurável com as referências.

Embora a diferença seja pequena em termos absolutos, o ROUGE-L reforça a superioridade relativa do Flan-T5-Small no conjunto avaliado.

---

## 3. Perplexidade

A Perplexidade foi a métrica que apresentou maior capacidade de discriminação entre os modelos.

### Resultados

| Modelo | Perplexidade |
|----------|-------------:|
| Flan-T5-Small | **1.66** |
| Qwen2-0.5B | 27.22 |
| GPT-NEO-125M | 2317.00 |
| BART-Tiny | 50253.99 |

### Fundamentação Teórica documentada nesse processo

A Perplexidade mede o grau de incerteza de um modelo ao prever a próxima palavra de uma sequência textual.

Matematicamente:

### Fórmula da Perplexidade
### Fórmula da Perplexidade

A perplexidade pode ser definida como:

> **PPL = exp(-(1/N) × Σ log(P(wᵢ)))**

Onde:

- **N** representa o número total de tokens;
- **P(wᵢ)** representa a probabilidade atribuída ao token correto;
- **Σ** representa o somatório das probabilidades logarítmicas dos tokens da sequência.

Quanto menor o valor da perplexidade, melhor a capacidade do modelo em prever a distribuição linguística dos dados.
# Comparação Individual dos Modelos

## 🥇 Flan-T5-Small

### Destaques

- Menor perplexidade do experimento (**1.66**);
- Maior ROUGE-L;
- Melhor estabilidade geral;
- Melhor aderência ao conjunto avaliado.

### Avaliação Técnica

O modelo demonstrou excelente capacidade de modelagem linguística, apresentando baixa incerteza preditiva e desempenho consistentemente superior em todas as métricas relevantes.

Os resultados sugerem que sua arquitetura encoder-decoder baseada em T5, aliada ao processo de instruction tuning realizado pelo Google, favoreceu significativamente sua adaptação à tarefa proposta.

---

## 🥈 Qwen2-0.5B

### Destaques

- Segunda menor perplexidade (**27.22**);
- Desempenho superior ao GPT-NEO-125M e BART-Tiny.

### Avaliação Técnica

Apesar de não apresentar correspondência textual detectável pelas métricas BLEU e ROUGE-L, o modelo demonstrou capacidade razoável de modelagem probabilística da linguagem.

Isso indica que o modelo produz sequências linguisticamente plausíveis, embora não alinhadas às respostas de referência utilizadas no benchmark.

---

## 🥉 GPT-NEO-125M - EleutherAI

### Destaques

- Pequena pontuação em ROUGE-L;
- Desempenho intermediário.

### Avaliação Técnica

A elevada perplexidade (**2317**) indica dificuldades substanciais em capturar a distribuição textual presente no conjunto de avaliação.

Embora tenha apresentado uma pequena correspondência estrutural com as referências, seu comportamento geral foi significativamente inferior ao dos dois primeiros colocados.

---

## 4º Lugar — BART-Tiny

### Destaques

- BLEU nulo;
- ROUGE-L nulo;
- Maior perplexidade observada.

### Avaliação Técnica

Os resultados indicam que a versão reduzida da arquitetura BART utilizada no experimento não possui capacidade representacional suficiente para a complexidade da tarefa avaliada.

---

# Ranking Final

| Posição | Modelo |
|----------|---------|
| 🥇 1º | Flan-T5-Small |
| 🥈 2º | Qwen2-0.5B |
| 🥉 3º | GPT-NEO-125M |
| 4º | BART-Tiny |

---

# Veredito Técnico

A análise conjunta das métricas evidencia que o **Flan-T5-Small** apresentou o melhor desempenho global entre todos os modelos avaliados.

Os principais fatores que sustentam essa conclusão são:

- Menor perplexidade do experimento (**1.66**);
- Maior valor de ROUGE-L;
- Maior estabilidade estatística;
- Melhor capacidade de modelagem linguística.

Embora BLEU e ROUGE-L tenham apresentado valores extremamente baixos para todos os modelos, a Perplexidade demonstrou forte capacidade de diferenciação, tornando-se a métrica mais relevante para este conjunto experimental.

Os resultados indicam que o Flan-T5-Small foi o único modelo capaz de representar adequadamente a distribuição textual presente no dataset, demonstrando maior consistência e menor incerteza durante o processo de geração.

---

# Conclusão

Com base nas evidências quantitativas obtidas, conclui-se que o **Flan-T5-Small** constitui o modelo base mais adequado para servir como baseline do projeto.

Sua superioridade em termos de modelagem probabilística e aderência textual sugere que futuras etapas de:

- Fine-tuning;
- Ajuste de hiperparâmetros;
- Treinamento supervisionado adicional;
- Avaliações qualitativas;

devem utilizar o Flan-T5-Small como principal referência experimental.

Em síntese, os resultados demonstram que arquiteturas menores, porém instruídas e especializadas, podem superar modelos maiores quando avaliadas em tarefas específicas de geração textual.
---

<a name="instalacao"></a>
## 🚀 Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/moiseslopesdasilva708811-ai/LLM-FineTuning-Psicologia-Educacional.git
cd LLM-FineTuning-Psicologia-Educacional
```

### 2. Criar ambiente virtual

Linux/Mac:

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

---

## 🔥 Fine-Tuning com LoRA

O notebook principal para treinamento é:

```bash
02_lora.ipynb
```

Nele são realizados:

- Carregamento do dataset
- Tokenização
- Configuração LoRA
- Treinamento
- Salvamento dos adaptadores

Os resultados são armazenados em:

```bash
train/
```

---

## 📚 Pipeline RAG

O notebook:

```bash
01_rag.ipynb
```

executa:

1. Ingestão de documentos
2. Processamento de texto
3. Vetorização
4. Criação do índice vetorial
5. Busca semântica

---

## 📈 Avaliação dos Modelos

O notebook:

```bash
03_avaliacao_modelo_finetuned.ipynb
```

permite:

- Comparar modelos treinados
- Medir tempo de inferência
- Avaliar qualidade das respostas
- Gerar métricas de benchmark

---

## Estrutura de Datasets

O projeto utiliza três datasets principais em formato `.jsonl`. Abaixo, detalhamos o conteúdo e o propósito de cada um:

### 1. `dataset_gerado.jsonl`
Este arquivo contém um conjunto estruturado de instruções e perguntas formatadas para treino. É o arquivo mais robusto para a base de conhecimento do modelo.
- **Formato:** `{"Instruction": "...", "Output": "..."}`
- **Conteúdo:** Perguntas sobre Psicologia Educacional formuladas com base em conceitos teóricos, incluindo:
    - O objeto de estudo da Psicologia Educacional.
    - A importância da perspectiva integrativa.
    - Influência de métodos experimentais (Wundt) na psicologia.
    - Dinâmicas familiares e competências socioemocionais.

### 2. `dataset_gerado_curado` realizado por CURADORIA MANUAL
- Utilizado para treinamento de todo o modelo
## 🌐 API FastAPI

O servidor principal encontra-se em:

```bash
main.py
```

Executar:

```bash
uvicorn main:app --reload
```

A API ficará disponível em:

```text
http://localhost:8000
```

Documentação automática:

```text
http://localhost:8000/docs
```

---

## 🖥️ Interface Web

A interface de usuário está localizada em:

```bash
static/index.html
```

Funcionalidades:

- Chat interativo
- Comunicação com a API FastAPI
- Exibição das respostas do modelo

---

## 📦 Dependências Principais

- Python 3.10+
- Transformers
- PEFT
- Torch
- Datasets
- Accelerate
- FastAPI
- Uvicorn
- SentenceTransformers
- FAISS

---

## Estrutura dos Adaptadores LoRA

Cada modelo treinado possui:

```text
adapter_config.json
```

Configurações do LoRA.

```text
adapter_model.safetensors
```

Pesos treinados.

```text
tokenizer.json
tokenizer_config.json
```

Arquivos de tokenização (quando aplicável).

---

## Melhorias Futuras

- Implementação de quantização (4-bit / 8-bit)
- Deploy com Docker
- Integração com banco vetorial dedicado
- Avaliação automática utilizando LLM-as-a-Judge
- Monitoramento de inferência

---


> [!NOTE]
>
> ### 🏗️ Decisões de Engenharia
>
> Este projeto foi estruturado com foco em **coesão arquitetural**, **manutenibilidade** e **facilidade de validação experimental**.
>
> A combinação entre **RAG (ChromaDB)**, **especialização via LoRA**, **API desacoplada em FastAPI** e uma **interface web hospedada** permite que cada componente seja desenvolvido, testado e evoluído de forma independente.
>
> Essa abordagem reflete uma preocupação com critérios de engenharia frequentemente exigidos em ambientes acadêmicos e profissionais, priorizando não apenas o desempenho do modelo, mas também a qualidade da solução como um todo.


## Licença

Este projeto é destinado a fins acadêmicos e de pesquisa. Ajuste a licença conforme a necessidade do seu ambiente de desenvolvimento.

---
---
### REFERÊNCIAS

[1] THOMAS, George B.; WEIR, Maurice D.; HASS, Joel. **Cálculo**. 12. ed. São Paulo: Pearson Education do Brasil, 2012.

[2] WINTERLE, Paulo. **Vetores e Geometria Analítica**. 2. ed. São Paulo: Pearson Education do Brasil, 2004.

[3] KODAKANDLA, N. **Scaling AI responsibly: Leveraging MLOps for sustainable machine learning deployments**. International Journal of Science and Research Archive, v. 13, n. 1, p. 3447–3455, 2024.

[4] BAYRAM, F.; AHMED, B. S. **Towards trustworthy machine learning in production: An overview of the robustness in mlops approach**. ACM Computing Surveys, v. 57, n. 5, p. 1–35, 2025.

[5] HANCHUK, D. O.; SEMERIKOV, S. O. **Implementing MLOps practices for effective machine learning model deployment: A meta synthesis**. In: AREdu, 2024, p. 329–337, 2024.

---


## Autor
**José Moisés Lopes da Silva** *Acadêmico em Engenharia da Computação - UFRN* [🔗 LinkedIn](https://www.linkedin.com/in/josé-moisés/) | [📧 josemoiseslopesdasilva580@gmail.com](mailto:seuemail@email.com)
