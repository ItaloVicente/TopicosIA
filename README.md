# Classificador de Comentários de Revisão de Código para Clones

Este projeto implementa um pipeline de Machine Learning para treinar, avaliar e utilizar um modelo capaz de classificar comentários de revisão de código. O foco principal é analisar discussões que ocorrem em revisões que contêm clones de código, permitindo uma análise categorizada e agrupada por projeto.

## Funcionalidades

- Treina e avalia múltiplos modelos de classificação a partir de um dataset rotulado.
- Identifica o melhor modelo com base em métricas como Acurácia, F1-Score e AUC.
- Prepara um novo conjunto de dados para classificação, filtrando discussões relevantes (associadas a clones).
- Classifica os novos comentários usando o modelo treinado.
- Gera um relatório final com a contagem de cada classificação por projeto.

## Estrutura do Projeto

```
.
├── comments.txt         # Comentários de treinamento rotulados
├── target.txt           # Rótulos (labels) para os comentários de treinamento
├── discussions/         # Pasta com os dados brutos de discussões a serem classificadas
├── type_clone/          # Pasta com informações sobre quais revisões contêm clones
|
├── create_embedding_dataset_training.py # Script para vetorizar os dados de treino
├── create_model.py                      # Script para treinar e avaliar os modelos
├── count_and_create_revisions_with_discussions.py # Script para preparar os dados de inferência
├── classification_discussions.py        # Script para classificar os novos dados
├── count_classifications.py             # Script para gerar o relatório final
|
└── requirements.txt     # Lista de dependências do Python
```

## Instalação

Antes de começar, certifique-se de que você tem o Python 3.x instalado.

1.  Clone este repositório para a sua máquina local.
2.  Navegue até a pasta do projeto e instale todas as bibliotecas necessárias executando o seguinte comando no seu terminal:

```bash
pip install -r requirements.txt
```

## Como Executar o Pipeline

O processo é dividido em três etapas principais. Siga a ordem abaixo para garantir que os artefatos de cada script sejam gerados corretamente para o próximo.

### Etapa 1: Treinamento e Seleção do Modelo

Nesta etapa, vamos usar os dados rotulados (`comments.txt` e `target.txt`) para treinar e encontrar o nosso melhor modelo de classificação.

**1. Gerar os Embeddings de Treinamento**
Este script lê os dados de treinamento, realiza a limpeza e os converte em uma representação vetorial (embedding) que os modelos de machine learning podem entender.

```bash
python create_embedding_dataset_training.py
```

**2. Treinar e Avaliar os Modelos**
Após gerar os embeddings, este script treina e avalia múltiplos modelos de classificação (ex: Regressão Logística, SVM, RandomForest, MLP, etc.). Ao final, ele indica o melhor modelo com base nas métricas de avaliação e o salva ou configura para ser usado na próxima etapa.

```bash
python create_model.py
```

### Etapa 2: Classificação das Novas Discussões

Agora, com um modelo treinado, vamos classificar os dados do seu projeto.

**3. Preparar os Dados para Classificação**
Este script processa os dados da pasta `discussions/`, filtrando apenas as discussões de revisões que contêm clones (conforme os dados da pasta `type_clone/`). Em seguida, ele vetoriza os comentários usando TF-IDF para prepará-los para a classificação.

```bash
python count_and_create_revisions_with_discussions.py
```

**4. Executar a Classificação**
Utilizando o melhor modelo identificado na Etapa 1, este script classifica cada comentário preparado no passo anterior e salva os resultados em um arquivo `embeddings_classification.csv`.

```bash
python classification_discussions.py
```

### Etapa 3: Geração do Relatório Final

**5. Contar as Classificações por Projeto**
O script final lê o arquivo `embeddings_classification.csv` com todos os comentários classificados e gera um relatório que totaliza a quantidade de cada tipo de classificação, **agrupado por projeto**.

```bash
python count_classifications.py
```

## Desempenho do Modelo

O script `create_model.py` avalia diversos algoritmos. Para este projeto, o melhor modelo identificado para a tarefa foi o **MLP (Multi-layer Perceptron)** com os seguintes resultados no conjunto de teste:

- **Acurácia:** 77%
- **F1-Score (Weighted):** 75%
- **AUC:** 0.88

## Saída Final

Ao final da execução de todos os scripts, o `count_classifications.py` exibirá no console um relatório detalhado mostrando a contagem de cada categoria de comentário para cada projeto analisado, permitindo uma análise aprofundada dos tipos de discussões que ocorrem em revisões de código com clones.
