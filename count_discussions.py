import pandas as pd
from collections import defaultdict

# Caminho para o arquivo com as classificações
csv_path = "embeddings_classification.csv"

# Mapeamento das classes para nomes legíveis (ajuste conforme necessário)
classe_labels = {
    13: "Others",
    4: "Error",
    8: "Code Organization/ Refactoring",
    10: "High Level Method Semantics & Design"
}

# Carrega o CSV
df = pd.read_csv(csv_path)
df["classe"] = df["classe"].astype(int)

# Filtra apenas mensagens classificadas
df_classified = df[df["classe"] != -1]

# Agrupa por projeto e classe
project_counts = defaultdict(lambda: defaultdict(int))

for _, row in df_classified.iterrows():
    projeto = row["project"]
    classe = row["classe"]
    project_counts[projeto][classe] += 1

# Imprime relatório
print("\n RELATÓRIO DE CLASSIFICAÇÕES POR PROJETO\n")

for projeto in sorted(project_counts):
    total = sum(project_counts[projeto].values())
    print(f" Projeto: {projeto}")
    for classe in sorted(project_counts[projeto]):
        nome = classe_labels.get(classe, f"Classe {classe}")
        qtd = project_counts[projeto][classe]
        print(f"   - {nome:15}: {qtd}")
    print(f"   Total de mensagens classificadas: {total}")
    print("-" * 40)

# Visão geral
print("\n Resumo Geral por Classe:")
geral = defaultdict(int)
for projeto in project_counts:
    for classe, qtd in project_counts[projeto].items():
        geral[classe] += qtd

for classe in sorted(geral):
    nome = classe_labels.get(classe, f"Classe {classe}")
    print(f"   - {nome:15}: {geral[classe]}")
print("\n Fim do relatório.")
