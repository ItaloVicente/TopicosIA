import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

# Treinamento do modelo MLP

training_file = "training_embeddings_tfidf.csv"
classes_relevantes = [1, 4, 8, 10]
class_to_idx = {cls: idx for idx, cls in enumerate(classes_relevantes)}
idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

df_train = pd.read_csv(training_file)
df_train = df_train[df_train["classe"] != 13]
df_train = df_train[df_train["classe"].isin(classes_relevantes)]
df_train["classe_mapped"] = df_train["classe"].map(class_to_idx)
df_train["embedding"] = df_train["embedding"].apply(json.loads)

X = np.array(df_train["embedding"].tolist())
y = df_train["classe_mapped"].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Balanceamento
train_df = pd.DataFrame({"embedding": X_train.tolist(), "classe_mapped": y_train})
classe_0 = train_df[train_df["classe_mapped"] == 0].sample(n=200, random_state=42)
outras = train_df[train_df["classe_mapped"] != 0]
train_df_bal = pd.concat([classe_0, outras], ignore_index=True)

X_bal = train_df_bal["embedding"].tolist()
y_bal = train_df_bal["classe_mapped"].tolist()

smote_strategy = {i: 200 for i in range(1, len(classes_relevantes))}
smote = SMOTE(sampling_strategy=smote_strategy, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_bal, y_bal)
X_train_resampled = np.array(X_train_resampled)

# PCA
pca = PCA(n_components=0.99, random_state=42)
X_train_pca = pca.fit_transform(X_train_resampled)
X_test_pca = pca.transform(X_test)

# MLP
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=5000, random_state=42)
mlp.fit(X_train_pca, y_train_resampled)

# Avaliação rápida (Apenas para checar se o modelo treinado tem as mesmas acaracterísticas do achado anteriormente)
acc = mlp.score(X_test_pca, y_test)
print(f"Acurácia no teste: {acc:.4f}")

# Classificação em batches (mais leve) para meu computador conseguir processar

classify_file = "embeddings_classification.csv"
df_classify = pd.read_csv(classify_file)
df_classify["classe"] = df_classify["classe"].astype(int)

# Identifica linhas a serem classificadas
to_classify_idx = df_classify[df_classify["classe"] == -1].index
print(f"Total a classificar: {len(to_classify_idx)}")

batch_size = 1000  # pode ajustar para 500 se continuar pesado
for start in range(0, len(to_classify_idx), batch_size):
    end = min(start + batch_size, len(to_classify_idx))
    batch_idx = to_classify_idx[start:end]

    print(f"Classificando mensagens {start} a {end}...")

    embeddings = df_classify.loc[batch_idx, "embedding"].apply(json.loads).tolist()
    X_batch = np.array(embeddings)
    X_batch_pca = pca.transform(X_batch)
    y_pred = mlp.predict(X_batch_pca)
    y_pred_orig = [idx_to_class[p] for p in y_pred]

    df_classify.loc[batch_idx, "classe"] = y_pred_orig

# Salva o CSV
df_classify.to_csv(classify_file, index=False)
print("✅ Classificação concluída e arquivo atualizado.")
