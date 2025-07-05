import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelBinarizer

# Tipos de vetorização (bert e tfidf)
L = ["training_embeddings_bert.csv", "training_embeddings_tfidf.csv"]

for training in L:
    df = pd.read_csv(training)
    df["classe"] = df["classe"].astype(int)

    # Filtra apenas classes relevantes para duplicação de código (na realidade, a classe 13 nao eh, estou apenas colocando para termos a classe "Outros")
    classes_relevantes = [13, 4, 8, 10]
    df = df[df["classe"].isin(classes_relevantes)]

    # Imprime a quantidade de exemplos por classe relevante
    contagem_classes = df["classe"].value_counts().sort_index()
    print("Quantidade por classe relevante:")
    print(contagem_classes)

    # Mapeia classes originais para índices contínuos
    class_to_idx = {cls: idx for idx, cls in enumerate(classes_relevantes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    df["classe_mapped"] = df["classe"].map(class_to_idx)

    # Converte embeddings de string para lista
    df["embedding"] = df["embedding"].apply(json.loads)

    # Separa X e y mapeados
    X = np.array(df["embedding"].tolist())
    y = df["classe_mapped"].tolist()

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Reconstrói DataFrame
    train_df = pd.DataFrame({"embedding": X_train.tolist(), "classe_mapped": y_train})

    # Reduz a classe 0 (que corresponde à original 1) para 200 exemplos
    classe_0 = train_df[train_df["classe_mapped"] == 0].sample(n=200, random_state=42)

    # Mantém as outras classes
    outras = train_df[train_df["classe_mapped"] != 0]

    # Junta para formar novo conjunto de treino balanceado
    train_df_bal = pd.concat([classe_0, outras], ignore_index=True)

    # Separa X e y balanceados
    X_bal = train_df_bal["embedding"].tolist()
    y_bal = train_df_bal["classe_mapped"].tolist()

    # SMOTE
    smote_strategy = {i: 200 for i in range(1, len(classes_relevantes))}
    smote = SMOTE(sampling_strategy=smote_strategy, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_bal, y_bal)
    X_train_resampled = np.array(X_train_resampled)

    print("\nTotal por classe relevante após balanceamento:")
    for classe_idx, qtd in Counter(y_train_resampled).items():
        print(f"Classe original {idx_to_class[classe_idx]} (mapeada {classe_idx}): {qtd} exemplos")

    print(f"\nTotal de exemplos no conjunto de treino após balanceamento: {len(y_train_resampled)}")

    # PCA para redução dimensional (≥ 99% da variância)
    pca = PCA(n_components=0.99)
    X_train_pca = pca.fit_transform(X_train_resampled)
    X_test_pca = pca.transform(X_test)
    print(f"\n Reduzido de {X.shape[1]} para {X_train_pca.shape[1]} dimensões com PCA.")

    # Atualiza dados para treino/teste com PCA
    X_train, X_test, y_train, y_test = X_train_pca, X_test_pca, y_train_resampled, y_test

    # Função de avaliação
    def evaluate_model(name, model):
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted')

        try:
            lb = LabelBinarizer()
            y_test_bin = lb.fit_transform(y_test)
            y_pred_prob = model.predict_proba(X_test)
            auc = roc_auc_score(y_test_bin, y_pred_prob, average='weighted', multi_class='ovr')
        except:
            auc = "N/A"

        print(f"\n Modelo: {name}")
        print(f"Acurácia : {acc:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"Precisão : {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"AUC      : {auc}")

    # Treinamento e avaliação dos modelos

    # Logistic Regression
    for C in [1, 10, 100]:
        for max_iter in [100, 300]:
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(X_train, y_train)
            evaluate_model(f"Logistic(C={C}, max_iter={max_iter})", model)

    # SVM
    for kernel in ['linear', 'rbf']:
        for C in [1, 10, 20, 50]:
            model = SVC(kernel=kernel, C=C, probability=True)
            model.fit(X_train, y_train)
            evaluate_model(f"SVM(kernel={kernel}, C={C})", model)

    # Random Forest
    for depth in [5, 10, 20]:
        model = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        evaluate_model(f"RandomForest(max_depth={depth})", model)

    # MLP
    for layers in [(100,), (100, 50), (200, 100)]:
        model = MLPClassifier(hidden_layer_sizes=layers, max_iter=5000, random_state=42)
        model.fit(X_train, y_train)
        evaluate_model(f"MLP(layers={layers})", model)

    # XGBoost
    for depth in [3, 6, 10]:
        model = XGBClassifier(eval_metric="mlogloss",
                              n_estimators=100, max_depth=depth, learning_rate=0.1)
        model.fit(X_train, y_train)
        evaluate_model(f"XGBoost(max_depth={depth})", model)

# A partir disso, vimos que o melhor modelo que se sobresaiu foi o SVM(linear, C = 1) com acurácia próxima de 73%, AUC = 0.86 e F1-score = 0.73
