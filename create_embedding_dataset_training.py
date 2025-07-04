import csv
import json
import re
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from transformers import BertTokenizer, BertModel
import torch
import joblib

# Caminhos dos arquivos
comments_path = "comments.txt"
labels_path = "target.txt"
output_csv = "training_embeddings_tfidf.csv"

# Função de limpeza dos comentários
def formatComment(comment):
    comment = re.sub(r"\*|\[|\]|#|!|,|\.|\"|;|\?|\(|\)|`.*?`", "", comment)
    comment = re.sub(r"\.|\(|\)|<|>", " ", comment)
    comment = ' '.join(comment.split())
    return comment

def formatComments(comments):
    for index, comment in enumerate(comments):
        comments[index] = formatComment(comment)

# Carrega os comentários e rótulos
with open(comments_path, encoding="utf-8") as f:
    comments = [line.strip() for line in f if line.strip()]

with open(labels_path, encoding="utf-8") as g:
    labels = [line.strip() for line in g if line.strip()]

# Limpa os comentários
formatComments(comments)

# Cria pipeline só para vetorização (CountVectorizer + TfidfTransformer)
vectorizer_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer())
])

print(" Ajustando pipeline de vetorização nos comentários limpos...")
tfidf_matrix = vectorizer_pipeline.fit_transform(comments)

print(" Shape dos vetores TF-IDF:", tfidf_matrix.shape)

# Salva os dados em CSV com a estrutura pedida
print(" Salvando vetores no CSV...")
with open(output_csv, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["mensagem_original", "mensagem_limpa", "classe", "embedding"])

    for i in tqdm(range(len(comments))):
        writer.writerow([
            comments[i],
            comments[i],
            int(labels[i]),
            json.dumps(tfidf_matrix[i].toarray().flatten().tolist())
        ])

print(f"\n Vetores TF-IDF salvos em: {output_csv}")
joblib.dump(vectorizer_pipeline, "vectorizer_pipeline.joblib")

comments_path = "comments.txt"
labels_path = "target.txt"
output_csv = "training_embeddings_bert.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to(device)
model.eval()

def formatComment(comment):
    comment = re.sub(r"\*|\[|\]|#|!|,|\.|\"|;|\?|\(|\)|`.*?`", "", comment)
    comment = re.sub(r"\.|\(|\)|<|>", " ", comment)
    comment = ' '.join(comment.split())
    return comment

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[0][0]
    return cls_embedding.cpu().numpy().tolist()

with open(comments_path, encoding="utf-8") as f:
    comments = [formatComment(line.strip()) for line in f if line.strip()]
with open(labels_path, encoding="utf-8") as g:
    labels = [line.strip() for line in g if line.strip()]

with open(output_csv, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["mensagem_original", "mensagem_limpa", "classe", "embedding"])
    for i in tqdm(range(len(comments))):
        emb = get_bert_embedding(comments[i])
        writer.writerow([comments[i], comments[i], int(labels[i]), json.dumps(emb)])

print(f"\n Vetores BERT salvos em: {output_csv}")