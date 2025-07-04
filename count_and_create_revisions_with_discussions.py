import os
import csv
import json
import re
import joblib
from tqdm import tqdm

CSV_PATH = "embeddings_classification.csv"
PIPELINE_PATH = "vectorizer_pipeline.joblib"  # pipeline salvo anteriormente

def is_relevant_comment(message):
    return "\n\n" in message.strip()

def clean_comment(comment):
    comment = re.sub(r"Patch Set \d+:\s?", "", comment)
    comment = re.sub(r"Line:\d+,\s?.*?->", "", comment)
    comment = re.sub(r"\b(?:[\w\-\.]+/)+[\w\-\.]+\.\w+\b", "", comment)
    comment = re.sub(r"http[s]?://\S+", "", comment)
    comment = re.sub(r"[^\w\s]", "", comment)
    comment = re.sub(r"\s+", " ", comment)
    return comment.strip()

def initialize_csv(path):
    if not os.path.exists(path):
        with open(path, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["project", "review", "revision", "mensagem", "mensagem_limpa", "embedding", "classe"])

def write_embedding_row(writer, project, review, revision, mensagem, mensagem_clean, embedding, classe: int = -1):
    writer.writerow([
        project,
        review,
        revision,
        mensagem,
        mensagem_clean,
        json.dumps(embedding),
        classe
    ])

def scan_type_clones_tfidf(clones_dir="type_clones/"):
    projects = ["platform.ui", "egit", "jgit", "couchbase-jvm-core", "couchbase-java-client", "spymemcached"]
    processed_pairs = set()

    data_for_vectorization = []
    data_raw = []  # tuplas (project, review, revision, mensagem_original, mensagem_limpa)

    for project in projects:
        csv_path = os.path.join(clones_dir, f"{project}.csv")
        print(f"\nProcessando projeto: {project}", flush=True)
        with open(csv_path, newline='', encoding='utf-8', errors='ignore') as csvfile:
            reader = csv.reader(csvfile)
            for row in tqdm(reader):
                if len(row) < 2:
                    continue
                review, revision = row[0].strip(), row[1].strip()
                key = (project, review, revision)
                if key in processed_pairs:
                    continue
                processed_pairs.add(key)

                discussion_path = os.path.join(f"discussion/{project}/{review}/{review}_rev{revision}_discussion.txt")
                if not os.path.exists(discussion_path):
                    continue

                with open(discussion_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                if "COMMENTS" not in content:
                    continue

                comments_section = content.split("COMMENTS", 1)[-1]
                comment_blocks = comments_section.split("Author:")

                for block in comment_blocks[1:]:
                    if "Message:" in block:
                        message_part = block.split("Message:", 1)[-1]
                        message_content = message_part.split("-----")[0].strip()
                        if is_relevant_comment(message_content):
                            clean = clean_comment(message_content)
                            if clean:
                                data_for_vectorization.append(clean)
                                data_raw.append((project, review, revision, message_content, clean))

    # Carrega pipeline TF-IDF já treinado
    print("\nCarregando pipeline TF-IDF salvo...")
    vectorizer_pipeline = joblib.load(PIPELINE_PATH)

    # Vetoriza com o pipeline carregado (sem fit)
    print("Vetorizando comentários com o pipeline treinado...")
    tfidf_matrix = vectorizer_pipeline.transform(data_for_vectorization)
    print("Shape dos vetores TF-IDF:", tfidf_matrix.shape)

    # Salva no CSV
    initialize_csv(CSV_PATH)
    with open(CSV_PATH, "a", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for i in tqdm(range(len(data_raw))):
            project, review, revision, mensagem_original, mensagem_limpa = data_raw[i]
            embedding = tfidf_matrix[i].toarray().flatten().tolist()
            write_embedding_row(writer, project, review, revision, mensagem_original, mensagem_limpa, embedding, -1)

    print(f"\nVetores TF-IDF salvos em: {CSV_PATH}")

# Executa a função principal
scan_type_clones_tfidf()
