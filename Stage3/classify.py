import os
import json
import pandas as pd
import numpy as np
import faiss
import torch

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def get_kb(report_path):
    chunks = []
    for filename in os.listdir(report_path):
        r = PdfReader(os.path.join(report_path, filename))
        text = ""
        for page in r.pages:
            text += page.extract_text()
        for i in range(0, len(text), 512):
                chunk = text[i:i+512]
                chunks.append(chunk)

    print(f"There are {len(chunks)} chunks for the kb")
    print(chunks[-2])
    return chunks

def embed_chunks(chunks, embedder):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    return embeddings

def index_embeddings(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def classify_article(text, index, chunks, embedder, tokenizer, model, n=1):
    text_embedding = embedder.encode([text])
    distances, indices = index.search(text_embedding, n)
    evidence = " ".join([chunks[i] for i in indices[0]])
    #print(evidence)

    inputs = tokenizer(evidence, text, return_tensors='pt', truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label_idx = torch.argmax(probs).item()

    label = model.config.id2label[label_idx]
    print(model.config.id2label)
    print(probs[0].tolist())
    print(label)

    if probs[0].tolist()[0] > probs[0].tolist()[2]:
        print("no!")
        return "no"
    else:
        print("yes!")
        return "yes"

    #return label

def classify_articles(art_path, output, index, chunks, embedder, tokenizer, model):
    predictions = []

    with open(art_path, "r", encoding="utf-8") as f:
        for l in f:
            article = json.loads(l)
            idx = article["Index"]
            text = article.get("Text")
            classification = classify_article(text, index, chunks, embedder, tokenizer, model)
            predictions.append({"index": idx, "real_news": classification})

    df = pd.DataFrame(predictions)
    df.to_csv(output, index=False, encoding="utf-8")

def main():

    report_folder = "arti/"
    articles_for_classification_path = "without_assessment.jsonl"
    output = "group58_stag22e3.csv"

    #load trained model
    tokenizer = AutoTokenizer.from_pretrained("./bart-mnli-claim-checker4")
    model = AutoModelForSequenceClassification.from_pretrained("./bart-mnli-claim-checker4")
    print(model.config.id2label)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # get chunks
    chunks = get_kb(report_folder)

    # generate embeddings and index
    # similar to https://medium.com/@nithishkotte353/building-a-retrieval-augmented-generation-rag-pipeline-using-python-597718d28722
    embeddings = embed_chunks(chunks, embedder)
    index = index_embeddings(embeddings)

    classify_articles(articles_for_classification_path, output, index, chunks, embedder, tokenizer, model)
    #predictions done!!

if __name__ == "__main__":
    main()
