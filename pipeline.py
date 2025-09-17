import os
from openai import OpenAI
import numpy as np
import pandas as pd
import chromadb

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


# 1. Load Dataset
def load_dataset(path: str, text_col: str = "text") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["id"] = [f"doc{i}" for i in range(len(df))]
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in dataset")
    return df


# 2. Generate Embeddings (batched)
def embed_texts(texts, model="text-embedding-3-small", batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        for item in resp.data:
            embeddings.append(item.embedding)
    return np.array(embeddings)


# 3. Add Embeddings to DataFrame
def add_embeddings_to_df(
    df: pd.DataFrame, text_col="text", model="text-embedding-3-small"
):
    texts = df[text_col].tolist()
    embeddings = embed_texts(texts, model=model)
    df["embedding"] = embeddings.tolist()
    return df


# 4. Initialize Chroma
def init_chroma(persist_dir: str = None):
    if persist_dir:
        client = chromadb.PersistentClient(path=persist_dir)
    else:
        client = chromadb.Client()
    return client


# 5. Add to Vector DB
def index_documents(
    df: pd.DataFrame,
    collection_name="medical_docs",
    persist_dir=None,
    text_col="default",
):
    client = init_chroma(persist_dir)
    collection = client.create_collection(collection_name)
    collection.add(
        documents=df[text_col].tolist(),
        embeddings=df["embedding"].tolist(),
        ids=df["id"].astype(str).tolist(),
    )
    return collection


# 6. Query Vector DB
def query_collection(collection, query: str, model="text-embedding-3-small", top_k=3):
    query_vector = client.embeddings.create(model=model, input=query).data[0].embedding
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "distances"],
    )
    return results


def run():
    # 1. Load dataset
    df = load_dataset("train-shortened.csv", text_col="Question")
    print("After Load")
    print("Columns:")
    print(df.columns.tolist())
    print("Head")
    print(df.head(10))

    # # 2. Generate embeddings
    df = add_embeddings_to_df(df, text_col="Question")
    print("After Add Embeddings")
    print("Columns:")
    print(df.columns.tolist())
    print("Head")
    print(df.head(10))

    # 3. Index in Chroma (persistent)
    collection = index_documents(
        df,
        collection_name="medical_docs",
        persist_dir="vectorstore/chroma",
        text_col="Question",
    )

    # # 4. Query
    q = "What should I worry about?"
    results = query_collection(collection, q, top_k=2)
    print("Query:", q)
    print("Results:", results["documents"])


if __name__ == "__main__":
    run()
