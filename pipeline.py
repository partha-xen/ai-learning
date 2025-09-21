import os
from openai import OpenAI
import numpy as np
import pandas as pd
import chromadb

# Constants
DATASET_PATH = "train-shortened.csv"
TEXT_COLUMN = "Question"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 16
COLLECTION_NAME = "medical_docs"
PERSIST_DIR = "vectorstore/chroma"
TOP_K_RESULTS = 3

# Menu options
MENU_OPTIONS = {
    "1": "Load, generate and store embeddings",
    "1f": "Load, generate and store embeddings (force overwrite)",
    "2": "Read embeddings from store and print stats",
    "3": "Query the database",
    "4": "Get RAG answer (AI-generated response)",
    "5": "Exit",
}

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


# 1. Load Dataset
def load_dataset(path: str, text_col: str = TEXT_COLUMN) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["id"] = [f"doc{i}" for i in range(len(df))]
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in dataset")
    return df


# 2. Generate Embeddings (batched)
def embed_texts(texts, model=EMBEDDING_MODEL, batch_size=BATCH_SIZE):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        for item in resp.data:
            embeddings.append(item.embedding)
    return np.array(embeddings)


# 3. Add Embeddings to DataFrame
def add_embeddings_to_df(df: pd.DataFrame, text_col=TEXT_COLUMN, model=EMBEDDING_MODEL):
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
    collection_name=COLLECTION_NAME,
    persist_dir=None,
    text_col=TEXT_COLUMN,
    force_overwrite=False,
):
    client = init_chroma(persist_dir)

    # Handle existing collection
    try:
        existing_collection = client.get_collection(collection_name)
        if existing_collection:
            if force_overwrite:
                print(f"⚠️  Deleting existing collection: {collection_name}")
                client.delete_collection(collection_name)
            else:
                print(f"❌ Collection '{collection_name}' already exists!")
                print(
                    "Use option '1f' to force overwrite, or choose a different collection name."
                )
                raise ValueError(
                    f"Collection '{collection_name}' already exists. Use force_overwrite=True to replace it."
                )
    except Exception:
        # Collection doesn't exist, which is fine
        pass

    # Create new collection
    collection = client.create_collection(collection_name)
    collection.add(
        documents=df[text_col].tolist(),
        embeddings=df["embedding"].tolist(),
        ids=df["id"].astype(str).tolist(),
    )
    return collection


# 6. Query Vector DB
def query_collection(
    collection, query: str, model=EMBEDDING_MODEL, top_k=TOP_K_RESULTS
):
    query_vector = client.embeddings.create(model=model, input=query).data[0].embedding
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "distances"],
    )
    return results


# 7. RAG Answer Generation
def rag_answer(collection, query: str, model=EMBEDDING_MODEL, top_k=2):
    """
    Generate an answer using RAG (Retrieval-Augmented Generation)

    Args:
        collection: ChromaDB collection to query
        query: User's question
        model: Embedding model to use
        top_k: Number of relevant documents to retrieve

    Returns:
        Generated answer string
    """
    # 1. Retrieve relevant documents using existing query_collection method
    results = query_collection(collection, query, model=model, top_k=top_k)
    retrieved_docs = results["documents"][0]  # list of top docs (strings)

    # 2. Construct prompt with retrieved context
    context_text = "\n".join(retrieved_docs)
    prompt = f"""Use the following medical information to answer the question.

Context:
{context_text}

Question: {query}

Answer:"""

    # 3. Get LLM to generate an answer using the context
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful medical assistant. Answer based on the provided context.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        temperature=0.2,
    )

    answer = completion.choices[0].message.content
    return answer.strip()


def run():
    while True:
        print("\n=== Vector Database Pipeline ===")
        for key, value in MENU_OPTIONS.items():
            print(f"{key}. {value}")

        choice = input("\nPlease select an option (1, 1f, 2-5): ").strip()

        if choice == "1":
            load_and_store_embeddings(force_overwrite=False)
        elif choice == "1f":
            load_and_store_embeddings(force_overwrite=True)
        elif choice == "2":
            read_and_print_stats()
        elif choice == "3":
            query_database()
        elif choice == "4":
            get_rag_answer()
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1, 1f, 2-5.")


def load_and_store_embeddings(force_overwrite=False):
    print("\n=== Loading and Storing Embeddings ===")
    if force_overwrite:
        print("⚠️  Force overwrite mode enabled - existing collection will be replaced")

    # 1. Load dataset
    df = load_dataset(DATASET_PATH, text_col=TEXT_COLUMN)
    print("After Load")
    print("Columns:")
    print(df.columns.tolist())
    print("Shape:", df.shape)
    print("Head")
    print(df.head(5))

    # 2. Generate embeddings
    print("\nGenerating embeddings...")
    df = add_embeddings_to_df(df, text_col=TEXT_COLUMN)
    print("After Add Embeddings")
    print("Columns:")
    print(df.columns.tolist())
    print("Embedding shape:", np.array(df["embedding"].iloc[0]).shape)

    # 3. Index in Chroma (persistent)
    print("\nIndexing in ChromaDB...")
    collection = index_documents(
        df,
        collection_name=COLLECTION_NAME,
        persist_dir=PERSIST_DIR,
        text_col=TEXT_COLUMN,
        force_overwrite=force_overwrite,
    )
    print(f"Successfully indexed {len(df)} documents!")
    print(f"Collection name: {COLLECTION_NAME}")
    print("Collection details: ", collection)
    print(f"Persist directory: {PERSIST_DIR}")


def read_and_print_stats():
    print("\n=== Reading from Store and Printing Stats ===")
    try:
        # Initialize Chroma client to read existing data
        chroma_client = init_chroma(persist_dir=PERSIST_DIR)
        collection = chroma_client.get_collection(COLLECTION_NAME)

        # Get collection stats
        count = collection.count()
        print(f"Total documents in collection: {count}")

        # Get a few sample documents
        sample_results = collection.get(limit=3, include=["documents"])
        print("\nSample documents:")
        for i, doc in enumerate(sample_results["documents"][:3]):
            print(f"{i+1}. {doc[:100]}...")

    except Exception as e:
        print(f"Error reading from store: {e}")
        print("Make sure you've run option 1 first to create the embeddings.")


def query_database():
    print("\n=== Query Database ===")
    try:
        # Initialize Chroma client
        chroma_client = init_chroma(persist_dir=PERSIST_DIR)
        collection = chroma_client.get_collection(COLLECTION_NAME)

        # Get user input
        q = input("Please enter your question: ").strip()

        if not q:
            print("No question entered. Returning to main menu.")
            return

        print(f"\nSearching for: '{q}'")
        results = query_collection(collection, q, top_k=TOP_K_RESULTS)

        print(f"\nFound {len(results['documents'][0])} similar documents:")
        for i, (doc, distance) in enumerate(
            zip(results["documents"][0], results["distances"][0])
        ):
            print(f"\n{i+1}. (Similarity: {1-distance:.3f})")
            print(f"   {doc}")

    except Exception as e:
        print(f"Error querying database: {e}")
        print("Make sure you've run option 1 first to create the embeddings.")


def get_rag_answer():
    print("\n=== Get RAG Answer ===")
    try:
        # Initialize Chroma client
        chroma_client = init_chroma(persist_dir=PERSIST_DIR)
        collection = chroma_client.get_collection(COLLECTION_NAME)

        # Get user input
        q = input("Please enter your question: ").strip()

        if not q:
            print("No question entered. Returning to main menu.")
            return

        print(f"\nGenerating AI answer for: '{q}'")
        print("Retrieving relevant documents and generating response...")

        # Get RAG answer
        answer = rag_answer(collection, q, top_k=2)

        print("\n🤖 AI Answer:")
        print("=" * 50)
        print(answer)
        print("=" * 50)

        # Also show the retrieved documents for transparency
        print("\n📚 Retrieved Documents:")
        results = query_collection(collection, q, top_k=2)
        for i, (doc, distance) in enumerate(
            zip(results["documents"][0], results["distances"][0])
        ):
            print(f"\n{i+1}. (Similarity: {1-distance:.3f})")
            print(f"   {doc[:150]}...")

    except Exception as e:
        print(f"Error generating RAG answer: {e}")
        print("Make sure you've run option 1 first to create the embeddings.")


if __name__ == "__main__":
    run()
