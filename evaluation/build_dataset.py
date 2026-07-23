import sys
import os
import json
import asyncio
import requests
from datasets import Dataset

# Setup path so we can import backend modules without modifying production code
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, BACKEND_DIR)

# We import the backend components only to run the retriever locally and get raw texts
from app.config import get_settings
# pyrefly: ignore [missing-import]
from app.rag.embeddings import get_embeddings
# pyrefly: ignore [missing-import]
from app.rag.vector_store import get_vector_store
# pyrefly: ignore [missing-import]
from app.rag.knowledge_graph import get_knowledge_graph
# pyrefly: ignore [missing-import]
from app.rag.retriever import build_retriever

API_URL = "http://localhost:8000/chat"

async def async_build_ragas_dataset(input_json="dataset.json", output_dir="ragas_dataset"):
    print(f"Reading dataset from {input_json}...")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    # Change to backend directory so .env is loaded correctly by settings
    original_cwd = os.getcwd()
    os.chdir(BACKEND_DIR)
    
    try:
        settings = get_settings()
        embeddings = get_embeddings()
        vector_store = get_vector_store()
        knowledge_graph = get_knowledge_graph()

        for idx, item in enumerate(data):
            question = item.get("question")
            ground_truth = item.get("ground_truth")
            category = item.get("category")

            if not question or not ground_truth:
                continue

            print(f"[{idx+1}/{len(data)}] Processing question: {question}")
            
            # 1. Get the answer from the production API (synchronous HTTP request)
            payload = {"question": question}
            if category:
                payload["category"] = category
                
            try:
                # We can use asyncio.to_thread to not block if we wanted, but sync is fine here
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                result = response.json()
                answer = result.get("answer", "")
            except Exception as e:
                print(f"  -> API Error: {e}")
                print("  -> Ensure backend is running at localhost:8000. Skipping...")
                continue

            # 2. Directly call the retriever to get raw text contexts
            try:
                retriever = build_retriever(
                    embeddings=embeddings, 
                    vector_store=vector_store, 
                    k=settings.default_match_count, 
                    category=category,
                    knowledge_graph=knowledge_graph
                )
                documents = await retriever.ainvoke(question)
                contexts = [doc.page_content for doc in documents]
            except Exception as e:
                print(f"  -> Retriever Error: {e}")
                contexts = []
                
            if not contexts:
                print(f"  -> Warning: No contexts returned for question '{question}'.")

            ragas_data["question"].append(question)
            ragas_data["answer"].append(answer)
            ragas_data["contexts"].append(contexts)
            ragas_data["ground_truth"].append(ground_truth)
    finally:
        os.chdir(original_cwd)

    if not ragas_data["question"]:
        print("No valid questions were processed. Dataset was not created.")
        return

    dataset = Dataset.from_dict(ragas_data)
    dataset.save_to_disk(output_dir)
    print(f"\nDataset successfully built and saved to {output_dir}")

def build_ragas_dataset(input_json="dataset.json", output_dir="ragas_dataset"):
    asyncio.run(async_build_ragas_dataset(input_json, output_dir))

if __name__ == "__main__":
    build_ragas_dataset()
