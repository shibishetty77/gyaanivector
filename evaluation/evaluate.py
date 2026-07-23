import sys
import os
import json
import pandas as pd
from datasets import load_from_disk

# Setup path so we can import backend modules without modifying production code
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, BACKEND_DIR)

from app.config import get_settings
# pyrefly: ignore [missing-import]
from app.services.llm import get_llm
# pyrefly: ignore [missing-import]
from app.rag.embeddings import get_embeddings

# --- RAGAS compatibility patch ---
import types
try:
    import langchain_community.chat_models
    import langchain_community.llms
    
    if not hasattr(langchain_community.chat_models, 'vertexai'):
        langchain_community.chat_models.vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
        sys.modules["langchain_community.chat_models.vertexai"] = langchain_community.chat_models.vertexai
    
    if not hasattr(langchain_community.chat_models.vertexai, 'ChatVertexAI'):
        langchain_community.chat_models.vertexai.ChatVertexAI = type("ChatVertexAI", (object,), {})
        
    if not hasattr(langchain_community.llms, 'VertexAI'):
        langchain_community.llms.VertexAI = type("VertexAI", (object,), {})
except Exception:
    pass
# ---------------------------------

from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas import EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

def run_evaluation(dataset_dir="ragas_dataset", output_json="results.json", output_csv="results.csv"):
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found.")
        print("Please run build_dataset.py first.")
        return

    print(f"Loading dataset from {dataset_dir}...")
    hf_dataset = load_from_disk(dataset_dir)
    
    # RAGAS 0.4.x expects a specific column schema (SingleTurnSample).
    # We rename our standard columns to match RAGAS's internal fields:
    column_mapping = {
        "question": "user_input",
        "answer": "response",
        "contexts": "retrieved_contexts",
        "ground_truth": "reference"
    }
    
    rename_dict = {k: v for k, v in column_mapping.items() if k in hf_dataset.column_names}
    if rename_dict:
        hf_dataset = hf_dataset.rename_columns(rename_dict)
    
    # Convert HF Dataset to EvaluationDataset
    eval_dataset = EvaluationDataset.from_hf_dataset(hf_dataset)
    
    print("Initializing local LLM and Embeddings from backend configuration...")
    # Change directory so config.py loads the correct .env file
    original_cwd = os.getcwd()
    os.chdir(BACKEND_DIR)
    try:
        settings = get_settings()
        langchain_llm = get_llm()
        langchain_embeddings = get_embeddings()
    finally:
        os.chdir(original_cwd)

    # Wrap LangChain components for RAGAS
    evaluator_llm = LangchainLLMWrapper(langchain_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

    print("Starting evaluation (this may take some time depending on dataset size)...")
    
    # Use standard 0.4.x metric classes
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]
    
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    
    df = result.to_pandas()
    
    print("\n" + "="*40)
    print("EVALUATION RESULTS SUMMARY")
    print("="*40)
    summary_series = df.mean(numeric_only=True)
    for metric_name, score in summary_series.items():
        print(f"{metric_name.replace('_', ' ').title():<20} : {score:.2f}")
    print("="*40)

    # Save results
    df.to_csv(output_csv, index=False)
    
    # Save JSON details
    final_output = {
        "summary": summary_series.to_dict(),
        "details": df.to_dict(orient="records")
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)
        
    print(f"\nResults successfully saved to {output_csv} and {output_json}")

if __name__ == "__main__":
    run_evaluation()
