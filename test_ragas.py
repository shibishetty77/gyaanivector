import asyncio
import argparse
import logging
import os
import sys
import pandas as pd
from datasets import Dataset

# Setup path so we can import backend modules
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "backend"))
sys.path.insert(0, BACKEND_DIR)

from app.config import get_settings
from app.rag.embeddings import get_embeddings
from app.rag.vector_store import get_vector_store
from app.rag.prompt import build_prompt_template, format_docs
from app.services.llm import get_llm
from langchain_core.runnables import RunnableLambda

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

# Need to import ragas classes. We will evaluate the EvaluationDataset compatible with 0.4.x
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Sample QA Pairs based on the knowledge base (ESP32 and components)
TEST_DATA = [
    {
        "question": "What are the main differences between the ESP32-WROOM and ESP32-WROOM-32E modules?",
        "ground_truth": "The ESP32-WROOM-32E features the newer ESP32-D0WD-V3 chip which offers improved stability, security (like secure boot V2), and slightly better power efficiency compared to the older chip in the standard ESP32-WROOM. It also has a modified antenna design for better RF performance."
    },
    {
        "question": "What is the operating voltage range for the ESP32-WROOM-32E?",
        "ground_truth": "The operating voltage range is typically 3.0V to 3.6V."
    },
    {
        "question": "How do I connect a DHT22 temperature sensor to an ESP32?",
        "ground_truth": "Connect the VCC pin of the DHT22 to the 3.3V pin of the ESP32, GND to GND, and the Data pin to a digital GPIO pin on the ESP32. A 10k ohm pull-up resistor is typically required between VCC and Data."
    },
    {
        "question": "What is the I2C address commonly used for the LCD 1602 display?",
        "ground_truth": "The most common I2C addresses for the LCD 1602 module (when used with an I2C adapter) are 0x27 or 0x3F."
    },
    {
        "question": "What is the difference in temperature reading accuracy between DHT11 and DHT22?",
        "ground_truth": "The DHT22 is more accurate, typically offering ±0.5°C temperature accuracy, whereas the DHT11 offers ±2.0°C accuracy."
    }
]

async def run_l1_retrieval(embeddings, vector_store):
    print("\n" + "="*50)
    print("LEVEL 1: RETRIEVAL UNIT TESTS (FAST, NO LLM)")
    print("="*50)
    
    # We will test vector optimization by checking different K values
    k_values_to_test = list(range(1, 8))
    
    for item in TEST_DATA:
        question = item["question"]
        print(f"\nQ: {question}")
        
        q_emb = embeddings.embed_query(question)
        
        for k in k_values_to_test:
            from app.rag.retriever import build_retriever
            retriever = build_retriever(embeddings, vector_store, k, category=None)
            
            # Using raw similarity search to also show scores
            rows = vector_store.similarity_search(q_emb, k=k, category=None)
            docs = retriever._rows_to_documents(rows)
            
            print(f"  [k={k}] Retrieved {len(docs)} documents.")
            for i, d in enumerate(docs):
                # Retrieve similarity if available from metadata or raw rows
                sim = rows[i].get("similarity", "N/A") if i < len(rows) else "N/A"
                print(f"    -> Doc {i+1} | Score: {sim} | Preview: {d.page_content[:60]}...")
                
    print("\n--- L1 Retrieval Test Complete ---")

async def build_evaluation_dataset(embeddings, vector_store, llm, k=4):
    print("\nGenerating answers for dataset using LLM...")
    dataset_rows = []
    
    prompt = build_prompt_template()
    chain = prompt | llm | __import__('langchain_core').output_parsers.StrOutputParser()
    from app.rag.retriever import build_retriever
    
    for item in TEST_DATA:
        question = item["question"]
        q_emb = embeddings.embed_query(question)
        
        retriever = build_retriever(embeddings, vector_store, k, category=None)
        rows = vector_store.similarity_search(q_emb, k=k, category=None)
        docs = retriever._rows_to_documents(rows)
        
        context = await RunnableLambda(format_docs).ainvoke(docs)
        answer = await chain.ainvoke({"question": question, "context": context, "chat_history": []})
        
        contexts_list = [d.page_content for d in docs]
        
        # RAGAS 0.4.x requires specific column names:
        # question -> user_input, answer -> response, contexts -> retrieved_contexts, ground_truth -> reference
        dataset_rows.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts_list,
            "reference": item["ground_truth"]
        })
        
    df = pd.DataFrame(dataset_rows)
    hf_dataset = Dataset.from_pandas(df)
    eval_dataset = EvaluationDataset.from_hf_dataset(hf_dataset)
    return eval_dataset

def run_ragas_evaluation(dataset, metrics_to_run, evaluator_llm, evaluator_embeddings, level_name):
    print("\n" + "="*50)
    print(f"LEVEL: {level_name}")
    print("="*50)
    
    result = evaluate(
        dataset=dataset,
        metrics=metrics_to_run,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    
    df = result.to_pandas()
    
    print("\n" + "-"*40)
    print("EVALUATION RESULTS SUMMARY")
    print("-"*40)
    summary_series = df.mean(numeric_only=True)
    for metric_name, score in summary_series.items():
        print(f"{metric_name.replace('_', ' ').title():<20} : {score:.2f}")
    print("-"*40)
    return df

async def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation tests.")
    parser.add_argument('--l1', action='store_true', help='Retrieval unit tests (fast, no LLM)')
    parser.add_argument('--l2', action='store_true', help='RAGAS metrics only')
    parser.add_argument('--l3', action='store_true', help='Hallucination checks only')
    
    args = parser.parse_args()
    
    # If no flags passed, run all
    run_all = not (args.l1 or args.l2 or args.l3)
    
    # Load core backend configurations
    original_cwd = os.getcwd()
    os.chdir(BACKEND_DIR)
    try:
        settings = get_settings()
        embeddings = get_embeddings()
        vector_store = get_vector_store()
        llm = get_llm()
    finally:
        os.chdir(original_cwd)

    # ---------------------------
    # LEVEL 1
    # ---------------------------
    if args.l1 or run_all:
        await run_l1_retrieval(embeddings, vector_store)

    # ---------------------------
    # Prepare Evaluators for L2 / L3
    # ---------------------------
    if args.l2 or args.l3 or run_all:
        evaluator_llm = LangchainLLMWrapper(llm)
        evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)
        
        # Build the runtime dataset (k=4 by default for standard generation)
        print("\n[Notice] Building dataset for evaluation... This will invoke the LLM.")
        dataset = await build_evaluation_dataset(embeddings, vector_store, llm, k=4)

        # ---------------------------
        # LEVEL 2
        # ---------------------------
        if args.l2 or run_all:
            # Full RAGAS suite
            metrics = [ContextPrecision(), ContextRecall(), AnswerRelevancy(), Faithfulness()]
            df_l2 = run_ragas_evaluation(dataset, metrics, evaluator_llm, evaluator_embeddings, "L2 (RAGAS METRICS)")

        # ---------------------------
        # LEVEL 3
        # ---------------------------
        if args.l3 or run_all:
            # Hallucination focused
            metrics = [Faithfulness(), AnswerRelevancy()]
            df_l3 = run_ragas_evaluation(dataset, metrics, evaluator_llm, evaluator_embeddings, "L3 (HALLUCINATION CHECKS)")
            
            print("\nVector Optimization Tips for Hallucinations:")
            print("- If Faithfulness is low, the LLM is fabricating facts not in the context. Try decreasing 'k' or increasing the similarity threshold to drop irrelevant context.")
            print("- If Answer Relevancy is low, the LLM's answer doesn't address the user's question. This often happens if the context itself is irrelevant. Review Context Precision.")

if __name__ == '__main__':
    asyncio.run(main())
