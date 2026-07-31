"""
RAGAS evaluation for the graph-based retrieval modes ("graph" and
"hybrid_graph"), mirroring test_ragas.py's structure so results are
directly comparable to the vector-only run.

Unlike test_ragas.py, this script goes through `build_retriever(...,
retrieval_mode=..., knowledge_graph=..., graph_store=...)` -- the exact
same retriever construction RAGService uses internally -- instead of
calling vector_store.similarity_search directly. That's what actually
exercises GraphStore.traverse() / extract_entities_from_query() (and, for
"hybrid_graph", RRF fusion with vector + BM25) rather than pure vector
search.

Usage:
    cd backend  # NOT required -- path setup below handles it either way
    python test_ragas_graph.py                      # runs all modes, prints comparison
    python test_ragas_graph.py --mode graph          # graph only
    python test_ragas_graph.py --mode hybrid_graph    # hybrid_graph only
    python test_ragas_graph.py --mode vector          # vector only (baseline, for comparison)
    python test_ragas_graph.py --mode graph --l3      # hallucination-focused metrics only
"""

import asyncio
import argparse
import logging
import os
import sys
import pandas as pd
from datasets import Dataset

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "backend"))
if os.path.isdir(BACKEND_DIR):
    sys.path.insert(0, BACKEND_DIR)
else:
    # Script is already sitting inside backend/
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import get_settings
from app.rag.embeddings import get_embeddings
from app.rag.vector_store import get_vector_store
from app.rag.prompt import build_prompt_template, format_docs
from app.rag.retriever import build_retriever
from app.rag.graph import get_graph_store
from app.rag.knowledge_graph import get_knowledge_graph
from app.services.llm import get_llm
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- RAGAS compatibility patch (same as test_ragas.py) --------------------
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
# ---------------------------------------------------------------------------

from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Same test set as test_ragas.py -- keep these identical so vector vs graph
# results are apples-to-apples. If you add questions here, add them to
# test_ragas.py too (or better: factor TEST_DATA out into a shared module).
TEST_DATA = [
    # ESP32 core knowledge (original set)
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
    },

    # --- Filtered subset from backend/services/master_prompt.py ---
    # Your current knowledge graph (27 entities, see graph_viewer / Supabase
    # `entities` table) only covers the ESP32 family plus generic component
    # types (dht11, dht22, sensor, gpio, led, i2c, i2c_device, display,
    # lcd-1602-i2c, single_wire, push-button, input, actuator, platform).
    # It has NO nodes for STM32F401, STM32F303, Arduino Uno, INA219, BME280,
    # or DS18B20 -- so all master_prompt.py questions about those were
    # dropped here (they would score seed_entities=0 / contexts=0 in every
    # retrieval mode, not because of a retrieval quality difference, but
    # because those entities simply aren't in the ingested knowledge base).
    # If you later ingest YAML content for those platforms/sensors and
    # re-run export_graph_to_supabase.py, the dropped questions can be
    # added back -- they're preserved in git history / the previous version
    # of this file.
    {
        "question": "What GPIO pin is used for DHT11 or DHT22 DATA on an ESP32?",
        "ground_truth": "GPIO pin 4 (the integer 4), defined in firmware as #define DATA_PIN 4."
    },
    {
        "question": "What canonical platform name corresponds to mentions like 'esp', 'wroom', or 'espressif'?",
        "ground_truth": "ESP32-WROOM."
    },
    {
        "question": "Which sensor is recommended for monitoring humidity?",
        "ground_truth": "The DHT22, used to monitor humidity."
    },
    {
        "question": "Which sensor is the default choice for general temperature monitoring?",
        "ground_truth": "The DHT22."
    },
    {
        "question": "How should ESP32 pin numbers be written in firmware code, versus how they appear on the wiring canvas?",
        "ground_truth": "In firmware (Arduino framework), pin numbers must be plain integers, e.g. #define DATA_PIN 4 — never identifiers like GPIO4 or GPIO_NUM_4. The 'GPIO4' label seen on the wiring canvas is only for visualization and is not a valid C identifier."
    },
    {
        "question": "How many wires are needed for a basic ESP32 to DHT22 connection, including the pull-up resistor?",
        "ground_truth": "5 wires total: Data, Power (VCC to 3.3V), Ground, plus 2 wires for the pull-up resistor (signal and VCC)."
    },
]


async def build_evaluation_dataset(
    embeddings, vector_store, llm, knowledge_graph, graph_store, retrieval_mode, k=4, rrf_k=60
):
    """Same shape as test_ragas.py's build_evaluation_dataset, but routes
    retrieval through build_retriever with the given retrieval_mode -- so
    "graph" exercises GraphStore.traverse(), "hybrid_graph" exercises RRF
    fusion of vector + BM25 + graph, etc. (same modes app.config.Settings
    documents: vector | bm25 | hybrid | graph | hybrid_graph)."""
    print(f"\nGenerating answers using retrieval_mode='{retrieval_mode}'...")
    dataset_rows = []

    prompt = build_prompt_template()
    chain = prompt | llm | StrOutputParser()

    retriever = build_retriever(
        embeddings, vector_store, k, category=None,
        knowledge_graph=knowledge_graph,
        retrieval_mode=retrieval_mode,
        rrf_k=rrf_k,
        graph_store=graph_store,
    )

    for item in TEST_DATA:
        question = item["question"]

        docs = await retriever.ainvoke(question)
        context = await RunnableLambda(format_docs).ainvoke(docs)
        answer = await chain.ainvoke({"question": question, "context": context, "chat_history": []})

        contexts_list = [d.page_content for d in docs]
        retrieved_via = [d.metadata.get("retrieved_via", "?") for d in docs]

        print(f"  Q: {question[:60]}...")
        print(f"     retrieved {len(docs)} docs via: {retrieved_via}")

        dataset_rows.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts_list,
            "reference": item["ground_truth"],
        })

    df = pd.DataFrame(dataset_rows)
    hf_dataset = Dataset.from_pandas(df)
    return EvaluationDataset.from_hf_dataset(hf_dataset)


def run_ragas_evaluation(dataset, metrics_to_run, evaluator_llm, evaluator_embeddings, label):
    print("\n" + "=" * 50)
    print(f"MODE: {label}")
    print("=" * 50)

    result = evaluate(
        dataset=dataset,
        metrics=metrics_to_run,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    df = result.to_pandas()
    summary = df.mean(numeric_only=True)

    print("-" * 40)
    for metric_name, score in summary.items():
        print(f"{metric_name.replace('_', ' ').title():<20} : {score:.2f}")
    print("-" * 40)
    return summary


async def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation for graph-based retrieval modes.")
    parser.add_argument(
        "--mode", choices=["vector", "graph", "hybrid_graph", "all"], default="all",
        help="Which retrieval mode to evaluate (default: all, runs and compares every mode)"
    )
    parser.add_argument("--l3", action="store_true", help="Hallucination-focused metrics only (Faithfulness + AnswerRelevancy)")
    parser.add_argument("--k", type=int, default=4, help="Number of documents/entities to retrieve (default: 4)")
    args = parser.parse_args()

    modes = ["vector", "graph", "hybrid_graph"] if args.mode == "all" else [args.mode]

    original_cwd = os.getcwd()
    os.chdir(BACKEND_DIR if os.path.isdir(BACKEND_DIR) else os.path.dirname(os.path.abspath(__file__)))
    try:
        settings = get_settings()
        embeddings = get_embeddings()
        vector_store = get_vector_store()
        llm = get_llm()
        knowledge_graph = get_knowledge_graph()
        graph_store = get_graph_store()
    finally:
        os.chdir(original_cwd)

    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    metrics = [Faithfulness(), AnswerRelevancy()] if args.l3 else \
        [ContextPrecision(), ContextRecall(), AnswerRelevancy(), Faithfulness()]

    all_results: dict[str, pd.Series] = {}

    for mode in modes:
        dataset = await build_evaluation_dataset(
            embeddings, vector_store, llm, knowledge_graph, graph_store,
            retrieval_mode=mode, k=args.k, rrf_k=settings.rrf_k,
        )
        summary = run_ragas_evaluation(dataset, metrics, evaluator_llm, evaluator_embeddings, mode.upper())
        all_results[mode] = summary

    if len(all_results) > 1:
        print("\n" + "=" * 50)
        print("COMPARISON ACROSS MODES")
        print("=" * 50)
        comparison_df = pd.DataFrame(all_results)
        print(comparison_df.round(3).to_string())


if __name__ == "__main__":
    asyncio.run(main())