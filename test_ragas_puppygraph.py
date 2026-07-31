"""
RAGAS evaluation for retrieval via PuppyGraph itself (real Bolt/Cypher
queries), as opposed to test_ragas_graph.py which evaluates the in-process
GraphStore.traverse().

Why this needs its own pipeline (PuppyGraph isn't wired into RAGService):
PuppyGraph has no natural-language understanding -- it only answers Cypher/
Gremlin queries. So per question, this script:

  1. Client-side entity extraction: word-boundary alias matching against the
     `entities` table (fetched once from Supabase into memory) -- mirrors
     GraphStore.extract_entities_from_query() so the two eval scripts are
     testing an equivalent "find seed entities" step.
  2. Runs a REAL Cypher query against PuppyGraph's Bolt endpoint to traverse
     1-2 hops out from those seed entities -- this is what's actually being
     evaluated: PuppyGraph's own graph engine, not local NetworkX.
  3. Maps the resulting entity IDs -> their doc_ids -> actual chunk content
     from the `knowledge_chunks` table in Supabase. (entities only carries
     metadata -- id/category/aliases -- not document text, so this hop is
     required to get real retrieved_contexts for RAGAS to judge.)
  4. Generates an answer with the same LLM + prompt chain as the other eval
     scripts, then runs the same RAGAS metrics -- so all three scripts
     (vector / graph / puppygraph) are directly comparable.

Prerequisites:
    pip install neo4j supabase --break-system-packages

    entities table needs a doc_ids column. If you created it before this
    existed, run in Supabase SQL Editor:
        alter table entities add column if not exists doc_ids text[];
    then re-run the updated export_graph_to_supabase.py so doc_ids gets
    populated (it was previously left out).

    PuppyGraph must be running with the entities/related_to schema already
    submitted (the setup you already did), reachable at its Bolt port
    (default 7687 from the docker run command used earlier).

    This script must sit in the same directory as test_ragas_graph.py --
    it imports TEST_DATA from there to keep the question set identical
    across all three eval scripts (vector / graph / puppygraph).

Usage:
    python test_ragas_puppygraph.py
    python test_ragas_puppygraph.py --l3
    python test_ragas_puppygraph.py --bolt-uri bolt://localhost:7687 --bolt-user puppygraph --bolt-password puppygraph123
"""

import argparse
import asyncio
import logging
import os
import re
import sys

import pandas as pd
from datasets import Dataset

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "backend"))
if os.path.isdir(BACKEND_DIR):
    sys.path.insert(0, BACKEND_DIR)
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse the exact same question set as test_ragas_graph.py so all three
# eval scripts (vector / graph / puppygraph) are directly comparable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from test_ragas_graph import TEST_DATA
except ImportError:
    raise SystemExit(
        "Could not import TEST_DATA from test_ragas_graph.py -- make sure "
        "test_ragas_puppygraph.py sits in the same directory as test_ragas_graph.py."
    )

from app.config import get_settings
from app.rag.prompt import build_prompt_template
from app.services.llm import get_llm
from langchain_core.output_parsers import StrOutputParser

# --- RAGAS compatibility patch (same as the other eval scripts) -----------
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
from app.rag.embeddings import get_embeddings
from supabase import Client, create_client
from neo4j import GraphDatabase

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: entity extraction -- mirrors GraphStore.extract_entities_from_query
# ---------------------------------------------------------------------------
def build_alias_index(entities_rows: list[dict]) -> dict[str, set[str]]:
    alias_index: dict[str, set[str]] = {}
    for row in entities_rows:
        entity_id = row["id"]
        for alias in (row.get("aliases") or []):
            alias_index.setdefault(alias.lower(), set()).add(entity_id)
        # id itself is always a matchable alias
        alias_index.setdefault(entity_id.lower(), set()).add(entity_id)
    return alias_index


def extract_entities_from_query(query: str, alias_index: dict[str, set[str]]) -> set[str]:
    found = set()
    query_lower = query.lower()
    for alias, ids in alias_index.items():
        pattern = rf"\b{re.escape(alias)}\b"
        if re.search(pattern, query_lower):
            found.update(ids)
    return found


# ---------------------------------------------------------------------------
# Step 2: real traversal via PuppyGraph (Cypher over Bolt)
# ---------------------------------------------------------------------------
def traverse_puppygraph(driver, seed_ids: set[str], max_hops: int = 2) -> set[str]:
    if not seed_ids:
        return set()

    query = f"""
    MATCH (a) WHERE a.id IN $ids
    OPTIONAL MATCH (a)-[:related_to*1..{max_hops}]-(b)
    RETURN a.id AS seed_id, collect(DISTINCT b.id) AS neighbor_ids
    """
    result_ids = set(seed_ids)
    with driver.session() as session:
        result = session.run(query, ids=list(seed_ids))
        for record in result:
            if record["seed_id"]:
                result_ids.add(record["seed_id"])
            for nid in record["neighbor_ids"] or []:
                if nid:
                    result_ids.add(nid)
    return result_ids


# ---------------------------------------------------------------------------
# Step 3: matched entity IDs -> doc_ids -> real chunk content from Supabase
# ---------------------------------------------------------------------------
def entities_to_contexts(
    entity_ids: set[str], entities_by_id: dict[str, dict], supabase: Client
) -> list[str]:
    doc_ids: set[str] = set()
    for eid in entity_ids:
        row = entities_by_id.get(eid)
        if row and row.get("doc_ids"):
            doc_ids.update(row["doc_ids"])

    if not doc_ids:
        return []

    resp = supabase.table("knowledge_chunks").select("doc_id,content").in_("doc_id", list(doc_ids)).execute()
    return [row["content"] for row in resp.data if row.get("content")]


# ---------------------------------------------------------------------------
# Build the RAGAS dataset by running the full PuppyGraph pipeline per question
# ---------------------------------------------------------------------------
async def build_evaluation_dataset(llm, driver, alias_index, entities_by_id, supabase: Client):
    print("\nGenerating answers using retrieval via PuppyGraph (Bolt/Cypher)...")
    dataset_rows = []

    prompt = build_prompt_template()
    chain = prompt | llm | StrOutputParser()

    for item in TEST_DATA:
        question = item["question"]

        seed_ids = extract_entities_from_query(question, alias_index)
        matched_ids = traverse_puppygraph(driver, seed_ids, max_hops=2)
        contexts_list = entities_to_contexts(matched_ids, entities_by_id, supabase)

        context_text = "\n\n".join(contexts_list) if contexts_list else "No relevant context found."
        answer = await chain.ainvoke({"question": question, "context": context_text, "chat_history": []})

        print(f"  Q: {question[:60]}...")
        print(f"     seed_entities={len(seed_ids)} traversed_entities={len(matched_ids)} contexts={len(contexts_list)}")

        dataset_rows.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts_list or ["No relevant context found."],
            "reference": item["ground_truth"],
        })

    df = pd.DataFrame(dataset_rows)
    hf_dataset = Dataset.from_pandas(df)
    return EvaluationDataset.from_hf_dataset(hf_dataset)


def run_ragas_evaluation(dataset, metrics_to_run, evaluator_llm, evaluator_embeddings):
    print("\n" + "=" * 50)
    print("MODE: PUPPYGRAPH (Bolt/Cypher retrieval)")
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
    return df


async def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation for retrieval via PuppyGraph.")
    parser.add_argument("--l3", action="store_true", help="Hallucination-focused metrics only (Faithfulness + AnswerRelevancy)")
    parser.add_argument("--bolt-uri", default="bolt://localhost:7687", help="PuppyGraph Bolt endpoint")
    parser.add_argument("--bolt-user", default="puppygraph", help="PuppyGraph Bolt username")
    parser.add_argument("--bolt-password", default="puppygraph123", help="PuppyGraph Bolt password")
    args = parser.parse_args()

    original_cwd = os.getcwd()
    os.chdir(BACKEND_DIR if os.path.isdir(BACKEND_DIR) else os.path.dirname(os.path.abspath(__file__)))
    try:
        settings = get_settings()
        llm = get_llm()
        embeddings = get_embeddings()
    finally:
        os.chdir(original_cwd)

    supabase: Client = create_client(settings.supabase_url, settings.supabase_service_key)

    print("Loading entities table from Supabase...")
    entities_resp = supabase.table("entities").select("id,aliases,doc_ids").execute()
    entities_rows = entities_resp.data
    entities_by_id = {row["id"]: row for row in entities_rows}
    alias_index = build_alias_index(entities_rows)
    print(f"Loaded {len(entities_rows)} entities, {len(alias_index)} aliases.")

    print(f"Connecting to PuppyGraph at {args.bolt_uri}...")
    driver = GraphDatabase.driver(args.bolt_uri, auth=(args.bolt_user, args.bolt_password))
    try:
        driver.verify_connectivity()
        print("Connected.")
    except Exception as e:
        raise SystemExit(f"Could not connect to PuppyGraph Bolt endpoint: {e}")

    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    metrics = [Faithfulness(), AnswerRelevancy()] if args.l3 else \
        [ContextPrecision(), ContextRecall(), AnswerRelevancy(), Faithfulness()]

    try:
        dataset = await build_evaluation_dataset(llm, driver, alias_index, entities_by_id, supabase)
        run_ragas_evaluation(dataset, metrics, evaluator_llm, evaluator_embeddings)
    finally:
        driver.close()


if __name__ == "__main__":
    asyncio.run(main())