import asyncio
import logging

from app.config import get_settings
from app.rag.chain import RAGService
from app.rag.embeddings import get_embeddings
from app.rag.prompt import build_prompt_template, format_docs
from app.rag.vector_store import get_vector_store
from app.services.llm import get_llm
from langchain_core.runnables import RunnableLambda

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

async def run_pipeline():
    print("Stage 1 - Configuration passed")
    
    print("Stage 2 - Instrument the pipeline: Using local logging in script")

    print("\n--- API ---")
    print("entered stage: API")
    question = "hello"
    k = 4
    
    print("\n--- Embedding generation ---")
    print("entered stage: Embedding generation")
    try:
        embeddings = get_embeddings()
        q_emb = embeddings.embed_query(question)
        print("important values: embedding length =", len(q_emb))
        print("completed stage: Embedding generation")
    except Exception as e:
        print("failure reason:", e)
        return

    print("\n--- Retriever ---")
    print("entered stage: Retriever")
    try:
        vector_store = get_vector_store()
        
        print("\n--- Vector store / Supabase RPC ---")
        print("entered stage: Vector store")
        print("Embedding length:", len(q_emb))
        print("Category:", None)
        print("k:", k)
        print("RPC name:", vector_store._match_function)
        
        rows = vector_store.similarity_search(q_emb, k=k, category=None)
        print("Rows returned:", len(rows))
        
        if rows:
            print("Similarity scores:")
            for r in rows:
                print(r.get("similarity"))
            print("Document IDs:")
            for r in rows:
                print(r.get("doc_id"))
        
        print("completed stage: Vector store")
        
        from app.rag.retriever import build_retriever
        retriever = build_retriever(embeddings, vector_store, k, category=None)
        
        print("documents before filtering:", len(rows))
        docs = retriever._rows_to_documents(rows)
        print("documents after filtering:", len(docs))
        print("documents after threshold:", len(docs))
        print("documents after deduplication:", len(docs))
        print("documents sent to chain:", len(docs))
        print("completed stage: Retriever")
        
    except Exception as e:
        print("failure reason:", e)
        return

    print("\n--- Context assembly ---")
    print("entered stage: Context assembly")
    try:
        print("number of chunks:", len(docs))
        total_chars = sum(len(d.page_content) for d in docs)
        print("total context characters:", total_chars)
        for i, d in enumerate(docs):
            print(f"first 200 chars of chunk {i}:", repr(d.page_content[:200]))
            
        context = await RunnableLambda(format_docs).ainvoke(docs)
        print("completed stage: Context assembly")
    except Exception as e:
        print("failure reason:", e)
        return

    print("\n--- LLM invocation ---")
    print("entered stage: LLM invocation")
    try:
        llm = get_llm()
        prompt = build_prompt_template()
        chain = prompt | llm | __import__('langchain_core').output_parsers.StrOutputParser()
        
        print("LLM started")
        answer = await chain.ainvoke({"question": question, "context": context, "chat_history": []})
        print("LLM finished")
        print("response length:", len(answer))
        
        print("completed stage: LLM invocation")
        
        print("\nAPI stage completed successfully")
        
    except Exception as e:
        print("failure reason:", e)
        return

if __name__ == '__main__':
    asyncio.run(run_pipeline())
