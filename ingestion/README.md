# Gyani AI — RAG retrieval & QA backend (LangChain + FastAPI + Supabase)

This is the **retrieval and generation** layer on top of your existing
ingestion pipeline. It does not touch, re-embed, or recreate anything you
already have:

- Reuses the existing `knowledge_chunks` table as-is
- Reuses the existing `match_knowledge_chunks` RPC function as-is
- Only ever generates ONE embedding per request — for the incoming
  question, with the same model your ingestion pipeline used
  (`all-MiniLM-L6-v2`, 384 dims, normalized) — so query vectors are
  comparable to the stored document vectors
- Document embeddings are never regenerated

## Project structure

```
backend/
    app/
        api/
            routes.py         # POST /chat, POST /chat/stream
        rag/
            embeddings.py     # query-time SentenceTransformer wrapper (LangChain Embeddings interface)
            vector_store.py   # thin wrapper around your existing Supabase RPC call
            retriever.py      # LangChain BaseRetriever: question -> embedding -> RPC -> Documents
            prompt.py         # the Gyani AI system prompt + context formatting
            chain.py          # RAGService: retrieval + LCEL chain + memory + streaming
        services/
            llm.py            # swappable LLM factory (gemini / openai / anthropic)
        config.py              # typed settings from env
        main.py                 # FastAPI app, startup wiring, CORS
    requirements.txt
    .env.example
```

## Setup

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env   # then fill in SUPABASE_URL, SUPABASE_SERVICE_KEY, GOOGLE_API_KEY
```

Run it:

```bash
uvicorn app.main:app --reload
```

First request that needs the embedding model will download it once from
Hugging Face (cached after that) if it isn't already cached from your
ingestion pipeline's runs.

## Endpoints

### `POST /chat`

```json
{
  "question": "How do I connect DHT22?",
  "category": "facts",       // optional filter: "facts" | "classes" | "eval"
  "k": 5,                    // optional, defaults to DEFAULT_MATCH_COUNT
  "session_id": "user-123"   // optional, enables multi-turn memory
}
```

Response:

```json
{
  "answer": "...",
  "sources": [
    {"source_path": "facts/components/dht22.yaml", "category": "facts", "similarity": 0.91}
  ]
}
```

### `POST /chat/stream`

Same request body. Returns a `text/event-stream` response:

```
event: sources
data: [{"source_path": "...", "category": "...", "similarity": 0.91}]

event: token
data: {"token": "The"}

event: token
data: {"token": " DHT22"}

...

event: done
data: {}
```

Sources are sent as the first event so a client can render "answering
from..." before generation finishes.

## Design notes

- **LCEL, not deprecated chains.** `prompt | llm | StrOutputParser()` is a
  `RunnableSequence` — the modern replacement for `LLMChain`/`RetrievalQA`.
  `RunnableLambda` wraps the context-formatting step.
- **Conversation memory** is a plain `dict[session_id, list[BaseMessage]]`,
  not `RunnableWithMessageHistory` — that class is deprecated upstream in
  favor of LangGraph's own persistence, so using it today would mean
  building on a documented dead end. The manual list-of-messages approach
  is simpler and maps directly onto LangGraph's state model (a message
  list + a checkpointer), so migrating later is a storage-layer swap, not a
  logic rewrite. It's in-memory and per-process right now; for persistence
  across restarts, back `_load_history`/`_save_turn` in `chain.py` with a
  Supabase table instead.
- **LLM is swappable** via `LLM_PROVIDER` in `.env` (`gemini` / `openai` /
  `anthropic`) — nothing outside `services/llm.py` references a specific
  provider.
- **Metadata filtering** (`category`) is threaded from the request all the
  way to the RPC's `filter_category` argument.
- **Hybrid retrieval ready**: `SupabaseRAGRetriever` is the single seam
  where a future keyword/BM25 branch would plug in alongside the vector
  search call, without touching the API layer or the chain.
- **Migrating to LangGraph later**: `RAGService`'s methods
  (`_retrieve`, generation, memory read/write) map almost 1:1 onto graph
  nodes (`retrieve`, `generate`) with `_session_store` becoming a
  checkpointer-backed state — the rewrite would be mostly relocation, not
  new logic.

## Tested

All modules were syntax-checked and the full request path was exercised
with fakes standing in for the embedding model, Supabase RPC call, and LLM
(no real network access in the environment this was built in): `/chat`
returns a correct answer + sources, `/chat/stream` emits `sources` → many
`token` → `done` events, and multi-turn memory correctly accumulates and is
passed back into the prompt on the next turn. What wasn't (and can't be,
without your real keys) exercised here: an actual Gemini API call and an
actual Supabase network round-trip — verify those once you plug in real
credentials.
