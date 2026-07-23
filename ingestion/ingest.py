#!/usr/bin/env python3
"""
ingest.py — load YAML knowledge files into a Supabase pgvector table.

Embeddings are generated LOCALLY with sentence-transformers (all-MiniLM-L6-v2,
384 dims) — no OpenAI key, no per-call cost, nothing leaves your machine
except the final vectors going to Supabase.

Setup (one time):
    1. In the Supabase SQL editor, run schema.sql from this folder.
    2. pip install -r requirements.txt
       (first run also downloads the ~80MB model from Hugging Face — needs
       internet once; cached locally after that)
    3. Set environment variables (or put them in a .env file next to this script):
         SUPABASE_URL=https://xxxx.supabase.co
         SUPABASE_SERVICE_KEY=<service_role key, NOT the anon key>
       Optional:
         EMBEDDING_MODEL=all-MiniLM-L6-v2   (default; 384 dims, matches schema.sql)

Usage:
    python ingest.py /path/to/knowledge_yaml_files.zip
    python ingest.py /path/to/already_unzipped_folder
"""

import os
import sys
import json
import zipfile
import tempfile
import argparse
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env support is optional


EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSIONS = 384  # must match the `vector(384)` column in schema.sql
TABLE_NAME = "knowledge_chunks"


def get_supabase_client():
    from supabase import create_client

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

    missing = [
        name for name, val in [
            ("SUPABASE_URL", supabase_url),
            ("SUPABASE_SERVICE_KEY", supabase_key),
        ] if not val
    ]
    if missing:
        sys.exit(
            f"Missing required environment variable(s): {', '.join(missing)}.\n"
            "Set them in your shell or in a .env file next to this script."
        )

    return create_client(supabase_url, supabase_key)


def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    print(f"Loading local embedding model '{EMBEDDING_MODEL}' (downloads once, then cached)...")
    return SentenceTransformer(EMBEDDING_MODEL)


def prepare_input_dir(path_str: str) -> Path:
    """Accept either a .zip file or an already-extracted directory; return a directory path."""
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        sys.exit(f"Path not found: {path}")

    if path.is_file() and path.suffix.lower() == ".zip":
        extract_dir = Path(tempfile.mkdtemp(prefix="knowledge_yaml_"))
        with zipfile.ZipFile(path) as zf:
            zf.extractall(extract_dir)
        return extract_dir

    if path.is_dir():
        return path

    sys.exit(f"Expected a .zip file or a directory, got: {path}")


def find_yaml_files(root: Path):
    return sorted(
        p for p in root.rglob("*") if p.suffix.lower() in (".yaml", ".yml") and p.is_file()
    )


def derive_category(root: Path, file_path: Path):
    """
    From a path like .../backend/knowledge/facts/components/dht22.yaml
    derive category='facts', subcategory='components'.
    Falls back gracefully for shallower paths.
    """
    rel_parts = file_path.relative_to(root).parts
    # Look for the 'knowledge' folder as an anchor; use the parts right after it.
    if "knowledge" in rel_parts:
        idx = rel_parts.index("knowledge")
        after = rel_parts[idx + 1:]
    else:
        after = rel_parts

    category = after[0] if len(after) >= 1 else None
    subcategory = after[1] if len(after) >= 3 else None  # only if there's a filename after it too
    return category, subcategory


def derive_doc_id(parsed: dict):
    if not isinstance(parsed, dict):
        return None
    for key in ("id", "class", "evaluation_id"):
        if key in parsed:
            return str(parsed[key])
    return None


def embed_texts(model, texts):
    """Batch-embed a list of texts locally, preserving order."""
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return [e.tolist() for e in embeddings]


def main():
    parser = argparse.ArgumentParser(description="Ingest YAML knowledge files into Supabase pgvector.")
    parser.add_argument("input_path", help="Path to the .zip file or an already-extracted folder")
    parser.add_argument("--batch-size", type=int, default=20, help="Files to embed per API call")
    parser.add_argument("--dry-run", action="store_true", help="Parse and print, but don't embed or write to Supabase")
    args = parser.parse_args()

    root = prepare_input_dir(args.input_path)
    yaml_files = find_yaml_files(root)

    if not yaml_files:
        sys.exit(f"No .yaml/.yml files found under {root}")

    print(f"Found {len(yaml_files)} YAML files under {root}")

    rows = []
    for file_path in yaml_files:
        raw_text = file_path.read_text(encoding="utf-8")

        try:
            parsed = yaml.safe_load(raw_text)
        except yaml.YAMLError as e:
            print(f"  ! Skipping {file_path} — invalid YAML: {e}")
            continue

        category, subcategory = derive_category(root, file_path)
        doc_id = derive_doc_id(parsed)
        source_path = str(file_path.relative_to(root))

        rows.append({
            "source_path": source_path,
            "category": category,
            "subcategory": subcategory,
            "doc_id": doc_id,
            "content": raw_text,
            "metadata": parsed if isinstance(parsed, dict) else {},
        })
        print(f"  - {source_path}  [category={category}, subcategory={subcategory}, id={doc_id}]")

    if args.dry_run:
        print("\nDry run: skipping embedding + upload.")
        return

    supabase = get_supabase_client()
    model = get_embedding_model()

    for i in range(0, len(rows), args.batch_size):
        batch = rows[i:i + args.batch_size]
        texts = [r["content"] for r in batch]
        print(f"Embedding batch {i // args.batch_size + 1} ({len(batch)} files)...")
        embeddings = embed_texts(model, texts)

        for row, embedding in zip(batch, embeddings):
            row["embedding"] = embedding
            # metadata must be JSON-serialisable for the jsonb column
            row["metadata"] = json.loads(json.dumps(row["metadata"], default=str))

        print(f"Upserting batch {i // args.batch_size + 1} to Supabase table '{TABLE_NAME}'...")
        supabase.table(TABLE_NAME).upsert(batch, on_conflict="source_path").execute()

    print(f"\nDone. {len(rows)} files embedded and upserted into '{TABLE_NAME}'.")


if __name__ == "__main__":
    main()