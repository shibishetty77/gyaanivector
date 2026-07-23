# RAGAS Evaluation Pipeline

This folder contains a completely independent evaluation pipeline for the existing RAG backend, using [Ragas](https://docs.ragas.io/).

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Make sure you have your judge LLM provider's API key set in your environment (e.g., `OPENAI_API_KEY`), as Ragas uses OpenAI by default for metric evaluation.

## How to Create `dataset.json`

Create or modify `dataset.json` to include your test cases. The format should be a JSON array of objects:

```json
[
    {
        "question": "What is the capital of France?",
        "ground_truth": "The capital of France is Paris.",
        "category": "facts"
    }
]
```

## How to Run `build_dataset.py`

This script reads `dataset.json`, calls your local API (`POST http://localhost:8000/chat`), and creates a HuggingFace `Dataset` compatible with Ragas.

**Note:** Depending on your FastAPI endpoints, you might need to adjust the request payload (e.g., changing `"message"` to `"query"`) and response parsing in `build_dataset.py`. If your production endpoint does not expose the retrieved contexts, you will need to import your retriever and call it directly inside the script instead of pulling contexts from the API response. **Do not modify the production endpoint itself.**

Run the script:
```bash
python build_dataset.py
```
This will call your local API for each question and generate a `ragas_dataset` folder on disk.

## How to Run `evaluate.py`

This script evaluates the generated dataset using Ragas metrics and prints the results to the console.

Run the script:
```bash
python evaluate.py
```
It will output the scores to the console (Faithfulness, Answer Relevancy, Context Precision, Context Recall) and save a detailed report containing both aggregated scores and row-by-row data to `evaluation_results.json`.

## How to Interpret the Scores

Metrics are scored between 0 and 1, with 1 being the best.

- **Faithfulness**: Measures if the generated answer is faithful to the retrieved context (checking for hallucinations).
- **Answer Relevancy**: Measures how relevant the generated answer is to the given question.
- **Context Precision**: Measures if the retrieved contexts are highly relevant to the question and if the most relevant chunks are ranked higher.
- **Context Recall**: Measures if the retrieved contexts contain all the necessary information to answer the question, compared to the `ground_truth`.
