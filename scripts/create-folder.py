from pathlib import Path

folders = [
    "app/ui",
    "app/ingestion",
    "app/embeddings",
    "app/vectorstores",
    "app/retrieval",
    "app/chains",
    "app/memory",
    "app/services",
    "app/utils",
    "app/config",

    "app/prompts/system",
    "app/prompts/tasks",
    "app/prompts/templates",

    "data/documents",

    "vector_db/faiss_index",
    "vector_db/metadata",

    "evals/tests",
    "evals/traces",
    "evals/scorecards",

    "tests/unit",
    "tests/integration",
    "tests/e2e",

    "deployment/docker",
    "deployment/kubernetes",

    "notebooks",
    "logs",
    "scripts"
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

print("✅ Production RAG folder structure created successfully!")