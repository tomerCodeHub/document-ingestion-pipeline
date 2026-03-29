# Document Ingestion & Vectorization Pipeline

A modular Python utility that extracts text from documents, chunks it with several strategies, generates semantic embeddings with the Google Gemini API, and stores vectors in PostgreSQL (pgvector). It is intended as the ingestion layer for RAG (Retrieval-Augmented Generation) workflows.

## Features

- **File extraction**: Reads text from `.pdf` (PyPDF2) and `.docx` (python-docx).
- **Chunking strategies**:
  - **Fixed-size with overlap**: Sliding window (500 characters, 50 overlap) in `index_documents.py`.
  - **Paragraph-based**: Splits on blank-line boundaries (with fallbacks for dense text).
  - **Sentence-based**: Splits on sentence boundaries (regex after `.`, `!`, `?`, and newlines).
- **Embeddings**: `gemini-embedding-001` via the `google-genai` client, with `output_dimensionality` set to **768**. Embeddings are stored as **`vector(768)`** in PostgreSQL using the **pgvector** extension.
- **Persistence**: Batch `INSERT` into PostgreSQL with parameterized queries (`executemany`) to avoid SQL injection.

## Pipeline Flow

The ingestion process follows a simple ETL-style pipeline:

1. Load environment variables from `.env`
2. Extract raw text from the input file (PDF/DOCX)
3. Normalize and clean the extracted text (for example CRLF normalization and trimming)
4. Split the text into chunks using the selected strategy
5. Generate embeddings for each chunk using the Gemini API (batched requests)
6. Store chunks and embeddings in PostgreSQL (pgvector)

## Chunking Strategy Tradeoffs

Each chunking strategy has different advantages depending on the use case:

- **Fixed-size with overlap**
  - **Pros**: Maintains context across chunk boundaries
  - **Cons**: May split sentences unnaturally

- **Paragraph-based**
  - **Pros**: Preserves semantic structure of the document
  - **Cons**: Paragraph sizes may vary significantly

- **Sentence-based**
  - **Pros**: High granularity, precise retrieval
  - **Cons**: May lose broader context across sentences

## Project Structure

- `index_documents.py` — Main ingestion pipeline
- `test_db.py` — PostgreSQL connectivity test
- `test_gemini.py` — Gemini API embedding test
- `requirements.txt` — Project dependencies
- `.env` — Environment configuration (not committed)

## Prerequisites

- Python 3.9 or higher.
- PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension enabled.
- A **`document_vectors`** table whose columns match what the pipeline inserts. In `index_documents.py`, rows are written with:

  ```sql
  INSERT INTO document_vectors (chunk_text, embedding, filename, split_strategy)
  VALUES (%s, %s, %s, %s);
  ```

  You therefore need at least those four columns (with valid embedding dimension **768** for `embedding`). A primary key **`id`** and optional **`created_at`** should also exist if you follow the recommended DDL below.

  | Column           | Description                                      |
  | ---------------- | ------------------------------------------------ |
  | `id`             | Surrogate primary key                            |
  | `chunk_text`     | Text of the chunk                                |
  | `embedding`      | Embedding vector (`vector(768)`)                 |
  | `filename`       | Original source file name                        |
  | `split_strategy` | Chunking strategy (`fixed`, `paragraph`, `sentence`) |
  | `created_at`     | Row creation time (optional; DB default)         |

  Use the [Database schema](#database-schema) section to create this table if it does not exist yet.

## Installation

```bash
git clone https://github.com/tomerCodeHub/document-ingestion-pipeline.git
cd document-ingestion-pipeline

python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Environment configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_google_gemini_api_key
POSTGRES_URL=postgresql://username:password@localhost:5432/your_database_name
```

The pipeline reads these variables in `index_documents.py` (and the test scripts use the same names).

## Database schema

Run in PostgreSQL (e.g. psql or pgAdmin). This matches the `INSERT` columns used by the app plus `id` and optional `created_at`:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE document_vectors (
    id SERIAL PRIMARY KEY,
    chunk_text TEXT NOT NULL,
    embedding vector(768),
    filename TEXT NOT NULL,
    split_strategy TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Usage

With the virtual environment activated:

```bash
python3 index_documents.py
```

Interactive flow:

1. **File path**: Local path to a PDF or DOCX.
2. **Chunking**: Choose `1` (fixed), `2` (paragraph), or `3` (sentence).
3. The script extracts text, calls Gemini for embeddings, and inserts rows into `document_vectors`.

Example input:

```
resumes/Tomer_Kornblum.pdf
```

If extraction yields no text, the script exits without calling the API or the database.

## Sanity checks

Before running the full pipeline, you can verify connectivity and credentials:

- **`test_db.py`**: Loads `.env`, connects with `POSTGRES_URL`, and runs `SELECT * FROM document_vectors LIMIT 1` to confirm the table is reachable.
- **`test_gemini.py`**: Loads `.env`, calls `gemini-embedding-001` with `output_dimensionality: 768`, and prints success plus the embedding vector length.

```bash
source venv/bin/activate
python3 test_db.py
python3 test_gemini.py
```

## Verifying ingestion

Example aggregation by file and strategy (dimensions should be 768 for all rows):

```sql
SELECT
    filename,
    split_strategy,
    COUNT(*) AS total_chunks,
    MAX(vector_dims(embedding)) AS dimensions
FROM document_vectors
GROUP BY filename, split_strategy;
```

## Example Similarity Search

After ingestion, you can retrieve similar chunks with pgvector distance operators. Compute a query embedding with the **same model and dimensionality** (`gemini-embedding-001`, 768), then run:

```sql
SELECT chunk_text
FROM document_vectors
ORDER BY embedding <-> '[your_query_embedding]'::vector
LIMIT 5;
```

Replace `[your_query_embedding]` with a comma-separated list of 768 floats (same format you would pass to pgvector). This supports retrieval for downstream RAG applications.

## Performance Notes

- The number of chunks directly impacts API usage and cost.
- Smaller chunks tend to improve retrieval precision but require more embeddings (more API calls and storage).
- Larger chunks reduce API calls but can blur boundaries and reduce retrieval precision.

## Limitations

- Large documents may hit API limits; embedding requests are batched (default batch size 100), which helps but does not remove all quota or payload constraints.
- PDF text extraction quality depends on document formatting (scanned PDFs without text layers will not extract well).
- There is no automatic retry for failed API calls.
- There is no deduplication of similar chunks across or within runs.

## Error Handling

- Invalid or missing file paths are detected before processing (`logging.error`).
- If extraction fails or returns empty content, the pipeline exits before calling the embedding API.
- Gemini API failures are caught, logged (`logging.error`), and the pipeline skips the database write when no vectors are produced.
- PostgreSQL errors trigger a transaction **rollback**; connection cleanup happens in a `finally` block.
- Chunk and embedding list length mismatches abort the insert before touching the database.

## Implementation notes

- **Client**: Uses `google.genai.Client` and `client.models.embed_content` (same pattern as `test_gemini.py`).
- **Batching**: Embeddings are requested in batches; database writes use `executemany` for multiple rows per transaction.
- **Layout**: Extraction, chunking, embedding, and DB persistence are separated into functions in `index_documents.py` for a simple ETL-style structure.

## Regenerating `requirements.txt`

After installing or upgrading packages in your active environment:

```bash
pip freeze > requirements.txt
```

Review the file afterward so only direct project dependencies are listed, if you prefer a minimal manifest.
