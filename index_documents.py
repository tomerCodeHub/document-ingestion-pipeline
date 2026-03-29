import os
import re
import logging
import psycopg2
import PyPDF2
from docx import Document
from dotenv import load_dotenv
from google import genai
from typing import List

# --- Config & Logging ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

load_dotenv()
POSTGRES_URL = os.getenv("POSTGRES_URL")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# --- Extraction ---

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    try:
        if ext == ".pdf":
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n\n".join([
                    t for page in reader.pages
                    if (t := page.extract_text())
                ])

        elif ext == ".docx":
            doc = Document(file_path)
            text = "\n\n".join([para.text for para in doc.paragraphs])

        text = text.replace('\r', '\n')
        return text.strip()

    except Exception as e:
        logging.error(f"Extraction Error: {e}")
        return ""

# --- Chunking ---

def chunk_fixed_size(text: str, size: int = 500, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += (size - overlap)

    return chunks


def chunk_by_paragraph(text: str) -> List[str]:
    paragraphs = re.split(r'\n{2,}', text)

    if len(paragraphs) <= 1:
        lines = text.split('\n')
        paragraphs = []
        current_chunk = []

        for line in lines:
            clean_line = line.strip()
            if not clean_line:
                continue

            current_chunk.append(clean_line)

            if clean_line.endswith('.') or len(current_chunk) >= 5:
                paragraphs.append(" ".join(current_chunk))
                current_chunk = []

        if current_chunk:
            paragraphs.append(" ".join(current_chunk))

    return [p.strip() for p in paragraphs if len(p.strip()) > 30]


def chunk_by_sentence(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+|\n', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]

# --- Embeddings (WITH BATCHING) ---

def get_embeddings(text_chunks: List[str], batch_size: int = 100) -> List[List[float]]:
    all_embeddings = []

    try:
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            logging.info(f"Embedding batch {i//batch_size + 1} ({len(batch)} chunks)")

            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=batch,
                config={'output_dimensionality': 768}
            )

            batch_embeddings = [item.values for item in result.embeddings]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    except Exception as e:
        logging.error(f"Gemini API Embedding Error: {e}")
        return []

# --- DB Persistence ---

def save_to_db(chunks: List[str], embeddings: List[List[float]], filename: str, strategy: str):
    if len(chunks) != len(embeddings):
        logging.error("Mismatch between chunks and embeddings. Aborting insert.")
        return

    conn = None

    try:
        conn = psycopg2.connect(POSTGRES_URL)
        cur = conn.cursor()

        insert_stmt = """
        INSERT INTO document_vectors (chunk_text, embedding, filename, split_strategy)
        VALUES (%s, %s, %s, %s);
        """

        data = [(c, e, filename, strategy) for c, e in zip(chunks, embeddings)]
        cur.executemany(insert_stmt, data)

        conn.commit()
        logging.info(f"Inserted {len(chunks)} rows into database.")

    except psycopg2.Error as e:
        logging.error(f"PostgreSQL Error: {e}")
        if conn:
            conn.rollback()

    finally:
        if conn:
            cur.close()
            conn.close()

# --- Main ---

def main():
    logging.info("=== Document Ingestion Pipeline ===")

    file_path = input("Enter the file path: ").strip()
    if not os.path.exists(file_path):
        logging.error(f"File '{file_path}' not found.")
        return

    print("\nSelect Chunking Strategy:")
    print("1. Fixed-size")
    print("2. Paragraph-based")
    print("3. Sentence-based")

    choice = input("Choice [1-3]: ").strip()

    strat_map = {
        "1": ("fixed", chunk_fixed_size),
        "2": ("paragraph", chunk_by_paragraph),
        "3": ("sentence", chunk_by_sentence)
    }

    if choice not in strat_map:
        logging.warning("Invalid choice. Defaulting to fixed-size.")
    
    strategy_name, chunk_func = strat_map.get(choice, strat_map["1"])

    content = extract_text(file_path)
    if not content:
        logging.error("No content extracted. Exiting.")
        return

    chunks = chunk_func(content, 500, 50) if choice == "1" else chunk_func(content)

    logging.info(f"Generated {len(chunks)} chunks. Starting embeddings...")

    vectors = get_embeddings(chunks)

    if not vectors:
        logging.error("Embedding generation failed. Exiting.")
        return

    save_to_db(chunks, vectors, os.path.basename(file_path), strategy_name)


if __name__ == "__main__":
    main()