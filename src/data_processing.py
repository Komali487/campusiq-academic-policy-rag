"""
CampusIQ - Academic Policy Document Processing

This module loads PDF policy documents, extracts text, cleans it,
and splits it into smaller chunks for the RAG pipeline.
"""

from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from all pages of a PDF."""
    reader = PdfReader(str(pdf_path))

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return "\n".join(pages)


def clean_text(text: str) -> str:
    """Clean unnecessary whitespace from extracted text."""
    lines = [line.strip() for line in text.splitlines()]

    cleaned_lines = [
        line for line in lines
        if line
    ]

    return "\n".join(cleaned_lines)


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150
) -> List[str]:
    """Split text into overlapping chunks."""
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def process_pdf(pdf_path: Path) -> List[Dict]:
    """Process one PDF and return document chunks."""
    text = extract_text_from_pdf(pdf_path)
    text = clean_text(text)

    chunks = chunk_text(text)

    documents = []

    for index, chunk in enumerate(chunks):
        documents.append(
            {
                "source": pdf_path.name,
                "chunk_id": index,
                "text": chunk,
            }
        )

    return documents


def process_all_pdfs() -> List[Dict]:
    """Process all PDFs inside the raw data directory."""
    all_documents = []

    if not RAW_DATA_DIR.exists():
        print("Raw data directory does not exist.")
        return all_documents

    pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data/raw/")
        return all_documents

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        documents = process_pdf(pdf_path)
        all_documents.extend(documents)

    return all_documents


if __name__ == "__main__":
    documents = process_all_pdfs()

    print(f"\nTotal chunks created: {len(documents)}")
