from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import PAPERS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_papers():
    """Load all PDF papers and preserve page-level metadata."""

    documents = []

    pdf_files = list(Path(PAPERS_DIR).glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {PAPERS_DIR}")
        return documents

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        pdf = pymupdf.open(pdf_file)

        for page_number, page in enumerate(pdf):
            text = page.get_text("text").strip()

            if not text:
                continue

            document = Document(
                page_content=text,
                metadata={
                    "source": pdf_file.name,
                    "page": page_number + 1,
                },
            )

            documents.append(document)

        pdf.close()

    print(f"\nLoaded {len(documents)} pages.")

    return documents


def split_documents(documents):
    """Split pages into smaller chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    return chunks


if __name__ == "__main__":
    documents = load_papers()

    if documents:
        chunks = split_documents(documents)

        print("\nFirst chunk:")
        print("-" * 60)
        print(chunks[0].page_content)

        print("\nMetadata:")
        print(chunks[0].metadata)