"""
Simple extraction example.
Processes a single PDF and extracts concepts.
"""
from pathlib import Path
from knowledge_extraction.core import DocumentProcessor, SemanticChunker

def main():
    # Setup
    pdf_path = Path("sample.pdf")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Process PDF
    print("Processing PDF...")
    processor = DocumentProcessor()
    document = processor.process_pdf(pdf_path)

    # Create chunks
    print("Creating semantic chunks...")
    chunker = SemanticChunker()
    chunks = chunker.chunk_document(document)

    print(f"✓ Extracted {len(chunks)} chunks")
    print(f"✓ Results saved to {output_dir}")

if __name__ == "__main__":
    main()
