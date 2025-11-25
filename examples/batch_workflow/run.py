"""
Batch Workflow Example

Process multiple PDFs and build a unified knowledge graph.
"""

import json
import logging
from pathlib import Path

from knowledge_extraction.core import DocumentProcessor, SemanticChunker, VectorStore
from knowledge_extraction.extraction.concept_extractor import create_batch_extraction_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def process_batch(
    pdf_dir: Path,
    output_dir: Path,
    max_files: int = None,
    sample_chunks: int = None
) -> None:
    """
    Process multiple PDFs in batch.

    Args:
        pdf_dir: Directory containing PDF files
        output_dir: Base output directory
        max_files: Limit number of files (for testing)
        sample_chunks: Limit chunks per file (for testing)
    """
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all PDFs
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_dir}")
        logger.info("Please add PDF files to the 'pdfs/' directory")
        return

    if max_files:
        pdf_files = pdf_files[:max_files]

    logger.info(f"Processing {len(pdf_files)} PDFs...")

    processor = DocumentProcessor()
    chunker = SemanticChunker()

    successful = 0
    skipped = 0

    for i, pdf_file in enumerate(pdf_files, 1):
        # Create output directory for this paper
        paper_output = output_dir / pdf_file.stem

        # Skip if already processed
        if (paper_output / "chunks.json").exists():
            logger.info(f"[{i}/{len(pdf_files)}] Skipping {pdf_file.name} (already processed)")
            skipped += 1
            continue

        logger.info(f"[{i}/{len(pdf_files)}] Processing {pdf_file.name}...")

        try:
            paper_output.mkdir(parents=True, exist_ok=True)

            # Process document
            document = processor.process_file(pdf_file)
            if not document:
                logger.warning(f"  Could not process {pdf_file.name}")
                continue

            # Save document
            processor.save_document(document, paper_output / "document.json")

            # Chunk document
            chunks = chunker.chunk_document(
                text=document['text'],
                source_file=pdf_file.name,
                page_mapping=document.get('page_mapping')
            )

            if sample_chunks:
                chunks = chunks[:sample_chunks]

            # Save chunks
            chunks_data = {
                'source_file': pdf_file.name,
                'metadata': document['metadata'],
                'num_chunks': len(chunks),
                'chunks': [c.to_dict() for c in chunks]
            }

            chunks_file = paper_output / "chunks.json"
            chunks_file.write_text(json.dumps(chunks_data, indent=2))

            # Index into vector store
            db_path = paper_output / "chroma_db"
            vector_store = VectorStore(db_path=str(db_path))
            vector_store.index_chunks([c.to_dict() for c in chunks])

            # Generate extraction batch
            extraction_batch = paper_output / "extraction_batch.json"
            create_batch_extraction_file([c.to_dict() for c in chunks], extraction_batch)

            logger.info(f"  ✓ Processed {len(chunks)} chunks")
            successful += 1

        except Exception as e:
            logger.error(f"  Error processing {pdf_file.name}: {e}")
            continue

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Batch Processing Complete!")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Failed: {len(pdf_files) - successful - skipped}")
    logger.info("=" * 60)

    logger.info("\nNext steps:")
    logger.info("1. Use Claude Code to process extraction batches:")
    logger.info(f"   cd {output_dir}")
    logger.info("   # Process each paper's extraction_batch.json")
    logger.info("\n2. Build unified knowledge graph:")
    logger.info(f"   python -m knowledge_extraction.scripts.build_graph {output_dir}/")


def main() -> None:
    """Main entry point."""
    # Configuration
    pdf_dir = Path("pdfs")
    output_dir = Path("output")

    # Create pdfs directory if it doesn't exist
    if not pdf_dir.exists():
        pdf_dir.mkdir()
        logger.info(f"Created {pdf_dir}/ directory")
        logger.info("Please add PDF files to this directory and run again")
        return

    # Process batch
    process_batch(
        pdf_dir=pdf_dir,
        output_dir=output_dir,
        max_files=None,  # Process all files
        sample_chunks=None  # Process all chunks
    )


if __name__ == "__main__":
    main()
