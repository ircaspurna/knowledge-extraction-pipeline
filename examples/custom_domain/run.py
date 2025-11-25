"""
Custom Domain Example

Shows how to use domain-specific extraction configuration.
"""

import json
import logging
from pathlib import Path

from knowledge_extraction.core import DocumentProcessor, SemanticChunker
from knowledge_extraction.extraction import ConceptExtractorMCP
from knowledge_extraction.extraction.concept_extractor import create_batch_extraction_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Demonstrate custom domain extraction."""
    # Setup
    custom_config = Path("custom_prompts.yaml")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Check for custom config
    if not custom_config.exists():
        logger.warning(f"Custom config not found: {custom_config}")
        logger.info("Using default configuration")
        extractor = ConceptExtractorMCP()
    else:
        logger.info(f"Loading custom configuration from {custom_config}")
        extractor = ConceptExtractorMCP(config_path=custom_config)

    # Look for sample PDF
    sample_pdf = Path("sample_paper.pdf")

    if not sample_pdf.exists():
        logger.error("Sample PDF not found!")
        logger.info("Please add a PDF file named 'sample_paper.pdf' to this directory")
        logger.info("\nExample structure:")
        logger.info("  examples/custom_domain/")
        logger.info("  ├── run.py (this file)")
        logger.info("  ├── custom_prompts.yaml (your domain config)")
        logger.info("  └── sample_paper.pdf (your test document)")
        return

    logger.info(f"Processing {sample_pdf}...")

    # Process document
    processor = DocumentProcessor()
    document = processor.process_file(sample_pdf)

    if not document:
        logger.error("Could not process PDF")
        return

    logger.info(f"  ✓ Extracted {len(document['text'])} characters")

    # Create semantic chunks
    chunker = SemanticChunker(
        target_chunk_size=600,
        min_chunk_size=100,
        max_chunk_size=1000
    )

    chunks = chunker.chunk_document(
        text=document['text'],
        source_file=sample_pdf.name,
        page_mapping=document.get('page_mapping')
    )

    logger.info(f"  ✓ Created {len(chunks)} semantic chunks")

    # Save chunks
    chunks_file = output_dir / "chunks.json"
    chunks_data = {
        'source_file': sample_pdf.name,
        'metadata': document['metadata'],
        'num_chunks': len(chunks),
        'chunks': [c.to_dict() for c in chunks]
    }

    chunks_file.write_text(json.dumps(chunks_data, indent=2))
    logger.info(f"  ✓ Saved chunks to {chunks_file}")

    # Generate extraction batch with custom prompts
    extraction_batch = output_dir / "extraction_batch.json"
    create_batch_extraction_file([c.to_dict() for c in chunks], extraction_batch)

    logger.info(f"  ✓ Generated extraction batch: {extraction_batch}")

    # Show sample prompt
    sample_chunk = chunks[0].to_dict()
    sample_prompt = extractor.generate_extraction_prompt(sample_chunk)

    logger.info("\n" + "=" * 60)
    logger.info("Sample Extraction Prompt (first 500 chars):")
    logger.info("=" * 60)
    logger.info(sample_prompt[:500] + "...")
    logger.info("=" * 60)

    logger.info("\nNext steps:")
    logger.info("1. Review the generated prompt above")
    logger.info("2. Adjust custom_prompts.yaml if needed")
    logger.info("3. Use Claude Code to process the extraction batch:")
    logger.info(f"   claude-code process {extraction_batch}")
    logger.info("\n4. The extraction will use your custom:")
    logger.info("   - Domain-specific categories")
    logger.info("   - Custom terminology")
    logger.info("   - Specialized validation rules")


if __name__ == "__main__":
    main()
