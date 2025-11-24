#!/usr/bin/env python3
"""
Batch Process Multiple PDFs through Knowledge Extraction Pipeline

Processes multiple PDFs automatically without manual intervention.
"""

import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

# Set up logging
logger = logging.getLogger(__name__)

# Import from installed package
from knowledge_extraction.core import DocumentProcessor, GraphBuilder, SemanticChunker, VectorStore
from knowledge_extraction.extraction import ConceptExtractorMCP, EntityResolverMCP
from knowledge_extraction.extraction.concept_extractor import create_batch_extraction_file


def print_header(text: str) -> None:
    """Print formatted section header"""
    logger.info("")
    logger.info("=" * 70)
    logger.info(text)
    logger.info("=" * 70)
    logger.info("")


def print_step(text: str) -> None:
    """Print step description"""
    logger.info(f"  {text}")


def process_single_pdf(
    input_file: Path,
    output_dir: Path,
    sample_chunks: int | None = None,
    extract_concepts: bool = True
) -> dict[str, Any]:
    """
    Process a single PDF through the pipeline.

    Args:
        input_file: Path to PDF file
        output_dir: Output directory for this PDF
        sample_chunks: Limit to N chunks for testing
        extract_concepts: Whether to generate extraction prompts

    Returns:
        dict with processing stats and status
    """
    # Validate inputs
    if not input_file.exists():
        raise FileNotFoundError(f"PDF not found: {input_file}")

    if not input_file.is_file():
        raise ValueError(f"Expected file, got: {input_file}")

    if sample_chunks is not None:
        if not isinstance(sample_chunks, int) or sample_chunks < 1:
            raise ValueError(f"sample_chunks must be positive int, got {sample_chunks}")

    result: dict[str, Any] = {
        'file': input_file.name,
        'success': False,
        'error': None,
        'stats': {},
        'output_dir': str(output_dir)
    }

    start_time = time.time()

    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n📄 Processing: {input_file.name}")

        # =====================================================================
        # PHASE 1: DOCUMENT PROCESSING
        # =====================================================================

        print_step("Extracting text from PDF...")
        processor = DocumentProcessor()
        document = processor.process_file(input_file)

        if not document:
            result['error'] = "Could not process document"
            return result

        logger.info(f"  ✓ Extracted {len(document['text']):,} characters, {document['metadata']['pages']} pages")

        # Save document
        doc_file = output_dir / "document.json"
        processor.save_document(document, doc_file)

        result['stats']['pages'] = document['metadata']['pages']
        result['stats']['chars'] = len(document['text'])

        # =====================================================================
        # PHASE 2: SEMANTIC CHUNKING
        # =====================================================================

        print_step("Chunking document semantically...")
        chunker = SemanticChunker(
            target_chunk_size=600,
            min_chunk_size=100,
            max_chunk_size=1000
        )

        chunks = chunker.chunk_document(
            text=document['text'],
            source_file=input_file.name,
            page_mapping=document.get('page_mapping')
        )

        if sample_chunks:
            chunks = chunks[:sample_chunks]

        logger.info(f"  ✓ Created {len(chunks)} semantic chunks")

        # Save chunks
        chunks_file = output_dir / "chunks.json"
        chunks_data = {
            'source_file': input_file.name,
            'metadata': document['metadata'],
            'num_chunks': len(chunks),
            'chunks': [c.to_dict() for c in chunks]
        }
        chunks_file.write_text(json.dumps(chunks_data, indent=2), encoding='utf-8')

        result['stats']['chunks'] = len(chunks)

        # =====================================================================
        # PHASE 3: VECTOR INDEXING
        # =====================================================================

        print_step("Indexing chunks into vector database...")
        db_path = output_dir / "chroma_db"
        vector_store = VectorStore(db_path=str(db_path))
        vector_store.index_chunks([c.to_dict() for c in chunks])

        # =====================================================================
        # PHASE 4: CONCEPT EXTRACTION (if enabled)
        # =====================================================================

        if extract_concepts:
            print_step("Generating extraction prompts...")

            # Initialize extractor (needed for save_concepts later)
            extractor = ConceptExtractorMCP()

            extraction_batch_file = output_dir / "extraction_batch.json"
            create_batch_extraction_file([c.to_dict() for c in chunks], extraction_batch_file)

            logger.info("")
            logger.info("  " + "="*66)
            logger.warning("  ⚠️  WARNING: Concept extraction requires manual Claude Code processing")
            logger.info("  " + "="*66)
            logger.info(f"  Extraction prompts saved to: {extraction_batch_file}")
            logger.info("  To extract concepts:")
            logger.info("    1. Use Claude Code to process the batch file")
            logger.info(f"    2. Save responses to: {output_dir / 'extraction_responses.json'}")
            logger.info("    3. Run parse step to convert responses to concepts")
            logger.info("  ")
            logger.info("  Skipping concept extraction for now (batch mode doesn't auto-process)")
            logger.info("  " + "="*66)
            logger.info("")

            # NOTE: In batch mode, we only generate the extraction batch file
            # Manual Claude Code processing is required to actually extract concepts
            # This is intentional - batch processing shouldn't make API calls automatically
            all_concepts: list[Any] = []

            # Save concepts (even if empty)
            concepts_file = output_dir / "concepts.json"
            extractor.save_concepts(all_concepts, concepts_file)
            result['stats']['concepts'] = len(all_concepts)

            # =====================================================================
            # PHASE 5: ENTITY RESOLUTION
            # =====================================================================

            if all_concepts:
                print_step("Resolving duplicate entities...")
                resolver = EntityResolverMCP()
                concepts_dict = [c.to_dict() for c in all_concepts]

                entities, ambiguous_pairs = resolver.resolve_entities_automatic(concepts_dict)

                entities_file = output_dir / "entities.json"
                resolver.save_entities(entities, entities_file)

                result['stats']['entities'] = len(entities)

                # =====================================================================
                # PHASE 6: KNOWLEDGE GRAPH BUILDING
                # =====================================================================

                if entities:
                    print_step("Building knowledge graph...")
                    builder = GraphBuilder()
                    graph = builder.build_graph([e.to_dict() for e in entities], [])

                    graph_json_file = output_dir / "knowledge_graph.json"
                    graph_graphml_file = output_dir / "knowledge_graph.graphml"

                    builder.export_json(graph_json_file)
                    builder.export_graphml(graph_graphml_file)

                    stats = builder.get_stats()
                    result['stats']['graph_nodes'] = stats['nodes']
                    result['stats']['graph_edges'] = stats['edges']

        # Success
        elapsed = time.time() - start_time
        result['success'] = True
        result['stats']['elapsed_seconds'] = elapsed

        logger.info(f"  ✅ Completed in {elapsed:.1f}s")

    except Exception as e:
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
        logger.error(f"  ❌ Error: {e}")

    return result


def batch_process(
    input_dir: Path,
    output_base_dir: Path,
    sample_chunks: int | None = None,
    max_files: int | None = None,
    extract_concepts: bool = False  # Disabled by default since we need Claude
) -> list[dict[str, Any]]:
    """
    Process all PDFs in a directory.

    Args:
        input_dir: Directory containing PDFs
        output_base_dir: Base directory for outputs
        sample_chunks: Only process N chunks per file (for testing)
        max_files: Only process first N files
        extract_concepts: Whether to extract concepts (requires Claude)

    Returns:
        List of result dicts for each processed file
    """
    # Validate inputs
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not input_dir.is_dir():
        raise ValueError(f"Input must be a directory, got: {input_dir}")

    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    if sample_chunks is not None:
        if not isinstance(sample_chunks, int) or sample_chunks < 1:
            raise ValueError(f"sample_chunks must be positive int, got {sample_chunks}")

    if max_files is not None:
        if not isinstance(max_files, int) or max_files < 1:
            raise ValueError(f"max_files must be positive int, got {max_files}")

    # Find all PDFs
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in: {input_dir}")
        logger.info("Please add PDF files to the directory and try again")
        return []

    if max_files:
        pdf_files = pdf_files[:max_files]

    print_header(f"BATCH PROCESSING: {len(pdf_files)} PDFs")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_base_dir}")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if sample_chunks:
        logger.info(f"Sample mode: {sample_chunks} chunks per file")

    # Process each file
    results = []
    successful = 0
    failed = 0
    skipped = 0

    for i, pdf_file in enumerate(pdf_files, 1):
        # Create output directory for this file
        output_dir = output_base_dir / pdf_file.stem

        # Check if already processed
        if output_dir.exists() and (output_dir / "document.json").exists():
            logger.info(f"\n[{i}/{len(pdf_files)}] ⏭️  Skipping {pdf_file.name} (already processed)")
            skipped += 1
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"[{i}/{len(pdf_files)}] {pdf_file.name}")
        logger.info(f"{'='*70}")

        # Process
        result = process_single_pdf(
            pdf_file,
            output_dir,
            sample_chunks=sample_chunks,
            extract_concepts=extract_concepts
        )

        results.append(result)

        if result['success']:
            successful += 1
        else:
            failed += 1

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print_header("✅ BATCH PROCESSING COMPLETE")
    logger.info(f"Total files: {len(pdf_files)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Failed: {failed}")

    # Save results
    results_file = output_base_dir / "batch_results.json"
    results_file.write_text(json.dumps({
        'timestamp': datetime.now().isoformat(),
        'total_files': len(pdf_files),
        'successful': successful,
        'failed': failed,
        'results': results
    }, indent=2), encoding='utf-8')

    logger.info(f"\n📊 Detailed results saved to: {results_file}")

    # Print summary table
    logger.info("\n📋 Summary:")
    logger.info(f"{'File':<50} {'Status':<10} {'Pages':<8} {'Chunks':<8}")
    logger.info("-" * 80)
    for result in results:
        status = "✓" if result['success'] else "✗"
        pages = result['stats'].get('pages', 0)
        chunks = result['stats'].get('chunks', 0)
        file_name = result['file'][:47] + "..." if len(result['file']) > 50 else result['file']
        logger.info(f"{file_name:<50} {status:<10} {pages:<8} {chunks:<8}")

    return results


def main() -> int:
    """Command-line interface"""
    import argparse

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    parser = argparse.ArgumentParser(
        description='Batch process PDFs through knowledge extraction pipeline'
    )
    parser.add_argument('input_dir', type=str, help='Directory containing PDFs')
    parser.add_argument('--output', type=str, default='./batch_output',
                       help='Output base directory')
    parser.add_argument('--sample', type=int, default=None,
                       help='Only process first N chunks per file (for testing)')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Only process first N files')
    parser.add_argument('--extract-concepts', action='store_true',
                       help='Extract concepts (requires Claude setup)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        results = batch_process(
            input_dir=args.input_dir,
            output_base_dir=args.output,
            sample_chunks=args.sample,
            max_files=args.max_files,
            extract_concepts=args.extract_concepts
        )

        # Return success if any files processed
        return 0 if results else 1

    except (ValueError, FileNotFoundError, TypeError) as e:
        logger.error(f"Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.debug("Full traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
