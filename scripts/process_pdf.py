#!/usr/bin/env python3
"""
Complete Knowledge Extraction Pipeline

Orchestrates the entire workflow from PDF to knowledge graph.
Uses Claude Code for LLM work (cost-effective).
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

# Import from installed package
from knowledge_extraction.core import DocumentProcessor, GraphBuilder, SemanticChunker, VectorStore
from knowledge_extraction.extraction import (
    ConceptExtractorMCP,
    EntityResolverMCP,
    RelationshipExtractor,
)
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


def run_complete_pipeline(
    input_file: str,
    output_dir: str = "./output",
    sample_chunks: int | None = None,
    skip_validation: bool = False
) -> bool:
    """
    Run complete extraction pipeline.

    Args:
        input_file: Path to PDF or document
        output_dir: Output directory
        sample_chunks: Only process N chunks (for testing)
        skip_validation: Skip concept validation

    Returns:
        True if successful, False otherwise
    """
    # Validate inputs
    if not input_file or not input_file.strip():
        raise ValueError("input_file cannot be empty")

    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not input_path.is_file():
        raise ValueError(f"Input must be a file, got directory: {input_path}")

    # Validate sample_chunks if provided
    if sample_chunks is not None:
        if not isinstance(sample_chunks, int):
            raise TypeError(f"sample_chunks must be int, got {type(sample_chunks).__name__}")
        if sample_chunks < 1:
            raise ValueError(f"sample_chunks must be positive, got {sample_chunks}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    print_header("KNOWLEDGE EXTRACTION PIPELINE (MCP VERSION)")
    logger.info(f"Input: {input_path.name}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # =========================================================================
    # PHASE 1: DOCUMENT PROCESSING
    # =========================================================================

    print_header("📄 PHASE 1: Document Processing")

    print_step("Extracting text from PDF...")
    processor = DocumentProcessor()
    document = processor.process_file(input_path)

    if not document:
        logger.error("Could not process document")
        return False

    logger.info(f"  ✓ Extracted {len(document['text']):,} characters")
    logger.info(f"  ✓ Pages: {document['metadata']['pages']}")
    logger.info(f"  ✓ Title: {document['metadata']['title']}")

    # Save document
    doc_file = output_path / "document.json"
    processor.save_document(document, doc_file)

    # =========================================================================
    # PHASE 2: SEMANTIC CHUNKING
    # =========================================================================

    print_header("✂️ PHASE 2: Semantic Chunking")

    print_step("Chunking document semantically...")
    chunker = SemanticChunker(
        target_chunk_size=600,
        min_chunk_size=100,
        max_chunk_size=1000
    )

    chunks = chunker.chunk_document(
        text=document['text'],
        source_file=input_path.name,
        page_mapping=document.get('page_mapping')
    )

    if sample_chunks:
        chunks = chunks[:sample_chunks]
        logger.warning(f"  ⚠️  Using only first {sample_chunks} chunks for testing")

    stats = chunker.get_stats(chunks)
    logger.info(f"  ✓ Created {len(chunks)} semantic chunks")
    logger.info(f"  ✓ Avg size: {stats['avg_words_per_chunk']:.0f} words")

    # Save chunks
    chunks_file = output_path / "chunks.json"
    chunks_data = {
        'source_file': input_path.name,
        'metadata': document['metadata'],
        'num_chunks': len(chunks),
        'chunks': [c.to_dict() for c in chunks]
    }
    chunks_file.write_text(json.dumps(chunks_data, indent=2), encoding='utf-8')
    logger.info(f"  ✓ Saved chunks to {chunks_file.name}")

    # =========================================================================
    # PHASE 3: VECTOR INDEXING
    # =========================================================================

    print_header("📊 PHASE 3: Vector Indexing")

    print_step("Indexing chunks into vector database...")
    db_path = output_path / "chroma_db"
    vector_store = VectorStore(db_path=str(db_path))

    vector_store.index_chunks([c.to_dict() for c in chunks])

    stats = vector_store.get_stats()
    logger.info(f"  ✓ Indexed {stats['total_chunks']} chunks")
    logger.info(f"  ✓ Database: {db_path}")

    # =========================================================================
    # PHASE 4: CONCEPT EXTRACTION (GENERATE PROMPTS)
    # =========================================================================

    print_header("🧠 PHASE 4: Concept Extraction (Generate Prompts)")

    print_step("Generating extraction prompts for Claude Code...")

    extraction_batch_file = output_path / "extraction_batch.json"
    create_batch_extraction_file([c.to_dict() for c in chunks], extraction_batch_file)

    logger.warning("\n  ⚠️  MANUAL STEP REQUIRED:")
    logger.info(f"  Use Claude Code to process: {extraction_batch_file}")
    logger.info(f"  Save responses to: {output_path / 'extraction_responses.json'}")
    logger.info("\n  In Claude Code, type:")
    logger.info(f"  \"Process {extraction_batch_file} and save responses to extraction_responses.json\"")
    logger.info("\n  Press Enter when done...")
    input()

    # =========================================================================
    # PHASE 5: PARSE EXTRACTION RESPONSES
    # =========================================================================

    print_header("🔍 PHASE 5: Parse Extraction Responses")

    responses_file = output_path / "extraction_responses.json"
    if not responses_file.exists():
        logger.error(f"  ❌ Error: {responses_file} not found")
        logger.info("  Please run Claude Code first to generate responses")
        return False

    print_step("Parsing Claude's extraction responses...")

    extractor = ConceptExtractorMCP()
    responses_data = json.loads(responses_file.read_text(encoding='utf-8'))

    all_concepts = []
    for response in responses_data.get('responses', []):
        metadata = response['metadata']
        concepts = extractor.parse_extraction_response(
            response['response_text'],
            metadata['chunk_id'],
            metadata['source_file'],
            metadata['page']
        )
        all_concepts.extend(concepts)

    concepts_file = output_path / "concepts.json"
    extractor.save_concepts(all_concepts, concepts_file)

    logger.info(f"  ✓ Parsed {len(all_concepts)} concepts")

    # =========================================================================
    # PHASE 6: ENTITY RESOLUTION
    # =========================================================================

    print_header("🔗 PHASE 6: Entity Resolution")

    print_step("Resolving duplicate entities...")

    resolver = EntityResolverMCP()
    concepts_dict = [c.to_dict() for c in all_concepts]

    entities, ambiguous_pairs = resolver.resolve_entities_automatic(concepts_dict)

    entities_file = output_path / "entities.json"
    resolver.save_entities(entities, entities_file)

    logger.info(f"  ✓ Resolved {len(all_concepts)} concepts → {len(entities)} unique entities")

    if ambiguous_pairs:
        ambiguous_file = output_path / "ambiguous_pairs.json"
        resolver.create_ambiguous_batch_file(ambiguous_pairs, ambiguous_file)

        logger.warning(f"\n  ⚠️  {len(ambiguous_pairs)} ambiguous pairs need Claude Code")
        logger.info(f"  Process {ambiguous_file} with Claude Code (optional)")
        logger.info("  Press Enter to continue without resolving ambiguous pairs...")
        input()

    # =========================================================================
    # PHASE 7: RELATIONSHIP EXTRACTION (GENERATE PROMPTS)
    # =========================================================================

    print_header("🕸️ PHASE 7: Relationship Extraction")

    print_step("Generating relationship classification prompts...")

    rel_extractor = RelationshipExtractor()
    rel_batch_file = output_path / "relationship_batch.json"

    rel_extractor.create_classification_batch(
        [e.to_dict() for e in entities],
        [c.to_dict() for c in chunks],
        rel_batch_file
    )

    logger.warning("\n  ⚠️  MANUAL STEP REQUIRED:")
    logger.info(f"  Use Claude Code to process: {rel_batch_file}")
    logger.info(f"  Save responses to: {output_path / 'relationship_responses.json'}")
    logger.info("\n  Press Enter when done (or skip for now)...")
    response = input()

    relationships = []
    relationships_file = output_path / "relationships.json"

    if response.lower() not in ['skip', 's']:
        # =====================================================================
        # PHASE 8: PARSE RELATIONSHIP RESPONSES
        # =====================================================================

        print_header("🔗 PHASE 8: Parse Relationship Responses")

        rel_responses_file = output_path / "relationship_responses.json"
        if rel_responses_file.exists():
            relationships = rel_extractor.parse_classification_batch(
                rel_responses_file,
                relationships_file
            )
            logger.info(f"  ✓ Extracted {len(relationships)} relationships")
        else:
            logger.warning("  ⚠️  No relationship responses found, creating empty file")
            relationships_file.write_text(json.dumps({'relationships': []}, indent=2))
    else:
        logger.warning("  ⚠️  Skipping relationship extraction")
        relationships_file.write_text(json.dumps({'relationships': []}, indent=2))

    # =========================================================================
    # PHASE 9: KNOWLEDGE GRAPH BUILDING
    # =========================================================================

    print_header("📈 PHASE 9: Knowledge Graph Building")

    print_step("Building NetworkX graph...")

    builder = GraphBuilder()
    graph = builder.build_graph(
        [e.to_dict() for e in entities],
        relationships
    )

    # Get top concepts
    top_concepts = builder.get_top_concepts(10)
    logger.info("\n  ✓ Top 10 Concepts (by centrality):")
    for i, (concept, score) in enumerate(top_concepts, 1):
        logger.info(f"    {i}. {concept}: {score:.3f}")

    # Export
    graph_json_file = output_path / "knowledge_graph.json"
    graph_graphml_file = output_path / "knowledge_graph.graphml"

    builder.export_json(graph_json_file)
    builder.export_graphml(graph_graphml_file)

    # =========================================================================
    # SUMMARY
    # =========================================================================

    elapsed = time.time() - start_time

    print_header("✅ PIPELINE COMPLETE")
    logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total time: {elapsed/60:.1f} minutes")

    logger.info("\n📁 Output Files:")
    logger.info(f"  📄 Document: {doc_file}")
    logger.info(f"  ✂️  Chunks: {chunks_file}")
    logger.info(f"  🧠 Concepts: {concepts_file}")
    logger.info(f"  🔗 Entities: {entities_file}")
    logger.info(f"  🕸️ Relationships: {relationships_file}")
    logger.info(f"  📈 Graph (JSON): {graph_json_file}")
    logger.info(f"  📈 Graph (GraphML): {graph_graphml_file}")
    logger.info(f"  📊 Vector DB: {db_path}")

    logger.info("\n📊 Statistics:")
    logger.info(f"  📚 Chunks: {len(chunks)}")
    logger.info(f"  💡 Concepts: {len(all_concepts)}")
    logger.info(f"  🔗 Unique Entities: {len(entities)}")
    logger.info(f"  🕸️ Relationships: {len(relationships)}")

    graph_stats = builder.get_stats()
    logger.info(f"  📈 Graph nodes: {graph_stats['nodes']}")
    logger.info(f"  📈 Graph edges: {graph_stats['edges']}")
    logger.info(f"  📈 Graph density: {graph_stats['density']:.3f}")

    logger.info("\n🎉 Your knowledge graph is ready!")
    logger.info(f"  - Open {graph_graphml_file} in Gephi or Cytoscape for visualization")
    logger.info(f"  - Query the vector database at {db_path}")
    logger.info("  - Use the entities and relationships for further analysis")

    return True


# =========================================================================
# CLI
# =========================================================================

def main() -> int:
    """Command-line interface"""
    import argparse

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    parser = argparse.ArgumentParser(
        description='Complete knowledge extraction pipeline (MCP version)'
    )
    parser.add_argument('input', type=str, help='Input PDF file')
    parser.add_argument('--output', type=str, default='./output',
                       help='Output directory')
    parser.add_argument('--sample', type=int, default=None,
                       help='Only process first N chunks (for testing)')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip concept validation')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        success = run_complete_pipeline(
            input_file=args.input,
            output_dir=args.output,
            sample_chunks=args.sample,
            skip_validation=args.skip_validation
        )
        return 0 if success else 1
    except (ValueError, FileNotFoundError, TypeError) as e:
        logger.error(f"Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.debug("Full traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
