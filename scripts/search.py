#!/usr/bin/env python3
"""
Semantic Search Across All Processed Documents

Search through all vector databases to find relevant content.
"""

import logging
from pathlib import Path
from typing import Any

# Set up logging
logger = logging.getLogger(__name__)
import json

# Import from installed package
from knowledge_extraction.core import VectorStore


def find_all_vector_dbs(base_dir: Path) -> list[tuple[str, Path]]:
    """
    Find all ChromaDB databases in the output directory.

    Returns:
        List of (document_name, db_path) tuples
    """
    databases = []

    for doc_dir in base_dir.iterdir():
        if doc_dir.is_dir():
            db_path = doc_dir / "chroma_db"
            if db_path.exists():
                databases.append((doc_dir.name, db_path))

    return sorted(databases)


def search_all_databases(
    base_dir: Path,
    query: str,
    top_k: int = 5,
    min_similarity: float = 0.0
) -> list[dict[str, Any]]:
    """
    Search across all vector databases.

    Args:
        base_dir: Base directory containing all processed documents
        query: Search query
        top_k: Number of results per database
        min_similarity: Minimum similarity threshold

    Returns:
        List of search results sorted by relevance
    """

    databases = find_all_vector_dbs(base_dir)

    logger.info(f"\n🔍 Searching {len(databases)} documents for: \"{query}\"")
    logger.info(f"{'='*70}\n")

    all_results = []

    for doc_name, db_path in databases:
        try:
            # Load vector store
            vector_store = VectorStore(db_path=str(db_path))

            # Search
            results = vector_store.search(query, n_results=top_k)

            # Add document name to results
            for result in results:
                if result['similarity'] >= min_similarity:
                    result['document'] = doc_name
                    all_results.append(result)

        except Exception as e:
            logger.error(f"⚠️  Error searching {doc_name}: {e}")

    # Sort by similarity
    all_results.sort(key=lambda x: x['similarity'], reverse=True)

    return all_results


def format_results(results: list[dict[str, Any]], max_results: int = 20) -> None:
    """Format and display search results"""

    if not results:
        logger.info("No results found.")
        return

    logger.info(f"Found {len(results)} results (showing top {min(max_results, len(results))}):\n")

    for i, result in enumerate(results[:max_results], 1):
        doc_name = result['document']
        chunk_id = result['chunk_id']
        page = result['metadata'].get('page', 'N/A')
        similarity = result['similarity']
        text = result['text']

        # Truncate long text
        if len(text) > 300:
            text = text[:300] + "..."

        logger.info(f"{i}. 📄 {doc_name}")
        logger.info(f"   Page: {page} | Chunk: {chunk_id} | Similarity: {similarity:.3f}")
        logger.info(f"   {text}")
        logger.info("")


def search_with_context(
    base_dir: Path,
    query: str,
    top_k: int = 3,
    context_window: int = 1
) -> list[dict[str, Any]]:
    """
    Search with surrounding context chunks.

    Args:
        base_dir: Base directory
        query: Search query
        top_k: Results per database
        context_window: Number of surrounding chunks to include

    Returns:
        Results with context
    """

    results = search_all_databases(base_dir, query, top_k=top_k)

    # For top results, load surrounding chunks
    for result in results[:10]:  # Only add context to top 10
        doc_dir = base_dir / result['document']
        chunks_file = doc_dir / "chunks.json"

        if chunks_file.exists():
            try:
                chunks_data = json.loads(chunks_file.read_text(encoding='utf-8'))
                chunks = chunks_data['chunks']

                # Find current chunk index
                chunk_id = result['chunk_id']
                for idx, chunk in enumerate(chunks):
                    if chunk['chunk_id'] == chunk_id:
                        # Get surrounding chunks
                        start = max(0, idx - context_window)
                        end = min(len(chunks), idx + context_window + 1)

                        context_chunks = [chunks[i]['text'] for i in range(start, end)]
                        result['context'] = '\n\n'.join(context_chunks)
                        break
            except Exception:
                pass

    return results


def interactive_search(base_dir: Path) -> None:
    """Interactive search loop"""

    databases = find_all_vector_dbs(base_dir)

    logger.info("="*70)
    logger.info("SEMANTIC SEARCH - Knowledge Base")
    logger.info("="*70)
    logger.info(f"\n📚 Loaded {len(databases)} documents")
    logger.info("💡 Type your search query or 'quit' to exit\n")

    while True:
        try:
            query = input("🔍 Search: ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                logger.info("\n👋 Goodbye!")
                break

            # Parse options
            top_k = 3
            max_results = 20

            if query.startswith('/'):
                # Command mode
                parts = query.split()
                if parts[0] == '/help':
                    logger.info("\nCommands:")
                    logger.info("  /help          - Show this help")
                    logger.info("  /stats         - Show database statistics")
                    logger.info("  /docs          - List all documents")
                    logger.info("  <query>        - Search for query")
                    logger.info("")
                    continue
                elif parts[0] == '/stats':
                    logger.info("\n📊 Database Statistics:")
                    logger.info(f"   Total documents: {len(databases)}")
                    total_chunks = 0
                    for doc_name, db_path in databases:
                        vs = VectorStore(db_path=str(db_path))
                        stats = vs.get_stats()
                        total_chunks += stats['total_chunks']
                    logger.info(f"   Total chunks indexed: {total_chunks}")
                    logger.info("")
                    continue
                elif parts[0] == '/docs':
                    logger.info(f"\n📚 Available Documents ({len(databases)}):")
                    for i, (doc_name, _) in enumerate(databases, 1):
                        logger.info(f"   {i}. {doc_name}")
                    logger.info("")
                    continue

            # Perform search
            results = search_all_databases(base_dir, query, top_k=top_k)
            format_results(results, max_results=max_results)

        except KeyboardInterrupt:
            logger.info("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"\n❌ Error: {e}\n")


def main() -> int:
    """Command-line interface"""
    import argparse

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    parser = argparse.ArgumentParser(
        description='Semantic search across processed documents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive search
  python3 semantic_search.py --base-dir PIPELINE_OUTPUT/my_batch/ --interactive

  # Single query
  python3 semantic_search.py --base-dir PIPELINE_OUTPUT/my_batch/ --query "cognitive load"

  # More results per document
  python3 semantic_search.py --base-dir my_batch/ --query "methodology" --top-k 5
        """
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        required=True,
        help='Base directory containing processed documents with chroma_db/'
    )
    parser.add_argument('--query', type=str, help='Search query')
    parser.add_argument('--top-k', type=int, default=3,
                       help='Results per document (default: 3)')
    parser.add_argument('--max-results', type=int, default=20,
                       help='Maximum total results to show (default: 20)')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive search mode')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Validate base_dir
        base_dir = Path(args.base_dir)

        if not base_dir.exists():
            logger.error(f"Directory not found: {base_dir}")
            logger.info("Please provide a valid directory containing processed documents")
            logger.info("Each subdirectory should have a chroma_db/ folder")
            return 1

        if not base_dir.is_dir():
            logger.error(f"Expected directory, got file: {base_dir}")
            return 1

        # Validate numeric parameters
        if args.top_k < 1:
            logger.error(f"top-k must be positive, got {args.top_k}")
            return 1

        if args.max_results < 1:
            logger.error(f"max-results must be positive, got {args.max_results}")
            return 1

        # Run search
        if args.interactive or not args.query:
            interactive_search(base_dir)
        else:
            results = search_all_databases(base_dir, args.query, top_k=args.top_k)
            format_results(results, max_results=args.max_results)

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.debug("Full traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
