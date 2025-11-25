#!/usr/bin/env python3
"""
Vector Store - ChromaDB wrapper for semantic search

Provides:
- Chunk indexing with embeddings
- Semantic search
- Metadata filtering
- Persistence
"""

import json
import logging
from pathlib import Path
from typing import Any

from ..utils.path_utils import validate_directory_path
from ..utils.retry import DATABASE_RETRY_CONFIG, retry

# Set up logging
logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    logger.warning("chromadb not installed. Install with: pip install chromadb")
    chromadb = None


class VectorStore:
    """
    Wrapper for ChromaDB vector database.
    
    Stores text chunks with embeddings for semantic search.
    """

    def __init__(
        self,
        db_path: str = "./chroma_db",
        collection_name: str = "documents"
    ) -> None:
        """
        Initialize vector store.

        Args:
            db_path: Path to ChromaDB directory
            collection_name: Name of collection
        """
        if not chromadb:
            raise ImportError("chromadb not installed")

        # Validate and sanitize database path (security check)
        try:
            self.db_path = validate_directory_path(db_path, create=True)
        except (ValueError, OSError) as e:
            raise ValueError(f"Invalid database path: {e}")

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        self.stats = {
            'chunks_indexed': 0,
            'queries_performed': 0
        }

    @retry(
        max_retries=DATABASE_RETRY_CONFIG.max_retries,
        base_delay=DATABASE_RETRY_CONFIG.base_delay,
        max_delay=DATABASE_RETRY_CONFIG.max_delay,
        retryable_exceptions=DATABASE_RETRY_CONFIG.retryable_exceptions
    )
    def index_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """
        Index chunks into vector store.

        Args:
            chunks: List of chunk dicts with 'text', 'chunk_id', etc.

        Returns:
            Number of chunks indexed

        Note:
            Automatically retries on transient database errors with exponential backoff.
        """
        # Validate inputs
        if not isinstance(chunks, list):
            raise ValueError(f"chunks must be a list, got {type(chunks).__name__}")

        if not chunks:
            logger.warning("No chunks to index")
            return 0

        # Validate chunk structure
        required_fields = ['text', 'chunk_id']
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValueError(f"Chunk {i} must be a dict, got {type(chunk).__name__}")

            missing = [f for f in required_fields if f not in chunk]
            if missing:
                raise ValueError(f"Chunk {i} missing required fields: {missing}")

            if not chunk['text'] or not chunk['text'].strip():
                raise ValueError(f"Chunk {i} has empty 'text' field")

            if not chunk['chunk_id']:
                raise ValueError(f"Chunk {i} has empty 'chunk_id' field")

        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []

        for chunk in chunks:
            documents.append(chunk['text'])

            # Extract metadata
            metadata = {
                'chunk_id': chunk['chunk_id'],
                'source_file': chunk.get('source_file', 'unknown'),
                'page': chunk.get('page', 1),
                'section': chunk.get('section', 'unknown'),
                'chunk_index': chunk.get('chunk_index', 0)
            }
            metadatas.append(metadata)
            ids.append(chunk['chunk_id'])

        # Add to collection (ChromaDB handles embedding generation)
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        self.stats['chunks_indexed'] += len(chunks)

        logger.info(f"Indexed {len(chunks)} chunks into vector store")
        return len(chunks)

    @retry(
        max_retries=DATABASE_RETRY_CONFIG.max_retries,
        base_delay=DATABASE_RETRY_CONFIG.base_delay,
        max_delay=DATABASE_RETRY_CONFIG.max_delay,
        retryable_exceptions=DATABASE_RETRY_CONFIG.retryable_exceptions
    )
    def search(
        self,
        query: str,
        n_results: int = 10,
        filter_metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Semantic search for similar chunks.

        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter (e.g., {'source_file': 'book.pdf'})

        Returns:
            List of matching chunks with similarity scores

        Note:
            Automatically retries on transient database errors with exponential backoff.
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("query cannot be empty or whitespace-only")

        if not isinstance(n_results, int) or n_results < 1:
            raise ValueError(f"n_results must be a positive integer, got {n_results}")

        if filter_metadata is not None and not isinstance(filter_metadata, dict):
            raise ValueError(f"filter_metadata must be a dict, got {type(filter_metadata).__name__}")

        # Query collection
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_metadata
        )

        self.stats['queries_performed'] += 1

        # Format results
        chunks = []
        if results['ids'] and len(results['ids']) > 0:
            for i in range(len(results['ids'][0])):
                # ChromaDB returns distances (lower = more similar)
                # Convert to similarity (higher = more similar) for consistency
                distance = results['distances'][0][i] if 'distances' in results else None
                similarity = (1 - distance) if distance is not None else None

                chunk = {
                    'chunk_id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': similarity,  # Consistent: higher = more similar
                    'distance': distance  # Keep for reference
                }
                chunks.append(chunk)

        return chunks

    @retry(
        max_retries=DATABASE_RETRY_CONFIG.max_retries,
        base_delay=DATABASE_RETRY_CONFIG.base_delay,
        max_delay=DATABASE_RETRY_CONFIG.max_delay,
        retryable_exceptions=DATABASE_RETRY_CONFIG.retryable_exceptions
    )
    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """
        Retrieve specific chunk by ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Chunk dict or None if not found

        Note:
            Automatically retries on transient database errors with exponential backoff.
        """
        try:
            results = self.collection.get(
                ids=[chunk_id]
            )

            if results['ids']:
                return {
                    'chunk_id': results['ids'][0],
                    'text': results['documents'][0],
                    'metadata': results['metadatas'][0]
                }
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {e}")

        return None

    def get_stats(self) -> dict[str, Any]:
        """Get vector store statistics"""
        collection_count = self.collection.count()

        return {
            'total_chunks': collection_count,
            'chunks_indexed': self.stats['chunks_indexed'],
            'queries_performed': self.stats['queries_performed'],
            'db_path': str(self.db_path)
        }

    def delete_collection(self) -> None:
        """Delete the collection (use with caution!)"""
        self.client.delete_collection(self.collection.name)
        logger.warning(f"Deleted collection: {self.collection.name}")


# =========================================================================
# CLI
# =========================================================================

def main() -> int:
    """Command-line interface"""
    import argparse

    # Configure logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    parser = argparse.ArgumentParser(
        description='Vector store for semantic search'
    )
    parser.add_argument('command', type=str,
                       choices=['index', 'search', 'stats', 'get'],
                       help='Command to execute')
    parser.add_argument('--chunks', type=str,
                       help='Path to chunks JSON file (for index)')
    parser.add_argument('--query', type=str,
                       help='Search query (for search)')
    parser.add_argument('--chunk-id', type=str,
                       help='Chunk ID to retrieve (for get)')
    parser.add_argument('--db', type=str, default='./chroma_db',
                       help='Path to ChromaDB directory')
    parser.add_argument('--n-results', type=int, default=10,
                       help='Number of search results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize vector store
    store = VectorStore(db_path=args.db)

    if args.command == 'index':
        if not args.chunks:
            print("Error: --chunks required for index command")
            return 1

        # Load chunks
        chunks_data = json.loads(Path(args.chunks).read_text(encoding='utf-8'))
        chunks = chunks_data.get('chunks', [])

        print(f"Indexing {len(chunks)} chunks...")
        store.index_chunks(chunks)

        # Show stats
        stats = store.get_stats()
        print("\nVector Store Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    elif args.command == 'search':
        if not args.query:
            print("Error: --query required for search command")
            return 1

        print(f"Searching for: {args.query}")
        results = store.search(args.query, n_results=args.n_results)

        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['chunk_id']}")
            print(f"   Source: {result['metadata']['source_file']}, Page: {result['metadata']['page']}")
            if result.get('similarity') is not None:
                print(f"   Similarity: {result['similarity']:.3f}")
            print(f"   Text: {result['text'][:150]}...")

    elif args.command == 'get':
        if not args.chunk_id:
            print("Error: --chunk-id required for get command")
            return 1

        chunk = store.get_chunk(args.chunk_id)
        if chunk:
            print(f"Chunk: {chunk['chunk_id']}")
            print(f"Source: {chunk['metadata']['source_file']}")
            print(f"Page: {chunk['metadata']['page']}")
            print(f"\nText:\n{chunk['text']}")
        else:
            print(f"Chunk not found: {args.chunk_id}")

    elif args.command == 'stats':
        stats = store.get_stats()
        print("Vector Store Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    exit(main())
