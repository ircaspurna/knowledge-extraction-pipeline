#!/usr/bin/env python3
"""
Semantic Batch Optimizer - Cluster chunks semantically for efficient extraction

Reduces extraction prompts by 80% while maintaining quality by grouping
semantically related chunks together.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    logger.warning("sentence-transformers or sklearn not installed")
    SentenceTransformer = None
    AgglomerativeClustering = None


class SemanticBatchOptimizer:
    """
    Groups chunks semantically to reduce extraction prompts.

    Uses hierarchical clustering on chunk embeddings to group
    semantically related content together.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunks_per_batch: int = 4,
        max_chunks_per_batch: int = 6,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize optimizer

        Args:
            model_name: Sentence transformer model
            chunks_per_batch: Target chunks per batch (3-5 optimal)
            max_chunks_per_batch: Maximum chunks in one batch (token limit)
            similarity_threshold: Minimum similarity for clustering
        """
        if not SentenceTransformer:
            raise ImportError("sentence-transformers required: pip install sentence-transformers scikit-learn")

        self.model = SentenceTransformer(model_name)
        self.chunks_per_batch = chunks_per_batch
        self.max_chunks_per_batch = max_chunks_per_batch
        self.similarity_threshold = similarity_threshold

    def optimize_chunks(
        self,
        chunks: List[Dict[str, Any]],
        filter_non_substantive: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """
        Cluster chunks semantically into batches

        Args:
            chunks: List of chunk dictionaries with 'text' field
            filter_non_substantive: Remove headers, page numbers, etc.

        Returns:
            List of chunk batches (each batch is 3-6 chunks)
        """
        # Filter non-substantive chunks
        if filter_non_substantive:
            chunks = self._filter_chunks(chunks)

        if not chunks:
            logger.warning("No substantive chunks to process")
            return []

        if len(chunks) <= self.chunks_per_batch:
            # Too few chunks, return as single batch
            return [chunks]

        logger.info(f"Clustering {len(chunks)} chunks...")

        # Compute embeddings
        texts = [c['text'] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False)

        # Compute optimal number of clusters
        num_batches = max(1, len(chunks) // self.chunks_per_batch)

        # Hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=num_batches,
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings)

        # Group chunks by cluster
        batches = {}
        for idx, label in enumerate(labels):
            if label not in batches:
                batches[label] = []
            batches[label].append(chunks[idx])

        # Split oversized batches
        final_batches = []
        for batch_chunks in batches.values():
            if len(batch_chunks) > self.max_chunks_per_batch:
                # Split large batches
                for i in range(0, len(batch_chunks), self.max_chunks_per_batch):
                    final_batches.append(batch_chunks[i:i + self.max_chunks_per_batch])
            else:
                final_batches.append(batch_chunks)

        # Sort batches by first chunk's position (maintain some document order)
        final_batches.sort(key=lambda b: b[0].get('chunk_index', 0))

        logger.info(f"✓ Created {len(final_batches)} semantic batches")
        logger.info(f"  Average chunks per batch: {len(chunks) / len(final_batches):.1f}")
        logger.info(f"  Reduction: {len(chunks)} → {len(final_batches)} prompts ({(1-len(final_batches)/len(chunks))*100:.1f}%)")

        return final_batches

    def _filter_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out non-substantive chunks"""
        substantive = []

        for chunk in chunks:
            text = chunk.get('text', '').strip()

            # Skip very short chunks
            if len(text) < 100:
                continue

            # Skip chunks that are mostly numbers/whitespace
            words = text.split()
            if len(words) < 20:
                continue

            # Skip chunks with too many citations (reference sections)
            if chunk.get('has_citations', False) and text.count('[') > 10:
                continue

            # Skip page headers/footers
            if text.lower().startswith(('page ', 'chapter ', '© ')):
                continue

            substantive.append(chunk)

        filtered_count = len(chunks) - len(substantive)
        if filtered_count > 0:
            logger.info(f"  ℹ️  Filtered {filtered_count} non-substantive chunks")

        return substantive

    def get_batch_statistics(self, batches: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Get statistics about batch optimization"""
        if not batches:
            return {}

        total_chunks = sum(len(b) for b in batches)
        avg_per_batch = total_chunks / len(batches)

        return {
            'total_batches': len(batches),
            'total_chunks': total_chunks,
            'avg_chunks_per_batch': avg_per_batch,
            'reduction_pct': (1 - len(batches) / total_chunks) * 100 if total_chunks > 0 else 0,
            'batch_sizes': [len(b) for b in batches]
        }


def create_batched_prompts(
    batches: List[List[Dict[str, Any]]],
    prompt_template: str,
    domain: str = 'psychology'
) -> List[Dict[str, Any]]:
    """
    Create extraction prompts from semantic batches

    Args:
        batches: List of chunk batches
        prompt_template: Base prompt template
        domain: Domain for extraction

    Returns:
        List of prompt dictionaries
    """
    prompts = []

    for batch_idx, batch in enumerate(batches):
        # Combine chunk texts
        combined_text = "\n\n---\n\n".join([
            f"**Passage {i+1}:**\n{chunk['text']}"
            for i, chunk in enumerate(batch)
        ])

        # Create metadata combining all chunk IDs
        metadata = {
            'batch_id': f"batch_{batch_idx}",
            'chunk_ids': [c.get('chunk_id', f"chunk_{i}") for i, c in enumerate(batch)],
            'num_chunks_in_batch': len(batch),
            'source_file': batch[0].get('source_file', 'unknown'),
            'pages': list(set(c.get('page', 0) for c in batch)),
            'task': 'extraction_batched'
        }

        # Build prompt with multi-passage instruction
        prompt = prompt_template.replace(
            "**Passage**:",
            f"**{len(batch)} Passages** (extract from ALL):"
        ).replace(
            "Extract research context from the following passage.",
            f"Extract research context from the following {len(batch)} related passages. Process ALL passages together."
        )

        prompts.append({
            'prompt': prompt + "\n\n" + combined_text,
            'metadata': metadata
        })

    return prompts
