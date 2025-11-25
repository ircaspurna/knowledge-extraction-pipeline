#!/usr/bin/env python3
"""
Semantic Chunker - TRUE content-aware document chunking

Uses deep semantic understanding:
- Embedding-based topic boundary detection
- Citation context preservation
- Semantic split points (not just structural)
- Cross-chunk semantic coherence tracking
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.path_utils import validate_file_path, validate_directory_path

# Set up logging
logger = logging.getLogger(__name__)

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import DBSCAN
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers or sklearn not installed - semantic features disabled")
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None


@dataclass
class Chunk:
    """Semantically coherent text chunk with metadata"""
    text: str
    chunk_id: str
    source_file: str
    page: int
    chunk_index: int
    section: str
    char_start: int
    char_end: int
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    semantic_coherence: float = 0.0
    has_citations: bool = False
    topic_cluster: int = -1  # Topic cluster ID

    def to_dict(self) -> dict[str, Any]:
        return {
            'text': self.text,
            'chunk_id': self.chunk_id,
            'source_file': self.source_file,
            'page': self.page,
            'chunk_index': self.chunk_index,
            'section': self.section,
            'char_start': self.char_start,
            'char_end': self.char_end,
            'prev_chunk_id': self.prev_chunk_id,
            'next_chunk_id': self.next_chunk_id,
            'semantic_coherence': self.semantic_coherence,
            'has_citations': self.has_citations,
            'topic_cluster': self.topic_cluster
        }


class SemanticChunker:
    """
    Chunk documents using TRUE semantic understanding.

    Strategy:
    1. Split text into sentences for fine-grained analysis
    2. Detect topic boundaries using embedding similarity
    3. Preserve citation context (don't split references)
    4. Merge semantically similar chunks
    5. Split at semantic boundaries (not just structural)
    6. Track cross-chunk coherence
    """

    def __init__(
        self,
        target_chunk_size: int = 600,  # Target words, not strict limit
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        similarity_threshold: float = 0.75,  # For merging similar chunks
        topic_shift_threshold: float = 0.65,  # For detecting topic changes
        embedding_model: str = "all-MiniLM-L6-v2",
        dbscan_eps: float = 0.3,  # DBSCAN neighborhood radius
        dbscan_min_samples: int = 2  # DBSCAN minimum cluster size
    ) -> None:
        """
        Initialize semantic chunker.

        Args:
            target_chunk_size: Preferred chunk size in words
            min_chunk_size: Don't create chunks smaller than this
            max_chunk_size: Split chunks larger than this
            similarity_threshold: Merge adjacent chunks if similarity > threshold
            topic_shift_threshold: Split when similarity drops below this
            embedding_model: Model for semantic similarity
            dbscan_eps: DBSCAN epsilon (neighborhood radius for clustering)
            dbscan_min_samples: DBSCAN minimum samples per cluster
        """
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self.topic_shift_threshold = topic_shift_threshold
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

        # Initialize embedding model
        if EMBEDDINGS_AVAILABLE:
            self.embedder = SentenceTransformer(embedding_model)
            logger.info(f"Semantic chunking enabled with {embedding_model}")
        else:
            self.embedder = None
            logger.warning("Semantic features disabled (sentence-transformers not available)")

    # =========================================================================
    # CITATION DETECTION
    # =========================================================================

    def detect_citations(self, text: str) -> list[tuple[int, int]]:
        """
        Detect citations in text.

        Returns: List of (start_pos, end_pos) for each citation
        """
        citations = []

        # Citation patterns
        patterns = [
            r'\([A-Z][a-z]+\s+et\s+al\.,?\s+\d{4}\)',  # (Smith et al., 2020)
            r'\([A-Z][a-z]+\s+&\s+[A-Z][a-z]+,?\s+\d{4}\)',  # (Smith & Jones, 2020)
            r'\([A-Z][a-z]+,?\s+\d{4}\)',  # (Smith, 2020)
            r'\[\d+\]',  # [1]
            r'\[\d+,\s*\d+\]',  # [1, 2]
            r'\[\d+-\d+\]',  # [1-3]
            r'see\s+(?:Figure|Table|Section|Chapter)\s+\d+',  # see Figure 5
            r'(?:Figure|Table|Section|Chapter)\s+\d+\s+shows',  # Figure 5 shows
            r'as\s+shown\s+in\s+(?:Figure|Table|Section)\s+\d+',  # as shown in Figure 5
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                citations.append((match.start(), match.end()))

        return citations

    def has_citation_context(self, text: str, start: int, end: int) -> bool:
        """
        Check if text segment has citation context that should be preserved.

        Args:
            text: Full text
            start: Start position of potential split
            end: End position of potential split
        """
        # Check 50 chars before and after potential split
        context_before = text[max(0, start-50):start]
        context_after = text[end:min(len(text), end+50)]

        citations_before = self.detect_citations(context_before)
        citations_after = self.detect_citations(context_after)

        # Don't split if there are citations nearby
        return len(citations_before) > 0 or len(citations_after) > 0

    # =========================================================================
    # SENTENCE SPLITTING
    # =========================================================================

    def split_into_sentences(self, text: str) -> list[tuple[str, int]]:
        """
        Split text into sentences with char positions.

        Returns: List of (sentence_text, char_start)
        """
        sentences = []

        # Sentence boundary detection (handles academic text)
        # Use multiple fixed-width negative lookbehinds for common abbreviations
        # Handles: Dr., Mr., Mrs., Ms., Prof., Sr., Jr., Fig., Vol., Ed., No., pp., vs., cf., al.
        sentence_pattern = r'(?<!Dr\.)(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)(?<!Fig\.)(?<!Tab\.)(?<!Vol\.)(?<!Ed\.)(?<!No\.)(?<!pp\.)(?<!vs\.)(?<!cf\.)(?<!al\.)(?<!Inc\.)(?<!Corp\.)(?<!Ltd\.)(?<!i\.e\.)(?<!e\.g\.)(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+'

        current_pos = 0
        for match in re.finditer(sentence_pattern, text):
            sent_end = match.start()
            if sent_end > current_pos:
                sent_text = text[current_pos:sent_end].strip()
                if sent_text:
                    sentences.append((sent_text, current_pos))
            current_pos = match.end()

        # Last sentence
        if current_pos < len(text):
            sent_text = text[current_pos:].strip()
            if sent_text:
                sentences.append((sent_text, current_pos))

        return sentences

    # =========================================================================
    # PARAGRAPH AND SECTION HANDLING
    # =========================================================================

    def split_into_paragraphs(self, text: str) -> list[tuple[str, int]]:
        """
        Split text into paragraphs with char positions.

        Returns: List of (paragraph_text, char_start)
        """
        paragraphs = []
        current_pos = 0

        # Split on double newlines or clear paragraph breaks
        para_pattern = r'\n\s*\n|\n(?=[A-Z])'

        for match in re.finditer(para_pattern, text):
            para_end = match.start()
            if para_end > current_pos:
                para_text = text[current_pos:para_end].strip()
                if para_text:  # Skip empty paragraphs
                    paragraphs.append((para_text, current_pos))
            current_pos = match.end()

        # Last paragraph
        if current_pos < len(text):
            para_text = text[current_pos:].strip()
            if para_text:
                paragraphs.append((para_text, current_pos))

        return paragraphs

    def extract_sections(self, text: str) -> list[tuple[str, str, int]]:
        """
        Extract sections from text.

        Returns: List of (section_title, section_text, char_start)
        """
        sections = []

        # Pattern for section headers (Markdown-style or numbered)
        section_pattern = r'^(#{1,3}\s+.+|(?:[0-9]+\.)+\s*.+|[A-Z][A-Z\s]{3,})\n'

        matches = list(re.finditer(section_pattern, text, re.MULTILINE))

        if not matches:
            # No sections found, treat entire text as one section
            return [("Main Content", text, 0)]

        for i, match in enumerate(matches):
            section_title = match.group(1).strip('# ').strip()
            section_start = match.end()

            # Find next section or end of text
            if i + 1 < len(matches):
                section_end = matches[i + 1].start()
            else:
                section_end = len(text)

            section_text = text[section_start:section_end].strip()
            sections.append((section_title, section_text, match.start()))

        return sections

    # =========================================================================
    # SEMANTIC ANALYSIS
    # =========================================================================

    def word_count(self, text: str) -> int:
        """Count words in text"""
        return len(text.split())

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if not self.embedder:
            return 0.0

        try:
            embeddings = self.embedder.encode([text1, text2])
            # Cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return 0.0

    def detect_topic_boundaries(
        self,
        sentences: list[tuple[str, int]]
    ) -> list[int]:
        """
        Detect topic boundaries using embedding similarity.

        Returns: List of sentence indices where topics change
        """
        if not self.embedder or len(sentences) < 3:
            return []

        # Compute embeddings for all sentences
        texts = [sent[0] for sent in sentences]

        try:
            embeddings = self.embedder.encode(texts)
        except Exception as e:
            logger.warning(f"Failed to compute embeddings: {e}")
            return []

        # Find boundaries where similarity drops
        boundaries = []

        for i in range(1, len(embeddings) - 1):
            # Compare current sentence with previous and next
            sim_prev = np.dot(embeddings[i], embeddings[i-1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1])
            )
            sim_next = np.dot(embeddings[i], embeddings[i+1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
            )

            # If similarity drops below threshold, this is a topic boundary
            if sim_prev < self.topic_shift_threshold or sim_next < self.topic_shift_threshold:
                boundaries.append(i)

        return boundaries

    def cluster_topics(
        self,
        texts: list[str]
    ) -> list[int]:
        """
        Cluster texts by topic using embeddings.

        Returns: List of cluster IDs for each text
        """
        if not self.embedder or len(texts) < 2:
            return [-1] * len(texts)

        try:
            embeddings = self.embedder.encode(texts)

            # Use DBSCAN clustering with configurable parameters
            clustering = DBSCAN(
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
                metric='cosine'
            )
            labels = clustering.fit_predict(embeddings)

            return labels.tolist()
        except Exception as e:
            logger.warning(f"Topic clustering failed: {e}")
            return [-1] * len(texts)

    # =========================================================================
    # SEMANTIC SPLITTING
    # =========================================================================

    def split_at_semantic_boundaries(
        self,
        text: str,
        char_start: int
    ) -> list[tuple[str, int]]:
        """
        Split text at semantic boundaries, not just structural ones.

        Returns: List of (chunk_text, char_start)
        """
        if not self.embedder:
            # Fallback to paragraph-based splitting
            return self.split_into_paragraphs(text)

        # Get sentences
        sentences = self.split_into_sentences(text)

        if len(sentences) <= 1:
            return [(text, char_start)]

        # Detect topic boundaries
        boundaries = self.detect_topic_boundaries(sentences)

        # Build chunks at topic boundaries
        chunks = []
        current_chunk_sentences = []
        current_chunk_start = char_start + sentences[0][1]

        for i, (sent_text, sent_offset) in enumerate(sentences):
            current_chunk_sentences.append(sent_text)

            # Calculate current chunk size
            current_text = ' '.join(current_chunk_sentences)
            current_size = self.word_count(current_text)

            # Check if we should split here
            should_split = False

            # Split at topic boundaries (only if chunk is large enough)
            if current_size >= self.min_chunk_size and i in boundaries:
                should_split = True

            # Force split if chunk exceeds maximum size
            if current_size > self.max_chunk_size:
                should_split = True

            # Check citation context - don't split citations
            if should_split:
                abs_pos = char_start + sent_offset
                if self.has_citation_context(text, abs_pos, abs_pos + len(sent_text)):
                    should_split = False

            if should_split and current_chunk_sentences:
                # Save current chunk
                chunk_text = ' '.join(current_chunk_sentences[:-1]) if len(current_chunk_sentences) > 1 else current_chunk_sentences[0]
                if chunk_text.strip():
                    chunks.append((chunk_text, current_chunk_start))

                # Start new chunk
                current_chunk_sentences = [sent_text]
                current_chunk_start = char_start + sent_offset

        # Add final chunk
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append((chunk_text, current_chunk_start))

        return chunks

    # =========================================================================
    # CHUNK MERGING
    # =========================================================================

    def merge_small_chunks(
        self,
        chunks: list[tuple[str, int, str]]
    ) -> list[tuple[str, int, str]]:
        """
        Merge adjacent small chunks if semantically similar.

        Args:
            chunks: List of (text, char_start, section)

        Returns: Merged chunks
        """
        if not self.embedder or len(chunks) <= 1:
            return chunks

        merged = []
        i = 0

        while i < len(chunks):
            current_text, current_pos, current_section = chunks[i]
            current_words = self.word_count(current_text)

            # Check if current chunk is too small
            if current_words < self.min_chunk_size and i + 1 < len(chunks):
                next_text, next_pos, next_section = chunks[i + 1]

                # Only merge if in same section
                if current_section == next_section:
                    # Check semantic similarity
                    similarity = self.calculate_similarity(current_text, next_text)

                    # Use relaxed threshold for undersized chunks (0.5 vs 0.75)
                    # This prioritizes minimum chunk size over perfect semantic coherence
                    merge_threshold = 0.5 if current_words < self.min_chunk_size else self.similarity_threshold

                    if similarity > merge_threshold:
                        # Merge - but only if it doesn't exceed max_chunk_size
                        merged_text = current_text + "\n\n" + next_text
                        merged_words = self.word_count(merged_text)

                        if merged_words <= self.max_chunk_size:
                            merged.append((merged_text, current_pos, current_section))
                            i += 2  # Skip next chunk
                            continue
                        # If merge would exceed max, keep current chunk as-is

            # Don't merge, keep current chunk
            merged.append((current_text, current_pos, current_section))
            i += 1

        return merged

    # =========================================================================
    # MAIN CHUNKING FUNCTION
    # =========================================================================

    def chunk_document(
        self,
        text: str,
        source_file: str,
        page_mapping: dict[int, tuple[int, int]] | None = None
    ) -> list[Chunk]:
        """
        Chunk document using TRUE semantic understanding.

        Args:
            text: Document text
            source_file: Source filename
            page_mapping: Optional dict of {char_pos: (page_num, page_start, page_end)}

        Returns: List of semantic chunks
        """
        # Validate inputs
        if not text or not text.strip():
            raise ValueError("Cannot chunk empty or whitespace-only text")

        if not source_file or not source_file.strip():
            raise ValueError("source_file cannot be empty")

        # Validate page_mapping structure if provided
        if page_mapping is not None:
            if not isinstance(page_mapping, dict):
                raise ValueError(f"page_mapping must be a dict, got {type(page_mapping).__name__}")

            for key, value in page_mapping.items():
                # Keys should be convertible to int (char positions)
                try:
                    int(key)
                except (ValueError, TypeError):
                    raise ValueError(f"page_mapping key must be numeric, got: {key}")

                # Values should be tuples/lists of length 3
                if not isinstance(value, (tuple, list)):
                    raise ValueError(f"page_mapping value for {key} must be tuple/list, got {type(value).__name__}")
                if len(value) != 3:
                    raise ValueError(f"page_mapping value for {key} must have 3 elements (page_num, start, end), got {len(value)}")

                # Elements should be numeric
                try:
                    int(value[0])  # page_num
                    int(value[1])  # page_start
                    int(value[2])  # page_end
                except (ValueError, TypeError):
                    raise ValueError(f"page_mapping value for {key} must contain numeric elements: {value}")

        # Extract sections
        sections = self.extract_sections(text)

        # Split each section semantically
        all_chunks = []

        for section_title, section_text, section_start in sections:
            # Use semantic splitting if available
            if self.embedder:
                section_chunks = self.split_at_semantic_boundaries(section_text, section_start)
            else:
                # Fallback to paragraph-based
                section_chunks = self.split_into_paragraphs(section_text)

            # Add section info
            for chunk_text, chunk_start in section_chunks:
                all_chunks.append((chunk_text, section_start + chunk_start, section_title))

        # Merge small chunks if semantically similar
        all_chunks = self.merge_small_chunks(all_chunks)

        # Detect citations
        chunk_has_citations = []
        for chunk_text, _, _ in all_chunks:
            citations = self.detect_citations(chunk_text)
            chunk_has_citations.append(len(citations) > 0)

        # Cluster topics
        chunk_texts = [chunk[0] for chunk in all_chunks]
        topic_clusters = self.cluster_topics(chunk_texts)

        # Create Chunk objects with metadata
        chunk_objects = []

        for i, ((chunk_text, char_start, section), has_cites, topic_id) in enumerate(
            zip(all_chunks, chunk_has_citations, topic_clusters)
        ):
            # Determine page number
            page = 1  # Default
            if page_mapping:
                # Convert keys to int in case they were loaded from JSON as strings
                # Find the LAST position <= char_start (don't break early)
                for pos, page_info in sorted((int(k), v) for k, v in page_mapping.items()):
                    page_num = page_info[0] if isinstance(page_info, (tuple, list)) else page_info
                    if char_start >= pos:
                        page = page_num
                    # No break - continue to find the last valid position

            # Create chunk ID
            chunk_id = f"{Path(source_file).stem}_p{page}_c{i}"

            # Calculate semantic coherence (similarity with previous chunk)
            coherence = 0.0
            if self.embedder and i > 0:
                prev_text = all_chunks[i-1][0]
                coherence = self.calculate_similarity(chunk_text, prev_text)

            chunk = Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                source_file=source_file,
                page=page,
                chunk_index=i,
                section=section,
                char_start=char_start,
                char_end=char_start + len(chunk_text),
                prev_chunk_id=f"{Path(source_file).stem}_p{page}_c{i-1}" if i > 0 else None,
                next_chunk_id=f"{Path(source_file).stem}_p{page}_c{i+1}" if i < len(all_chunks) - 1 else None,
                semantic_coherence=coherence,
                has_citations=has_cites,
                topic_cluster=topic_id
            )

            chunk_objects.append(chunk)

        return chunk_objects

    def get_stats(self, chunks: list[Chunk]) -> dict[str, Any]:
        """Get chunking statistics"""
        if not chunks:
            return {}

        word_counts = [self.word_count(c.text) for c in chunks]
        coherences = [c.semantic_coherence for c in chunks]

        return {
            'num_chunks': len(chunks),
            'avg_words_per_chunk': sum(word_counts) / len(word_counts),
            'min_words': min(word_counts),
            'max_words': max(word_counts),
            'avg_semantic_coherence': sum(coherences) / len(coherences) if coherences else 0.0,
            'sections': len(set(c.section for c in chunks)),
            'chunks_with_citations': sum(1 for c in chunks if c.has_citations),
            'num_topics': len(set(c.topic_cluster for c in chunks if c.topic_cluster != -1))
        }


# =========================================================================
# CLI
# =========================================================================

def main() -> int:
    """Command-line interface"""
    import argparse
    import json

    # Configure logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    parser = argparse.ArgumentParser(
        description='Semantically chunk documents (TRUE semantic understanding)'
    )
    parser.add_argument('input', type=str, help='Input text file')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--target-size', type=int, default=600,
                       help='Target chunk size in words')
    parser.add_argument('--min-size', type=int, default=100,
                       help='Minimum chunk size')
    parser.add_argument('--max-size', type=int, default=1000,
                       help='Maximum chunk size')
    parser.add_argument('--stats', action='store_true',
                       help='Print chunking statistics')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate and read input (security check)
    try:
        input_path = validate_file_path(
            args.input,
            allowed_extensions=['.txt', '.md', '.json'],
            must_exist=True
        )
        text = input_path.read_text(encoding='utf-8')
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return 1

    # Create chunker
    chunker = SemanticChunker(
        target_chunk_size=args.target_size,
        min_chunk_size=args.min_size,
        max_chunk_size=args.max_size
    )

    # Chunk document
    print(f"Chunking: {input_path.name}")
    chunks = chunker.chunk_document(text, input_path.name)

    # Print stats
    if args.stats:
        stats = chunker.get_stats(chunks)
        print("\nChunking Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    # Save output
    if args.output:
        try:
            output_path = Path(args.output)
            output_dir = validate_directory_path(output_path.parent, create=True)
            output_path = output_dir / output_path.name

            output_data = {
                'source_file': input_path.name,
                'num_chunks': len(chunks),
                'chunks': [c.to_dict() for c in chunks]
            }
            output_path.write_text(json.dumps(output_data, indent=2), encoding='utf-8')
            print(f"\n✓ Saved {len(chunks)} chunks to: {output_path}")
        except (ValueError, OSError) as e:
            print(f"Error saving output: {e}")
            return 1
    else:
        # Print sample chunks
        for i, chunk in enumerate(chunks[:5], 1):
            print(f"\n--- Chunk {i} ---")
            print(f"Section: {chunk.section}")
            print(f"Words: {chunker.word_count(chunk.text)}")
            print(f"Page: {chunk.page}")
            print(f"Coherence: {chunk.semantic_coherence:.2f}")
            print(f"Has citations: {chunk.has_citations}")
            print(f"Topic cluster: {chunk.topic_cluster}")
            print(f"\n{chunk.text[:200]}...")

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return 0


if __name__ == "__main__":
    exit(main())
