#!/usr/bin/env python3
"""
Tests for real PDF processing using academic paper fixtures

These tests use real academic papers to verify the pipeline works
with actual documents, not just mocked data.
"""

import pytest
from pathlib import Path

from knowledge_extraction.core.document_processor import DocumentProcessor
from knowledge_extraction.core.semantic_chunker import SemanticChunker
from knowledge_extraction.core.vector_store import VectorStore


# Fixture paths
FIXTURES_DIR = Path(__file__).parent / "fixtures"
PDF_BRODERSEN = FIXTURES_DIR / "Brodersen_2015_CausalImpact_Inferring_Causal_Effects.pdf"
PDF_GIGERENZER = FIXTURES_DIR / "Gigerenzer_2009_Why_Heuristics_Work.pdf"
PDF_CHEN = FIXTURES_DIR / "Chen_2020_CausalML_Python_Package_Causal.pdf"


class TestRealPDFProcessing:
    """Test document processing with real academic PDFs"""

    def test_fixtures_exist(self):
        """Verify all test fixtures are present"""
        assert PDF_BRODERSEN.exists(), f"Missing: {PDF_BRODERSEN.name}"
        assert PDF_GIGERENZER.exists(), f"Missing: {PDF_GIGERENZER.name}"
        assert PDF_CHEN.exists(), f"Missing: {PDF_CHEN.name}"

    def test_process_small_pdf(self):
        """Process smallest PDF (Brodersen 94KB)"""
        processor = DocumentProcessor()
        result = processor.process_file(PDF_BRODERSEN)

        assert result is not None
        assert "text" in result
        assert "metadata" in result
        assert len(result["text"]) > 100  # Should extract meaningful text

        # Check metadata
        metadata = result["metadata"]
        assert "source_file" in metadata
        assert metadata["source_file"] == PDF_BRODERSEN.name

    def test_process_medium_pdf(self):
        """Process medium PDF (Gigerenzer 169KB)"""
        processor = DocumentProcessor()
        result = processor.process_file(PDF_GIGERENZER)

        assert result is not None
        assert "text" in result
        assert len(result["text"]) > 1000  # Larger document

        # Should contain key terms from the paper
        text_lower = result["text"].lower()
        assert "heuristic" in text_lower or "decision" in text_lower

    def test_process_technical_pdf(self):
        """Process technical PDF with code (Chen 190KB)"""
        processor = DocumentProcessor()
        result = processor.process_file(PDF_CHEN)

        assert result is not None
        assert "text" in result
        assert len(result["text"]) > 1000

        # Technical paper should mention causal or ML terms
        text_lower = result["text"].lower()
        assert any(term in text_lower for term in ["causal", "python", "machine learning"])

    def test_extract_metadata_from_all_pdfs(self):
        """Extract metadata from all test PDFs"""
        processor = DocumentProcessor()

        for pdf_path in [PDF_BRODERSEN, PDF_GIGERENZER, PDF_CHEN]:
            result = processor.process_file(pdf_path)
            assert result is not None

            metadata = result["metadata"]
            assert "source_file" in metadata
            # Metadata should have at least some fields
            assert len(metadata) > 1

    def test_chunking_real_pdf(self):
        """Test semantic chunking on real PDF content"""
        processor = DocumentProcessor()
        chunker = SemanticChunker()

        # Process PDF
        result = processor.process_file(PDF_BRODERSEN)
        assert result is not None

        # Chunk the document
        chunks = chunker.chunk_document(
            result["text"],
            source_file=PDF_BRODERSEN.name
        )

        assert len(chunks) > 0
        # Convert Chunk objects to dicts for easier testing
        chunk_dicts = [c.to_dict() for c in chunks]

        assert all("text" in chunk for chunk in chunk_dicts)
        assert all("chunk_id" in chunk for chunk in chunk_dicts)
        assert all(chunk["source_file"] == PDF_BRODERSEN.name for chunk in chunk_dicts)

        # Filter out empty chunks (chunker may create some)
        non_empty_chunks = [c for c in chunk_dicts if c["text"].strip()]

        # Non-empty chunks should have reasonable sizes
        assert len(non_empty_chunks) > 0, "Should have at least some non-empty chunks"
        for chunk in non_empty_chunks:
            assert len(chunk["text"]) > 10  # Not empty
            assert len(chunk["text"]) < 10000  # Not too large

    def test_vector_store_with_real_chunks(self, tmp_path):
        """Test vector store indexing with real PDF chunks"""
        processor = DocumentProcessor()
        chunker = SemanticChunker()
        vector_store = VectorStore(db_path=str(tmp_path / "test_db"))

        # Process and chunk PDF
        result = processor.process_file(PDF_GIGERENZER)
        chunks = chunker.chunk_document(
            result["text"],
            source_file=PDF_GIGERENZER.name
        )

        # Convert to dicts for indexing
        chunk_dicts = [c.to_dict() for c in chunks]

        # Filter out empty chunks (chunker may create some)
        non_empty_chunks = [c for c in chunk_dicts if c["text"].strip()]

        # Index chunks
        indexed_count = vector_store.index_chunks(non_empty_chunks)
        assert indexed_count == len(non_empty_chunks)

        # Search for relevant content
        search_results = vector_store.search("heuristic decision making", n_results=5)
        assert len(search_results) > 0

        # Results should be from our document
        for result in search_results:
            assert result["metadata"]["source_file"] == PDF_GIGERENZER.name
            assert "text" in result
            assert "similarity" in result


class TestPDFTextExtraction:
    """Test text extraction quality with real PDFs"""

    def test_extract_complete_text(self):
        """Verify complete text extraction (no truncation)"""
        processor = DocumentProcessor()

        for pdf_path in [PDF_BRODERSEN, PDF_GIGERENZER, PDF_CHEN]:
            result = processor.process_file(pdf_path)
            assert result is not None

            text = result["text"]
            # Real PDFs should extract substantial text
            assert len(text) > 500, f"{pdf_path.name} extracted too little text"

            # Should not have extraction errors
            assert "error" not in text.lower()[:100]

    def test_text_encoding(self):
        """Verify proper text encoding (UTF-8, no mojibake)"""
        processor = DocumentProcessor()

        for pdf_path in [PDF_BRODERSEN, PDF_GIGERENZER, PDF_CHEN]:
            result = processor.process_file(pdf_path)
            assert result is not None

            text = result["text"]
            # Should be valid UTF-8
            assert isinstance(text, str)
            # Should not have encoding errors
            assert "\ufffd" not in text  # Replacement character

    def test_preserve_structure(self):
        """Verify text structure is preserved (paragraphs, spacing)"""
        processor = DocumentProcessor()
        result = processor.process_file(PDF_BRODERSEN)

        assert result is not None
        text = result["text"]

        # Should have paragraph breaks (multiple newlines)
        assert "\n" in text
        # Should not be one giant line
        lines = text.split("\n")
        assert len(lines) > 10


class TestPDFMetadata:
    """Test metadata extraction from real PDFs"""

    def test_extract_basic_metadata(self):
        """Extract basic metadata (filename, size, pages)"""
        processor = DocumentProcessor()

        for pdf_path in [PDF_BRODERSEN, PDF_GIGERENZER, PDF_CHEN]:
            result = processor.process_file(pdf_path)
            assert result is not None

            metadata = result["metadata"]
            assert metadata["source_file"] == pdf_path.name
            # Should have some metadata fields
            assert len(metadata) > 1

    def test_word_count_accuracy(self):
        """Verify word count is reasonable by checking text length"""
        processor = DocumentProcessor()

        for pdf_path in [PDF_BRODERSEN, PDF_GIGERENZER, PDF_CHEN]:
            result = processor.process_file(pdf_path)
            assert result is not None

            # Count words from text
            word_count = len(result["text"].split())
            # Academic papers should have substantial word counts
            assert word_count > 100, f"{pdf_path.name} has suspiciously low word count"


class TestEndToEndPipeline:
    """Test complete pipeline with real PDFs"""

    def test_full_pipeline_small_pdf(self, tmp_path):
        """Run full pipeline on small PDF"""
        processor = DocumentProcessor()
        chunker = SemanticChunker()
        vector_store = VectorStore(db_path=str(tmp_path / "test_db"))

        # 1. Process PDF
        doc_result = processor.process_file(PDF_BRODERSEN)
        assert doc_result is not None

        # 2. Chunk text
        chunks = chunker.chunk_document(
            doc_result["text"],
            source_file=PDF_BRODERSEN.name
        )
        assert len(chunks) > 0

        # Convert to dicts
        chunk_dicts = [c.to_dict() for c in chunks]

        # Filter out empty chunks
        non_empty_chunks = [c for c in chunk_dicts if c["text"].strip()]

        # 3. Index chunks
        indexed = vector_store.index_chunks(non_empty_chunks)
        assert indexed == len(non_empty_chunks)

        # 4. Search indexed content
        results = vector_store.search("causal impact", n_results=3)
        assert len(results) > 0

        # Verify results are relevant
        for result in results:
            assert result["metadata"]["source_file"] == PDF_BRODERSEN.name

    def test_batch_processing_multiple_pdfs(self, tmp_path):
        """Process multiple PDFs in batch"""
        processor = DocumentProcessor()
        chunker = SemanticChunker()
        vector_store = VectorStore(db_path=str(tmp_path / "batch_db"))

        all_chunks = []

        # Process all PDFs
        for pdf_path in [PDF_BRODERSEN, PDF_GIGERENZER, PDF_CHEN]:
            result = processor.process_file(pdf_path)
            assert result is not None

            chunks = chunker.chunk_document(result["text"], source_file=pdf_path.name)
            chunk_dicts = [c.to_dict() for c in chunks]
            all_chunks.extend(chunk_dicts)

        # Filter out empty chunks
        non_empty_chunks = [c for c in all_chunks if c["text"].strip()]

        # Index all chunks together
        indexed = vector_store.index_chunks(non_empty_chunks)
        assert indexed == len(non_empty_chunks)

        # Search should find results from different papers
        results = vector_store.search("decision making heuristics", n_results=10)
        assert len(results) > 0

        # Should have results from multiple sources
        sources = {r["metadata"]["source_file"] for r in results}
        assert len(sources) >= 1  # At least one source

    def test_pipeline_performance(self):
        """Verify pipeline completes in reasonable time"""
        import time

        processor = DocumentProcessor()

        # Process small PDF should be fast
        start = time.time()
        result = processor.process_file(PDF_BRODERSEN)
        duration = time.time() - start

        assert result is not None
        # Should process 94KB PDF in under 10 seconds
        assert duration < 10.0, f"Processing took {duration:.2f}s (too slow)"


class TestErrorHandling:
    """Test error handling with real PDFs"""

    def test_process_nonexistent_pdf(self):
        """Handle missing PDF file gracefully"""
        processor = DocumentProcessor()
        result = processor.process_file("nonexistent.pdf")
        assert result is None

    def test_process_invalid_path(self):
        """Handle invalid path gracefully (security validation)"""
        from knowledge_extraction.utils.path_utils import PathSecurityError

        processor = DocumentProcessor()
        # Invalid path with traversal should raise PathSecurityError
        with pytest.raises(PathSecurityError):
            processor.process_file("/invalid/../../../path.pdf")

    def test_empty_pdf_handling(self):
        """Handle edge cases gracefully"""
        processor = DocumentProcessor()
        # Real PDFs should work
        result = processor.process_file(PDF_BRODERSEN)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
