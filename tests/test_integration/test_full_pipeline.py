"""Integration test for full pipeline."""

from pathlib import Path
from unittest.mock import Mock, patch


class TestFullPipelineIntegration:
    """Test full pipeline integration."""

    @patch('knowledge_extraction.core.document_processor.PdfReader')
    @patch('knowledge_extraction.core.vector_store.chromadb')
    def test_document_to_vector_store(
        self,
        mock_chromadb: Mock,
        mock_pdf_reader: Mock,
        tmp_path: Path
    ) -> None:
        """Test pipeline from document processing to vector indexing."""
        from knowledge_extraction.core import DocumentProcessor, SemanticChunker, VectorStore

        # Mock PDF
        mock_page = Mock()
        mock_page.extract_text.return_value = "This is test content. " * 50
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {'/Title': 'Test PDF'}
        mock_pdf_reader.return_value = mock_pdf

        # Mock ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Create test PDF file
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"PDF content")

        # Step 1: Process document
        processor = DocumentProcessor()
        document = processor.process_pdf(test_pdf)

        assert document is not None
        assert 'text' in document
        assert len(document['text']) > 0

        # Step 2: Chunk document
        chunker = SemanticChunker(target_chunk_size=100)
        chunks = chunker.chunk_document(
            text=document['text'],
            source_file=test_pdf.name
        )

        assert len(chunks) > 0

        # Step 3: Index chunks
        db_path = tmp_path / "chroma_db"
        vector_store = VectorStore(db_path=str(db_path))
        vector_store.index_chunks([c.to_dict() for c in chunks])

        # Verify indexing was called
        mock_collection.add.assert_called_once()

    def test_concept_extraction_to_graph_integration(self) -> None:
        """Test pipeline from concepts to knowledge graph."""
        from knowledge_extraction.core import GraphBuilder
        from knowledge_extraction.extraction import EntityResolverMCP

        # Create test concepts
        concepts = [
            {
                'term': 'Machine Learning',
                'definition': 'AI subset',
                'category': 'method',
                'source_file': 'test.pdf',
                'importance': 'high',
                'justification': 'Core concept',
                'quote': 'ML is important',
                'chunk_id': 'chunk_1',
                'page': 1
            },
            {
                'term': 'Deep Learning',
                'definition': 'Neural networks',
                'category': 'method',
                'source_file': 'test.pdf',
                'importance': 'high',
                'justification': 'Advanced technique',
                'quote': 'DL uses neural networks',
                'chunk_id': 'chunk_2',
                'page': 2
            },
            {
                'term': 'Machine Learning',  # Duplicate
                'definition': 'Learning from data',
                'category': 'method',
                'source_file': 'test2.pdf',
                'importance': 'critical',
                'justification': 'Fundamental',
                'quote': 'ML learns from data',
                'chunk_id': 'chunk_3',
                'page': 1
            }
        ]

        # Step 1: Resolve entities
        resolver = EntityResolverMCP()
        entities, ambiguous_pairs = resolver.resolve_entities_automatic(concepts)

        # Should merge duplicate ML
        assert len(entities) == 2

        # Step 2: Build graph
        builder = GraphBuilder()
        graph = builder.build_graph(
            [e.to_dict() for e in entities],
            []  # No relationships for this test
        )

        assert graph is not None
        assert len(graph.nodes()) == 2

        # Step 3: Get stats
        stats = builder.get_stats()
        assert stats['nodes'] == 2
        assert 'density' in stats


class TestBatchProcessingIntegration:
    """Test batch processing integration."""

    @patch('knowledge_extraction.core.document_processor.PdfReader')
    def test_batch_document_processing(
        self,
        mock_pdf_reader: Mock,
        tmp_path: Path
    ) -> None:
        """Test processing multiple documents."""
        from knowledge_extraction.core import DocumentProcessor, SemanticChunker

        # Mock PDF
        mock_page = Mock()
        mock_page.extract_text.return_value = "Document content. " * 30
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {'/Title': 'Test'}
        mock_pdf_reader.return_value = mock_pdf

        # Create test PDFs
        pdf1 = tmp_path / "doc1.pdf"
        pdf2 = tmp_path / "doc2.pdf"
        pdf1.write_bytes(b"PDF1")
        pdf2.write_bytes(b"PDF2")

        processor = DocumentProcessor()
        chunker = SemanticChunker()

        all_chunks = []

        # Process both documents
        for pdf_file in [pdf1, pdf2]:
            document = processor.process_pdf(pdf_file)
            assert document is not None

            chunks = chunker.chunk_document(
                text=document['text'],
                source_file=pdf_file.name
            )
            all_chunks.extend(chunks)

        # Should have chunks from both documents
        assert len(all_chunks) > 0

        # Verify chunks have different source files
        source_files = {chunk.source_file for chunk in all_chunks}
        assert len(source_files) == 2


class TestEndToEndWorkflow:
    """Test end-to-end workflow (simplified)."""

    @patch('knowledge_extraction.core.document_processor.PdfReader')
    @patch('knowledge_extraction.core.vector_store.chromadb')
    def test_minimal_end_to_end(
        self,
        mock_chromadb: Mock,
        mock_pdf_reader: Mock,
        tmp_path: Path
    ) -> None:
        """Test minimal end-to-end workflow."""
        from knowledge_extraction.core import (
            DocumentProcessor,
            GraphBuilder,
            SemanticChunker,
            VectorStore,
        )
        from knowledge_extraction.extraction import EntityResolverMCP

        # Setup mocks
        mock_page = Mock()
        mock_page.extract_text.return_value = "Test content about machine learning and neural networks. " * 20
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {'/Title': 'Test'}
        mock_pdf_reader.return_value = mock_pdf

        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Create test file
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"PDF")

        # Phase 1: Document Processing
        processor = DocumentProcessor()
        document = processor.process_pdf(test_pdf)
        assert document is not None

        # Phase 2: Chunking
        chunker = SemanticChunker(target_chunk_size=50)
        chunks = chunker.chunk_document(
            text=document['text'],
            source_file=test_pdf.name
        )
        assert len(chunks) > 0

        # Phase 3: Vector Indexing
        db_path = tmp_path / "db"
        vector_store = VectorStore(db_path=str(db_path))
        vector_store.index_chunks([c.to_dict() for c in chunks])

        # Phase 4: Concept Extraction (mocked - normally done by Claude)
        # Simulate extracted concepts
        concepts = [
            {
                'term': 'Machine Learning',
                'definition': 'Learning from data',
                'category': 'method',
                'source_file': test_pdf.name,
                'importance': 'high',
                'justification': 'Core topic',
                'quote': 'machine learning',
                'chunk_id': chunks[0].chunk_id,
                'page': 1
            }
        ]

        # Phase 5: Entity Resolution
        resolver = EntityResolverMCP()
        entities, _ = resolver.resolve_entities_automatic(concepts)
        assert len(entities) > 0

        # Phase 6: Graph Building
        builder = GraphBuilder()
        graph = builder.build_graph([e.to_dict() for e in entities], [])
        assert graph is not None

        # Verify we completed all phases
        stats = builder.get_stats()
        assert stats['nodes'] > 0
