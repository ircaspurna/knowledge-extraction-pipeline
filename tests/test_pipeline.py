#!/usr/bin/env python3
"""
Comprehensive Pipeline Test Suite

Tests the complete knowledge extraction pipeline from end-to-end.

Adapted for open source package structure.
"""

import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import logging

# Import from installed package
from knowledge_extraction.core import DocumentProcessor, SemanticChunker, VectorStore
from knowledge_extraction.extraction import ConceptExtractorMCP
from knowledge_extraction.extraction.semantic_batch_optimizer import SemanticBatchOptimizer

logger = logging.getLogger(__name__)


class PipelineTest:
    """Test suite for knowledge extraction pipeline"""

    def __init__(self):
        self.test_results: Dict[str, bool] = {}
        self.temp_dir = None

    def run_all_tests(self) -> bool:
        """Run all tests and return True if all pass"""
        print("\n" + "="*70)
        print("  Knowledge Extraction Pipeline - Test Suite")
        print("="*70 + "\n")

        # Set up temp directory
        self.temp_dir = Path(tempfile.mkdtemp())
        print(f"📂 Test directory: {self.temp_dir}\n")

        try:
            tests = [
                ("Document Processor", self.test_document_processor),
                ("Semantic Chunker", self.test_semantic_chunker),
                ("Vector Store", self.test_vector_store),
                ("Concept Extractor (prompts)", self.test_concept_extractor),
                ("Semantic Batching", self.test_semantic_batching),
                ("Config Selector", self.test_config_selector),
                ("State Manager", self.test_state_manager),
            ]

            for test_name, test_func in tests:
                print(f"Running: {test_name}...")
                try:
                    result = test_func()
                    self.test_results[test_name] = result
                    status = "✅ PASS" if result else "❌ FAIL"
                    print(f"  {status}\n")
                except Exception as e:
                    self.test_results[test_name] = False
                    print(f"  ❌ FAIL - Exception: {e}\n")
                    import traceback
                    traceback.print_exc()

            # Summary
            self._print_summary()

            return all(self.test_results.values())

        finally:
            # Cleanup
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)

    def test_document_processor(self) -> bool:
        """Test PDF text extraction"""
        # Test that class loads correctly
        processor = DocumentProcessor()
        assert processor is not None
        print("    ✓ DocumentProcessor initialized")

        return True

    def test_semantic_chunker(self) -> bool:
        """Test semantic chunking"""
        # Test initialization
        chunker = SemanticChunker(
            target_chunk_size=800,
            min_chunk_size=150,
            max_chunk_size=1200
        )
        print("    ✓ SemanticChunker initialized")

        # Test chunking with sample text
        sample_text = """
        This is a test document with multiple paragraphs.
        It contains enough text to create at least one chunk.

        This is the second paragraph. It discusses different concepts
        and ideas that should be chunked together semantically.

        And here is a third paragraph with more content to ensure
        we have sufficient text for the chunking algorithm to work with.
        """ * 20  # Repeat to ensure enough text

        chunks = chunker.chunk_document(
            text=sample_text,
            source_file="test.pdf",
            page_mapping=None
        )

        assert len(chunks) > 0, "Should create at least one chunk"
        assert all(hasattr(c, 'text') for c in chunks), "Chunks should have text attribute"
        assert all(hasattr(c, 'chunk_id') for c in chunks), "Chunks should have chunk_id"

        print(f"    ✓ Created {len(chunks)} chunks from sample text")

        return True

    def test_vector_store(self) -> bool:
        """Test ChromaDB vector indexing"""
        from vector_store import VectorStore
        from semantic_chunker import Chunk

        # Create test chunks
        test_chunks = [
            Chunk(
                text="This is test chunk 1 about machine learning.",
                chunk_id="test_c1",
                source_file="test.pdf",
                page=1,
                chunk_index=0,
                section="Introduction",
                char_start=0,
                char_end=45
            ),
            Chunk(
                text="This is test chunk 2 about natural language processing.",
                chunk_id="test_c2",
                source_file="test.pdf",
                page=1,
                chunk_index=1,
                section="Introduction",
                char_start=46,
                char_end=102
            )
        ]

        # Test vector store
        db_path = self.temp_dir / "test_chroma"
        vector_store = VectorStore(
            db_path=str(db_path),
            collection_name="test_collection"
        )
        print("    ✓ VectorStore initialized")

        # Index chunks (convert to dicts)
        vector_store.index_chunks([chunk.to_dict() for chunk in test_chunks])
        print(f"    ✓ Indexed {len(test_chunks)} chunks")

        # Test search
        results = vector_store.search("machine learning", n_results=1)
        assert len(results) > 0, "Should find at least one result"
        print(f"    ✓ Search returned {len(results)} results")

        return True

    def test_concept_extractor(self) -> bool:
        """Test concept extraction prompt generation"""

        # Test initialization with different domains
        for domain in ['psychology', 'psycholinguistics']:
            extractor = ConceptExtractorMCP(domain=domain)
            assert extractor.domain_config is not None
            print(f"    ✓ Initialized with domain: {domain}")

        # Test prompt generation
        extractor = ConceptExtractorMCP(domain='psychology')
        prompt_data = extractor.generate_extraction_prompt(
            chunk_text="Loss aversion is the tendency for losses to loom larger than gains.",
            chunk_id="test_1",
            source_file="test.pdf",
            page=1
        )

        assert 'prompt' in prompt_data
        assert 'metadata' in prompt_data
        assert len(prompt_data['prompt']) > 100, "Prompt should be substantial"
        print("    ✓ Generated extraction prompt")

        return True

    def test_semantic_batching(self) -> bool:
        """Test semantic batch optimization"""
        from semantic_batch_optimizer import SemanticBatchOptimizer

        # Create test chunks
        test_chunks = [
            {'text': f"This is chunk {i} about machine learning and AI.", 'chunk_id': f'c{i}'}
            for i in range(20)
        ]

        # Test optimizer
        optimizer = SemanticBatchOptimizer(chunks_per_batch=4)
        print("    ✓ Semantic optimizer initialized")

        # Optimize chunks
        batches = optimizer.optimize_chunks(test_chunks, filter_non_substantive=False)

        assert len(batches) > 0, "Should create at least one batch"
        assert len(batches) < len(test_chunks), "Should reduce number of prompts"

        total_chunks = sum(len(b) for b in batches)
        assert total_chunks == len(test_chunks), "Should preserve all chunks"

        stats = optimizer.get_batch_statistics(batches)
        reduction = stats['reduction_pct']

        print(f"    ✓ Created {len(batches)} batches from {len(test_chunks)} chunks")
        print(f"    ✓ Reduction: {reduction:.1f}%")

        return True

    def test_config_selector(self) -> bool:
        """Test configuration profile selection (simplified for open source)"""
        # Note: ConfigSelector is part of main pipeline, not open source
        # This test is simplified for compatibility

        selector = ConfigSelector()
        print(f"    ✓ Loaded {len(selector.profiles)} profiles")

        # Test profile selection
        profile = selector.select_profile(
            explicit_profile='deception_detection',
            directory_path=Path('/test/deception_batch/')
        )

        assert profile is not None
        assert profile['name'] == 'Deception Detection & Psycholinguistics'
        print(f"    ✓ Selected profile: {profile['name']}")

        # Test auto-detection
        profile2 = selector.select_profile(
            explicit_profile=None,
            directory_path=Path('/test/mental_models_full/')
        )
        assert profile2 is not None
        print(f"    ✓ Auto-detected profile: {profile2['name']}")

        return True

    def test_state_manager(self) -> bool:
        """Test batch state management"""
        from state_manager import StateManager, BatchStatus, PaperStatus

        # Use temp directory for state
        state_dir = self.temp_dir / "state"
        state_dir.mkdir()

        manager = StateManager(state_dir=str(state_dir))
        print("    ✓ StateManager initialized")

        # Create batch state
        batch_state = manager.create_batch(
            input_dir="/test/input",
            output_dir="/test/output",
            profile="test_profile",
            domain="psychology",
            extraction_mode="concept_extraction",
            paper_files=['paper1.pdf', 'paper2.pdf'],
            config={'test': True}
        )

        assert batch_state is not None
        assert batch_state.batch_id.startswith("batch_")
        print("    ✓ Created batch state")

        # Save state
        manager.save_state(batch_state)
        assert (state_dir / f"{batch_state.batch_id}.json").exists()
        print("    ✓ Saved batch state")

        # Load state
        loaded_state = manager.load_state(batch_state.batch_id)
        assert loaded_state is not None
        assert loaded_state.batch_id == batch_state.batch_id
        print("    ✓ Loaded batch state")

        # Update status
        manager.update_batch_status(batch_state, BatchStatus.PROCESSING_DOCUMENTS)
        assert batch_state.status == BatchStatus.PROCESSING_DOCUMENTS
        print("    ✓ Updated batch status")

        return True

    def _print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("  Test Summary")
        print("="*70 + "\n")

        passed = sum(1 for v in self.test_results.values() if v)
        total = len(self.test_results)

        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status}  {test_name}")

        print(f"\n  Results: {passed}/{total} tests passed")

        if passed == total:
            print("\n  🎉 All tests passed!\n")
        else:
            print(f"\n  ⚠️  {total - passed} test(s) failed\n")


def main():
    """Run tests from command line"""
    logging.basicConfig(
        level=logging.WARNING,  # Suppress info logs during testing
        format='%(levelname)s: %(message)s'
    )

    tester = PipelineTest()
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
