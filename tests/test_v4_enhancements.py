#!/usr/bin/env python3
"""
Test Suite for v4.0 Enhancements

Tests semantic batching and monitoring features added in v4.0.
"""

import tempfile
import shutil
from pathlib import Path

from knowledge_extraction.core import SemanticChunker, VectorStore, ProgressMonitor
from knowledge_extraction.extraction import ConceptExtractorMCP
from knowledge_extraction.extraction.semantic_batch_optimizer import SemanticBatchOptimizer


def test_semantic_batching():
    """Test that semantic batching reduces prompts by 65-70%"""
    print("\n🧬 Testing Semantic Batching...")

    # Create test chunks
    test_chunks = [
        {'text': f"This is chunk {i} about machine learning and artificial intelligence.",
         'chunk_id': f'c{i}'}
        for i in range(100)
    ]

    # Test optimizer
    optimizer = SemanticBatchOptimizer(chunks_per_batch=4)
    batches = optimizer.optimize_chunks(test_chunks, filter_non_substantive=False)

    stats = optimizer.get_batch_statistics(batches)

    # Validate reduction
    assert stats['reduction_pct'] >= 60, f"Expected ≥60% reduction, got {stats['reduction_pct']:.1f}%"
    assert len(batches) < len(test_chunks), "Should reduce number of prompts"

    print(f"  ✓ {len(test_chunks)} chunks → {len(batches)} prompts ({stats['reduction_pct']:.1f}% reduction)")
    print(f"  ✓ Average chunks per batch: {stats['avg_chunks_per_batch']:.1f}")


def test_progress_monitoring():
    """Test progress monitoring and metrics collection"""
    print("\n📊 Testing Progress Monitoring...")

    batch_id = "test_batch_001"
    total_papers = 10

    # Initialize monitor
    monitor = ProgressMonitor(batch_id=batch_id, total_papers=total_papers)
    monitor.start_batch()

    # Simulate processing 3 papers
    for i in range(3):
        filename = f"paper_{i}.pdf"
        monitor.start_paper(filename)

        monitor.record_paper_stats(
            filename=filename,
            text_length=5000,
            num_pages=10,
            chunks_created=50,
            semantic_batches=15,
            batching_reduction_pct=70.0
        )

        monitor.complete_paper(filename, success=True)

    # Complete batch
    monitor.complete_batch()

    # Validate metrics
    assert monitor.metrics.papers_succeeded == 3
    assert monitor.metrics.total_chunks == 150
    assert monitor.metrics.total_prompts == 45

    # Test report generation
    report = monitor.generate_final_report()
    assert "Success Rate" in report
    assert "Semantic Batching" in report

    # Test metrics export
    temp_dir = Path(tempfile.mkdtemp())
    try:
        metrics_file = temp_dir / "metrics.json"
        monitor.export_metrics(metrics_file)
        assert metrics_file.exists()
        print(f"  ✓ Monitoring initialized and tracked 3 papers")
        print(f"  ✓ Final report generated successfully")
        print(f"  ✓ Metrics exported to JSON")
    finally:
        shutil.rmtree(temp_dir)


def test_semantic_batching_integration():
    """Test semantic batching integration with concept extractor"""
    print("\n🔗 Testing Semantic Batching Integration...")

    # Create test chunks
    test_chunks = [
        {
            'text': f"Sample text about deception detection {i}.",
            'chunk_id': f'chunk_{i}',
            'source_file': 'test.pdf',
            'page': 1
        }
        for i in range(20)
    ]

    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Test with semantic batching enabled
        from knowledge_extraction.extraction.concept_extractor import create_batch_extraction_file

        output_file = temp_dir / "extraction_batch.json"
        create_batch_extraction_file(
            test_chunks,
            output_file,
            use_semantic_batching=True,
            chunks_per_batch=4
        )

        # Read and validate output
        import json
        with open(output_file) as f:
            batch_data = json.load(f)

        assert batch_data['semantic_batching_enabled'] is True
        assert len(batch_data['prompts']) < len(test_chunks)

        reduction = (1 - len(batch_data['prompts']) / len(test_chunks)) * 100

        print(f"  ✓ {len(test_chunks)} chunks → {len(batch_data['prompts'])} prompts")
        print(f"  ✓ Reduction: {reduction:.1f}%")
        print(f"  ✓ Batch file created successfully")

    finally:
        shutil.rmtree(temp_dir)


def test_chunk_positions_stay_within_text():
    """Regression (v4.0.3): chunk char_start must never exceed document length.

    Earlier versions double-added section_start in chunk_document when the
    semantic-boundary splitter was active (the default whenever sentence-
    transformers is available). Chunk positions drifted upward, pushing late
    chunks past every page_mapping entry — so the chunker's "last position <=
    char_start" lookup fell through and assigned the document's max page
    number to ~half the chunks. Guards against that pathology returning.
    """
    print("\n📐 Testing Chunk Position Drift (v4.0.3 regression)...")

    # Build a long multi-section document so section_start accumulates.
    # Headers match the section pattern in extract_sections.
    sections = []
    for i in range(20):
        sections.append(f"## Section {i+1}\n\n")
        sections.append(
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            * 8
        )
        sections.append("\n\n")
    text = "".join(sections)

    # Build a valid page_mapping that covers the whole text (20 pretend pages).
    page_size = len(text) // 20
    page_mapping = {
        i * page_size: (i + 1, i * page_size, (i + 1) * page_size)
        for i in range(20)
    }

    chunker = SemanticChunker(
        target_chunk_size=100,
        min_chunk_size=30,
        max_chunk_size=300,
    )
    chunks = chunker.chunk_document(
        text=text,
        source_file="multisection.pdf",
        page_mapping=page_mapping,
    )

    assert chunks, "Should produce at least one chunk"
    text_len = len(text)
    for c in chunks:
        assert 0 <= c.char_start < text_len, (
            f"chunk {c.chunk_id} has char_start={c.char_start} "
            f"outside [0, {text_len})"
        )

    # Stronger invariant: no single page should capture >50% of chunks.
    # If the bug were back, chunks would all pile at the max page (20).
    from collections import Counter
    page_counts = Counter(c.page for c in chunks)
    dominant_page, dominant_count = page_counts.most_common(1)[0]
    assert dominant_count / len(chunks) < 0.5, (
        f"page {dominant_page} got {dominant_count}/{len(chunks)} chunks "
        f"({100*dominant_count/len(chunks):.0f}%) — likely double-add regression"
    )

    print(f"  ✓ All {len(chunks)} chunk char_starts within [0, {text_len})")
    print(f"  ✓ Page distribution healthy "
          f"(top page {dominant_page}: {dominant_count} chunks, "
          f"{100*dominant_count/len(chunks):.0f}%)")


def test_batched_passage_index_attribution():
    """Regression (v4.0.4): each concept extracted from a batched prompt must
    be attributed to its true source chunk via passage_index, not silently
    defaulted to the first chunk in the batch."""
    print("\n🎯 Testing Per-Concept Attribution in Batched Prompts (v4.0.4 regression)...")

    import json as _json
    from knowledge_extraction.extraction.concept_extractor import ConceptExtractorMCP
    from knowledge_extraction.extraction.semantic_batch_optimizer import create_batched_prompts

    # ----- 1. Batched-prompt metadata: pages must be a parallel LIST -----
    chunks = [
        {"text": "First passage about alpha.", "chunk_id": "p10c0",
         "source_file": "x.pdf", "page": 10},
        {"text": "Second passage about beta.", "chunk_id": "p10c1",
         "source_file": "x.pdf", "page": 10},  # same page as above
        {"text": "Third passage about gamma.", "chunk_id": "p11c0",
         "source_file": "x.pdf", "page": 11},
    ]
    prompts = create_batched_prompts(
        batches=[chunks],
        prompt_template="**Passage**: Placeholder.\nExtract research context from the following passage.",
    )
    md = prompts[0]["metadata"]
    assert isinstance(md["pages"], list) and len(md["pages"]) == 3, \
        "pages must be a parallel list (length matches chunk_ids), not a set"
    assert md["pages"] == [10, 10, 11], "pages must preserve order and duplicates"
    assert "passage_index" in prompts[0]["prompt"], \
        "batched prompt must instruct the LLM to emit passage_index"

    # ----- 2. Parser must route concepts via passage_index -----
    extractor = ConceptExtractorMCP()
    response_text = _json.dumps({
        "concepts": [
            {"term": "Alpha", "definition": "Alpha def.", "category": "method",
             "importance": "high", "justification": "From passage 1",
             "quote": "alpha", "passage_index": 1},
            {"term": "Beta", "definition": "Beta def.", "category": "method",
             "importance": "high", "justification": "From passage 2",
             "quote": "beta", "passage_index": 2},
            {"term": "Gamma", "definition": "Gamma def.", "category": "method",
             "importance": "high", "justification": "From passage 3",
             "quote": "gamma", "passage_index": 3},
        ]
    })
    concepts = extractor.parse_extraction_response(
        response_text=response_text,
        chunk_id="p10c0",      # response-level fallback (the buggy default)
        source_file="x.pdf",
        page=10,
        chunk_ids=["p10c0", "p10c1", "p11c0"],
        pages=[10, 10, 11],
    )
    by_term = {c.term: (c.chunk_id, c.page) for c in concepts}
    assert by_term["Alpha"] == ("p10c0", 10)
    assert by_term["Beta"]  == ("p10c1", 10)
    assert by_term["Gamma"] == ("p11c0", 11), \
        "Concept from passage 3 must be attributed to p11c0, not the first chunk"

    # ----- 3. Legacy responses without passage_index still work -----
    legacy_response = _json.dumps({
        "concepts": [
            {"term": "Legacy1", "definition": "d", "category": "method",
             "importance": "high", "justification": "j", "quote": "q"},
            {"term": "Legacy2", "definition": "d", "category": "method",
             "importance": "high", "justification": "j", "quote": "q"},
        ]
    })
    legacy_concepts = extractor.parse_extraction_response(
        response_text=legacy_response,
        chunk_id="p10c0",
        source_file="x.pdf",
        page=10,
        chunk_ids=["p10c0", "p10c1"],
        pages=[10, 11],
    )
    for c in legacy_concepts:
        assert c.chunk_id == "p10c0" and c.page == 10, \
            "Without passage_index, must fall back to response-level chunk_id/page"

    print(f"  ✓ Batched metadata: pages parallel to chunk_ids (preserves duplicates)")
    print(f"  ✓ Per-concept passage_index correctly routes attribution")
    print(f"  ✓ Legacy responses (no passage_index) still parse via fallback")


def main():
    """Run all v4.0 enhancement tests"""
    print("\n" + "="*70)
    print("  Knowledge Extraction Pipeline v4.0 - Enhancement Test Suite")
    print("="*70)

    tests = [
        ("Semantic Batching", test_semantic_batching),
        ("Progress Monitoring", test_progress_monitoring),
        ("Semantic Batching Integration", test_semantic_batching_integration),
        ("Chunk Position Drift (v4.0.3 regression)", test_chunk_positions_stay_within_text),
        ("Batched Passage Index Attribution (v4.0.4 regression)", test_batched_passage_index_attribution),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            test_func()
            results[test_name] = True
            print(f"  ✅ PASS: {test_name}\n")
        except Exception as e:
            results[test_name] = False
            print(f"  ❌ FAIL: {test_name}")
            print(f"     Error: {e}\n")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("  Test Summary")
    print("="*70 + "\n")

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {test_name}")

    print(f"\n  Results: {passed}/{total} tests passed")

    if all(results.values()):
        print("\n  🎉 All tests passed!")
        return 0
    else:
        print("\n  ❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
