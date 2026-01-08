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

    return True


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

    return True


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

    return True


def main():
    """Run all v4.0 enhancement tests"""
    print("\n" + "="*70)
    print("  Knowledge Extraction Pipeline v4.0 - Enhancement Test Suite")
    print("="*70)

    tests = [
        ("Semantic Batching", test_semantic_batching),
        ("Progress Monitoring", test_progress_monitoring),
        ("Semantic Batching Integration", test_semantic_batching_integration),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
            print(f"  {'✅ PASS' if success else '❌ FAIL'}: {test_name}\n")
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
