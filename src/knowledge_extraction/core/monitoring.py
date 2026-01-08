#!/usr/bin/env python3
"""
Enhanced Monitoring and Progress Tracking

Provides:
- Real-time progress tracking
- Performance metrics
- Resource monitoring
- Statistical summaries
- Detailed reporting

Author: Claude Code
Version: 1.0.0
Date: 2026-01-08
"""

import time
import json
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PhaseMetrics:
    """Metrics for a single processing phase"""
    phase_name: str
    started_at: float = 0.0
    completed_at: float = 0.0
    duration: float = 0.0
    items_processed: int = 0
    items_failed: int = 0
    errors: List[str] = field(default_factory=list)

    def start(self) -> None:
        """Mark phase as started"""
        self.started_at = time.time()

    def complete(self) -> None:
        """Mark phase as completed"""
        self.completed_at = time.time()
        if self.started_at > 0:
            self.duration = self.completed_at - self.started_at

    def add_error(self, error: str) -> None:
        """Record an error"""
        self.errors.append(error)
        self.items_failed += 1

    def add_success(self) -> None:
        """Record a successful item"""
        self.items_processed += 1


@dataclass
class PaperMetrics:
    """Metrics for a single paper"""
    filename: str
    started_at: float = 0.0
    completed_at: float = 0.0
    duration: float = 0.0

    # Phase timings
    phase_durations: Dict[str, float] = field(default_factory=dict)

    # Document stats
    text_length: int = 0
    num_pages: int = 0

    # Chunking stats
    chunks_created: int = 0
    chunks_filtered: int = 0
    avg_chunk_size: float = 0.0

    # Batching stats
    semantic_batches: int = 0
    batching_reduction_pct: float = 0.0

    # Vector indexing
    vectors_indexed: int = 0

    # Status
    success: bool = False
    error_message: Optional[str] = None

    def start(self) -> None:
        """Mark paper processing as started"""
        self.started_at = time.time()

    def complete(self, success: bool = True, error: Optional[str] = None) -> None:
        """Mark paper processing as completed"""
        self.completed_at = time.time()
        if self.started_at > 0:
            self.duration = self.completed_at - self.started_at
        self.success = success
        self.error_message = error


@dataclass
class BatchMetrics:
    """Comprehensive batch processing metrics"""
    batch_id: str
    started_at: float = 0.0
    completed_at: float = 0.0
    duration: float = 0.0

    # High-level stats
    total_papers: int = 0
    papers_succeeded: int = 0
    papers_failed: int = 0

    # Resource usage
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    cpu_percent: float = 0.0

    # Processing stats
    total_chunks: int = 0
    total_chunks_filtered: int = 0
    total_prompts: int = 0
    total_vectors: int = 0

    # Performance
    papers_per_minute: float = 0.0
    chunks_per_second: float = 0.0
    avg_paper_duration: float = 0.0

    # Semantic batching effectiveness
    overall_reduction_pct: float = 0.0
    avg_chunks_per_batch: float = 0.0

    # Phase metrics
    phase_metrics: Dict[str, PhaseMetrics] = field(default_factory=dict)

    # Paper-level metrics
    paper_metrics: Dict[str, PaperMetrics] = field(default_factory=dict)

    def start(self) -> None:
        """Mark batch processing as started"""
        self.started_at = time.time()
        self._update_memory_usage()

    def complete(self) -> None:
        """Mark batch processing as completed"""
        self.completed_at = time.time()
        if self.started_at > 0:
            self.duration = self.completed_at - self.started_at
        self._calculate_derived_metrics()

    def _update_memory_usage(self) -> None:
        """Update memory usage statistics"""
        process = psutil.Process()
        current_mb = process.memory_info().rss / (1024 * 1024)
        self.peak_memory_mb = max(self.peak_memory_mb, current_mb)

        # Update average (simple moving average)
        if self.avg_memory_mb == 0:
            self.avg_memory_mb = current_mb
        else:
            self.avg_memory_mb = (self.avg_memory_mb + current_mb) / 2

        self.cpu_percent = process.cpu_percent()

    def _calculate_derived_metrics(self) -> None:
        """Calculate derived metrics from raw data"""
        if self.duration > 0:
            self.papers_per_minute = (self.papers_succeeded / self.duration) * 60
            self.chunks_per_second = self.total_chunks / self.duration

        if self.papers_succeeded > 0:
            total_duration = sum(
                m.duration for m in self.paper_metrics.values() if m.success
            )
            self.avg_paper_duration = total_duration / self.papers_succeeded

        if self.total_chunks > 0 and self.total_prompts > 0:
            self.overall_reduction_pct = (
                (1 - (self.total_prompts / self.total_chunks)) * 100
            )
            self.avg_chunks_per_batch = self.total_chunks / self.total_prompts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert phase_metrics objects
        data['phase_metrics'] = {
            name: asdict(metrics)
            for name, metrics in self.phase_metrics.items()
        }
        # Convert paper_metrics objects
        data['paper_metrics'] = {
            name: asdict(metrics)
            for name, metrics in self.paper_metrics.items()
        }
        return data


class ProgressMonitor:
    """
    Enhanced progress monitoring and metrics collection

    Features:
    - Real-time progress tracking
    - Performance metrics
    - Resource monitoring
    - Statistical summaries
    - JSON export
    """

    def __init__(self, batch_id: str, total_papers: int):
        """
        Initialize progress monitor

        Args:
            batch_id: Batch identifier
            total_papers: Total number of papers to process
        """
        self.batch_id = batch_id
        self.metrics = BatchMetrics(
            batch_id=batch_id,
            total_papers=total_papers
        )
        self.current_paper: Optional[str] = None
        self.current_phase: Optional[str] = None

    def start_batch(self) -> None:
        """Start monitoring batch processing"""
        self.metrics.start()
        logger.info(f"📊 Monitoring started for batch: {self.batch_id}")

    def complete_batch(self) -> None:
        """Complete batch monitoring and generate report"""
        self.metrics.complete()
        logger.info(f"📊 Monitoring completed for batch: {self.batch_id}")

    def start_paper(self, filename: str) -> None:
        """Start monitoring a paper"""
        self.current_paper = filename
        if filename not in self.metrics.paper_metrics:
            self.metrics.paper_metrics[filename] = PaperMetrics(filename=filename)
        self.metrics.paper_metrics[filename].start()

    def complete_paper(
        self,
        filename: str,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """Complete paper monitoring"""
        if filename in self.metrics.paper_metrics:
            self.metrics.paper_metrics[filename].complete(success, error)

            if success:
                self.metrics.papers_succeeded += 1
            else:
                self.metrics.papers_failed += 1

        self.current_paper = None
        self.metrics._update_memory_usage()

    def start_phase(self, phase_name: str) -> None:
        """Start monitoring a phase"""
        self.current_phase = phase_name
        if phase_name not in self.metrics.phase_metrics:
            self.metrics.phase_metrics[phase_name] = PhaseMetrics(phase_name=phase_name)
        self.metrics.phase_metrics[phase_name].start()

    def complete_phase(self, phase_name: str) -> None:
        """Complete phase monitoring"""
        if phase_name in self.metrics.phase_metrics:
            self.metrics.phase_metrics[phase_name].complete()
        self.current_phase = None

    def record_paper_stats(
        self,
        filename: str,
        text_length: int = 0,
        num_pages: int = 0,
        chunks_created: int = 0,
        chunks_filtered: int = 0,
        semantic_batches: int = 0,
        batching_reduction_pct: float = 0.0,
        vectors_indexed: int = 0
    ) -> None:
        """Record statistics for a paper"""
        if filename not in self.metrics.paper_metrics:
            self.metrics.paper_metrics[filename] = PaperMetrics(filename=filename)

        paper = self.metrics.paper_metrics[filename]
        paper.text_length = text_length
        paper.num_pages = num_pages
        paper.chunks_created = chunks_created
        paper.chunks_filtered = chunks_filtered
        paper.semantic_batches = semantic_batches
        paper.batching_reduction_pct = batching_reduction_pct
        paper.vectors_indexed = vectors_indexed

        # Update batch totals
        self.metrics.total_chunks += chunks_created
        self.metrics.total_chunks_filtered += chunks_filtered
        self.metrics.total_prompts += semantic_batches
        self.metrics.total_vectors += vectors_indexed

    def get_progress_summary(self) -> str:
        """Get current progress summary"""
        elapsed = time.time() - self.metrics.started_at if self.metrics.started_at > 0 else 0

        summary = []
        summary.append(f"\n{'='*70}")
        summary.append(f"  Progress Summary - {self.batch_id}")
        summary.append(f"{'='*70}\n")

        # Overall progress
        completed = self.metrics.papers_succeeded + self.metrics.papers_failed
        pct = (completed / self.metrics.total_papers * 100) if self.metrics.total_papers > 0 else 0
        summary.append(f"📈 Progress: {completed}/{self.metrics.total_papers} papers ({pct:.1f}%)")
        summary.append(f"   Succeeded: {self.metrics.papers_succeeded}")
        summary.append(f"   Failed: {self.metrics.papers_failed}")

        # Timing
        if elapsed > 0:
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            summary.append(f"\n⏱️  Elapsed: {elapsed_str}")

            if self.metrics.papers_succeeded > 0:
                eta_seconds = (elapsed / completed) * (self.metrics.total_papers - completed)
                eta_str = str(timedelta(seconds=int(eta_seconds)))
                summary.append(f"   ETA: {eta_str}")

        # Performance
        if elapsed > 60:
            papers_per_min = (completed / elapsed) * 60
            summary.append(f"\n⚡ Performance:")
            summary.append(f"   {papers_per_min:.2f} papers/minute")

        # Resources
        summary.append(f"\n💾 Resources:")
        summary.append(f"   Memory: {self.metrics.avg_memory_mb:.1f} MB avg, {self.metrics.peak_memory_mb:.1f} MB peak")
        summary.append(f"   CPU: {self.metrics.cpu_percent:.1f}%")

        summary.append(f"\n{'='*70}\n")

        return "\n".join(summary)

    def generate_final_report(self) -> str:
        """Generate comprehensive final report"""
        report = []
        report.append(f"\n{'='*70}")
        report.append(f"  Batch Processing Report")
        report.append(f"{'='*70}\n")

        report.append(f"📋 Batch: {self.batch_id}")
        report.append(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Duration
        duration_str = str(timedelta(seconds=int(self.metrics.duration)))
        report.append(f"\n⏱️  Duration: {duration_str}")

        # Success rate
        total = self.metrics.papers_succeeded + self.metrics.papers_failed
        success_rate = (self.metrics.papers_succeeded / total * 100) if total > 0 else 0
        report.append(f"\n✅ Success Rate: {success_rate:.1f}%")
        report.append(f"   Succeeded: {self.metrics.papers_succeeded}/{total}")
        report.append(f"   Failed: {self.metrics.papers_failed}/{total}")

        # Processing stats
        report.append(f"\n📊 Processing Statistics:")
        report.append(f"   Total chunks: {self.metrics.total_chunks:,}")
        report.append(f"   Filtered: {self.metrics.total_chunks_filtered:,}")
        report.append(f"   Semantic batches: {self.metrics.total_prompts:,}")
        report.append(f"   Vectors indexed: {self.metrics.total_vectors:,}")

        # Semantic batching effectiveness
        if self.metrics.overall_reduction_pct > 0:
            report.append(f"\n🧬 Semantic Batching:")
            report.append(f"   Reduction: {self.metrics.overall_reduction_pct:.1f}%")
            report.append(f"   Avg chunks/batch: {self.metrics.avg_chunks_per_batch:.1f}")
            original = self.metrics.total_chunks - self.metrics.total_chunks_filtered
            report.append(f"   {original:,} chunks → {self.metrics.total_prompts:,} prompts")

        # Performance metrics
        report.append(f"\n⚡ Performance:")
        report.append(f"   {self.metrics.papers_per_minute:.2f} papers/minute")
        report.append(f"   {self.metrics.chunks_per_second:.2f} chunks/second")
        report.append(f"   {self.metrics.avg_paper_duration:.1f}s avg/paper")

        # Resource usage
        report.append(f"\n💾 Resource Usage:")
        report.append(f"   Peak memory: {self.metrics.peak_memory_mb:.1f} MB")
        report.append(f"   Avg memory: {self.metrics.avg_memory_mb:.1f} MB")

        # Top papers by processing time
        if self.metrics.paper_metrics:
            sorted_papers = sorted(
                [(name, m.duration) for name, m in self.metrics.paper_metrics.items() if m.success],
                key=lambda x: x[1],
                reverse=True
            )[:5]

            if sorted_papers:
                report.append(f"\n⏱️  Slowest Papers:")
                for name, duration in sorted_papers:
                    report.append(f"   {name}: {duration:.1f}s")

        # Failed papers
        failed = [
            name for name, m in self.metrics.paper_metrics.items()
            if not m.success
        ]
        if failed:
            report.append(f"\n❌ Failed Papers:")
            for name in failed:
                error = self.metrics.paper_metrics[name].error_message
                report.append(f"   {name}")
                if error:
                    report.append(f"      Error: {error}")

        report.append(f"\n{'='*70}\n")

        return "\n".join(report)

    def export_metrics(self, output_path: Path) -> None:
        """Export metrics to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.metrics.to_dict(), f, indent=2)

        logger.info(f"📊 Exported metrics to: {output_path}")
