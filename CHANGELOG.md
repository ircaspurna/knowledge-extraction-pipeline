# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.1] - 2026-01-26

### 🐛 Bug Fixes

#### Relationship Extractor - Invalid Chunk ID Warning System

**Fixed:** Silent failure when entities have invalid/unknown chunk_ids during relationship extraction

**Problem:**
- `find_co_occurrences()` silently skipped entities with `chunk_id: "unknown"` or missing chunk_ids
- No warning to users when relationship detection was significantly impacted
- Could result in 0 relationships being found despite having valid entities

**Solution:**
- Added validation tracking for invalid chunk_ids
- Warns when >10% of evidence items have invalid chunk_ids
- Provides actionable fix suggestion (run chunk_id_repair.py)
- Informational message when 1-10% invalid

**Impact:**
- Early detection of data quality issues
- Users now alerted to run repair tools before relationship extraction
- Prevents silent failures in relationship detection

**Example output:**
```
⚠️ 842/842 evidence items (100.0%) have invalid chunk_ids!
   This will significantly reduce relationship detection.
💡 Consider running chunk_id_repair.py to fix entities.json
```

**Code quality:**
- ✅ Passes `mypy --strict` type checking
- ✅ No breaking changes to API
- ✅ Backward compatible

**Files changed:**
- `src/knowledge_extraction/extraction/relationship_extractor.py`

### ✨ Added

#### Relationship Type Classification Script

**New:** `scripts/type_relationships.py` - Standalone utility to upgrade generic relationships to semantic types

**Purpose:**
- Upgrades "RELATED" or "CO_OCCURS" relationships to semantic types based on entity categories
- Uses 40+ category-pair mappings (method→concept = APPLIES_TO, theory→phenomenon = EXPLAINS, etc.)
- Complements the existing `infer_relationships_tfidf.py` (which creates NEW relationships)

**Usage:**
```bash
python3 scripts/type_relationships.py entities.json relationships.json

# Result: relationships.json updated with semantic types
# Example: 76.5% of relationships upgraded (3,599/4,707)
```

**Features:**
- Automatic backup creation before modification
- Supports both list and dict JSON formats
- Comprehensive statistics reporting
- 25+ semantic relationship types:
  - VARIANT_OF, USES, STUDIES, IMPLEMENTS, EVALUATES
  - EXPLAINS, QUANTIFIES, PRODUCES, APPLIES_TO
  - GOVERNS, GUIDES, MEASURES, etc.

**When to use:**
- After extracting co-occurrence relationships
- When entities have category assignments
- To add semantic meaning to generic "RELATED" relationships

**Files added:**
- `scripts/type_relationships.py`

## [4.0.0] - 2026-01-08

### 🎉 Major Features

#### 💰 Semantic Batching - 70% Cost Reduction

**NEW:** Intelligent hierarchical clustering reduces API prompts by 65-75% through semantic grouping

- Uses sentence-transformers (all-MiniLM-L6-v2) for local embedding generation
- AgglomerativeClustering groups semantically related chunks before extraction
- Automatic non-substantive content filtering (tables, headers, references)
- Configurable batch sizes and thresholds

**Performance:**
- **Before:** 835 chunks = 835 prompts
- **After:** 835 chunks → 212 prompts (74.6% reduction)
- **Cost savings:** ~$594 per 413 papers

**Example cost breakdown:**

| Scenario | Chunks | Without Batching | With Batching | Savings |
|----------|--------|-----------------|---------------|---------|
| 10 papers | 495 | $22.27 | $7.79 | 65% ($14.48) |
| 50 papers | 2,475 | $111.37 | $38.95 | 65% ($72.42) |
| 413 papers | 20,400 | $918 | $324 | 65% ($594) |

#### 📊 Enhanced Monitoring & Progress Tracking

**NEW:** Production-ready monitoring with comprehensive metrics

- Real-time progress tracking with ETAs
- Resource monitoring (CPU, memory usage via psutil)
- Per-paper and batch-level statistics
- Checkpoint reporting (configurable intervals)
- JSON metrics export for analytics
- Performance metrics (papers/minute, chunks/second)

#### 🎛️ CLI Configuration Flags

**NEW:** Full user control over v4.0 features

```bash
# Disable semantic batching (not recommended)
python3 scripts/batch_process.py pdfs/ --no-semantic-batching

# Adjust batch size (default: 4)
python3 scripts/batch_process.py pdfs/ --chunks-per-batch 6

# Disable monitoring
python3 scripts/batch_process.py pdfs/ --no-monitoring
```

**Available flags:**
- `--no-semantic-batching` - Disable semantic batching (increases costs by 3x)
- `--chunks-per-batch N` - Target chunks per semantic batch (default: 4)
- `--no-monitoring` - Disable progress monitoring

### 🐛 Bug Fixes

#### Critical: Empty Chunk Filtering

**Fixed:** SemanticChunker edge case producing chunks with empty text

- **Issue:** Chunker occasionally creates chunks where `char_start == char_end`, resulting in empty text
- **Impact:** VectorStore validation errors during indexing
- **Solution:** Automatic filtering of empty chunks immediately after chunking
- **Detection:** Logs warning with count of filtered chunks

### 📦 Dependencies

**Added:**
- `sentence-transformers>=2.2.0` - Local embedding generation for semantic clustering
- `scikit-learn>=1.3.0` - AgglomerativeClustering for semantic grouping
- `psutil>=5.9.0` - Resource monitoring (CPU, memory usage)

### 🧪 Testing

**Added comprehensive test suite:**
- `tests/test_v4_enhancements.py` - 3 focused tests for v4.0 features
- Real-world validation with 10 academic PDFs
- 100% success rate, 74.6% average reduction

### 📝 Documentation

**Updated:**
- `README.md` - Added v4.0 feature highlights, cost breakdowns, usage examples
- `scripts/batch_process.py` - Enhanced docstrings for all new parameters
- Added comprehensive changelog documentation

### ⚙️ API Changes

**New Function Parameters:**

All new parameters have sensible defaults - **no breaking changes**

```python
# batch_process() - New parameters
use_semantic_batching: bool = True
chunks_per_batch: int = 4
enable_monitoring: bool = True

# process_single_pdf() - New parameters
use_semantic_batching: bool = True
chunks_per_batch: int = 4
```

### 🔄 Migration Guide

#### Upgrading from v3.x to v4.0

**Installation:**
```bash
pip install -e ".[dev]"
```

**No breaking changes:**
- All existing scripts continue to work without modifications
- Semantic batching enabled by default (can be disabled)
- Monitoring enabled by default (can be disabled)
- Output format unchanged

### 📊 Performance

**Benchmarks (10 academic PDFs):**
- **Total processing time:** 2 minutes 47 seconds
- **Throughput:** 3.57 papers/minute, 4.97 chunks/second
- **Memory usage:** Peak 1290 MB, Average 707 MB
- **Semantic batching reduction:** 74.6% average (range: 66.7% - 86.3%)
- **Success rate:** 100% (10/10 papers)

---

## [3.0.0] - 2025-12-01

### Added
- **4 New MCP Tools** for complete pipeline coverage:
  - `batch_process_pdfs` - Process multiple PDFs in parallel with automatic chunking and extraction prompt generation
  - `create_graph_visualization` - Generate interactive HTML visualizations with Cytoscape.js (handles 10K+ nodes)
  - `search_semantic_documents` - Semantic search across all processed documents using ChromaDB vector databases
  - `get_graph_statistics` - Comprehensive graph analysis (PageRank, centrality, degree distribution, categories)
- **Complete MCP Workflow Guide** (`docs/WORKFLOW.md`) with step-by-step instructions for all use cases
- Updated MCP server to version 3.0 with 14 total tools (100% MCP-native pipeline)

### Improved
- MCP server now provides complete coverage - no manual Python scripts needed
- Batch PDF processing with progress tracking and error recovery
- Advanced graph visualization with filtering, search, and offline support
- Cross-document semantic search with similarity scoring

### Documentation
- Added comprehensive workflow guide (457 lines)
- Updated README with new tools and MCP-native approach
- Examples for batch processing and visualization workflows

## [2.2.0] - 2025-11-23

### Added
- Initial open source release
- Complete PDF processing pipeline
- MCP-based concept extraction
- Entity resolution with embeddings
- Knowledge graph building with NetworkX
- Neo4j import and visualization
- Semantic search with ChromaDB
- Comprehensive test suite
- Full documentation

### Features
- Process academic PDFs with page tracking
- Extract concepts using Claude MCP
- Build interactive knowledge graphs
- Export to Neo4j, GraphML, JSON
- Semantic search across documents
- Configurable extraction prompts (YAML)
- Domain-aware extraction rules

### Documentation
- Complete user guide
- API reference
- Tutorial examples
- Quick start guide
- Contributing guidelines

## [Unreleased]

### Planned
- Streaming extraction support
- Multi-document cross-referencing
- Automatic prompt optimization
- Web UI for pipeline management
- Additional visualization backends
