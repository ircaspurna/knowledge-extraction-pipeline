# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.4] - 2026-05-22

### Fixed

#### Per-Concept Provenance in Batched Extraction Prompts

**Severity:** High — affected provenance accuracy of every concept extracted via semantic batching (the default cost-saving path).

**Problem:** `create_batched_prompts()` bundles 4–6 chunks per prompt under headings `**Passage 1:**`, `**Passage 2:**`, etc. The LLM was never asked to tag each emitted concept with which Passage it came from. The parser, lacking that signal, defaulted every concept in a batched response to the first chunk in the batch (`chunk_ids[0]` / `pages[0]`). Concepts genuinely from Passage 3 of a 4-passage batch were silently attributed to Passage 1 — losing chunk-level provenance and skewing page citations downstream.

Compounding the issue, batch metadata stored `pages` as a deduplicated set, so even if the LLM had emitted a passage index, the parser would have had no way to recover the per-passage page number.

**Fix:**
- `semantic_batch_optimizer.py: create_batched_prompts()`:
  - Emit `pages` as a parallel list (same length and order as `chunk_ids`), not `list(set(...))`.
  - Append an explicit per-concept attribution instruction to multi-chunk prompts: each emitted concept must include a `passage_index` integer (1..N) identifying which numbered Passage it came from.
- `concept_extractor.py: parse_extraction_response()` — accept optional `chunk_ids` and `pages` parallel lists. When a concept carries `passage_index`, route its `chunk_id` and `page` via `chunk_ids[passage_index-1]` / `pages[passage_index-1]`. Fall back to the response-level `chunk_id`/`page` when `passage_index` is absent (legacy responses, single-chunk prompts).
- `mcp/server.py: handle_parse_extraction_responses()` — pass the per-batch `chunk_ids` and `pages` lists from response metadata into the parser.

**Backward compatibility:** existing `extraction_responses.json` files without `passage_index` continue to parse correctly via the fallback path. Single-chunk prompts (no batching) are unaffected.

**Files changed:**
- `src/knowledge_extraction/extraction/semantic_batch_optimizer.py`
- `src/knowledge_extraction/extraction/concept_extractor.py`
- `src/knowledge_extraction/mcp/server.py`
- `tests/test_v4_enhancements.py` (regression test)

**Migration:** existing `concepts.json` files built from batched extraction responses have inflated `chunk_id` / `page` attribution to the first chunk in each batch (typically ~75% of concepts in 4-chunk batches). Re-parse `extraction_responses.json` with the v4.0.4 parser to recover correct attribution — but only the concepts whose responses include `passage_index` will be re-routed; legacy responses continue to use the first-chunk fallback. For correct provenance on past batches, re-extract concepts through the patched prompt template.

---

## [4.0.3] - 2026-05-20

### Fixed

#### Chunk Page Attribution — `char_start` Drift in Multi-Section Documents

**Severity:** High — affected page metadata on every chunked document with detected section headers; books were impacted most heavily.

**Problem:** In `SemanticChunker.chunk_document()`, when the `sentence-transformers` embedder was available (i.e., almost always), `char_start` positions for chunks drifted upward by approximately the cumulative `section_start` offset of every preceding section. Late chunks in long multi-section documents ended up with `char_start` values *past the end of the source text*, so the chunker's "last position ≤ `char_start` wins" page lookup fell through to the document's maximum page number. Empirically: ~46% of chunks in a 503-page book were tagged with `page=503` (the last page).

**Root cause:** Inconsistent contracts between the two splitting helpers. `split_at_semantic_boundaries(text, char_start)` returns chunks with **absolute** positions (it adds `char_start` internally), while `split_into_paragraphs(text)` returns **relative** positions. `chunk_document` previously called both as if they returned relative positions, so the semantic-boundary branch got `section_start` added *twice*. A smaller secondary issue: `extract_sections` stored `match.start()` (position of the section header) rather than the actual start of the (stripped) `section_text`, contributing a small per-section offset.

**Fix:**
- `semantic_chunker.py: chunk_document` — handle the two splitter contracts separately. The semantic-boundary branch trusts `chunk_start` as absolute; the paragraph branch adds `section_start` to its relative offsets.
- `semantic_chunker.py: extract_sections` — the third tuple element is now the actual position where `section_text` begins (`match.end() + leading_whitespace`), not `match.start()`.

#### Empty Pages Collide in `page_mapping`

**Problem:** All three PDF processors (pypdf, pdfplumber, PyMuPDF) wrote `page_mapping[page_start]` unconditionally, including for pages that extracted as empty (image-only, OCR-failed, blank). Empty pages don't grow the `text` buffer, so consecutive empties produce identical `page_start` keys; the dict insertion overwrites earlier entries, leaving only the last (highest-numbered) empty page mapped at that position. This compounded with the chunker bug above.

**Fix:** Each PDF backend now only writes a `page_mapping` entry for non-empty pages. A new `logger.warning` fires when >20% of pages extract empty (a signal that OCR or image-only content is degrading page-citation fidelity).

**Files changed:**
- `src/knowledge_extraction/core/semantic_chunker.py`
- `src/knowledge_extraction/core/document_processor.py`
- `tests/test_v4_enhancements.py` (regression test)

**Validation on a 503-page book (Vrij 2008):** chunks-at-max-page dropped from 734 (46.5%) to 1 (0.1%). Chunk distribution across 100-page ranges is now uniform (~20% per bucket) instead of piling at the end.

---

## [4.0.2] - 2026-02-08

### Fixed

#### Bare Exception Handlers
- **graph_builder.py**: `get_top_concepts()` bare `except:` replaced with `except Exception as e:` with proper logging and fallback message
- **mcp/server.py**: `handle_get_graph_statistics()` bare `except:` in PageRank section replaced with `except Exception as e:` with warning log

#### Chunk Serialization in Batch Processing
- **mcp/server.py**: `handle_batch_process_pdfs()` now correctly calls `chunker.chunk_document(text, source_file, page_mapping)` with proper arguments and serializes chunks via `[c.to_dict() for c in chunks]` instead of passing raw objects
- **mcp/server.py**: `handle_create_semantic_chunks()` now correctly calls `chunker.chunk_document(text=..., source_file=..., page_mapping=...)` with proper arguments and serializes via `.to_dict()`

### Improved

#### Semantic Chunker - Local Model Caching
- **semantic_chunker.py**: Embedding model initialization now prefers local HuggingFace cache (`HF_HUB_OFFLINE=1`) before attempting network download, avoiding SSL failures in restricted environments
- Provides actionable fix suggestion if both cache and network loading fail

#### Configurable Semantic Thresholds in Topic Profiles
- Semantic chunker thresholds (`similarity_threshold`, `topic_shift_threshold`, `dbscan_eps`) are now configurable per topic profile via `config/topic_profiles.yaml`

### Added

#### graph_viz_smart Fallback Import
- **mcp/server.py**: Now attempts to import `SmartGraphVisualizer` from `graph_viz_smart` first, falling back to `UltraFastGraphVisualizer` from `graph_viz` if not available
- Matches main pipeline import pattern for forward compatibility

**Files changed:**
- `src/knowledge_extraction/mcp/server.py`
- `src/knowledge_extraction/core/graph_builder.py`
- `src/knowledge_extraction/core/semantic_chunker.py`

---

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
