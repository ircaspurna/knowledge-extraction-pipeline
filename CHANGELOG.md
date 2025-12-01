# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
