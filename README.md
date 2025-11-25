# Knowledge Extraction Pipeline

> Transform academic PDFs into interactive knowledge graphs using Claude MCP

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/ircaspurna/knowledge-extraction-pipeline/actions/workflows/quality.yml/badge.svg)](https://github.com/ircaspurna/knowledge-extraction-pipeline/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A complete, production-ready system for extracting structured knowledge from academic PDFs and building professional knowledge graphs.**

## ✨ Features

- 📄 **PDF Processing** - Extract text from PDFs with page-level tracking
- 🧠 **Concept Extraction** - Claude MCP-powered semantic extraction
- 🔗 **Entity Resolution** - Smart deduplication using embeddings
- 📊 **Knowledge Graphs** - NetworkX graphs with Neo4j export
- 🔍 **Semantic Search** - Vector-based document search with ChromaDB
- 🎨 **Visualization** - Interactive Cytoscape.js & Neo4j Browser

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ircaspurna/knowledge-extraction-pipeline.git
cd knowledge-extraction-pipeline

# Install the package (development mode)
pip install -e .

# Or install directly
pip install .
```

### Process Your First PDF

```bash
# Complete pipeline: PDF → Chunks → Concepts → Entities → Graph
python scripts/process_pdf.py paper.pdf --output ./output/

# The script will:
# 1. Extract text and create semantic chunks
# 2. Generate extraction prompts → saves to extraction_batch.json
# 3. YOU: Ask Claude Code to process extraction_batch.json (see below)
# 4. Parse Claude's responses → saves to entities.json
# 5. Resolve entities and build knowledge graph

# Search across processed documents
python scripts/search.py ./output/chroma_db "cognitive load"

# Build topic graph from multiple papers
python scripts/build_graph.py ./output/ --output topic_graph.json

# Import to Neo4j for advanced visualization (optional)
python scripts/import_neo4j.py ./output/knowledge_graph.json
```

## 📖 Documentation

### Examples

- **[Batch Workflow](examples/batch_workflow/)** - Process multiple PDFs in parallel
- **[Custom Domain](examples/custom_domain/)** - Customize extraction for your domain

### Core Modules

- **Document Processor** (`knowledge_extraction.core.DocumentProcessor`) - PDF text extraction
- **Semantic Chunker** (`knowledge_extraction.core.SemanticChunker`) - Intelligent text chunking
- **Vector Store** (`knowledge_extraction.core.VectorStore`) - ChromaDB-backed vector search
- **Graph Builder** (`knowledge_extraction.core.GraphBuilder`) - NetworkX graph construction
- **Concept Extractor** (`knowledge_extraction.extraction.ConceptExtractorMCP`) - MCP-based concept extraction
- **Entity Resolver** (`knowledge_extraction.extraction.EntityResolverMCP`) - Entity deduplication
- **Relationship Extractor** (`knowledge_extraction.extraction.RelationshipExtractor`) - Relationship classification

## 🏗️ Architecture

```
Input PDF
    ↓
[Document Processor] → Extract text, preserve structure
    ↓
[Semantic Chunker] → Create semantic chunks
    ↓
[Vector Store] → Index with ChromaDB
    ↓
[Concept Extractor] → Extract concepts (Claude MCP)
    ↓
[Entity Resolver] → Deduplicate with embeddings
    ↓
[Graph Builder] → Build NetworkX graph
    ↓
[Visualization] → Neo4j / Cytoscape.js
```

## 🎯 Use Cases

- **Literature Reviews** - Extract concepts from 100+ papers automatically
- **Research Synthesis** - Build unified knowledge graphs across disciplines
- **Knowledge Management** - Organize internal research documents
- **Systematic Reviews** - Structure extraction for meta-analysis

## 📊 Performance

- Processes **300-page book in ~15 minutes**
- Extracts **500-1,000 concepts per book**
- **40-60% cost savings** vs direct API calls (using MCP)
- Handles **10,000+ entity graphs** in Neo4j

## 💡 Key Innovation: MCP Integration

This pipeline uses **Model Context Protocol (MCP)** for concept extraction, providing:
- **Lower costs** compared to direct API calls (40-60% savings)
- **Better quality** through structured prompts
- **Full control** over extraction workflow
- **Interactive review** of extractions before finalizing

### How MCP Works (The "Manual Claude Step")

The pipeline generates prompts that you process through Claude Code:

```bash
# 1. Generate extraction prompts (automated)
python scripts/process_pdf.py paper.pdf --output ./output/
# Creates: extraction_batch.json with 50-100 prompts

# 2. Process with Claude Code (interactive)
# In Claude Code chat, say:
# "Process the extraction prompts in ./output/extraction_batch.json
#  and save responses to ./output/extraction_responses.json"

# 3. Parse responses (automated)
python scripts/parse_responses.py ./output/extraction_responses.json
# Creates: entities.json with extracted concepts
```

**Why this approach?**
- ✅ You review extractions as they happen
- ✅ Costs appear in Claude Code usage (transparent billing)
- ✅ Can pause/resume long extractions
- ✅ Iteratively refine prompts if quality is low

## 🛠️ Development

### Setup Development Environment

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest pytest-cov ruff mypy
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run smoke tests only
pytest tests/test_smoke.py -v

# Run integration tests
pytest tests/test_integration/ -v

# Run with coverage
pytest tests/ --cov=knowledge_extraction --cov-report=html
```

### Code Quality

```bash
# Lint code
python -m ruff check src/ scripts/

# Auto-fix linting issues
python -m ruff check src/ scripts/ --fix

# Type checking
python -m mypy src/
```

### Test Status

✅ **All tests passing** (14/14)
- 10 smoke tests (package structure validation)
- 4 integration tests (end-to-end workflows)
- Test coverage available for core modules

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{knowledge_extraction_pipeline,
  author = {Spurna, Irena},
  title = {Knowledge Extraction Pipeline: MCP-Based Academic Knowledge Graph Construction},
  year = {2025},
  url = {https://github.com/ircaspurna/knowledge-extraction-pipeline},
  version = {2.2.0}
}
```

## 🙏 Acknowledgments

- Built with [Anthropic's Claude MCP](https://github.com/anthropics/mcp)
- Uses [sentence-transformers](https://www.sbert.net/) for embeddings
- Powered by [NetworkX](https://networkx.org/) for graph operations
- Visualization with [Cytoscape.js](https://js.cytoscape.org/)

## 📞 Support

- **Examples**: Check [examples/](examples/) for working code
- **Issues**: Open an issue on GitHub
- **Questions**: Start a discussion on GitHub

---

**Built for researchers and knowledge workers who need quality extraction without the API costs.**

*Save money. Keep quality. Use MCP.*
