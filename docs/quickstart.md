# Quick Start Guide

Get started with Knowledge Extraction Pipeline in 5 minutes.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/knowledge-extraction-pipeline.git
cd knowledge-extraction-pipeline
pip install -r requirements.txt
```

## Your First Extraction

### 1. Process a PDF

```bash
python scripts/process_pdf.py path/to/paper.pdf --output ./output/
```

This extracts text and creates semantic chunks.

### 2. Extract Concepts

The extraction uses Claude MCP. Generate prompts:

```python
from knowledge_extraction.extraction import ConceptExtractorMCP
import json

extractor = ConceptExtractorMCP()
with open('output/chunks.json') as f:
    chunks = json.load(f)

# Generate extraction prompts
extractor.create_batch_extraction_file(
    chunks,
    'extraction_batch.json'
)
```

Then process with Claude Code and parse responses.

### 3. Build Knowledge Graph

```bash
python scripts/build_graph.py ./output/
```

This creates a NetworkX graph and exports to JSON/GraphML.

### 4. Visualize (Optional)

Import to Neo4j:

```bash
python scripts/import_neo4j.py ./output/knowledge_graph.json
```

Then open Neo4j Browser at http://localhost:7474

## Next Steps

- Read the [User Guide](user_guide/) for detailed workflows
- Check [Examples](../examples/) for working code
- See [API Reference](api_reference/) for module documentation
