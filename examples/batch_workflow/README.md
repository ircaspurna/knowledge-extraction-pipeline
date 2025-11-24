# Batch Workflow Example

This example demonstrates how to process multiple PDFs in a single batch.

## What This Example Shows

- Processing multiple academic papers automatically
- Organizing outputs by paper
- Generating extraction batches for all papers
- Building a unified knowledge graph from all papers

## Prerequisites

```bash
# Install the package
pip install -e ../..

# Or if installing from PyPI
pip install knowledge-extraction-pipeline
```

## Files

- `run.py` - Main batch processing script
- `pdfs/` - Directory to place your PDF files (create this)
- `output/` - Generated outputs (created automatically)

## Usage

### Step 1: Add PDFs

```bash
mkdir pdfs
# Copy your PDF files into the pdfs/ directory
```

### Step 2: Run Batch Processing

```bash
python run.py
```

This will:
1. Process all PDFs in the `pdfs/` directory
2. Extract text and create semantic chunks for each
3. Index into vector databases
4. Generate extraction batch files

### Step 3: Extract Concepts (Claude Code)

```bash
# For each paper, process the extraction batch file with Claude Code
# Example:
# claude-code process output/paper1/extraction_batch.json
```

### Step 4: Build Unified Graph

After concept extraction, build a single knowledge graph from all papers:

```bash
python -m knowledge_extraction.scripts.build_graph output/
```

## Output Structure

```
output/
├── paper1/
│   ├── document.json
│   ├── chunks.json
│   ├── chroma_db/
│   ├── extraction_batch.json
│   ├── concepts.json (after Step 3)
│   └── entities.json (after resolution)
├── paper2/
│   └── ...
├── all_concepts.json (combined)
├── entities.json (deduplicated)
└── knowledge_graph.json (unified graph)
```

## Configuration

Edit `run.py` to customize:

```python
# Limit processing (for testing)
max_files = 5  # Only process first 5 PDFs

# Adjust chunk sizes
target_chunk_size = 600  # Target words per chunk

# Sample mode (faster testing)
sample_chunks = 10  # Only process first 10 chunks per paper
```

## Tips

- **Start Small**: Test with 2-3 papers first
- **Monitor Progress**: Check logs for processing status
- **Skip Processed**: Script skips already-processed papers
- **Parallel Concepts**: Process extraction batches in parallel for speed

## Next Steps

1. Export to Neo4j for visualization
2. Run semantic search across all papers
3. Analyze cross-document concept connections
