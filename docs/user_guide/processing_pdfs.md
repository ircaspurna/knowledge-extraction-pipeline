# Processing PDFs

Learn how to extract text from academic PDFs with structure preservation.

## Basic Usage

```python
from knowledge_extraction.core import DocumentProcessor

processor = DocumentProcessor()
document = processor.process_pdf("paper.pdf")

print(f"Pages: {len(document.pages)}")
print(f"Total text: {len(document.full_text)} characters")
```

## Features

- **Page tracking** - Know which page each text segment comes from
- **Structure preservation** - Maintains paragraphs, sections
- **Metadata extraction** - Title, authors, abstract (when available)

## Advanced Options

### Custom chunk size

```python
from knowledge_extraction.core import SemanticChunker

chunker = SemanticChunker(
    chunk_size=1000,  # characters
    overlap=200
)
chunks = chunker.chunk_document(document)
```

### Filtering

Remove non-substantive content:

```python
chunks = chunker.chunk_document(
    document,
    filter_non_substantive=True
)
```

## Troubleshooting

### "Failed to extract text"

Try different PDF libraries in DocumentProcessor settings.

### "Chunks too small"

Increase `chunk_size` parameter or adjust `min_word_count`.
