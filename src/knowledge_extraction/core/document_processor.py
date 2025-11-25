#!/usr/bin/env python3
"""
Document Processor - Extract text from documents with page tracking

Supports:
- PDF files (with page numbers) - Multiple libraries with fallback
- EPUB files (e-books) - Chapter tracking
- Text files (plain text, markdown)
- Metadata extraction

PDF Processing Strategy:
1. Try pypdf (fastest, most compatible)
2. Fallback to pdfplumber (better quality, handles tables)
3. Fallback to PyMuPDF/fitz (most robust, best quality)
"""

import json
import logging
from pathlib import Path
from typing import Any

from ..utils.path_utils import validate_directory_path, validate_file_path

# Set up logging
logger = logging.getLogger(__name__)

# PDF Libraries (try multiple for robustness)
PDF_LIBRARIES = []

try:
    from pypdf import PdfReader
    PDF_LIBRARIES.append('pypdf')
except ImportError:
    PdfReader = None

try:
    import pdfplumber
    PDF_LIBRARIES.append('pdfplumber')
except ImportError:
    pdfplumber = None

try:
    import fitz  # PyMuPDF
    PDF_LIBRARIES.append('pymupdf')
except ImportError:
    fitz = None

# EPUB Libraries
try:
    import ebooklib
    from bs4 import BeautifulSoup
    from ebooklib import epub
    EPUB_AVAILABLE = True
except ImportError:
    EPUB_AVAILABLE = False
    logger.debug("EPUB support not available. Install: pip install ebooklib beautifulsoup4 lxml")


class DocumentProcessor:
    """Extract text from documents with provenance tracking"""

    def __init__(self, pdf_library: str | None = None) -> None:
        """
        Initialize document processor

        Args:
            pdf_library: Preferred PDF library ('pypdf', 'pdfplumber', 'pymupdf')
                        If None, will try all available libraries in order
        """
        self.stats = {
            'files_processed': 0,
            'total_pages': 0,
            'total_chars': 0
        }

        self.pdf_library_preference = pdf_library

        # Log available libraries
        if PDF_LIBRARIES:
            logger.debug(f"Available PDF libraries: {', '.join(PDF_LIBRARIES)}")
        else:
            logger.warning("No PDF libraries installed! Install at least one: pypdf, pdfplumber, or pymupdf")

        if EPUB_AVAILABLE:
            logger.debug("EPUB support available")

    # =========================================================================
    # PDF PROCESSING (with fallback)
    # =========================================================================

    def process_pdf(self, file_path: Path) -> dict[str, Any] | None:
        """
        Extract text from PDF with page tracking.

        Tries multiple PDF libraries in order for robustness:
        1. pypdf (fastest)
        2. pdfplumber (better quality)
        3. PyMuPDF (most robust)

        Returns:
            {
                'text': str,
                'metadata': {
                    'title': str,
                    'author': str,
                    'pages': int,
                    'source_file': str,
                    'pdf_library': str  # Which library was used
                },
                'page_mapping': {
                    char_pos: (page_num, page_start, page_end)
                }
            }
        """
        # Validate inputs
        if not file_path:
            raise ValueError("file_path cannot be None or empty")

        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        if file_path.suffix.lower() not in ['.pdf']:
            raise ValueError(f"File must be a PDF, got: {file_path.suffix}")

        # Try PDF libraries in order
        libraries_to_try = []

        if self.pdf_library_preference:
            # Try preferred library first
            if self.pdf_library_preference in PDF_LIBRARIES:
                libraries_to_try.append(self.pdf_library_preference)
            else:
                logger.warning(f"Preferred library '{self.pdf_library_preference}' not available")

        # Add remaining libraries as fallbacks
        for lib in ['pypdf', 'pdfplumber', 'pymupdf']:
            if lib in PDF_LIBRARIES and lib not in libraries_to_try:
                libraries_to_try.append(lib)

        if not libraries_to_try:
            logger.error("No PDF libraries available")
            return None

        # Try each library until one succeeds
        last_error = None
        for lib_name in libraries_to_try:
            try:
                logger.debug(f"Trying PDF extraction with {lib_name}")

                if lib_name == 'pypdf':
                    result = self._process_pdf_pypdf(file_path)
                elif lib_name == 'pdfplumber':
                    result = self._process_pdf_pdfplumber(file_path)
                elif lib_name == 'pymupdf':
                    result = self._process_pdf_pymupdf(file_path)
                else:
                    continue

                if result:
                    result['metadata']['pdf_library'] = lib_name
                    logger.debug(f"Successfully extracted PDF with {lib_name}")
                    return result

            except Exception as e:
                last_error = e
                logger.debug(f"Failed with {lib_name}: {e}")
                continue

        # All libraries failed
        logger.error(f"All PDF libraries failed. Last error: {last_error}")
        return None

    def _process_pdf_pypdf(self, file_path: Path) -> dict[str, Any] | None:
        """Process PDF using pypdf library"""
        if PdfReader is None:
            return None

        reader = PdfReader(file_path)

        # Extract metadata
        metadata = {
            'title': reader.metadata.get('/Title', file_path.stem) if reader.metadata else file_path.stem,
            'author': reader.metadata.get('/Author', 'Unknown') if reader.metadata else 'Unknown',
            'pages': len(reader.pages),
            'source_file': file_path.name
        }

        # Extract text with page tracking
        text = ""
        page_mapping = {}

        for page_num, page in enumerate(reader.pages, 1):
            page_start = len(text)

            # Extract text from page
            try:
                page_text = page.extract_text()
                if not page_text or not page_text.strip():
                    logger.warning(f"Page {page_num} extracted as empty (pypdf)")
                    page_text = ""
            except Exception as e:
                logger.warning(f"Could not extract text from page {page_num} (pypdf): {e}")
                page_text = ""

            if page_text:
                text += page_text
                # Add spacing between pages
                if page_num < len(reader.pages):
                    text += "\n\n"

            page_end = len(text)
            page_mapping[page_start] = (page_num, page_start, page_end)

        self.stats['files_processed'] += 1
        self.stats['total_pages'] += len(reader.pages)
        self.stats['total_chars'] += len(text)

        return {
            'text': text,
            'metadata': metadata,
            'page_mapping': page_mapping
        }

    def _process_pdf_pdfplumber(self, file_path: Path) -> dict[str, Any] | None:
        """Process PDF using pdfplumber library (better quality, handles tables)"""
        if pdfplumber is None:
            return None

        with pdfplumber.open(file_path) as pdf:
            # Extract metadata
            metadata = {
                'title': pdf.metadata.get('Title', file_path.stem) if pdf.metadata else file_path.stem,
                'author': pdf.metadata.get('Author', 'Unknown') if pdf.metadata else 'Unknown',
                'pages': len(pdf.pages),
                'source_file': file_path.name
            }

            # Extract text with page tracking
            text = ""
            page_mapping = {}

            for page_num, page in enumerate(pdf.pages, 1):
                page_start = len(text)

                try:
                    page_text = page.extract_text()
                    if not page_text or not page_text.strip():
                        logger.warning(f"Page {page_num} extracted as empty (pdfplumber)")
                        page_text = ""
                except Exception as e:
                    logger.warning(f"Could not extract text from page {page_num} (pdfplumber): {e}")
                    page_text = ""

                if page_text:
                    text += page_text
                    if page_num < len(pdf.pages):
                        text += "\n\n"

                page_end = len(text)
                page_mapping[page_start] = (page_num, page_start, page_end)

            self.stats['files_processed'] += 1
            self.stats['total_pages'] += len(pdf.pages)
            self.stats['total_chars'] += len(text)

            return {
                'text': text,
                'metadata': metadata,
                'page_mapping': page_mapping
            }

    def _process_pdf_pymupdf(self, file_path: Path) -> dict[str, Any] | None:
        """Process PDF using PyMuPDF/fitz library (most robust, best quality)"""
        if fitz is None:
            return None

        doc = fitz.open(file_path)

        # Extract metadata
        metadata_dict = doc.metadata
        metadata = {
            'title': metadata_dict.get('title', file_path.stem) or file_path.stem,
            'author': metadata_dict.get('author', 'Unknown') or 'Unknown',
            'pages': len(doc),
            'source_file': file_path.name
        }

        # Extract text with page tracking
        text = ""
        page_mapping = {}

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_start = len(text)

            try:
                page_text = page.get_text()
                if not page_text or not page_text.strip():
                    logger.warning(f"Page {page_num + 1} extracted as empty (PyMuPDF)")
                    page_text = ""
            except Exception as e:
                logger.warning(f"Could not extract text from page {page_num + 1} (PyMuPDF): {e}")
                page_text = ""

            if page_text:
                text += page_text
                if page_num < len(doc) - 1:
                    text += "\n\n"

            page_end = len(text)
            page_mapping[page_start] = (page_num + 1, page_start, page_end)

        doc.close()

        self.stats['files_processed'] += 1
        self.stats['total_pages'] += len(doc)
        self.stats['total_chars'] += len(text)

        return {
            'text': text,
            'metadata': metadata,
            'page_mapping': page_mapping
        }

    # =========================================================================
    # EPUB PROCESSING
    # =========================================================================

    def process_epub(self, file_path: Path) -> dict[str, Any] | None:
        """
        Extract text from EPUB with chapter tracking.

        EPUB is a ZIP archive containing XHTML/HTML files.
        We extract and concatenate all content files.

        Returns:
            {
                'text': str,
                'metadata': {
                    'title': str,
                    'author': str,
                    'pages': int,  # Number of chapters
                    'source_file': str
                },
                'page_mapping': {
                    char_pos: (chapter_num, chapter_start, chapter_end)
                }
            }
        """
        # Validate inputs
        if not file_path:
            raise ValueError("file_path cannot be None or empty")

        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        if file_path.suffix.lower() not in ['.epub']:
            raise ValueError(f"File must be an EPUB, got: {file_path.suffix}")

        if not EPUB_AVAILABLE:
            logger.error("EPUB support not available. Install: pip install ebooklib beautifulsoup4 lxml")
            return None

        try:
            book = epub.read_epub(str(file_path))

            # Extract metadata
            title = 'Unknown'
            author = 'Unknown'

            title_meta = book.get_metadata('DC', 'title')
            if title_meta:
                title = title_meta[0][0]

            author_meta = book.get_metadata('DC', 'creator')
            if author_meta:
                author = author_meta[0][0]

            # Count chapters
            chapters = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

            metadata = {
                'title': title or file_path.stem,
                'author': author or 'Unknown',
                'pages': len(chapters),  # Chapters treated as "pages"
                'source_file': file_path.name
            }

            # Extract text with chapter tracking
            text = ""
            page_mapping = {}
            chapter_num = 0

            for item in chapters:
                chapter_start = len(text)
                chapter_num += 1

                try:
                    # Parse HTML content
                    soup = BeautifulSoup(item.get_content(), 'html.parser')

                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()

                    # Get text
                    chapter_text = soup.get_text(separator=' ', strip=True)

                    # Clean up whitespace
                    lines = (line.strip() for line in chapter_text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    chapter_text = '\n'.join(chunk for chunk in chunks if chunk)

                    if chapter_text:
                        text += chapter_text + "\n\n"

                except Exception as e:
                    logger.warning(f"Could not extract text from chapter {chapter_num}: {e}")
                    continue

                chapter_end = len(text)
                page_mapping[chapter_start] = (chapter_num, chapter_start, chapter_end)

            self.stats['files_processed'] += 1
            self.stats['total_pages'] += chapter_num
            self.stats['total_chars'] += len(text)

            return {
                'text': text,
                'metadata': metadata,
                'page_mapping': page_mapping
            }

        except Exception as e:
            logger.error(f"Error processing EPUB {file_path}: {e}")
            return None

    # =========================================================================
    # TEXT PROCESSING
    # =========================================================================

    def process_text(self, file_path: Path) -> dict[str, Any] | None:
        """
        Process plain text file.

        Returns same structure as process_pdf for consistency.
        """
        # Validate inputs
        if not file_path:
            raise ValueError("file_path cannot be None or empty")

        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        try:
            text = file_path.read_text(encoding='utf-8')

            metadata = {
                'title': file_path.stem,
                'author': 'Unknown',
                'pages': 1,
                'source_file': file_path.name
            }

            # Single page mapping
            page_mapping = {
                0: (1, 0, len(text))
            }

            self.stats['files_processed'] += 1
            self.stats['total_pages'] += 1
            self.stats['total_chars'] += len(text)

            return {
                'text': text,
                'metadata': metadata,
                'page_mapping': page_mapping
            }

        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return None

    # =========================================================================
    # UNIFIED INTERFACE
    # =========================================================================

    def process_file(self, file_path: Path) -> dict[str, Any] | None:
        """
        Process any supported file type.

        Automatically detects file type and uses appropriate processor.

        Supported formats:
        - PDF (.pdf) - Multiple libraries with fallback
        - EPUB (.epub) - E-books with chapter tracking
        - Text (.txt, .md) - Plain text
        """
        # Validate and sanitize input path (security check)
        try:
            file_path = validate_file_path(
                file_path,
                allowed_extensions=['.pdf', '.epub', '.txt', '.md'],
                must_exist=True
            )
        except (ValueError, FileNotFoundError) as e:
            logger.error(f"Invalid file path: {e}")
            return None

        suffix = file_path.suffix.lower()

        if suffix == '.pdf':
            return self.process_pdf(file_path)
        elif suffix == '.epub':
            return self.process_epub(file_path)
        elif suffix in ['.txt', '.md']:
            return self.process_text(file_path)
        else:
            logger.error(f"Unsupported file type: {suffix}")
            logger.info("Supported types: .pdf, .epub, .txt, .md")
            return None

    def save_document(self, document: dict[str, Any], output_path: Path) -> None:
        """Save processed document to JSON"""
        # Validate output directory (security check)
        try:
            output_path = Path(output_path)
            output_dir = validate_directory_path(output_path.parent, create=True)
            output_path = output_dir / output_path.name
        except (ValueError, OSError) as e:
            logger.error(f"Invalid output path: {e}")
            raise

        # Convert page_mapping keys to strings for JSON
        if 'page_mapping' in document:
            document['page_mapping'] = {
                str(k): v for k, v in document['page_mapping'].items()
            }

        output_path.write_text(json.dumps(document, indent=2), encoding='utf-8')
        logger.info(f"Saved document to: {output_path}")

    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics"""
        return self.stats


# =========================================================================
# CLI
# =========================================================================

def main() -> int:
    """Command-line interface"""
    import argparse

    # Configure logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    parser = argparse.ArgumentParser(
        description='Extract text from documents with page tracking'
    )
    parser.add_argument('input', type=str, help='Input file (PDF, EPUB, or TXT)')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--stats', action='store_true',
                       help='Print processing statistics')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--pdf-library', type=str,
                       choices=['pypdf', 'pdfplumber', 'pymupdf'],
                       help='Preferred PDF library (will fallback to others if needed)')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Process file
    processor = DocumentProcessor(pdf_library=args.pdf_library)
    document = processor.process_file(Path(args.input))

    if not document:
        return 1

    # Print stats
    if args.stats:
        print("\nDocument Statistics:")
        print(f"  Title: {document['metadata']['title']}")
        print(f"  Author: {document['metadata']['author']}")
        print(f"  Pages: {document['metadata']['pages']}")
        print(f"  Characters: {len(document['text']):,}")
        print(f"  Words: {len(document['text'].split()):,}")
        if 'pdf_library' in document['metadata']:
            print(f"  PDF Library: {document['metadata']['pdf_library']}")

    # Save output
    if args.output:
        processor.save_document(document, Path(args.output))
    else:
        # Print first 500 chars
        print("\nFirst 500 characters:")
        print(document['text'][:500])
        print("...")

    return 0


if __name__ == "__main__":
    exit(main())
