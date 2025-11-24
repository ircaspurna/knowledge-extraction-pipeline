"""
Knowledge Extraction Pipeline

A complete system for extracting knowledge from academic PDFs and building
interactive knowledge graphs using Claude MCP.
"""

__version__ = "2.2.0"
__author__ = "IRI"
__license__ = "MIT"

from knowledge_extraction.core.document_processor import DocumentProcessor
from knowledge_extraction.core.graph_builder import GraphBuilder
from knowledge_extraction.core.semantic_chunker import SemanticChunker
from knowledge_extraction.extraction.concept_extractor import ConceptExtractorMCP
from knowledge_extraction.extraction.entity_resolver import EntityResolverMCP

__all__ = [
    "DocumentProcessor",
    "SemanticChunker",
    "GraphBuilder",
    "ConceptExtractorMCP",
    "EntityResolverMCP",
]
