#!/usr/bin/env python3
"""
Core graph building functions - Importable versions of fast_batch_resolution.py

This module provides clean API for graph building that can be called from MCP server.
Uses the optimized algorithms from fast_batch_resolution.py.
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx  # type: ignore

# Set up logging
logger = logging.getLogger(__name__)


def normalize_term(term: str) -> str:
    """Normalize term for matching"""
    # Validate inputs
    if not isinstance(term, str):
        raise ValueError(f"term must be a string, got {type(term).__name__}")
    if not term:
        return ""
    return re.sub(r'\s+', ' ', term.lower().strip())


def exact_string_resolution(concepts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """
    Phase 1: Exact string matching (very fast, O(n))
    Returns entities and mapping of concept indices to entity IDs
    """
    # Validate inputs
    if not isinstance(concepts, list):
        raise ValueError(f"concepts must be a list, got {type(concepts).__name__}")

    for i, concept in enumerate(concepts):
        if not isinstance(concept, dict):
            raise ValueError(f"Concept {i} must be a dict, got {type(concept).__name__}")

    logger.info(f"Resolving {len(concepts)} concepts using exact string matching...")

    # Group by normalized term
    term_groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for i, concept in enumerate(concepts):
        norm_term = normalize_term(concept.get('term', ''))
        if norm_term:
            term_groups[norm_term].append((i, concept))

    # Create entities from groups
    entities: list[dict[str, Any]] = []
    concept_to_entity: dict[int, int] = {}

    for norm_term, group in term_groups.items():
        # Use first concept as representative
        base_concept = group[0][1].copy()

        # Merge evidence from all occurrences
        base_concept['canonical_term'] = base_concept.get('term', norm_term)
        base_concept['aliases'] = []
        base_concept['evidence_count'] = len(group)
        base_concept['source_papers'] = []

        # Collect unique source files
        sources = set()
        for idx, concept in group:
            src = concept.get('source_file', '')
            if src:
                sources.add(src)
            concept_to_entity[idx] = len(entities)

        base_concept['source_papers'] = list(sources)
        entities.append(base_concept)

    logger.info(f"Created {len(entities)} entities from exact string matching")
    return entities, concept_to_entity


def known_alias_resolution(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Phase 2: Merge known aliases (ML = Machine Learning, etc.)
    """
    # Validate inputs
    if not isinstance(entities, list):
        raise ValueError(f"entities must be a list, got {type(entities).__name__}")

    for i, entity in enumerate(entities):
        if not isinstance(entity, dict):
            raise ValueError(f"Entity {i} must be a dict, got {type(entity).__name__}")

    logger.info(f"Applying known alias resolution to {len(entities)} entities...")

    # Common aliases in academic literature
    known_aliases = {
        'ml': 'machine learning',
        'ai': 'artificial intelligence',
        'nn': 'neural network',
        'cnn': 'convolutional neural network',
        'rnn': 'recurrent neural network',
        'nlp': 'natural language processing',
        'cv': 'computer vision',
    }

    # Build reverse mapping
    alias_to_canonical: dict[str, str] = {}
    for alias, canonical in known_aliases.items():
        alias_to_canonical[alias] = canonical
        alias_to_canonical[canonical] = canonical

    # Group entities
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in entities:
        term = normalize_term(entity.get('term', ''))
        canonical = alias_to_canonical.get(term, term)
        groups[canonical].append(entity)

    # Merge groups
    merged_entities: list[dict[str, Any]] = []
    for canonical_term, group in groups.items():
        if len(group) == 1:
            merged_entities.append(group[0])
        else:
            # Merge multiple entities
            base = group[0].copy()
            base['aliases'] = [e['term'] for e in group if e['term'] != base['term']]
            base['evidence_count'] = sum(e.get('evidence_count', 1) for e in group)

            # Merge source papers
            all_sources = set()
            for e in group:
                all_sources.update(e.get('source_papers', []))
            base['source_papers'] = list(all_sources)

            merged_entities.append(base)

    logger.info(f"Merged aliases: {len(entities)} → {len(merged_entities)} entities")
    return merged_entities


def build_graph_from_entities(entities: list[dict[str, Any]]) -> nx.Graph:
    """Build NetworkX graph from entities with co-occurrence edges"""
    # Validate inputs
    if not isinstance(entities, list):
        raise ValueError(f"entities must be a list, got {type(entities).__name__}")

    from ..core.graph_builder import GraphBuilder

    builder = GraphBuilder()
    graph = builder.build_graph(entities)

    return graph


def build_knowledge_graph(
    entities: list[dict[str, Any]],
    output_dir: Path,
    title: str = "Knowledge Graph"
) -> tuple[Path, Path, dict[str, Any]]:
    """
    Complete graph building workflow

    Args:
        entities: List of entity dictionaries
        output_dir: Where to save outputs
        title: Graph title

    Returns:
        (graph_json_path, graphml_path, statistics)
    """
    # Validate inputs
    if not isinstance(entities, list):
        raise ValueError(f"entities must be a list, got {type(entities).__name__}")
    if not entities:
        raise ValueError("entities list cannot be empty")
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    if not output_dir.exists():
        raise ValueError(f"output_dir does not exist: {output_dir}")

    logger.info(f"Building knowledge graph from {len(entities)} entities...")

    from ..core.graph_builder import GraphBuilder

    # Build graph
    builder = GraphBuilder()
    graph = builder.build_graph(entities)

    # Export JSON
    graph_file = output_dir / 'knowledge_graph.json'
    builder.export_json(graph, graph_file, title=title)

    # Export GraphML
    graphml_file = output_dir / 'knowledge_graph.graphml'
    nx.write_graphml(graph, str(graphml_file))

    # Create visualization
    try:
        cytoscape_file = output_dir / 'knowledge_graph_cytoscape.html'
        builder.render_cytoscape(graph, cytoscape_file, title=title, node_size_by='centrality')
    except Exception:
        pass  # Visualization is optional

    # Get statistics
    stats = {
        'nodes': graph.number_of_nodes(),
        'edges': graph.number_of_edges(),
        'density': nx.density(graph),
        'categories': {},
        'importance_levels': {}
    }

    # Category and importance distribution
    for node, data in graph.nodes(data=True):
        cat = data.get('category', 'unknown')
        imp = data.get('importance', 'medium')
        stats['categories'][cat] = stats['categories'].get(cat, 0) + 1
        stats['importance_levels'][imp] = stats['importance_levels'].get(imp, 0) + 1

    # Top concepts by centrality
    if stats['nodes'] > 0:
        centrality = nx.degree_centrality(graph)
        top_concepts = sorted(centrality.items(), key=lambda x: -x[1])[:20]
        stats['top_concepts'] = [
            {'term': term, 'centrality': float(cent)}
            for term, cent in top_concepts
        ]

    logger.info(f"✓ Graph built successfully: {stats['nodes']} nodes, {stats['edges']} edges")
    return graph_file, graphml_file, stats


def process_topic_directory(
    topic_dir: Path,
    title: str | None = None,
    output_dir: Path | None = None
) -> tuple[Path, Path, dict[str, Any]]:
    """
    Complete workflow: Load concepts → Resolve → Build graph

    This is the main entry point for MCP server.

    Args:
        topic_dir: Directory containing paper subdirectories with concepts.json
        title: Graph title (defaults to directory name)
        output_dir: Where to save outputs (defaults to topic_dir)

    Returns:
        (graph_json_path, graphml_path, statistics)
    """
    # Validate inputs
    if not isinstance(topic_dir, Path):
        topic_dir = Path(topic_dir)
    if not topic_dir.exists():
        raise ValueError(f"topic_dir does not exist: {topic_dir}")
    if not topic_dir.is_dir():
        raise ValueError(f"topic_dir is not a directory: {topic_dir}")

    if output_dir is None:
        output_dir = topic_dir
    elif not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    if title is None:
        title = topic_dir.name.replace('_', ' ').title()

    logger.info(f"Processing topic directory: {topic_dir}")

    # Load all concepts from topic directory
    all_concepts: list[dict[str, Any]] = []
    paper_count = 0

    for paper_dir in sorted(topic_dir.iterdir()):
        if not paper_dir.is_dir():
            continue

        concepts_file = paper_dir / 'concepts_ENRICHED.json'
        if not concepts_file.exists():
            concepts_file = paper_dir / 'concepts.json'

        if concepts_file.exists():
            with open(concepts_file, encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    concepts = data
                elif isinstance(data, dict) and 'concepts' in data:
                    concepts = data['concepts']
                else:
                    continue

                all_concepts.extend(concepts)
                paper_count += 1

    if len(all_concepts) == 0:
        raise ValueError(f"No concepts found in {topic_dir}")

    logger.info(f"Loaded {len(all_concepts)} concepts from {paper_count} papers")

    # Entity resolution
    entities, _ = exact_string_resolution(all_concepts)
    final_entities = known_alias_resolution(entities)

    # Save entities
    entities_file = output_dir / 'entities.json'
    with open(entities_file, 'w', encoding='utf-8') as f:
        json.dump(final_entities, f, indent=2, ensure_ascii=False)

    # Build graph
    graph_file, graphml_file, stats = build_knowledge_graph(
        final_entities,
        output_dir,
        title
    )

    # Add paper and concept counts to stats
    stats['papers_processed'] = paper_count
    stats['input_concepts'] = len(all_concepts)
    stats['final_entities'] = len(final_entities)
    stats['reduction_percentage'] = 100 * (1 - len(final_entities) / len(all_concepts))

    return graph_file, graphml_file, stats


def process_entities_file(
    entities_file: Path,
    title: str | None = None
) -> tuple[Path, Path, dict[str, Any]]:
    """
    Build graph from existing entities.json file

    Args:
        entities_file: Path to entities.json
        title: Graph title

    Returns:
        (graph_json_path, graphml_path, statistics)
    """
    # Validate inputs
    if not isinstance(entities_file, Path):
        entities_file = Path(entities_file)
    if not entities_file.exists():
        raise FileNotFoundError(f"entities_file not found: {entities_file}")
    if not entities_file.is_file():
        raise ValueError(f"entities_file is not a file: {entities_file}")

    logger.info(f"Processing entities file: {entities_file}")

    output_dir = entities_file.parent

    if title is None:
        title = entities_file.parent.name.replace('_', ' ').title()

    # Load entities
    try:
        with open(entities_file, encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in entities file: {e}")

    if isinstance(data, list):
        entities = data
    elif isinstance(data, dict):
        entities = data.get('entities', data.get('concepts', []))
    else:
        raise ValueError(f"Invalid entities file format: {entities_file}")

    if not entities:
        raise ValueError(f"No entities found in {entities_file}")

    logger.info(f"Loaded {len(entities)} entities from file")

    # Build graph
    graph_file, graphml_file, stats = build_knowledge_graph(
        entities,
        output_dir,
        title
    )

    logger.info("✓ Processing complete")
    return graph_file, graphml_file, stats
