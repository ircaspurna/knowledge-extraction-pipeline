#!/usr/bin/env python3
"""
Core Neo4j import functions - Importable version of import_to_neo4j.py

This module provides clean API for Neo4j import that can be called from MCP server.
Uses the optimized batch import from import_to_neo4j.py.
"""

import json
import logging
from pathlib import Path
from typing import Any

# Set up logging
logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase  # type: ignore
except ImportError:
    GraphDatabase = None


class Neo4jImporter:
    """Professional Neo4j importer with batching and indexes"""

    def __init__(self, uri: str, user: str, password: str) -> None:
        if GraphDatabase is None:
            raise ImportError("neo4j driver not installed. Install with: pip3 install neo4j")

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.uri = uri
        self.user = user

    def close(self) -> None:
        """Close driver connection"""
        self.driver.close()

    def clear_database(self) -> None:
        """Clear all nodes and relationships"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("✓ Cleared existing data")

    def create_constraints(self) -> None:
        """Create constraints and indexes for better performance"""
        with self.driver.session() as session:
            # Create uniqueness constraint on term
            try:
                session.run("""
                    CREATE CONSTRAINT concept_term IF NOT EXISTS
                    FOR (c:Concept) REQUIRE c.term IS UNIQUE
                """)
                logger.info("✓ Created uniqueness constraint on term")
            except Exception as e:
                logger.warning(f"Warning: Could not create constraint: {e}")

            # Create indexes
            try:
                session.run("CREATE INDEX concept_category IF NOT EXISTS FOR (c:Concept) ON (c.category)")
                session.run("CREATE INDEX concept_importance IF NOT EXISTS FOR (c:Concept) ON (c.importance)")
                logger.info("✓ Created indexes")
            except Exception as e:
                logger.warning(f"Warning: Could not create indexes: {e}")

    def import_nodes(self, nodes: list[dict[str, Any]]) -> int:
        """
        Import nodes in batches

        Returns: Number of nodes imported
        """
        batch_size = 1000
        total = len(nodes)

        with self.driver.session() as session:
            for i in range(0, total, batch_size):
                batch = nodes[i:i+batch_size]

                # Prepare batch with proper types
                prepared_batch = []
                for node in batch:
                    prepared_node = {
                        'term': str(node.get('term', node.get('id', ''))),
                        'category': str(node.get('category', 'unknown')),
                        'importance': str(node.get('importance', 'medium')),
                        'definition': str(node.get('definition', '')),
                        'confidence': float(node.get('confidence', 0.0))
                    }
                    prepared_batch.append(prepared_node)

                session.run("""
                    UNWIND $batch AS node
                    CREATE (c:Concept {
                        term: node.term,
                        category: node.category,
                        importance: node.importance,
                        definition: node.definition,
                        confidence: node.confidence
                    })
                """, batch=prepared_batch)

                logger.info(f"  Imported {min(i+batch_size, total)}/{total} nodes...")

        logger.info(f"✓ Imported {total:,} nodes")
        return total

    def import_edges(self, edges: list[dict[str, Any]]) -> tuple[int, int]:
        """
        Import edges in batches

        Returns: (semantic_count, similarity_count)
        """
        batch_size = 1000

        # Group by edge class
        semantic_edges = [e for e in edges if e.get('edge_class') == 'semantic' or e.get('relationship_type')]
        similarity_edges = [e for e in edges if e.get('edge_class') == 'similarity']

        logger.info(f"Importing {len(semantic_edges)} semantic relationships...")
        semantic_count = self._import_semantic_edges(semantic_edges, batch_size)

        logger.info(f"Importing {len(similarity_edges):,} similarity edges...")
        similarity_count = self._import_similarity_edges(similarity_edges, batch_size)

        return semantic_count, similarity_count

    def _import_semantic_edges(self, edges: list[dict[str, Any]], batch_size: int) -> int:
        """Import semantic relationship edges"""
        total = len(edges)
        if total == 0:
            return 0

        with self.driver.session() as session:
            for i in range(0, total, batch_size):
                batch = edges[i:i+batch_size]

                prepared_batch = []
                for edge in batch:
                    prepared_edge = {
                        'source': str(edge.get('source', '')),
                        'target': str(edge.get('target', '')),
                        'type': str(edge.get('relationship_type', edge.get('type', 'RELATED_TO'))),
                        'weight': float(edge.get('weight', 1.0))
                    }
                    prepared_batch.append(prepared_edge)

                # Create relationships dynamically based on type
                session.run("""
                    UNWIND $batch AS edge
                    MATCH (a:Concept {term: edge.source})
                    MATCH (b:Concept {term: edge.target})
                    CALL apoc.create.relationship(a, edge.type, {weight: edge.weight}, b) YIELD rel
                    RETURN count(rel)
                """, batch=prepared_batch)

                logger.info(f"  Imported {min(i+batch_size, total)}/{total} semantic edges...")

        logger.info(f"✓ Imported {total:,} semantic edges")
        return total

    def _import_similarity_edges(self, edges: list[dict[str, Any]], batch_size: int) -> int:
        """Import similarity edges"""
        total = len(edges)
        if total == 0:
            return 0

        with self.driver.session() as session:
            for i in range(0, total, batch_size):
                batch = edges[i:i+batch_size]

                prepared_batch = []
                for edge in batch:
                    prepared_edge = {
                        'source': str(edge.get('source', '')),
                        'target': str(edge.get('target', '')),
                        'weight': float(edge.get('weight', 0.0))
                    }
                    prepared_batch.append(prepared_edge)

                session.run("""
                    UNWIND $batch AS edge
                    MATCH (a:Concept {term: edge.source})
                    MATCH (b:Concept {term: edge.target})
                    CREATE (a)-[:SIMILAR_TO {weight: edge.weight}]->(b)
                """, batch=prepared_batch)

                logger.info(f"  Imported {min(i+batch_size, total)}/{total} similarity edges...")

        logger.info(f"✓ Imported {total:,} similarity edges")
        return total


def import_graph_to_neo4j(
    graph_file: Path,
    neo4j_password: str,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    clear_existing: bool = True
) -> dict[str, Any]:
    """
    Import knowledge graph to Neo4j

    Args:
        graph_file: Path to knowledge_graph.json
        neo4j_password: Neo4j database password
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        clear_existing: Whether to clear existing data first

    Returns:
        Dictionary with import statistics
    """
    # Load graph data
    with open(graph_file, encoding='utf-8') as f:
        graph_data = json.load(f)

    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])

    if len(nodes) == 0:
        raise ValueError(f"No nodes found in {graph_file}")

    # Initialize importer
    importer = Neo4jImporter(neo4j_uri, neo4j_user, neo4j_password)

    try:
        # Clear existing data if requested
        if clear_existing:
            importer.clear_database()

        # Create constraints and indexes
        importer.create_constraints()

        # Import nodes
        nodes_imported = importer.import_nodes(nodes)

        # Import edges
        semantic_count, similarity_count = importer.import_edges(edges)

        # Build statistics
        stats = {
            'nodes_imported': nodes_imported,
            'semantic_edges_imported': semantic_count,
            'similarity_edges_imported': similarity_count,
            'total_edges_imported': semantic_count + similarity_count,
            'neo4j_uri': neo4j_uri,
            'graph_file': str(graph_file)
        }

        logger.info("✅ Import complete!")
        return stats

    finally:
        importer.close()


def verify_neo4j_connection(
    neo4j_password: str,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j"
) -> bool:
    """
    Verify Neo4j connection

    Returns: True if connection successful
    """
    if GraphDatabase is None:
        return False

    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False
