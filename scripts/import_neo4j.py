#!/usr/bin/env python3
"""
Import knowledge graph into Neo4j
"""

import json
import logging
from pathlib import Path
from typing import Any

# Set up logging
logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase
except ImportError:
    logger.error("Error: neo4j driver not installed")
    logger.info("Install with: pip3 install neo4j")
    exit(1)


class Neo4jImporter:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
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

    def import_nodes(self, nodes: list[dict[str, Any]]) -> None:
        """Import nodes in batches"""
        batch_size = 1000
        total = len(nodes)

        with self.driver.session() as session:
            for i in range(0, total, batch_size):
                batch = nodes[i:i+batch_size]

                session.run("""
                    UNWIND $batch AS node
                    CREATE (c:Concept {
                        term: node.term,
                        category: node.category,
                        importance: node.importance,
                        definition: node.definition,
                        confidence: node.confidence
                    })
                """, batch=batch)

                logger.info(f"  Imported {min(i+batch_size, total)}/{total} nodes...")

        logger.info(f"✓ Imported {total:,} nodes")

    def import_edges(self, edges: list[dict[str, Any]]) -> None:
        """Import edges in batches"""
        batch_size = 1000
        total = len(edges)

        # Group by edge class
        semantic_edges = [e for e in edges if e.get('edge_class') == 'semantic']
        similarity_edges = [e for e in edges if e.get('edge_class') == 'similarity']

        logger.info(f"Importing {len(semantic_edges)} semantic relationships...")
        self._import_semantic_edges(semantic_edges, batch_size)

        logger.info(f"Importing {len(similarity_edges):,} similarity edges...")
        self._import_similarity_edges(similarity_edges, batch_size)

    def _import_semantic_edges(self, edges: list[dict[str, Any]], batch_size: int) -> None:
        """Import semantic relationship edges"""
        with self.driver.session() as session:
            for i in range(0, len(edges), batch_size):
                batch = edges[i:i+batch_size]

                session.run("""
                    UNWIND $batch AS edge
                    MATCH (a:Concept {term: edge.source})
                    MATCH (b:Concept {term: edge.target})
                    CREATE (a)-[r:RELATES_TO {
                        type: edge.relationship_type,
                        weight: edge.weight,
                        direction: edge.direction,
                        explanation: edge.explanation,
                        confidence: edge.confidence
                    }]->(b)
                """, batch=batch)

                logger.info(f"  Imported {min(i+batch_size, len(edges))}/{len(edges)} semantic edges...")

    def _import_similarity_edges(self, edges: list[dict[str, Any]], batch_size: int) -> None:
        """Import similarity edges"""
        with self.driver.session() as session:
            for i in range(0, len(edges), batch_size):
                batch = edges[i:i+batch_size]

                session.run("""
                    UNWIND $batch AS edge
                    MATCH (a:Concept {term: edge.source})
                    MATCH (b:Concept {term: edge.target})
                    CREATE (a)-[r:SIMILAR_TO {
                        similarity: edge.weight
                    }]->(b)
                """, batch=batch)

                if i % 5000 == 0 and i > 0:
                    logger.info(f"  Imported {i:,}/{len(edges):,} similarity edges...")

        logger.info(f"✓ Imported {len(edges):,} similarity edges")

    def create_sample_queries(self, output_path: Path) -> None:
        """Save sample queries for reference"""
        queries = """
# NEO4J SAMPLE QUERIES FOR KNOWLEDGE GRAPH

## 1. Overview Queries

### Count nodes by category
MATCH (c:Concept)
RETURN c.category, count(*) as count
ORDER BY count DESC

### Count relationships by type
MATCH ()-[r:RELATES_TO]->()
RETURN r.type, count(*) as count
ORDER BY count DESC

### Find critical concepts
MATCH (c:Concept {importance: 'critical'})
RETURN c.term, c.category, c.definition
LIMIT 25


## 2. Exploration Queries

### Find all methods
MATCH (c:Concept {category: 'method'})
RETURN c.term, c.definition
LIMIT 50

### Find concepts related to cognitive load
MATCH (c:Concept)
WHERE toLower(c.term) CONTAINS 'cognitive'
RETURN c.term, c.category, c.importance, c.definition
LIMIT 20

### Find all tools and their relationships
MATCH (t:Concept {category: 'tool'})-[r:RELATES_TO]-(other)
RETURN t.term, type(r), r.type, other.term, other.category
LIMIT 30


## 3. Relationship Queries

### Find concepts that ENABLE other concepts
MATCH (a:Concept)-[r:RELATES_TO {type: 'ENABLES'}]->(b:Concept)
RETURN a.term, b.term, r.explanation
LIMIT 20

### Find all relationships for a specific concept
MATCH path = (c:Concept {term: 'Machine Learning'})-[r:RELATES_TO]-(other)
RETURN path
LIMIT 25

### Find paths between two concepts
MATCH path = shortestPath(
  (a:Concept {term: 'LIWC Tool'})-[*..5]-(b:Concept {term: 'Cognitive Load'})
)
RETURN path
LIMIT 5


## 4. Analysis Queries

### Find most connected concepts (by degree)
MATCH (c:Concept)-[r]-()
WITH c, count(r) as connections
WHERE connections > 5
RETURN c.term, c.category, c.importance, connections
ORDER BY connections DESC
LIMIT 30

### Find concepts with both semantic and similarity connections
MATCH (c:Concept)-[r1:RELATES_TO]-()
MATCH (c)-[r2:SIMILAR_TO]-()
WITH c, count(distinct r1) as semantic, count(distinct r2) as similar
RETURN c.term, c.category, semantic, similar
ORDER BY semantic DESC
LIMIT 20

### Find concept clusters (community detection preview)
MATCH (c:Concept {category: 'method'})-[r:RELATES_TO]-(other:Concept)
WITH c, collect(distinct other.term) as related_concepts, count(r) as rel_count
WHERE rel_count > 2
RETURN c.term, related_concepts, rel_count
ORDER BY rel_count DESC
LIMIT 15


## 5. Domain-Specific Queries

### Find analytical methods
MATCH (m:Concept {category: 'method'})
WHERE toLower(m.term) CONTAINS 'analysis' OR toLower(m.definition) CONTAINS 'methodology'
RETURN m.term, m.importance, m.definition
LIMIT 20

### Find problems and their connected solutions
MATCH (p:Concept {category: 'problem'})-[r:RELATES_TO]-(s:Concept {category: 'method'})
RETURN p.term as problem, s.term as potential_solution, r.explanation
LIMIT 15

### Find findings about specific topics
MATCH (f:Concept {category: 'finding'})
WHERE toLower(f.term) CONTAINS 'accuracy' OR toLower(f.term) CONTAINS 'detection'
RETURN f.term, f.importance, f.definition
LIMIT 20


## 6. Visualization Queries

### Subgraph around a concept (2-hop neighborhood)
MATCH path = (c:Concept {term: 'Episodic Memory'})-[*..2]-(other)
RETURN path
LIMIT 50

### Critical concepts and their immediate connections
MATCH path = (c:Concept {importance: 'critical'})-[r:RELATES_TO]-(other)
RETURN path
LIMIT 100

### Methods and tools network
MATCH path = (m:Concept {category: 'method'})-[r:RELATES_TO]-(t:Concept {category: 'tool'})
RETURN path
LIMIT 50


## 7. Advanced Pattern Queries

### Find concepts that REQUIRE something and what they ENABLE
MATCH (a)-[r1:RELATES_TO {type: 'REQUIRES'}]->(req)
MATCH (a)-[r2:RELATES_TO {type: 'ENABLES'}]->(enabled)
RETURN a.term as concept, req.term as requires, enabled.term as enables
LIMIT 10

### Find contradictory concepts
MATCH (a:Concept)-[r:RELATES_TO {type: 'CONTRADICTS'}]-(b:Concept)
RETURN a.term, b.term, r.explanation
LIMIT 10

### Find theoretical frameworks and their components
MATCH (theory:Concept {category: 'theory'})<-[r:RELATES_TO {type: 'PART_OF'}]-(component)
RETURN theory.term, collect(component.term) as components
"""

        with open(output_path, 'w') as f:
            f.write(queries)
        logger.info(f"\n✓ Saved sample queries to: {output_path}")


def main() -> int:
    import argparse
    import getpass

    parser = argparse.ArgumentParser(
        description='Import knowledge graph to Neo4j database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import with default settings
  python3 import_to_neo4j.py knowledge_graph_FINAL.json

  # Specify output directory for sample queries
  python3 import_to_neo4j.py graph.json --queries-output ./NEO4J_QUERIES.md

  # Custom Neo4j URI
  python3 import_to_neo4j.py graph.json --uri bolt://localhost:7687
        """
    )

    parser.add_argument(
        'graph_file',
        type=str,
        help='Path to knowledge_graph JSON file (e.g., knowledge_graph_FINAL.json)'
    )
    parser.add_argument(
        '--uri',
        type=str,
        default='bolt://localhost:7687',
        help='Neo4j URI (default: bolt://localhost:7687)'
    )
    parser.add_argument(
        '--user',
        type=str,
        default='neo4j',
        help='Neo4j username (default: neo4j)'
    )
    parser.add_argument(
        '--queries-output',
        type=str,
        default=None,
        help='Where to save sample queries (default: same directory as graph_file)'
    )

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 80)
    logger.info("NEO4J IMPORT - KNOWLEDGE GRAPH")
    logger.info("=" * 80)
    logger.info("")

    # Configuration
    NEO4J_URI = args.uri
    NEO4J_USER = args.user
    NEO4J_PASSWORD = getpass.getpass("Enter Neo4j password: ")

    # Load graph data
    graph_file = Path(args.graph_file)

    if not graph_file.exists():
        logger.error(f"Graph file not found: {graph_file}")
        logger.info("Please provide a valid path to knowledge_graph_FINAL.json")
        return 1

    logger.info(f"\nLoading graph from: {graph_file}")

    try:
        with open(graph_file) as f:
            graph_data = json.load(f)

        if 'nodes' not in graph_data or 'edges' not in graph_data:
            logger.error("❌ Error: Invalid graph file format")
            logger.info("Expected JSON with 'nodes' and 'edges' keys")
            return 1

        nodes = graph_data['nodes']
        edges = graph_data['edges']

        logger.info(f"✓ Loaded {len(nodes):,} nodes and {len(edges):,} edges")
        logger.info("")
    except json.JSONDecodeError as e:
        logger.error(f"❌ Error: Invalid JSON file: {e}")
        return 1
    except Exception as e:
        logger.error(f"❌ Error loading graph file: {e}")
        return 1

    # Connect to Neo4j
    logger.info(f"Connecting to Neo4j at {NEO4J_URI}...")
    try:
        importer = Neo4jImporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        logger.info("✓ Connected")
    except Exception as e:
        logger.info(f"✗ Connection failed: {e}")
        logger.info("\nMake sure Neo4j is running and credentials are correct.")
        logger.info("Start Neo4j Desktop or run: neo4j start")
        return 1

    logger.info("")
    logger.info("=" * 80)
    logger.info("IMPORT PROCESS")
    logger.info("=" * 80)
    logger.info("")

    # Clear existing data
    response = input("Clear existing data in Neo4j? (y/N): ").strip().lower()
    if response == 'y':
        logger.info("Clearing database...")
        importer.clear_database()
        logger.info("")

    # Create constraints and indexes
    logger.info("Creating constraints and indexes...")
    importer.create_constraints()
    logger.info("")

    # Import nodes
    logger.info(f"Importing {len(nodes):,} nodes...")
    importer.import_nodes(nodes)
    logger.info("")

    # Import edges
    logger.info(f"Importing {len(edges):,} edges...")
    importer.import_edges(edges)
    logger.info("")

    # Save sample queries
    if args.queries_output:
        queries_file = Path(args.queries_output)
    else:
        # Default: same directory as graph file
        queries_file = graph_file.parent / 'NEO4J_SAMPLE_QUERIES.md'

    importer.create_sample_queries(queries_file)

    # Close connection
    importer.close()

    logger.info("")
    logger.info("=" * 80)
    logger.info("IMPORT COMPLETE ✓")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Successfully imported:")
    logger.info(f"  - {len(nodes):,} nodes (Concept)")
    logger.info(f"  - {len(edges):,} edges (RELATES_TO, SIMILAR_TO)")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Open Neo4j Browser: http://localhost:7474")
    logger.info("  2. Run sample queries from: NEO4J_SAMPLE_QUERIES.md")
    logger.info("  3. Explore with Neo4j Bloom (if available)")
    logger.info("")

    return 0


if __name__ == '__main__':
    exit(main())
