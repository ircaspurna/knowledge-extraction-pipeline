#!/usr/bin/env python3
"""
Graph Builder - Construct knowledge graph from entities and relationships

Creates NetworkX graph with:
- Nodes: Entities with attributes
- Edges: Typed relationships
- Metrics: Centrality, communities, etc.
"""

import json
import logging
from pathlib import Path
from typing import Any

# Set up logging
logger = logging.getLogger(__name__)

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    logger.warning("networkx not installed. Install with: pip install networkx")
    nx = None
    NX_AVAILABLE = False


class GraphBuilder:
    """Build and analyze knowledge graph"""

    def __init__(self) -> None:
        """Initialize graph builder"""
        self.graph: Any = None  # Will be nx.DiGraph after build_graph()

        self.stats: dict[str, Any] = {
            'nodes': 0,
            'edges': 0,
            'density': 0.0,
            'avg_degree': 0.0,
            'components': 0
        }

    def build_graph(
        self,
        entities: list[dict[str, Any]],
        relationships: list[dict[str, Any]]
    ) -> Any:  # Returns nx.DiGraph but nx might be None
        """
        Build directed graph from entities and relationships.

        Args:
            entities: List of entity dicts
            relationships: List of relationship dicts

        Returns:
            NetworkX DiGraph
        """
        if not nx:
            raise ImportError("networkx not installed")

        # Validate inputs
        if not isinstance(entities, list):
            raise ValueError(f"entities must be a list, got {type(entities).__name__}")
        if not isinstance(relationships, list):
            raise ValueError(f"relationships must be a list, got {type(relationships).__name__}")

        # Validate entity structure
        for i, entity in enumerate(entities):
            if not isinstance(entity, dict):
                raise ValueError(f"Entity {i} must be a dict, got {type(entity).__name__}")
            if 'canonical_term' not in entity:
                raise ValueError(f"Entity {i} missing required field 'canonical_term'")
            if not entity['canonical_term']:
                raise ValueError(f"Entity {i} has empty 'canonical_term'")

        # Validate relationship structure
        for i, rel in enumerate(relationships):
            if not isinstance(rel, dict):
                raise ValueError(f"Relationship {i} must be a dict, got {type(rel).__name__}")
            if 'source' not in rel:
                raise ValueError(f"Relationship {i} missing required field 'source'")
            if 'target' not in rel:
                raise ValueError(f"Relationship {i} missing required field 'target'")
            if not rel['source']:
                raise ValueError(f"Relationship {i} has empty 'source'")
            if not rel['target']:
                raise ValueError(f"Relationship {i} has empty 'target'")

        # Create directed graph
        self.graph = nx.DiGraph()

        # Add nodes (entities)
        for entity in entities:
            term = entity['canonical_term']

            # Node attributes
            attrs = {
                'term': term,
                'aliases': entity.get('aliases', []),
                'definitions': entity.get('definitions', []),
                'primary_category': entity.get('primary_category', 'unknown'),
                'primary_importance': entity.get('primary_importance', 'medium'),
                'avg_confidence': entity.get('avg_confidence', 0.0),
                'evidence_count': len(entity.get('evidence', [])),
                'sources': entity.get('sources', [])
            }

            self.graph.add_node(term, **attrs)

        # Add edges (relationships)
        for rel in relationships:
            source = rel['source']
            target = rel['target']

            # Only add if both nodes exist
            if source in self.graph and target in self.graph:
                # Edge attributes
                attrs = {
                    'type': rel.get('type', 'RELATED'),
                    'strength': rel.get('strength', 0.5),
                    'confidence': rel.get('confidence', 0.5),
                    'explanation': rel.get('explanation', ''),
                    'evidence': rel.get('evidence', [])
                }

                self.graph.add_edge(source, target, **attrs)

        # Calculate stats
        self._calculate_stats()

        logger.info(f"Built knowledge graph: {self.stats['nodes']} nodes, {self.stats['edges']} edges")
        logger.debug(f"Graph density: {self.stats['density']:.3f}, avg degree: {self.stats['avg_degree']:.2f}")

        return self.graph

    def _calculate_stats(self) -> None:
        """Calculate graph statistics"""
        if not self.graph:
            return

        self.stats['nodes'] = self.graph.number_of_nodes()
        self.stats['edges'] = self.graph.number_of_edges()

        # Density
        if self.stats['nodes'] > 1:
            self.stats['density'] = nx.density(self.graph)

        # Average degree
        if self.stats['nodes'] > 0:
            degrees = [d for n, d in self.graph.degree()]
            self.stats['avg_degree'] = sum(degrees) / len(degrees)

        # Connected components
        self.stats['components'] = nx.number_weakly_connected_components(self.graph)

    def get_top_concepts(self, n: int = 10) -> list[tuple[str, float]]:
        """
        Get top N concepts by centrality.
        
        Args:
            n: Number of concepts to return
        
        Returns:
            List of (concept, centrality_score) tuples
        """
        if not self.graph or self.graph.number_of_nodes() == 0:
            return []

        # Calculate PageRank (handles directed graphs well)
        try:
            centrality = nx.pagerank(self.graph)
            top = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:n]
            return top
        except:
            # Fallback to degree centrality
            centrality = nx.degree_centrality(self.graph)
            top = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:n]
            return top

    def get_concept_neighbors(
        self,
        concept: str,
        depth: int = 1
    ) -> list[str]:
        """
        Get neighboring concepts within depth hops.
        
        Args:
            concept: Concept term
            depth: How many hops to traverse
        
        Returns:
            List of neighbor concept terms
        """
        if not self.graph or concept not in self.graph:
            return []

        # Get all nodes within depth hops
        neighbors = set()

        for i in range(1, depth + 1):
            # Successors (outgoing edges)
            for node in nx.single_source_shortest_path_length(self.graph, concept, cutoff=i):
                if node != concept:
                    neighbors.add(node)

            # Predecessors (incoming edges)
            for node in nx.single_source_shortest_path_length(self.graph.reverse(), concept, cutoff=i):
                if node != concept:
                    neighbors.add(node)

        return list(neighbors)

    def get_relationship_path(
        self,
        source: str,
        target: str,
        max_hops: int = 3
    ) -> list[tuple[str, str, dict[str, Any]]] | None:
        """
        Find path between two concepts.
        
        Args:
            source: Source concept
            target: Target concept
            max_hops: Maximum path length
        
        Returns:
            List of (source, target, edge_data) tuples or None
        """
        if not self.graph or source not in self.graph or target not in self.graph:
            return None

        try:
            path = nx.shortest_path(self.graph, source, target)

            if len(path) - 1 > max_hops:
                return None

            # Get edge data for path
            path_edges = []
            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i+1])
                path_edges.append((path[i], path[i+1], edge_data))

            return path_edges
        except nx.NetworkXNoPath:
            return None

    def export_json(self, output_path: Path) -> None:
        """
        Export graph to JSON format.

        Format:
        {
            'nodes': [...],
            'edges': [...],
            'stats': {...}
        }
        """
        if not self.graph:
            logger.error("No graph to export")
            return

        output_path = Path(output_path)

        # Export nodes
        nodes = []
        for node, attrs in self.graph.nodes(data=True):
            node_data = {'id': node}
            node_data.update(attrs)
            nodes.append(node_data)

        # Export edges
        edges = []
        for source, target, attrs in self.graph.edges(data=True):
            edge_data = {
                'source': source,
                'target': target
            }
            edge_data.update(attrs)
            edges.append(edge_data)

        # Create JSON
        data = {
            'nodes': nodes,
            'edges': edges,
            'stats': self.stats
        }

        output_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        logger.info(f"Exported graph to JSON: {output_path}")

    def export_graphml(self, output_path: Path) -> None:
        """Export graph to GraphML format (for Gephi, Cytoscape)"""
        if not self.graph:
            logger.error("No graph to export")
            return

        output_path = Path(output_path)

        # Create a copy and convert list attributes to strings for GraphML
        g_copy = self.graph.copy()
        for node, attrs in g_copy.nodes(data=True):
            for key, value in list(attrs.items()):
                if isinstance(value, list):
                    # Convert lists to comma-separated strings
                    attrs[key] = ', '.join(str(v) for v in value) if value else ''

        for source, target, attrs in g_copy.edges(data=True):
            for key, value in list(attrs.items()):
                if isinstance(value, list):
                    attrs[key] = ', '.join(str(v) for v in value) if value else ''

        nx.write_graphml(g_copy, str(output_path))
        logger.info(f"Exported graph to GraphML: {output_path}")

    def get_stats(self) -> dict[str, Any]:
        """Get graph statistics"""
        return self.stats


# =========================================================================
# CLI
# =========================================================================

def main() -> None:
    """Command-line interface"""
    import argparse

    # Configure logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    parser = argparse.ArgumentParser(
        description='Build knowledge graph from entities and relationships'
    )
    parser.add_argument('--entities', type=str, required=True,
                       help='Path to entities JSON file')
    parser.add_argument('--relationships', type=str, required=True,
                       help='Path to relationships JSON file')
    parser.add_argument('--output-json', type=str, default='knowledge_graph.json',
                       help='Output JSON file')
    parser.add_argument('--output-graphml', type=str,
                       help='Output GraphML file (optional)')
    parser.add_argument('--top-concepts', type=int, default=10,
                       help='Number of top concepts to show')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load data
    entities_data = json.loads(Path(args.entities).read_text(encoding='utf-8'))
    relationships_data = json.loads(Path(args.relationships).read_text(encoding='utf-8'))

    entities = entities_data.get('entities', [])
    relationships = relationships_data.get('relationships', [])

    print(f"Loaded {len(entities)} entities and {len(relationships)} relationships")

    # Build graph
    builder = GraphBuilder()
    graph = builder.build_graph(entities, relationships)

    # Show top concepts
    print(f"\nTop {args.top_concepts} Concepts (by centrality):")
    top_concepts = builder.get_top_concepts(args.top_concepts)
    for i, (concept, score) in enumerate(top_concepts, 1):
        print(f"  {i}. {concept}: {score:.3f}")

    # Export
    builder.export_json(Path(args.output_json))

    if args.output_graphml:
        builder.export_graphml(Path(args.output_graphml))

    print("\n✓ Knowledge graph complete!")
    print(f"  JSON: {args.output_json}")
    if args.output_graphml:
        print(f"  GraphML: {args.output_graphml}")


if __name__ == "__main__":
    main()
