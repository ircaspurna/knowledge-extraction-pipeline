#!/usr/bin/env python3
"""
Build Topic Graph - Automate phases 5-6-9 for a topic

Combines concepts from all papers in a topic directory,
resolves entities, builds graph, and visualizes.

Usage:
    python3 build_topic_graph.py /path/to/topic_directory/
    python3 build_topic_graph.py /path/to/topic_directory/ --title "Deception Detection"
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

# Set up logging
logger = logging.getLogger(__name__)
import argparse


def build_topic_graph(
    topic_dir: Path,
    title: str | None = None,
    semantic_threshold: float = 0.90,
    node_size_by: str = 'centrality'
) -> None:
    """
    Build knowledge graph from all papers in topic directory

    Args:
        topic_dir: Path to topic directory containing paper subdirectories
        title: Graph title (defaults to topic directory name)
        semantic_threshold: Threshold for entity resolution (0.90 = within topic)
        node_size_by: How to size nodes ('centrality', 'evidence', 'fixed')
    """
    # Import pipeline modules from installed package
    from knowledge_extraction.core import GraphBuilder
    from knowledge_extraction.extraction import EntityResolverMCP

    # Try to import graph_visualizer (optional visualization feature)
    try:
        from knowledge_extraction.visualization import GraphVisualizer
    except ImportError:
        logger.error("❌ Error: Could not import graph_visualizer")
        logger.info("   Visualization requires the 'viz' extras: pip install knowledge-extraction-pipeline[viz]")
        sys.exit(1)

    topic_dir = Path(topic_dir).resolve()

    if not topic_dir.exists():
        logger.error(f"❌ Error: Directory not found: {topic_dir}")
        sys.exit(1)

    logger.info(f"📂 Building topic graph for: {topic_dir.name}")
    logger.info("=" * 60)

    # Step 1: Combine all concepts from topic
    logger.info("\n1️⃣  Combining concepts from all papers...")
    all_concepts = []
    paper_count = 0

    for paper_dir in topic_dir.iterdir():
        if not paper_dir.is_dir():
            continue

        # Prefer enriched concepts if available
        concepts_file = paper_dir / 'concepts_ENRICHED.json'
        if not concepts_file.exists():
            concepts_file = paper_dir / 'concepts.json'

        if concepts_file.exists():
            with open(concepts_file) as f:
                data = json.load(f)
                # Handle both formats: list of concepts or dict with 'concepts' key
                if isinstance(data, list):
                    concepts = data
                elif isinstance(data, dict) and 'concepts' in data:
                    concepts = data['concepts']
                else:
                    logger.warning(f"   ⚠️ {paper_dir.name}: Unknown format, skipping")
                    continue

                all_concepts.extend(concepts)
                paper_count += 1
                logger.info(f"   ✓ {paper_dir.name}: {len(concepts)} concepts")

    if paper_count == 0:
        logger.error("❌ Error: No concepts.json files found in subdirectories")
        logger.info("   Make sure you've run phase 4 (concept extraction) first!")
        sys.exit(1)

    logger.info(f"\n   📊 Total: {len(all_concepts)} concepts from {paper_count} papers")

    # Save combined concepts
    all_concepts_file = topic_dir / 'all_concepts.json'
    with open(all_concepts_file, 'w') as f:
        json.dump(all_concepts, f, indent=2)
    logger.info(f"   💾 Saved: {all_concepts_file}")

    # Step 2: Resolve entities
    logger.info("\n2️⃣  Resolving entities...")
    logger.info(f"   Using semantic threshold: {semantic_threshold}")

    resolver = EntityResolverMCP()
    entities, ambiguous_pairs = resolver.resolve_entities_automatic(all_concepts)

    logger.info(f"   ✓ Resolved to {len(entities)} unique entities")
    if ambiguous_pairs:
        logger.warning(f"   ⚠️  {len(ambiguous_pairs)} ambiguous pairs (similarity 0.85-{semantic_threshold})")
        logger.info("      Review manually or let Claude Code decide")

    # Save entities
    entities_file = topic_dir / 'entities.json'
    with open(entities_file, 'w') as f:
        # Convert ResolvedEntity objects to dicts
        entities_dicts: list[dict[str, Any]] = [e.to_dict() for e in entities]
        json.dump(entities_dicts, f, indent=2)
    logger.info(f"   💾 Saved: {entities_file}")

    # Step 3: Build graph
    logger.info("\n3️⃣  Building knowledge graph...")

    builder = GraphBuilder()
    # Build graph without relationships (entities only)
    graph = builder.build_graph(entities_dicts, relationships=[])

    logger.info(f"   ✓ Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")

    # Get stats
    stats = builder.get_stats()
    logger.info("\n   📈 Statistics:")
    logger.info(f"      Average degree: {stats['avg_degree']:.2f}")
    logger.info(f"      Density: {stats['density']:.4f}")
    logger.info(f"      Connected components: {stats['connected_components']}")

    # Export graph
    graph_json = topic_dir / 'knowledge_graph.json'
    graph_graphml = topic_dir / 'knowledge_graph.graphml'

    builder.export_json(graph_json)
    builder.export_graphml(graph_graphml)

    logger.info("\n   💾 Saved:")
    logger.info(f"      JSON: {graph_json}")
    logger.info(f"      GraphML: {graph_graphml}")

    # Step 4: Visualize
    logger.info("\n4️⃣  Creating visualizations...")

    if title is None:
        title = f"{topic_dir.name.replace('_', ' ').title()} Knowledge Graph"

    viz = GraphVisualizer(graph)

    # Cytoscape (interactive)
    cytoscape_file = topic_dir / 'knowledge_graph_cytoscape.html'
    viz.render_cytoscape(
        cytoscape_file,
        title=title,
        node_size_by=node_size_by
    )

    # Matplotlib (static, if <500 nodes)
    if len(graph.nodes()) < 500:
        matplotlib_file = topic_dir / 'knowledge_graph.png'
        viz.render_matplotlib(
            matplotlib_file,
            title=title,
            node_size_by=node_size_by,
            figsize=(20, 16),
            dpi=300
        )

    # Summary report
    summary_file = topic_dir / 'knowledge_graph_summary.png'
    viz.generate_summary_report(summary_file)

    logger.info("\n✅ Done!")
    logger.info("=" * 60)
    logger.info("\n📊 Open visualization:")
    logger.info(f"   {cytoscape_file}")
    logger.info("\n🔍 For large graphs (>1000 nodes), use Gephi:")
    logger.info(f"   Open {graph_graphml} in Gephi")

    # Show top concepts
    logger.info("\n🏆 Top 10 concepts by centrality:")
    top_concepts = builder.get_top_concepts(10)
    for i, (concept, score) in enumerate(top_concepts, 1):
        logger.info(f"   {i:2d}. {concept} ({score:.4f})")


def main() -> int:
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    parser = argparse.ArgumentParser(
        description='Build topic graph from processed papers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build graph for deception detection papers
  python3 build_topic_graph.py /Users/IRI/Knowledge\\ Base/PIPELINE_OUTPUT/deception_detection/

  # With custom title
  python3 build_topic_graph.py deception_detection/ --title "Deception Detection Literature"

  # Higher threshold for cross-domain topics
  python3 build_topic_graph.py mixed_topics/ --threshold 0.92

Directory structure expected:
  topic_directory/
  ├── Paper_1/
  │   └── concepts.json
  ├── Paper_2/
  │   └── concepts.json
  └── ...
        """
    )

    parser.add_argument(
        'topic_dir',
        type=str,
        help='Path to topic directory containing paper subdirectories'
    )

    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='Graph title (defaults to directory name)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.90,
        help='Semantic similarity threshold for entity resolution (default: 0.90)'
    )

    parser.add_argument(
        '--node-size-by',
        type=str,
        default='centrality',
        choices=['centrality', 'evidence', 'fixed'],
        help='How to size nodes (default: centrality)'
    )

    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Validate topic_dir
        topic_dir = Path(args.topic_dir)
        if not topic_dir.exists():
            logger.error(f"Directory not found: {topic_dir}")
            return 1

        if not topic_dir.is_dir():
            logger.error(f"Expected directory, got file: {topic_dir}")
            return 1

        # Validate threshold
        if not 0 <= args.threshold <= 1:
            logger.error(f"Threshold must be between 0 and 1, got {args.threshold}")
            return 1

        build_topic_graph(
            topic_dir,
            title=args.title,
            semantic_threshold=args.threshold,
            node_size_by=args.node_size_by
        )
        return 0

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.debug("Full traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
