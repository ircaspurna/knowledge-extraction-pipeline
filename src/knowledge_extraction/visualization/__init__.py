"""Graph visualization modules."""

try:
    from knowledge_extraction.visualization.graph_viz import (
        UltraFastGraphVisualizer as GraphVisualizer,
    )
    __all__ = ["GraphVisualizer"]
except ImportError:
    # Visualization dependencies not installed
    __all__ = []
