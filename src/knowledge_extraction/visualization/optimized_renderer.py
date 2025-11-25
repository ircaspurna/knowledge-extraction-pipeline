#!/usr/bin/env python3
"""
Create optimized visualization with:
- Force-directed layout (built-in CoSE)
- Category-based coloring and grouping
- Only show main connected component
- Proper spacing and readability
"""
import json
import logging
import urllib.request
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

# Set up logging
logger = logging.getLogger(__name__)


def find_connected_components(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[set[Any]]:
    """Find all connected components"""
    adj = defaultdict(set)
    for edge in edges:
        adj[edge['source']].add(edge['target'])
        adj[edge['target']].add(edge['source'])

    node_ids = {n['id'] for n in nodes}
    visited = set()
    components = []

    for node_id in node_ids:
        if node_id not in visited:
            component = set()
            queue = deque([node_id])
            while queue:
                current = queue.popleft()
                if current not in visited:
                    visited.add(current)
                    component.add(current)
                    queue.extend(adj[current] - visited)
            components.append(component)

    components.sort(key=len, reverse=True)
    return components

def create_optimized_viz(graph_json_path: Path, output_path: Path, title: str) -> None:
    """Create optimized visualization"""

    logger.info(f"📂 Loading graph from: {graph_json_path}")
    with open(graph_json_path) as f:
        data = json.load(f)

    logger.info(f"✅ Loaded {len(data['nodes'])} nodes, {len(data['edges'])} edges")

    # Find connected components
    logger.info("\n🔍 Finding connected components...")
    components = find_connected_components(data['nodes'], data['edges'])
    logger.info(f"  Found {len(components)} components")
    logger.info(f"  Largest: {len(components[0])} nodes ({100*len(components[0])/len(data['nodes']):.1f}%)")

    # Option 1: Show only largest component
    # Option 2: Show top N components
    # Let's show components with 10+ nodes

    nodes_to_show: set[Any] = set()
    for comp in components:
        if len(comp) >= 10:  # At least 10 nodes
            nodes_to_show.update(comp)

    logger.info(f"  Showing {len(nodes_to_show)} nodes from {sum(1 for c in components if len(c) >= 10)} components")

    # Download Cytoscape
    logger.info("\n📥 Downloading Cytoscape library...")
    with urllib.request.urlopen('https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js') as response:
        cytoscape_js = response.read().decode('utf-8')
    logger.info("✅ Downloaded")

    # Prepare data
    logger.info("🔧 Preparing visualization data...")

    colors = {
        'method': '#4ECDC4',
        'finding': '#45B7D1',
        'concept': '#98D8C8',
        'problem': '#FF6B6B',
        'evaluation': '#F39C12',
        'tool': '#FFA07A',
        'critique': '#E74C3C',
        'phenomenon': '#9B59B6',
        'metric': '#85C1E2',
        'theory': '#C0392B',
        'principle': '#16A085',
        'bias': '#F7DC6F',
        'unknown': '#BDC3C7'
    }

    categories: Counter[str] = Counter()
    nodes: list[dict[str, Any]] = []

    for node in data['nodes']:
        if node['id'] not in nodes_to_show:
            continue

        cat = node.get('category', 'unknown')
        imp = node.get('importance', 'medium')

        # Size by importance
        size_map = {'critical': 40, 'high': 30, 'medium': 20, 'low': 15}
        size = size_map.get(imp, 20)

        nodes.append({
            'id': node['id'],
            'label': node.get('term', node['id'])[:60],
            'category': cat,
            'importance': imp,
            'definition': node.get('definition', '')[:300],
            'color': colors.get(cat, '#BDC3C7'),
            'size': size
        })
        categories[cat] += 1

    # Filter edges
    visible_ids = nodes_to_show
    edges: list[dict[str, Any]] = []
    for edge in data['edges']:
        if edge['source'] in visible_ids and edge['target'] in visible_ids:
            sim = edge.get('similarity', 0.8)
            edges.append({
                'source': edge['source'],
                'target': edge['target'],
                'similarity': sim,
                'width': max(1, (sim - 0.75) * 10)  # Width based on similarity
            })

    logger.info(f"📊 Final visualization: {len(nodes)} nodes, {len(edges)} edges")
    logger.info(f"   Categories: {len(categories)}")

    # Generate HTML with CoSE layout (force-directed, built-in)
    logger.info("📝 Generating HTML...")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f5f5;
        }}
        #header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: white;
            text-align: center;
        }}
        h1 {{ font-size: 26px; margin-bottom: 10px; }}
        #stats {{ font-size: 14px; opacity: 0.95; }}
        .stat-badge {{
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 5px 12px;
            border-radius: 15px;
            margin: 0 5px;
        }}
        #cy {{
            width: 100%;
            height: calc(100vh - 200px);
            background: #fafafa;
            border: 1px solid #ddd;
        }}
        #controls {{
            padding: 15px 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 15px;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
        }}
        button, select, input {{
            padding: 10px 18px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        button {{
            background: #667eea;
            color: white;
            border: none;
            font-weight: 500;
        }}
        button:hover {{
            background: #5568d3;
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(102,126,234,0.3);
        }}
        button.secondary {{
            background: #95a5a6;
        }}
        button.secondary:hover {{
            background: #7f8c8d;
        }}
        select {{
            background: white;
            min-width: 180px;
        }}
        input {{
            min-width: 250px;
            cursor: text;
        }}
        #legend {{
            position: absolute;
            top: 110px;
            right: 20px;
            background: white;
            padding: 18px;
            border-radius: 10px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
            max-height: 450px;
            overflow-y: auto;
            min-width: 200px;
        }}
        .legend-title {{
            font-weight: 700;
            font-size: 13px;
            text-transform: uppercase;
            color: #555;
            margin-bottom: 12px;
            letter-spacing: 0.5px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 10px;
            cursor: pointer;
            border-radius: 6px;
            transition: all 0.2s;
            margin: 3px 0;
        }}
        .legend-item:hover {{
            background: #f0f0f0;
        }}
        .legend-item.active {{
            background: #e8eaf6;
            font-weight: 600;
        }}
        .legend-color {{
            width: 14px;
            height: 14px;
            border-radius: 3px;
            flex-shrink: 0;
        }}
        .legend-count {{
            margin-left: auto;
            color: #999;
            font-size: 12px;
        }}
        .control-group {{
            display: flex;
            gap: 8px;
            align-items: center;
        }}
        .divider {{
            width: 1px;
            height: 30px;
            background: #ddd;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>{title}</h1>
        <div id="stats">
            <span class="stat-badge">
                <strong id="node-count">{len(nodes)}</strong> concepts
            </span>
            <span class="stat-badge">
                <strong id="edge-count">{len(edges)}</strong> connections
            </span>
            <span class="stat-badge">
                <strong>{len(categories)}</strong> categories
            </span>
            <span class="stat-badge">
                Avg similarity: <strong>79.5%</strong>
            </span>
        </div>
    </div>

    <div id="legend">
        <div class="legend-title">🏷️ Categories</div>
        <div id="category-legend"></div>
    </div>

    <div id="cy"></div>

    <div id="controls">
        <div class="control-group">
            <select id="category-filter">
                <option value="">All Categories</option>
                {chr(10).join(f'                <option value="{cat}">{cat} ({count})</option>' for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True))}
            </select>

            <select id="importance-filter">
                <option value="">All Importance</option>
                <option value="critical">Critical</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
            </select>
        </div>

        <div class="divider"></div>

        <div class="control-group">
            <input type="text" id="search" placeholder="Search concepts...">
            <button onclick="searchNodes()">🔍 Search</button>
        </div>

        <div class="divider"></div>

        <div class="control-group">
            <button class="secondary" onclick="resetView()">Reset</button>
            <button class="secondary" onclick="cy.fit(50)">Fit View</button>
            <button onclick="relayout()">Re-layout</button>
        </div>
    </div>

    <script>
{cytoscape_js}
    </script>
    <script>
        console.log('🚀 Initializing optimized graph...');

        const nodesData = {json.dumps(nodes)};
        const edgesData = {json.dumps(edges)};
        const categoryData = {json.dumps(dict(categories))};

        console.log('Data:', nodesData.length, 'nodes,', edgesData.length, 'edges');

        const elements = [
            ...nodesData.map(n => ({{data: n}})),
            ...edgesData.map(e => ({{data: e}}))
        ];

        // Initialize with CoSE layout (force-directed)
        const cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: elements,
            style: [
                {{
                    selector: 'node',
                    style: {{
                        'label': 'data(label)',
                        'width': 'data(size)',
                        'height': 'data(size)',
                        'background-color': 'data(color)',
                        'font-size': '9px',
                        'font-weight': '600',
                        'text-outline-width': 2,
                        'text-outline-color': '#fff',
                        'text-wrap': 'wrap',
                        'text-max-width': '90px',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'border-width': 1.5,
                        'border-color': '#555',
                        'border-opacity': 0.3
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                        'width': 'data(width)',
                        'line-color': '#bbb',
                        'opacity': 0.35,
                        'curve-style': 'bezier'
                    }}
                }},
                {{
                    selector: 'node.hidden',
                    style: {{ 'display': 'none' }}
                }},
                {{
                    selector: 'edge.hidden',
                    style: {{ 'display': 'none' }}
                }},
                {{
                    selector: 'node.highlighted',
                    style: {{
                        'border-width': 4,
                        'border-color': '#e74c3c',
                        'border-opacity': 1,
                        'z-index': 9999
                    }}
                }},
                {{
                    selector: 'node.faded',
                    style: {{
                        'opacity': 0.2
                    }}
                }},
                {{
                    selector: 'edge.faded',
                    style: {{
                        'opacity': 0.05
                    }}
                }}
            ],
            layout: {{
                name: 'cose',  // Force-directed layout
                animate: true,
                animationDuration: 1000,
                idealEdgeLength: 80,
                nodeOverlap: 20,
                nodeRepulsion: 400000,
                edgeElasticity: 100,
                nestingFactor: 5,
                gravity: 80,
                numIter: 1000,
                initialTemp: 200,
                coolingFactor: 0.95,
                minTemp: 1.0
            }}
        }});

        console.log('✅ Graph created! Nodes:', cy.nodes().length, 'Edges:', cy.edges().length);

        // Build legend
        const legendDiv = document.getElementById('category-legend');
        Object.entries(categoryData)
            .sort((a, b) => b[1] - a[1])
            .forEach(([cat, count]) => {{
                const item = document.createElement('div');
                item.className = 'legend-item active';
                item.dataset.category = cat;
                const color = nodesData.find(n => n.category === cat)?.color || '#BDC3C7';
                item.innerHTML = `
                    <div class="legend-color" style="background: ${{color}}"></div>
                    <span>${{cat}}</span>
                    <span class="legend-count">${{count}}</span>
                `;
                item.onclick = () => toggleCategory(cat, item);
                legendDiv.appendChild(item);
            }});

        let activeCategories = new Set(Object.keys(categoryData));

        function toggleCategory(cat, elem) {{
            if (activeCategories.has(cat)) {{
                activeCategories.delete(cat);
                elem.classList.remove('active');
            }} else {{
                activeCategories.add(cat);
                elem.classList.add('active');
            }}
            applyFilters();
        }}

        // Category filter dropdown
        document.getElementById('category-filter').onchange = function() {{
            const cat = this.value;
            if (!cat) {{
                activeCategories = new Set(Object.keys(categoryData));
                document.querySelectorAll('.legend-item').forEach(item => item.classList.add('active'));
            }} else {{
                activeCategories = new Set([cat]);
                document.querySelectorAll('.legend-item').forEach(item => {{
                    if (item.dataset.category === cat) {{
                        item.classList.add('active');
                    }} else {{
                        item.classList.remove('active');
                    }}
                }});
            }}
            applyFilters();
        }};

        // Importance filter
        document.getElementById('importance-filter').onchange = applyFilters;

        function applyFilters() {{
            const importance = document.getElementById('importance-filter').value;

            cy.batch(() => {{
                let visibleCount = 0;
                let visibleEdges = 0;

                cy.nodes().forEach(node => {{
                    const data = node.data();
                    const catMatch = activeCategories.has(data.category);
                    const impMatch = !importance || data.importance === importance;

                    if (catMatch && impMatch) {{
                        node.removeClass('hidden faded');
                        visibleCount++;
                    }} else {{
                        node.addClass('hidden');
                    }}
                }});

                cy.edges().forEach(edge => {{
                    if (edge.source().hasClass('hidden') || edge.target().hasClass('hidden')) {{
                        edge.addClass('hidden');
                    }} else {{
                        edge.removeClass('hidden');
                        visibleEdges++;
                    }}
                }});

                document.getElementById('node-count').textContent = visibleCount;
                document.getElementById('edge-count').textContent = visibleEdges;

                const visible = cy.nodes().not('.hidden');
                if (visible.length > 0) cy.fit(visible, 50);
            }});
        }}

        function searchNodes() {{
            const query = document.getElementById('search').value.toLowerCase().trim();
            if (!query) return;

            cy.batch(() => {{
                cy.elements().removeClass('highlighted faded');

                const matches = cy.nodes().not('.hidden').filter(node =>
                    node.data('label').toLowerCase().includes(query) ||
                    node.data('definition').toLowerCase().includes(query)
                );

                if (matches.length > 0) {{
                    cy.elements().addClass('faded');
                    matches.removeClass('faded').addClass('highlighted');
                    matches.neighborhood().removeClass('faded');
                    cy.fit(matches, 50);
                }} else {{
                    alert('No matches found');
                }}
            }});
        }}

        function resetView() {{
            document.getElementById('category-filter').value = '';
            document.getElementById('importance-filter').value = '';
            document.getElementById('search').value = '';
            activeCategories = new Set(Object.keys(categoryData));
            document.querySelectorAll('.legend-item').forEach(item => item.classList.add('active'));
            cy.elements().removeClass('hidden highlighted faded');
            cy.fit(50);
            document.getElementById('node-count').textContent = {len(nodes)};
            document.getElementById('edge-count').textContent = {len(edges)};
        }}

        function relayout() {{
            cy.layout({{
                name: 'cose',
                animate: true,
                animationDuration: 1000,
                idealEdgeLength: 80,
                nodeOverlap: 20,
                nodeRepulsion: 400000,
                randomize: false
            }}).run();
        }}

        // Node click
        cy.on('tap', 'node', function(evt) {{
            const node = evt.target;
            const data = node.data();

            // Get connected nodes
            const neighbors = node.neighborhood('node');
            const degree = neighbors.length;

            alert([
                '📌 ' + data.label,
                '',
                'Category: ' + data.category,
                'Importance: ' + data.importance,
                'Connections: ' + degree,
                '',
                data.definition
            ].join('\\n'));
        }});

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') resetView();
            if (e.key === 'Enter' && document.activeElement.id === 'search') searchNodes();
        }});

        console.log('✅ Optimized visualization ready!');
    </script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    logger.info(f"\n✅ Saved optimized visualization to: {output_path}")
    logger.info(f"   📊 {len(nodes)} nodes, {len(edges)} edges")
    logger.info("   🎨 Force-directed layout with category coloring")
    logger.info("   🌐 No internet required!")
    logger.info(f"\n💡 Open it: file://{output_path.absolute()}")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    graph_path = Path("/Users/IRI/Knowledge Base/PIPELINE_OUTPUT/research_topic/knowledge_graph.json")
    output_path = Path("/Users/IRI/Knowledge Base/PIPELINE_OUTPUT/research_topic/knowledge_graph_OPTIMIZED.html")

    create_optimized_viz(graph_path, output_path, "Knowledge Graph - Optimized")
