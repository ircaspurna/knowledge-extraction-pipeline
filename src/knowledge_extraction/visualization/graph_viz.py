#!/usr/bin/env python3
"""
Ultra-Fast Graph Visualizer - Maximum Performance with Advanced Filtering

New Features:
- Multi-category filtering (select multiple categories at once)
- Importance level filtering
- Evidence count filtering
- Interactive legend (click to filter)
- Live node count updates
- Performance optimizations for 10K+ nodes
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx  # type: ignore[import-untyped]


class UltraFastGraphVisualizer:
    """Ultra-high-performance visualization with advanced filtering"""

    CATEGORY_COLORS = {
        'method': '#4ECDC4',      # Teal
        'finding': '#45B7D1',     # Blue
        'concept': '#98D8C8',     # Green
        'problem': '#FF6B6B',     # Red
        'evaluation': '#F39C12',  # Orange
        'tool': '#FFA07A',        # Light Orange
        'critique': '#E74C3C',    # Dark Red
        'phenomenon': '#9B59B6',  # Purple
        'metric': '#85C1E2',      # Light Blue
        'theory': '#C0392B',      # Dark Red
        'principle': '#16A085',   # Teal-Green
        'bias': '#F7DC6F',        # Yellow
        'general': '#95A5A6',     # Gray
        'unknown': '#BDC3C7'      # Light Gray
    }

    IMPORTANCE_LEVELS = ['high', 'medium', 'low']

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    @classmethod
    def from_json(cls, json_path: Path) -> 'UltraFastGraphVisualizer':
        """Load graph from JSON"""
        print(f"📂 Loading graph from: {json_path}")
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)

        G = nx.DiGraph()

        for node in data['nodes']:
            node_id = node['id']
            G.add_node(node_id, **{k: v for k, v in node.items() if k != 'id'})

        for edge in data['edges']:
            G.add_edge(edge['source'], edge['target'],
                      **{k: v for k, v in edge.items() if k not in ['source', 'target']})

        print(f"✅ Loaded {len(G.nodes())} nodes and {len(G.edges())} edges")
        return cls(G)

    def render_ultra_fast(
        self,
        output_path: Path,
        title: str = "Knowledge Graph",
        max_nodes: int = 10000
    ) -> None:
        """
        Render ultra-optimized visualization with advanced filtering
        """
        print("\n🚀 Generating ultra-fast visualization...")

        # Calculate centrality for node importance
        print("  📊 Computing node centrality...")
        centrality = nx.pagerank(self.graph)

        # Prepare node data
        print("  🔧 Preparing node data...")
        nodes_data = []
        category_counts: Counter[str] = Counter()
        importance_counts: Counter[str] = Counter()

        for node_id in self.graph.nodes():
            attrs = self.graph.nodes[node_id]
            category = attrs.get('category', attrs.get('primary_category', 'unknown'))
            importance = attrs.get('importance', attrs.get('primary_importance', 'medium'))
            evidence_count = attrs.get('evidence_count', 0)
            confidence = attrs.get('confidence', attrs.get('avg_confidence', 0.0))

            # Get definition
            if 'definition' in attrs:
                definition = attrs['definition']
            elif 'definitions' in attrs and attrs['definitions']:
                definition = attrs['definitions'][0]
            else:
                definition = 'No definition'

            # Calculate size based on centrality
            cent = centrality.get(node_id, 0)
            size = int(cent * 1000) + 20

            nodes_data.append({
                'id': node_id,
                'label': node_id,
                'category': category,
                'importance': importance,
                'evidence_count': evidence_count,
                'confidence': confidence,
                'definition': definition,
                'centrality': cent,  # Add centrality for stats panel
                'size': min(size, 60),  # Cap max size
                'color': self.CATEGORY_COLORS.get(category, '#95A5A6')
            })

            category_counts[category] += 1
            importance_counts[importance] += 1

        # Prepare edge data
        print("  🔗 Preparing edges...")
        edges_data = []
        for u, v in self.graph.edges():
            attrs = self.graph.edges[u, v]
            edges_data.append({
                'source': u,
                'target': v,
                'type': attrs.get('type', 'RELATED'),
                'strength': attrs.get('strength', 0.5),
                'width': max(1, attrs.get('strength', 0.5) * 3)
            })

        # Generate HTML
        print("  📝 Generating HTML...")
        html = self._generate_ultra_html(
            nodes_data,
            edges_data,
            category_counts,
            importance_counts,
            title
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"\n✅ Saved ultra-fast visualization to: {output_path}")
        print(f"   📈 {len(nodes_data)} nodes, {len(edges_data)} edges")
        print(f"   🏷️  {len(category_counts)} categories")
        print(f"\n💡 Open in your browser: file://{output_path.absolute()}")

    def _generate_ultra_html(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        category_counts: Counter[str],
        importance_counts: Counter[str],
        title: str
    ) -> str:
        """Generate ultra-optimized HTML with advanced filtering"""

        categories_sorted = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        importance_sorted = sorted(importance_counts.items(), key=lambda x: self.IMPORTANCE_LEVELS.index(x[0]) if x[0] in self.IMPORTANCE_LEVELS else 999)

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js"></script>
    <script src="https://unpkg.com/cytoscape-cose-bilkent@4.1.0/cytoscape-cose-bilkent.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f5f5;
            overflow: hidden;
        }}

        #header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px 20px;
            color: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}

        h1 {{
            font-size: 22px;
            font-weight: 600;
            margin-bottom: 5px;
        }}

        #stats {{
            font-size: 13px;
            opacity: 0.9;
            display: flex;
            gap: 20px;
            margin-top: 5px;
        }}

        .stat-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}

        .stat-value {{
            font-weight: 600;
            font-size: 15px;
        }}

        #cy {{
            width: 100%;
            height: calc(100vh - 200px);
            background: white;
        }}

        #controls {{
            padding: 15px 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
            box-shadow: 0 -2px 8px rgba(0,0,0,0.05);
        }}

        .controls-row {{
            display: flex;
            gap: 25px;
            align-items: flex-start;
            flex-wrap: wrap;
        }}

        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        .control-group label {{
            font-weight: 600;
            font-size: 12px;
            color: #555;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .control-row {{
            display: flex;
            gap: 8px;
            align-items: center;
        }}

        select, input {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 13px;
            background: white;
            min-width: 150px;
        }}

        select:focus, input:focus {{
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }}

        button {{
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            background: #667eea;
            color: white;
        }}

        button:hover {{
            background: #5568d3;
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        }}

        button:active {{
            transform: translateY(0);
        }}

        button.secondary {{
            background: #95a5a6;
        }}

        button.secondary:hover {{
            background: #7f8c8d;
        }}

        #legend {{
            position: absolute;
            top: 90px;
            right: 20px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            max-width: 250px;
            max-height: calc(100vh - 350px);
            overflow-y: auto;
            z-index: 100;
        }}

        #stats-panel {{
            position: absolute;
            top: 90px;
            left: 20px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            max-width: 300px;
            max-height: calc(100vh - 350px);
            overflow-y: auto;
            z-index: 100;
        }}

        #stats-panel.collapsed {{
            width: 40px;
            padding: 10px;
            cursor: pointer;
        }}

        #stats-panel.collapsed .stats-content {{
            display: none;
        }}

        #stats-panel.collapsed .stats-toggle {{
            transform: rotate(180deg);
        }}

        .stats-toggle {{
            float: right;
            cursor: pointer;
            font-size: 18px;
            transition: transform 0.3s;
            user-select: none;
        }}

        .stats-section {{
            margin: 15px 0;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }}

        .stats-section:last-child {{
            border-bottom: none;
        }}

        .stats-section h3 {{
            font-size: 12px;
            text-transform: uppercase;
            color: #667eea;
            margin-bottom: 8px;
            letter-spacing: 0.5px;
        }}

        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            font-size: 12px;
        }}

        .stat-label {{
            color: #555;
        }}

        .stat-value {{
            font-weight: 600;
            color: #2c3e50;
        }}

        .top-concept {{
            font-size: 11px;
            padding: 3px 0;
            color: #555;
            cursor: pointer;
        }}

        .top-concept:hover {{
            color: #667eea;
            font-weight: 600;
        }}

        .category-bar {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 4px 0;
            font-size: 11px;
        }}

        .category-bar-fill {{
            height: 8px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 4px;
            flex: 1;
        }}

        .legend-title {{
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            color: #555;
            margin-bottom: 10px;
            letter-spacing: 0.5px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 6px 8px;
            margin: 3px 0;
            cursor: pointer;
            border-radius: 4px;
            transition: all 0.2s;
        }}

        .legend-item:hover {{
            background: #f5f5f5;
        }}

        .legend-item.selected {{
            background: #e8eaf6;
            font-weight: 600;
        }}

        .legend-item-left {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .legend-color {{
            width: 14px;
            height: 14px;
            border-radius: 3px;
            flex-shrink: 0;
        }}

        .legend-count {{
            color: #7f8c8d;
            font-size: 11px;
            min-width: 30px;
            text-align: right;
        }}

        #filter-info {{
            background: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 10px 15px;
            margin-bottom: 15px;
            border-radius: 4px;
            font-size: 12px;
            display: none;
        }}

        #filter-info.active {{
            display: block;
        }}

        .filter-badge {{
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            margin: 2px;
        }}

        .checkbox-group {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}

        .checkbox-label {{
            display: flex;
            align-items: center;
            gap: 5px;
            cursor: pointer;
            font-size: 13px;
        }}

        input[type="checkbox"] {{
            min-width: auto;
            cursor: pointer;
        }}

        input[type="range"] {{
            min-width: 150px;
        }}

        .range-value {{
            font-weight: 600;
            color: #667eea;
            min-width: 40px;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>{title}</h1>
        <div id="stats">
            <div class="stat-item">
                <span class="stat-value" id="visible-nodes">{len(nodes)}</span>
                <span>/ {len(nodes)} concepts</span>
            </div>
            <div class="stat-item">
                <span class="stat-value" id="visible-edges">{len(edges)}</span>
                <span>/ {len(edges)} relationships</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">{len(category_counts)}</span>
                <span>categories</span>
            </div>
        </div>
    </div>

    <div id="stats-panel">
        <span class="stats-toggle" onclick="toggleStatsPanel()">◀</span>
        <div class="stats-content">
            <div class="legend-title">📊 Graph Statistics</div>

            <div class="stats-section">
                <h3>Overview</h3>
                <div class="stat-row">
                    <span class="stat-label">Total Nodes:</span>
                    <span class="stat-value">{len(nodes):,}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Total Edges:</span>
                    <span class="stat-value">{len(edges):,}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Categories:</span>
                    <span class="stat-value">{len(category_counts)}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Avg. Connections:</span>
                    <span class="stat-value" id="avg-connections">-</span>
                </div>
            </div>

            <div class="stats-section">
                <h3>Top Concepts</h3>
                <div id="top-concepts-list"></div>
            </div>

            <div class="stats-section">
                <h3>Category Distribution</h3>
                <div id="category-distribution"></div>
            </div>
        </div>
    </div>

    <div id="legend">
        <div class="legend-title">🏷️ Categories (click to filter)</div>
        <div id="category-legend"></div>
    </div>

    <div id="cy"></div>

    <div id="controls">
        <div id="filter-info">
            <strong>Active Filters:</strong>
            <div id="filter-badges"></div>
        </div>

        <div class="controls-row">
            <div class="control-group">
                <label>🎯 Importance</label>
                <div class="checkbox-group">
                    <label class="checkbox-label">
                        <input type="checkbox" value="high" checked> High
                    </label>
                    <label class="checkbox-label">
                        <input type="checkbox" value="medium" checked> Medium
                    </label>
                    <label class="checkbox-label">
                        <input type="checkbox" value="low" checked> Low
                    </label>
                </div>
            </div>

            <div class="control-group">
                <label>📊 Min Evidence Count</label>
                <div class="control-row">
                    <input type="range" id="evidence-slider" min="0" max="10" value="0" step="1">
                    <span class="range-value" id="evidence-value">0</span>
                </div>
            </div>

            <div class="control-group">
                <label>🔍 Search</label>
                <div class="control-row">
                    <input type="text" id="search-input" placeholder="Search concepts...">
                    <button onclick="searchNodes()">Search</button>
                </div>
            </div>

            <div class="control-group">
                <label>🎨 Layout</label>
                <div class="control-row">
                    <select id="layout-select">
                        <option value="cose-bilkent">Force-Directed</option>
                        <option value="circle">Circle</option>
                        <option value="grid">Grid</option>
                        <option value="concentric">Concentric</option>
                    </select>
                    <button onclick="applyLayout()">Apply</button>
                </div>
            </div>

            <div class="control-group">
                <label>Actions</label>
                <div class="control-row">
                    <button onclick="resetAllFilters()">Reset All</button>
                    <button class="secondary" onclick="cy.fit(50)">Fit View</button>
                    <button class="secondary" onclick="exportPNG()">Export PNG</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Data
        const nodesData = {json.dumps(nodes)};
        const edgesData = {json.dumps(edges)};
        const categoryData = {json.dumps(dict(categories_sorted))};
        
        // Active filters
        let activeCategories = new Set(Object.keys(categoryData));
        let activeImportance = new Set(['high', 'medium', 'low']);
        let minEvidence = 0;

        // Build elements
        const elements = [
            ...nodesData.map(n => ({{data: n}})),
            ...edgesData.map(e => ({{data: e}}))
        ];

        // Initialize Cytoscape
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
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'font-size': '9px',
                        'font-weight': '600',
                        'text-outline-width': 1.5,
                        'text-outline-color': '#fff',
                        'border-width': 2,
                        'border-color': '#555',
                        'text-wrap': 'wrap',
                        'text-max-width': '80px'
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                        'width': 'data(width)',
                        'line-color': '#95a5a6',
                        'target-arrow-color': '#95a5a6',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'opacity': 0.4,
                        'arrow-scale': 0.8
                    }}
                }},
                {{
                    selector: 'node:selected',
                    style: {{
                        'border-width': 4,
                        'border-color': '#000',
                        'z-index': 9999
                    }}
                }},
                {{
                    selector: 'node.highlighted',
                    style: {{
                        'border-width': 4,
                        'border-color': '#e74c3c',
                        'z-index': 9999
                    }}
                }},
                {{
                    selector: 'node.faded',
                    style: {{
                        'opacity': 0.15
                    }}
                }},
                {{
                    selector: 'edge.faded',
                    style: {{
                        'opacity': 0.05
                    }}
                }},
                {{
                    selector: 'node.hidden',
                    style: {{
                        'display': 'none'
                    }}
                }},
                {{
                    selector: 'edge.hidden',
                    style: {{
                        'display': 'none'
                    }}
                }}
            ],
            layout: {{
                name: 'cose-bilkent',
                animate: false,
                idealEdgeLength: 100,
                nodeRepulsion: 4500,
                randomize: false
            }}
        }});

        // Build legend
        const categoryLegend = document.getElementById('category-legend');
        Object.entries(categoryData).forEach(([cat, count]) => {{
            const item = document.createElement('div');
            item.className = 'legend-item selected';
            item.dataset.category = cat;
            
            const color = nodesData.find(n => n.category === cat)?.color || '#95A5A6';
            
            item.innerHTML = `
                <div class="legend-item-left">
                    <div class="legend-color" style="background: ${{color}}"></div>
                    <span>${{cat}}</span>
                </div>
                <span class="legend-count">${{count}}</span>
            `;
            
            item.onclick = () => toggleCategory(cat, item);
            categoryLegend.appendChild(item);
        }});

        // Toggle category filter
        function toggleCategory(cat, elem) {{
            if (activeCategories.has(cat)) {{
                activeCategories.delete(cat);
                elem.classList.remove('selected');
            }} else {{
                activeCategories.add(cat);
                elem.classList.add('selected');
            }}
            applyFilters();
        }}

        // Importance checkboxes
        document.querySelectorAll('input[type="checkbox"]').forEach(cb => {{
            cb.onchange = () => {{
                const val = cb.value;
                if (cb.checked) {{
                    activeImportance.add(val);
                }} else {{
                    activeImportance.delete(val);
                }}
                applyFilters();
            }};
        }});

        // Evidence slider
        const evidenceSlider = document.getElementById('evidence-slider');
        const evidenceValue = document.getElementById('evidence-value');
        evidenceSlider.oninput = () => {{
            minEvidence = parseInt(evidenceSlider.value);
            evidenceValue.textContent = minEvidence;
            applyFilters();
        }};

        // Apply all filters
        function applyFilters() {{
            cy.batch(() => {{
                let visibleCount = 0;
                let visibleEdges = 0;

                cy.nodes().forEach(node => {{
                    const data = node.data();
                    const categoryMatch = activeCategories.has(data.category);
                    const importanceMatch = activeImportance.has(data.importance);
                    const evidenceMatch = data.evidence_count >= minEvidence;

                    if (categoryMatch && importanceMatch && evidenceMatch) {{
                        node.removeClass('hidden');
                        visibleCount++;
                    }} else {{
                        node.addClass('hidden');
                    }}
                }});

                cy.edges().forEach(edge => {{
                    const srcHidden = edge.source().hasClass('hidden');
                    const tgtHidden = edge.target().hasClass('hidden');
                    if (srcHidden || tgtHidden) {{
                        edge.addClass('hidden');
                    }} else {{
                        edge.removeClass('hidden');
                        visibleEdges++;
                    }}
                }});

                // Update stats
                document.getElementById('visible-nodes').textContent = visibleCount;
                document.getElementById('visible-edges').textContent = visibleEdges;

                // Update filter info
                updateFilterInfo();

                // Fit to visible nodes
                const visible = cy.nodes().not('.hidden');
                if (visible.length > 0) {{
                    cy.fit(visible, 50);
                }}
            }});
        }}

        // Update filter info display
        function updateFilterInfo() {{
            const info = document.getElementById('filter-info');
            const badges = document.getElementById('filter-badges');
            const filters = [];

            if (activeCategories.size < Object.keys(categoryData).length) {{
                filters.push(`Categories: ${{Array.from(activeCategories).join(', ')}}`);
            }}
            if (activeImportance.size < 3) {{
                filters.push(`Importance: ${{Array.from(activeImportance).join(', ')}}`);
            }}
            if (minEvidence > 0) {{
                filters.push(`Min evidence: ${{minEvidence}}`);
            }}

            if (filters.length > 0) {{
                badges.innerHTML = filters.map(f => `<span class="filter-badge">${{f}}</span>`).join('');
                info.classList.add('active');
            }} else {{
                info.classList.remove('active');
            }}
        }}

        // Search
        function searchNodes() {{
            const query = document.getElementById('search-input').value.toLowerCase().trim();
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

        // Reset all filters
        function resetAllFilters() {{
            // Reset categories
            activeCategories = new Set(Object.keys(categoryData));
            document.querySelectorAll('.legend-item').forEach(item => {{
                item.classList.add('selected');
            }});

            // Reset importance
            activeImportance = new Set(['high', 'medium', 'low']);
            document.querySelectorAll('input[type="checkbox"]').forEach(cb => {{
                cb.checked = true;
            }});

            // Reset evidence
            minEvidence = 0;
            evidenceSlider.value = 0;
            evidenceValue.textContent = 0;

            // Reset search
            document.getElementById('search-input').value = '';
            
            cy.batch(() => {{
                cy.elements().removeClass('highlighted faded hidden');
            }});

            applyFilters();
            cy.fit(50);
        }}

        // Layout change
        function applyLayout() {{
            const layout = document.getElementById('layout-select').value;
            const visible = cy.nodes().not('.hidden');
            
            cy.layout({{
                name: layout,
                animate: true,
                animationDuration: 500,
                fit: true,
                padding: 50
            }}).run();
        }}

        // Export PNG
        function exportPNG() {{
            const png = cy.png({{
                output: 'blob',
                bg: 'white',
                full: true,
                scale: 2
            }});
            const url = URL.createObjectURL(png);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'knowledge_graph.png';
            a.click();
            URL.revokeObjectURL(url);
        }}

        // Node click details
        cy.on('tap', 'node', function(evt) {{
            const node = evt.target;
            const data = node.data();
            const msg = [
                `📌 ${{data.label}}`,
                ``,
                `Category: ${{data.category}}`,
                `Importance: ${{data.importance}}`,
                `Evidence: ${{data.evidence_count}} sources`,
                `Confidence: ${{(data.confidence * 100).toFixed(0)}}%`,
                ``,
                `Definition:`,
                data.definition
            ].join('\\n');
            alert(msg);
        }});

        // Toggle statistics panel
        function toggleStatsPanel() {{
            const panel = document.getElementById('stats-panel');
            panel.classList.toggle('collapsed');
        }}

        // Populate statistics
        function populateStatistics() {{
            // Calculate average connections
            const totalConnections = edgesData.length * 2; // Each edge counts for both nodes
            const avgConnections = (totalConnections / nodesData.length).toFixed(1);
            document.getElementById('avg-connections').textContent = avgConnections;

            // Top concepts by centrality
            const topConceptsList = document.getElementById('top-concepts-list');
            const sortedByCentrality = [...nodesData]
                .sort((a, b) => (b.centrality || 0) - (a.centrality || 0))
                .slice(0, 10);

            sortedByCentrality.forEach((node, idx) => {{
                const div = document.createElement('div');
                div.className = 'top-concept';
                div.textContent = `${{idx + 1}}. ${{node.label}}`;
                div.onclick = () => {{
                    // Find and highlight this node
                    cy.elements().removeClass('highlighted faded');
                    const cyNode = cy.getElementById(node.id);
                    if (cyNode.length > 0) {{
                        cy.elements().addClass('faded');
                        cyNode.removeClass('faded').addClass('highlighted');
                        cyNode.neighborhood().removeClass('faded');
                        cy.fit(cyNode, 100);
                    }}
                }};
                topConceptsList.appendChild(div);
            }});

            // Category distribution with bars
            const categoryDist = document.getElementById('category-distribution');
            const maxCount = Math.max(...Object.values(categoryData));

            Object.entries(categoryData)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 8)  // Top 8 categories
                .forEach(([cat, count]) => {{
                    const barDiv = document.createElement('div');
                    barDiv.className = 'category-bar';
                    const percentage = (count / maxCount * 100);
                    barDiv.innerHTML = `
                        <span style="min-width: 80px; font-size: 10px;">${{cat}}</span>
                        <div style="flex: 1; background: #f0f0f0; border-radius: 4px; height: 8px;">
                            <div style="width: ${{percentage}}%; background: #667eea; height: 100%; border-radius: 4px;"></div>
                        </div>
                        <span style="min-width: 30px; text-align: right; font-size: 10px;">${{count}}</span>
                    `;
                    categoryDist.appendChild(barDiv);
                }});
        }}

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') {{
                resetAllFilters();
            }}
            if (e.key === 'Enter' && document.activeElement.id === 'search-input') {{
                searchNodes();
            }}
        }});

        // Populate statistics panel
        populateStatistics();

        console.log('🚀 Ultra-Fast Knowledge Graph loaded successfully!');
        console.log(`📊 ${{nodesData.length}} concepts, ${{edgesData.length}} relationships`);
        console.log(`🏷️  ${{Object.keys(categoryData).length}} categories`);
    </script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate ultra-fast knowledge graph visualization')
    parser.add_argument('input_json', type=Path, help='Path to knowledge_graph.json')
    parser.add_argument('--output', '-o', type=Path, help='Output HTML file path')
    parser.add_argument('--title', '-t', default='Knowledge Graph', help='Graph title')
    parser.add_argument('--max-nodes', type=int, default=10000, help='Maximum nodes to display')

    args = parser.parse_args()

    if not args.input_json.exists():
        print(f"❌ Error: Input file not found: {args.input_json}")
        return

    # Default output path
    if args.output is None:
        args.output = args.input_json.parent / 'knowledge_graph_ultra_fast.html'

    # Load and visualize
    viz = UltraFastGraphVisualizer.from_json(args.input_json)
    viz.render_ultra_fast(args.output, title=args.title, max_nodes=args.max_nodes)


if __name__ == '__main__':
    main()
