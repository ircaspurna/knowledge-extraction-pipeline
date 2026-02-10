#!/usr/bin/env python3
"""
Lightweight Relationship Inferencer (TF-IDF based)

IMPORTANT: This is an OFFLINE FALLBACK for relationship inference, not the primary
approach. Use this only when:
  - You don't have access to Claude/LLM API for co-occurrence relationship extraction
  - You need a quick, dependency-free approximation of relationships
  - You're working offline or want a zero-cost baseline

Primary approach (preferred):
  1. Use the MCP tool `create_relationship_batch` to find co-occurring entity pairs
     in chunks and generate classification prompts for Claude
  2. Have Claude classify relationship types with context from the source text
  3. Use `parse_relationship_responses` to extract structured relationships
  4. Build graph with `build_knowledge_graph` which creates co-occurrence edges

This script instead:
  - Computes TF-IDF vectors from entity definitions/terms
  - Uses cosine similarity to find related pairs (O(n^2) pairwise comparison)
  - Infers relationship types from category pairs using hardcoded rules
  - Works entirely offline with no API calls

Limitations vs Claude-based approach:
  - No understanding of semantic context or nuance
  - Relationship types are guessed from categories, not from source text
  - TF-IDF similarity conflates lexical overlap with conceptual relatedness
  - O(n^2) pairwise comparison; slow for large graphs (1000+ nodes)
  - Cannot detect causal, contradictory, or hierarchical relationships reliably

Dependencies: standard library only (no sklearn, no sentence-transformers).
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import math


# Relationship type inference based on category pairs
CATEGORY_RELATIONSHIPS = {
    ('method', 'concept'): 'APPLIES_TO',
    ('method', 'problem'): 'ADDRESSES',
    ('method', 'phenomenon'): 'STUDIES',
    ('method', 'metric'): 'USES',
    ('method', 'method'): 'VARIANT_OF',
    ('method', 'theory'): 'IMPLEMENTS',
    ('method', 'tool'): 'USES',
    ('theory', 'concept'): 'DEFINES',
    ('theory', 'phenomenon'): 'EXPLAINS',
    ('theory', 'method'): 'GROUNDS',
    ('theory', 'principle'): 'CONTAINS',
    ('theory', 'theory'): 'RELATED_TO',
    ('concept', 'concept'): 'RELATED_TO',
    ('concept', 'phenomenon'): 'MANIFESTS_AS',
    ('concept', 'method'): 'MEASURED_BY',
    ('concept', 'metric'): 'QUANTIFIED_BY',
    ('phenomenon', 'phenomenon'): 'RELATED_TO',
    ('phenomenon', 'concept'): 'INVOLVES',
    ('phenomenon', 'method'): 'STUDIED_BY',
    ('principle', 'method'): 'GUIDES',
    ('principle', 'concept'): 'GOVERNS',
    ('principle', 'principle'): 'RELATED_TO',
    ('metric', 'concept'): 'MEASURES',
    ('metric', 'phenomenon'): 'QUANTIFIES',
    ('metric', 'method'): 'EVALUATES',
    ('problem', 'method'): 'SOLVED_BY',
    ('problem', 'concept'): 'INVOLVES',
    ('problem', 'phenomenon'): 'CAUSED_BY',
    ('tool', 'method'): 'IMPLEMENTS',
    ('tool', 'concept'): 'OPERATES_ON',
}

IMPORTANCE_WEIGHTS = {
    'critical': 1.0,
    'high': 0.8,
    'medium': 0.6,
    'low': 0.4
}

# Stop words for TF-IDF
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can',
    'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
    'which', 'who', 'whom', 'what', 'where', 'when', 'why', 'how', 'all',
    'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'not', 'only', 'same', 'so', 'than', 'too', 'very', 'just', 'also',
    'into', 'over', 'after', 'before', 'between', 'under', 'through', 'during',
    'above', 'below', 'about', 'against', 'within', 'without', 'along', 'across'
}


def tokenize(text: str) -> List[str]:
    """Simple tokenization"""
    text = text.lower()
    # Keep alphanumeric and hyphens
    tokens = re.findall(r'[a-z][a-z\-]*[a-z]|[a-z]', text)
    # Remove stop words and short tokens
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


def compute_tfidf(documents: List[List[str]]) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """Compute TF-IDF vectors for documents"""
    # Document frequency
    df = Counter()
    for doc in documents:
        df.update(set(doc))
    
    n_docs = len(documents)
    
    # IDF
    idf = {term: math.log(n_docs / (count + 1)) + 1 for term, count in df.items()}
    
    # TF-IDF vectors
    tfidf_vectors = []
    for doc in documents:
        tf = Counter(doc)
        doc_len = len(doc) if doc else 1
        vector = {term: (count / doc_len) * idf[term] for term, count in tf.items()}
        tfidf_vectors.append(vector)
    
    return tfidf_vectors, idf


def cosine_similarity(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    """Compute cosine similarity between two sparse vectors"""
    # Get common terms
    common = set(v1.keys()) & set(v2.keys())
    if not common:
        return 0.0
    
    # Dot product
    dot = sum(v1[t] * v2[t] for t in common)
    
    # Magnitudes
    mag1 = math.sqrt(sum(v ** 2 for v in v1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in v2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot / (mag1 * mag2)


def get_definition(node: dict) -> str:
    """Extract definition text from a node"""
    defs = node.get('definitions', [])
    if defs and isinstance(defs, list):
        return defs[0]
    return node.get('definition', node.get('term', node['id']))


def infer_relationship_type(cat1: str, cat2: str) -> str:
    """Infer relationship type from category pair"""
    rel = CATEGORY_RELATIONSHIPS.get((cat1, cat2))
    if rel:
        return rel
    rel = CATEGORY_RELATIONSHIPS.get((cat2, cat1))
    if rel:
        return rel
    return 'RELATED_TO'


def infer_relationships(
    nodes: List[dict],
    threshold: float = 0.3,
    max_edges_per_node: int = 5,
    existing_edges: Set[Tuple[str, str]] = None
) -> List[dict]:
    """Infer relationships using TF-IDF similarity"""
    
    if existing_edges is None:
        existing_edges = set()
    
    # Prepare documents (term + definition + category)
    print("  📝 Tokenizing definitions...")
    documents = []
    for node in nodes:
        term = node.get('term', node['id'])
        definition = get_definition(node)
        category = node.get('primary_category', '')
        text = f"{term} {definition} {category}"
        documents.append(tokenize(text))
    
    # Compute TF-IDF
    print("  🧮 Computing TF-IDF vectors...")
    tfidf_vectors, _ = compute_tfidf(documents)
    
    # Find similar pairs
    print(f"  🔍 Finding similar pairs (threshold={threshold})...")
    n = len(nodes)
    pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(tfidf_vectors[i], tfidf_vectors[j])
            if sim >= threshold:
                pairs.append((i, j, sim))
    
    # Sort by similarity
    pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"     Found {len(pairs)} candidate pairs")
    
    # Build edges with limits
    new_edges = []
    edges_per_node = defaultdict(int)
    
    for i, j, sim in pairs:
        node_i = nodes[i]
        node_j = nodes[j]
        id_i = node_i['id']
        id_j = node_j['id']
        
        # Skip existing
        if (id_i, id_j) in existing_edges or (id_j, id_i) in existing_edges:
            continue
        
        # Limit per node
        if edges_per_node[id_i] >= max_edges_per_node or edges_per_node[id_j] >= max_edges_per_node:
            continue
        
        cat_i = node_i.get('primary_category', 'concept')
        cat_j = node_j.get('primary_category', 'concept')
        rel_type = infer_relationship_type(cat_i, cat_j)
        
        imp_i = IMPORTANCE_WEIGHTS.get(node_i.get('primary_importance', 'medium'), 0.6)
        imp_j = IMPORTANCE_WEIGHTS.get(node_j.get('primary_importance', 'medium'), 0.6)
        strength = sim * (imp_i + imp_j) / 2
        
        new_edges.append({
            'source': id_i,
            'target': id_j,
            'type': rel_type,
            'strength': round(strength, 3),
            'confidence': round(sim, 3),
            'inferred': True,
            'method': 'tfidf_similarity'
        })
        
        edges_per_node[id_i] += 1
        edges_per_node[id_j] += 1
        existing_edges.add((id_i, id_j))
    
    return new_edges


def ensure_connectivity(
    nodes: List[dict],
    edges: List[dict],
    tfidf_vectors: List[Dict[str, float]] = None,
    min_degree: int = 1
) -> List[dict]:
    """Ensure all nodes have minimum connectivity"""
    
    # Recompute vectors if not provided
    if tfidf_vectors is None:
        documents = []
        for node in nodes:
            term = node.get('term', node['id'])
            definition = get_definition(node)
            text = f"{term} {definition}"
            documents.append(tokenize(text))
        tfidf_vectors, _ = compute_tfidf(documents)
    
    # Build adjacency
    node_ids = [n['id'] for n in nodes]
    node_idx = {nid: i for i, nid in enumerate(node_ids)}
    
    connected = defaultdict(set)
    for edge in edges:
        src, tgt = edge['source'], edge['target']
        connected[src].add(tgt)
        connected[tgt].add(src)
    
    # Find isolated nodes
    isolated = [nid for nid in node_ids if len(connected[nid]) < min_degree]
    print(f"  🔗 Connecting {len(isolated)} isolated nodes...")
    
    new_edges = []
    existing = {(e['source'], e['target']) for e in edges}
    existing.update({(e['target'], e['source']) for e in edges})
    
    for nid in isolated:
        idx = node_idx[nid]
        node = nodes[idx]
        
        # Find best match
        best_sim = 0
        best_idx = -1
        
        for other_idx, other_id in enumerate(node_ids):
            if other_id == nid or other_id in connected[nid]:
                continue
            sim = cosine_similarity(tfidf_vectors[idx], tfidf_vectors[other_idx])
            if sim > best_sim:
                best_sim = sim
                best_idx = other_idx
        
        if best_idx >= 0 and best_sim > 0.15:  # Lower threshold for connectivity
            other_node = nodes[best_idx]
            other_id = other_node['id']
            
            if (nid, other_id) not in existing and (other_id, nid) not in existing:
                cat1 = node.get('primary_category', 'concept')
                cat2 = other_node.get('primary_category', 'concept')
                
                new_edges.append({
                    'source': nid,
                    'target': other_id,
                    'type': infer_relationship_type(cat1, cat2),
                    'strength': round(best_sim * 0.6, 3),
                    'confidence': round(best_sim, 3),
                    'inferred': True,
                    'method': 'connectivity_repair'
                })
                
                existing.add((nid, other_id))
                connected[nid].add(other_id)
                connected[other_id].add(nid)
    
    return new_edges


def main():
    parser = argparse.ArgumentParser(
        description='Offline fallback: infer relationships using TF-IDF similarity. '
                    'For best results, use Claude-based co-occurrence extraction instead '
                    '(create_relationship_batch + parse_relationship_responses).'
    )
    parser.add_argument('input_json', type=Path, help='Input knowledge_graph.json')
    parser.add_argument('--output', '-o', type=Path, help='Output JSON path')
    parser.add_argument('--threshold', '-t', type=float, default=0.3,
                        help='Similarity threshold (default: 0.3)')
    parser.add_argument('--max-edges', '-m', type=int, default=5,
                        help='Max new edges per node (default: 5)')
    parser.add_argument('--ensure-connected', '-c', action='store_true',
                        help='Ensure all nodes have at least one edge')
    
    args = parser.parse_args()
    
    if not args.input_json.exists():
        print(f"❌ File not found: {args.input_json}")
        return
    
    if args.output is None:
        args.output = args.input_json.parent / 'knowledge_graph_enriched.json'
    
    print("=" * 60)
    print("TF-IDF RELATIONSHIP INFERENCER (offline fallback)")
    print("Note: For best results, use Claude-based co-occurrence")
    print("      extraction via create_relationship_batch instead.")
    print("=" * 60)
    
    # Load graph
    print(f"\n📂 Loading: {args.input_json}")
    with open(args.input_json, encoding='utf-8') as f:
        data = json.load(f)
    
    nodes = data['nodes']
    original_edges = data['edges']
    
    print(f"   Nodes: {len(nodes)}")
    print(f"   Existing edges: {len(original_edges)}")
    
    # Get existing edges
    existing_edges = {(e['source'], e['target']) for e in original_edges}
    existing_edges.update({(e['target'], e['source']) for e in original_edges})
    
    # Infer relationships
    print(f"\n🔗 Inferring relationships...")
    new_edges = infer_relationships(
        nodes,
        threshold=args.threshold,
        max_edges_per_node=args.max_edges,
        existing_edges=existing_edges
    )
    
    print(f"   New edges from similarity: {len(new_edges)}")
    
    # Combine
    all_edges = original_edges + new_edges
    
    # Ensure connectivity
    if args.ensure_connected:
        print(f"\n🔧 Ensuring connectivity...")
        connectivity_edges = ensure_connectivity(nodes, all_edges)
        all_edges.extend(connectivity_edges)
        print(f"   Connectivity edges added: {len(connectivity_edges)}")
    
    # Save
    enriched = {
        'nodes': nodes,
        'edges': all_edges,
        'metadata': {
            'original_edges': len(original_edges),
            'inferred_edges': len(all_edges) - len(original_edges),
            'similarity_threshold': args.threshold
        }
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(enriched, f, indent=2)
    
    print(f"\n✅ Enriched graph saved!")
    print(f"   📁 {args.output}")
    print(f"   📈 Total edges: {len(all_edges)} ({len(original_edges)} original + {len(all_edges) - len(original_edges)} inferred)")
    
    # Edge type distribution
    edge_types = Counter(e['type'] for e in all_edges)
    print(f"\n   Edge types:")
    for etype, count in edge_types.most_common(10):
        print(f"      {etype}: {count}")


if __name__ == '__main__':
    main()
