#!/usr/bin/env python3
"""
MCP-Native Entity Resolver - Returns prompts for Claude Code

Semantic deduplication without direct API calls.
Uses a two-phase vectorized approach for scalability:
  Phase 1: O(n) hash-based exact + alias matching
  Phase 2: Batch embedding + chunked cosine similarity + Union-Find

This handles 25K+ concepts in seconds. The previous pairwise approach
was O(n^2) and would run for hours on large datasets.
"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Set up logging
logger = logging.getLogger(__name__)

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.warning("sentence-transformers not installed - semantic matching disabled")
    np = None
    SentenceTransformer = None


@dataclass
class ResolvedEntity:
    """Canonical entity with aliases and merged evidence"""
    canonical_term: str
    aliases: list[str]
    definitions: list[str]
    categories: dict[str, int]
    importance_scores: dict[str, int]
    evidence: list[dict[str, Any]]
    sources: set[str]
    avg_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            'canonical_term': self.canonical_term,
            'aliases': self.aliases,
            'definitions': self.definitions,
            'categories': dict(self.categories),
            'importance_scores': dict(self.importance_scores),
            'evidence': self.evidence,
            'sources': list(self.sources),
            'avg_confidence': self.avg_confidence,
            'primary_category': max(self.categories.items(), key=lambda x: x[1])[0] if self.categories else 'unknown',
            'primary_importance': max(self.importance_scores.items(), key=lambda x: x[1])[0] if self.importance_scores else 'medium'
        }


class EntityResolverMCP:
    """
    Resolve duplicate entities using embeddings + Claude Code for ambiguous cases.

    Strategy (two-phase vectorized):
    1. Exact string matches via hash grouping — O(n) (automatic)
    2. Known alias matches — O(n) (automatic)
    3. Batch embedding + chunked cosine similarity — O(n^2) but vectorized (automatic)
       - Similarity >= semantic_match_threshold (0.90): auto-merge
       - Similarity in [ambiguous_threshold, semantic_match_threshold): ambiguous → Claude Code
    """

    MERGE_DECISION_PROMPT = """Decide if these two concepts refer to the SAME entity:

**Concept A**: {term_a}
Definition: {definition_a}
Context: {context_a}

**Concept B**: {term_b}
Definition: {definition_b}
Context: {context_b}

Are these the same concept? Consider:
- Do they refer to the same theoretical construct?
- Are they synonyms or just related concepts?
- Is one an acronym/abbreviation of the other?

**Examples**:
- "Machine Learning" and "ML" → SAME
- "Neural Networks" and "Deep Learning" → DIFFERENT (related but distinct)
- "Loss Aversion" and "Loss-Aversion" → SAME
- "Anchoring" and "Anchoring Bias" → SAME

**Output Format** (JSON):
```json
{{
  "decision": "SAME" | "DIFFERENT",
  "reasoning": "One sentence explanation",
  "confidence": 0.0-1.0
}}
```

Decision:"""

    # Chunk size for blocked similarity computation. Controls memory usage:
    # 2000 x 2000 x 4 bytes = ~16MB per block, well within limits.
    _SIMILARITY_BLOCK_SIZE = 2000

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        exact_match_threshold: float = 0.95,
        semantic_match_threshold: float = 0.90,
        ambiguous_threshold: float = 0.85
    ) -> None:
        """
        Initialize entity resolver.

        Args:
            embedding_model: Sentence transformer model
            exact_match_threshold: Similarity threshold for automatic merge
            semantic_match_threshold: Threshold for high-confidence match
            ambiguous_threshold: Below this, ask Claude Code to decide
        """
        # Initialize embedder
        if SentenceTransformer:
            self.embedder = SentenceTransformer(embedding_model)
        else:
            self.embedder = None
            logger.warning("Semantic matching disabled (sentence-transformers not available)")

        self.exact_match_threshold = exact_match_threshold
        self.semantic_match_threshold = semantic_match_threshold
        self.ambiguous_threshold = ambiguous_threshold

        # Known aliases (can be extended)
        self.known_aliases = {
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'nn': 'neural network',
            'dl': 'deep learning',
            'nlp': 'natural language processing',
            'cv': 'computer vision'
        }

        # Stats
        self.stats = {
            'concepts_processed': 0,
            'exact_matches': 0,
            'semantic_matches': 0,
            'llm_decisions_needed': 0,
            'entities_created': 0
        }

        # Store ambiguous pairs for Claude Code
        self.ambiguous_pairs: list[dict[str, Any]] = []

    def normalize_term(self, term: str) -> str:
        """Normalize term for matching"""
        normalized = term.lower().strip()
        normalized = normalized.replace('-', ' ').replace('_', ' ')
        normalized = ' '.join(normalized.split())
        return normalized

    def check_alias_match(self, term1: str, term2: str) -> bool:
        """Check if terms are known aliases"""
        norm1 = self.normalize_term(term1)
        norm2 = self.normalize_term(term2)

        if norm1 in self.known_aliases and self.known_aliases[norm1] == norm2:
            return True
        if norm2 in self.known_aliases and self.known_aliases[norm2] == norm1:
            return True

        if norm1 in self.known_aliases and norm2 in self.known_aliases:
            return self.known_aliases[norm1] == self.known_aliases[norm2]

        return False

    def calculate_similarity(
        self,
        term1: str,
        term2: str,
        def1: str = "",
        def2: str = ""
    ) -> float:
        """Calculate semantic similarity between two concepts.

        Note: This method encodes one pair at a time. For bulk resolution,
        resolve_entities_automatic() uses batch encoding internally,
        which is orders of magnitude faster.
        """
        if not self.embedder:
            # Fall back to simple string similarity
            norm1 = self.normalize_term(term1)
            norm2 = self.normalize_term(term2)
            if norm1 == norm2:
                return 1.0
            return 0.0

        try:
            # Combine term and definition
            text1 = f"{term1}. {def1[:200]}" if def1 else term1
            text2 = f"{term2}. {def2[:200]}" if def2 else term2

            # Get embeddings
            embeddings = self.embedder.encode([text1, text2])

            # Cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )

            return float(similarity)
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return 0.0

    def generate_merge_decision_prompt(
        self,
        concept1: dict[str, Any],
        concept2: dict[str, Any],
        similarity: float
    ) -> dict[str, Any]:
        """
        Generate prompt for Claude Code to decide if concepts should merge.

        Returns:
            {
                'prompt': str,
                'metadata': {
                    'concept1_term': str,
                    'concept2_term': str,
                    'similarity': float,
                    'task': 'merge_decision'
                }
            }
        """
        prompt = self.MERGE_DECISION_PROMPT.format(
            term_a=concept1['term'],
            definition_a=concept1.get('definition', ''),
            context_a=concept1.get('quote', '')[:200],
            term_b=concept2['term'],
            definition_b=concept2.get('definition', ''),
            context_b=concept2.get('quote', '')[:200]
        )

        return {
            'prompt': prompt,
            'metadata': {
                'concept1_term': concept1['term'],
                'concept2_term': concept2['term'],
                'similarity': similarity,
                'task': 'merge_decision'
            }
        }

    def parse_merge_decision(self, response_text: str) -> tuple[bool, str, float]:
        """
        Parse Claude's merge decision.

        Returns: (should_merge, reasoning, confidence)
        """
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                decision_data = json.loads(json_match.group(1))
            else:
                decision_data = json.loads(response_text)

            decision = decision_data.get('decision', 'DIFFERENT')
            reasoning = decision_data.get('reasoning', '')
            confidence = decision_data.get('confidence', 0.5)

            return decision == 'SAME', reasoning, confidence

        except Exception as e:
            logger.error(f"Error parsing merge decision: {e}")
            return False, f"Parse error: {e}", 0.0

    # ------------------------------------------------------------------
    # Union-Find helpers for transitive merging
    # ------------------------------------------------------------------

    @staticmethod
    def _uf_find(parent: list[int], x: int) -> int:
        """Find root with path compression."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    @staticmethod
    def _uf_union(parent: list[int], rank: list[int], x: int, y: int) -> bool:
        """Union by rank. Returns True if a merge happened."""
        rx, ry = EntityResolverMCP._uf_find(parent, x), EntityResolverMCP._uf_find(parent, y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    # ------------------------------------------------------------------
    # Core resolution (two-phase vectorized)
    # ------------------------------------------------------------------

    def resolve_entities_automatic(
        self,
        concepts: list[dict[str, Any]]
    ) -> tuple[list[ResolvedEntity], list[dict[str, Any]]]:
        """
        Resolve entities using automatic methods only.
        Returns resolved entities and list of ambiguous pairs for Claude Code.

        Uses a two-phase approach for scalability:
          Phase 1: O(n) hash-based exact + alias matching
          Phase 2: Vectorized batch embedding + chunked cosine similarity

        Args:
            concepts: List of concept dicts (must have 'term' and 'definition')

        Returns:
            (resolved_entities, ambiguous_pairs_for_claude)
        """
        # Validate inputs
        if not isinstance(concepts, list):
            raise ValueError(f"concepts must be a list, got {type(concepts).__name__}")

        for i, concept in enumerate(concepts):
            if not isinstance(concept, dict):
                raise ValueError(f"Concept {i} must be a dict, got {type(concept).__name__}")
            missing = [f for f in ('term', 'definition') if f not in concept]
            if missing:
                raise ValueError(f"Concept {i} missing required fields: {missing}")
            if not concept['term'] or not concept['term'].strip():
                raise ValueError(f"Concept {i} has empty 'term' field")

        logger.info(f"Resolving entities from {len(concepts)} concepts...")

        # ---------------------------------------------------------------
        # Phase 1: Hash-based exact + alias grouping  (O(n))
        # ---------------------------------------------------------------
        logger.info("Phase 1: exact + alias matching...")

        # Map normalized term → list of concept indices
        norm_groups: dict[str, list[int]] = defaultdict(list)
        for idx, c in enumerate(concepts):
            norm = self.normalize_term(c['term'])
            if norm:
                # Check alias expansion
                expanded = self.known_aliases.get(norm, norm)
                norm_groups[expanded].append(idx)

        logger.info(f"Phase 1: {len(concepts)} concepts → {len(norm_groups)} groups")
        self.stats['exact_matches'] = len(concepts) - len(norm_groups)

        # Build one representative concept per group for Phase 2
        group_keys = list(norm_groups.keys())
        group_reps: list[dict[str, Any]] = []  # one per group
        for key in group_keys:
            indices = norm_groups[key]
            # Pick the concept with the longest definition as representative
            rep = max((concepts[i] for i in indices),
                      key=lambda c: len(c.get('definition', '')))
            group_reps.append(rep)

        n_groups = len(group_reps)

        # ---------------------------------------------------------------
        # Phase 2: Vectorized semantic similarity on group representatives
        # ---------------------------------------------------------------
        # Union-Find over group indices
        parent = list(range(n_groups))
        rank = [0] * n_groups
        ambiguous_pairs: list[dict[str, Any]] = []

        if self.embedder and n_groups > 1:
            logger.info(f"Phase 2: batch-encoding {n_groups} terms...")

            # Build texts for embedding (term + short definition)
            texts = []
            for rep in group_reps:
                defn = rep.get('definition', '')[:200]
                texts.append(f"{rep['term']}. {defn}" if defn else rep['term'])

            # Batch encode all at once (fast)
            embeddings = self.embedder.encode(
                texts, batch_size=512, show_progress_bar=False,
                normalize_embeddings=True
            )

            # Chunked cosine similarity (normalized → dot product)
            bs = self._SIMILARITY_BLOCK_SIZE
            n_blocks = (n_groups + bs - 1) // bs

            logger.info(f"Phase 2: computing similarity ({n_blocks} blocks)...")

            for bi in range(0, n_groups, bs):
                bi_end = min(bi + bs, n_groups)
                emb_i = embeddings[bi:bi_end]

                for bj in range(bi, n_groups, bs):
                    bj_end = min(bj + bs, n_groups)
                    emb_j = embeddings[bj:bj_end]

                    sim_block = emb_i @ emb_j.T

                    if bi == bj:
                        # Same block — upper triangle only
                        rows, cols = np.where(np.triu(sim_block, k=1) >= self.ambiguous_threshold)
                    else:
                        rows, cols = np.where(sim_block >= self.ambiguous_threshold)

                    for r, c in zip(rows, cols):
                        gi, gj = bi + int(r), bj + int(c)
                        similarity = float(sim_block[r, c])

                        if similarity >= self.semantic_match_threshold:
                            # High confidence — auto-merge
                            if self._uf_union(parent, rank, gi, gj):
                                self.stats['semantic_matches'] += 1
                        else:
                            # Ambiguous — record for Claude Code
                            ambiguous_pairs.append({
                                'concept1': group_reps[gi],
                                'concept2': group_reps[gj],
                                'similarity': similarity,
                                'index1': gi,
                                'index2': gj
                            })
                            self.stats['llm_decisions_needed'] += 1

                logger.debug(f"  Block row {bi // bs + 1}/{n_blocks} done")

        elif not self.embedder:
            logger.info("Phase 2 skipped (no embedder available)")

        # ---------------------------------------------------------------
        # Build final entities from Union-Find groups
        # ---------------------------------------------------------------
        merged_groups: dict[int, list[int]] = defaultdict(list)
        for gi in range(n_groups):
            root = self._uf_find(parent, gi)
            merged_groups[root].append(gi)

        entities: list[ResolvedEntity] = []

        for root, group_indices in merged_groups.items():
            # Collect all original concept indices across all groups being merged
            all_concept_indices: list[int] = []
            for gi in group_indices:
                key = group_keys[gi]
                all_concept_indices.extend(norm_groups[key])

            # Build entity from all concepts
            all_concepts = [concepts[i] for i in all_concept_indices]

            # Pick canonical term from most frequent original casing
            term_counts: dict[str, int] = defaultdict(int)
            for c in all_concepts:
                term_counts[c['term'].strip()] += 1
            canonical_term = max(term_counts, key=term_counts.get)

            aliases = list(set(c['term'].strip() for c in all_concepts))

            definitions: list[str] = []
            seen_defs: set[str] = set()
            for c in all_concepts:
                d = c.get('definition', '').strip()
                if d and d not in seen_defs:
                    seen_defs.add(d)
                    definitions.append(d)

            categories: dict[str, int] = defaultdict(int)
            importance_scores: dict[str, int] = defaultdict(int)
            sources: set[str] = set()
            confidences: list[float] = []
            evidence: list[dict[str, Any]] = []

            for c in all_concepts:
                categories[c.get('category', 'general')] += 1
                importance_scores[c.get('importance', 'medium')] += 1
                sources.add(c.get('source_file', 'unknown'))
                confidences.append(c.get('confidence', 0.5))
                evidence.append({
                    'chunk_id': c.get('source_chunk_id', c.get('chunk_id', '')),
                    'quote': c.get('quote', ''),
                    'page': c.get('page', 1)
                })

            entity = ResolvedEntity(
                canonical_term=canonical_term,
                aliases=aliases,
                definitions=definitions[:10],
                categories=categories,
                importance_scores=importance_scores,
                evidence=evidence,
                sources=sources,
                avg_confidence=sum(confidences) / len(confidences) if confidences else 0.0
            )
            entities.append(entity)

        self.stats['concepts_processed'] = len(concepts)
        self.stats['entities_created'] = len(entities)

        logger.info(f"Resolution complete: {len(concepts)} concepts → {len(entities)} entities")
        logger.info(f"Exact/alias matches: {self.stats['exact_matches']}, "
                     f"Semantic merges: {self.stats['semantic_matches']}")
        if ambiguous_pairs:
            logger.info(f"Ambiguous pairs (need Claude): {len(ambiguous_pairs)}")

        return entities, ambiguous_pairs

    def create_ambiguous_batch_file(
        self,
        ambiguous_pairs: list[dict[str, Any]],
        output_path: Path
    ) -> Path:
        """
        Create batch file with merge decision prompts for Claude Code.

        Args:
            ambiguous_pairs: List of ambiguous concept pairs
            output_path: Where to save batch file

        Returns:
            Path to batch file
        """
        batch_data: dict[str, Any] = {
            'task': 'entity_merge_decisions',
            'total_pairs': len(ambiguous_pairs),
            'prompts': []
        }

        for pair in ambiguous_pairs:
            prompt_data = self.generate_merge_decision_prompt(
                pair['concept1'],
                pair['concept2'],
                pair['similarity']
            )
            prompt_data['metadata']['index1'] = pair['index1']
            prompt_data['metadata']['index2'] = pair['index2']
            batch_data['prompts'].append(prompt_data)

        output_path = Path(output_path)
        output_path.write_text(json.dumps(batch_data, indent=2), encoding='utf-8')

        logger.info(f"Created ambiguous pairs batch file: {output_path}")
        logger.info(f"Pairs needing Claude decision: {len(ambiguous_pairs)}")

        return output_path

    def save_entities(
        self,
        entities: list[ResolvedEntity],
        output_path: Path
    ) -> None:
        """Save resolved entities to JSON"""
        output_path = Path(output_path)

        data = {
            'num_entities': len(entities),
            'resolution_stats': self.stats,
            'entities': [e.to_dict() for e in entities]
        }

        output_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        logger.info(f"Saved {len(entities)} entities to: {output_path}")


# =========================================================================
# CLI
# =========================================================================

def main() -> int:
    """Command-line interface"""
    import argparse

    # Configure logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    parser = argparse.ArgumentParser(
        description='Resolve duplicate entities (MCP version)'
    )
    parser.add_argument('input', type=str, help='Input JSON file with concepts')
    parser.add_argument('--output', type=str, default='entities.json',
                       help='Output JSON file for resolved entities')
    parser.add_argument('--ambiguous-output', type=str, default='ambiguous_pairs.json',
                       help='Output file for ambiguous pairs needing Claude')
    parser.add_argument('--semantic-threshold', type=float, default=0.90,
                       help='Semantic similarity threshold for automatic merge')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load concepts
    input_path = Path(args.input)
    concepts_data = json.loads(input_path.read_text(encoding='utf-8'))
    concepts = concepts_data.get('concepts', [])

    print(f"Loaded {len(concepts)} concepts from {input_path.name}")

    # Create resolver
    resolver = EntityResolverMCP(
        semantic_match_threshold=args.semantic_threshold
    )

    # Resolve entities automatically
    entities, ambiguous_pairs = resolver.resolve_entities_automatic(concepts)

    # Save entities
    resolver.save_entities(entities, Path(args.output))

    # Save ambiguous pairs for Claude Code
    if ambiguous_pairs:
        batch_file = resolver.create_ambiguous_batch_file(
            ambiguous_pairs,
            Path(args.ambiguous_output)
        )

        print("\nNext steps:")
        print(f"1. Use Claude Code to process: {batch_file}")
        print("2. Claude Code will decide which pairs to merge")
        print("3. Apply merge decisions to update entities")
        print("\nExample Claude Code command:")
        print(f"  mcp use knowledge_extraction resolve_ambiguous {batch_file}")
    else:
        print("\n✓ No ambiguous pairs - entity resolution complete!")

    # Print sample entities
    print("\nSample resolved entities:")
    for entity in entities[:3]:
        print(f"\n  Canonical: {entity.canonical_term}")
        if len(entity.aliases) > 1:
            print(f"  Aliases: {', '.join(entity.aliases[1:])}")
        print(f"  Category: {entity.to_dict()['primary_category']}")
        print(f"  Evidence: {len(entity.evidence)} sources")

    return 0


if __name__ == "__main__":
    exit(main())
