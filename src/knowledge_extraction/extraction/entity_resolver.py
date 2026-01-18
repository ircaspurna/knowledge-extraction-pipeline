#!/usr/bin/env python3
"""
MCP-Native Entity Resolver - Returns prompts for Claude Code

Semantic deduplication without direct API calls.
"""

import json
import logging
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
    
    Strategy:
    1. Exact string matches (automatic)
    2. Known alias matches (automatic)
    3. High embedding similarity > 0.90 (automatic)
    4. Ambiguous cases 0.85-0.90 (generate prompt for Claude Code)
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
        """Calculate semantic similarity between concepts."""
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
            import re
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

    def resolve_entities_automatic(
        self,
        concepts: list[dict[str, Any]]
    ) -> tuple[list[ResolvedEntity], list[dict[str, Any]]]:
        """
        Resolve entities using automatic methods only.
        Returns resolved entities and list of ambiguous pairs for Claude Code.

        Args:
            concepts: List of concept dicts

        Returns:
            (resolved_entities, ambiguous_pairs_for_claude)
        """
        # Validate inputs
        if not isinstance(concepts, list):
            raise ValueError(f"concepts must be a list, got {type(concepts).__name__}")

        # Validate concept structure
        required_fields = ['term', 'definition']
        for i, concept in enumerate(concepts):
            if not isinstance(concept, dict):
                raise ValueError(f"Concept {i} must be a dict, got {type(concept).__name__}")

            missing = [f for f in required_fields if f not in concept]
            if missing:
                raise ValueError(f"Concept {i} missing required fields: {missing}")

            if not concept['term'] or not concept['term'].strip():
                raise ValueError(f"Concept {i} has empty 'term' field")

        logger.info(f"Resolving entities from {len(concepts)} concepts...")
        logger.debug("Using automatic methods (string matching + embeddings)")

        # Track which concepts have been merged
        merged_indices = set()
        entities = []
        ambiguous_pairs = []

        for i, concept1 in enumerate(concepts):
            if i in merged_indices:
                continue

            # Start new entity
            canonical_term = concept1['term']
            aliases = [concept1['term']]
            definitions = [concept1.get('definition', '')]
            categories: dict[str, int] = defaultdict(int)
            importance_scores: dict[str, int] = defaultdict(int)
            evidence = []
            sources = set()
            confidences = []

            # Add concept1 data
            categories[concept1.get('category', 'general')] += 1
            importance_scores[concept1.get('importance', 'medium')] += 1
            sources.add(concept1.get('source_file', 'unknown'))
            confidences.append(concept1.get('confidence', 0.5))

            evidence.append({
                'chunk_id': concept1.get('source_chunk_id', concept1.get('chunk_id', '')),
                'quote': concept1.get('quote', ''),
                'page': concept1.get('page', 1)
            })

            # Check for duplicates
            for j, concept2 in enumerate(concepts[i+1:], i+1):
                if j in merged_indices:
                    continue

                should_merge = False
                merge_reason = ""

                # Level 1: Exact string match
                norm1 = self.normalize_term(concept1['term'])
                norm2 = self.normalize_term(concept2['term'])

                if norm1 == norm2:
                    should_merge = True
                    merge_reason = "exact_match"
                    self.stats['exact_matches'] += 1

                # Level 2: Known alias match
                elif self.check_alias_match(concept1['term'], concept2['term']):
                    should_merge = True
                    merge_reason = "alias_match"
                    self.stats['exact_matches'] += 1

                # Level 3: Semantic similarity
                else:
                    similarity = self.calculate_similarity(
                        concept1['term'],
                        concept2['term'],
                        concept1.get('definition', ''),
                        concept2.get('definition', '')
                    )

                    if similarity >= self.exact_match_threshold:
                        # Very high similarity - merge automatically
                        should_merge = True
                        merge_reason = f"semantic_high ({similarity:.2f})"
                        self.stats['semantic_matches'] += 1

                    elif similarity >= self.ambiguous_threshold:
                        # Ambiguous - need Claude Code to decide
                        ambiguous_pairs.append({
                            'concept1': concept1,
                            'concept2': concept2,
                            'similarity': similarity,
                            'index1': i,
                            'index2': j
                        })
                        self.stats['llm_decisions_needed'] += 1
                        continue  # Don't merge yet

                # Merge if decision is positive
                if should_merge:
                    merged_indices.add(j)

                    if concept2['term'] not in aliases:
                        aliases.append(concept2['term'])

                    if concept2.get('definition') not in definitions:
                        definitions.append(concept2['definition'])

                    categories[concept2.get('category', 'general')] += 1
                    importance_scores[concept2.get('importance', 'medium')] += 1
                    sources.add(concept2.get('source_file', 'unknown'))
                    confidences.append(concept2.get('confidence', 0.5))

                    evidence.append({
                        'chunk_id': concept2.get('source_chunk_id', concept2.get('chunk_id', '')),
                        'quote': concept2.get('quote', ''),
                        'page': concept2.get('page', 1),
                        'merge_reason': merge_reason
                    })

            # Create resolved entity
            entity = ResolvedEntity(
                canonical_term=canonical_term,
                aliases=aliases,
                definitions=definitions,
                categories=categories,
                importance_scores=importance_scores,
                evidence=evidence,
                sources=sources,
                avg_confidence=sum(confidences) / len(confidences) if confidences else 0.0
            )

            entities.append(entity)
            self.stats['entities_created'] += 1

            if i % 10 == 0:
                logger.debug(f"Processed {i}/{len(concepts)} concepts...")

        self.stats['concepts_processed'] = len(concepts)

        logger.info(f"Automatic resolution complete: {len(entities)} entities created")
        logger.info(f"Exact matches: {self.stats['exact_matches']}, Semantic: {self.stats['semantic_matches']}")
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
