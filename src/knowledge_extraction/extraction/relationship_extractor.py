#!/usr/bin/env python3
"""
Relationship Extractor - MCP Native Version

Extracts typed relationships between entities using co-occurrence
and Claude Code for classification.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class RelationshipExtractor:
    """
    Extract typed relationships between entities.
    
    Strategy:
    1. Find co-occurring entities in chunks
    2. Generate classification prompts for Claude Code
    3. Parse Claude's relationship classifications
    """

    RELATIONSHIP_TYPES = [
        'CAUSES',        # X causes Y
        'ENABLES',       # X enables/facilitates Y
        'PREVENTS',      # X prevents/blocks Y
        'REQUIRES',      # X requires Y
        'CONTRADICTS',   # X contradicts Y
        'EXTENDS',       # X extends/builds-on Y
        'PART_OF',       # X is part of Y
        'EXAMPLE_OF',    # X is an example of Y
        'RELATED',       # General relationship
    ]

    CLASSIFICATION_PROMPT = """Classify the relationship between these two concepts:

**Concept A**: {term_a}
Definition: {definition_a}

**Concept B**: {term_b}
Definition: {definition_b}

**Context** (they appear together in this passage):
{context}

What is the relationship between A and B?

**Relationship Types**:
- CAUSES: A causes B (e.g., "Stress causes poor decisions")
- ENABLES: A enables/facilitates B (e.g., "Heuristics enable fast thinking")
- PREVENTS: A prevents/blocks B (e.g., "Deliberation prevents impulsive choices")
- REQUIRES: A requires B (e.g., "System 2 requires cognitive effort")
- CONTRADICTS: A contradicts B (e.g., "Fast thinking contradicts careful analysis")
- EXTENDS: A extends/builds on B (e.g., "Prospect theory extends utility theory")
- PART_OF: A is part of B (e.g., "Anchoring is part of System 1")
- EXAMPLE_OF: A is an example of B (e.g., "Loss aversion is an example of cognitive bias")
- RELATED: General relationship (use only if none above fit)

**Output Format** (JSON):
```json
{{
  "relationship_type": "CAUSES",
  "direction": "A->B" | "B->A" | "bidirectional",
  "strength": 0.0-1.0,
  "explanation": "One sentence explaining this relationship",
  "confidence": 0.0-1.0
}}
```

Classify the relationship:"""

    def __init__(self, context_window: int = 3) -> None:
        """
        Initialize relationship extractor.

        Args:
            context_window: How many chunks on either side to consider
        """
        self.context_window = context_window

        self.stats: dict[str, Any] = {
            'co_occurrences_found': 0,
            'relationships_extracted': 0,
            'by_type': defaultdict(int)
        }

    def find_co_occurrences(
        self,
        entities: list[dict[str, Any]],
        chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Find entities that appear together in same/nearby chunks.

        Args:
            entities: List of entity dicts with 'canonical_term', 'evidence'
            chunks: List of chunk dicts with 'chunk_id', 'text'

        Returns:
            List of co-occurrence dicts with entity pairs and context
        """
        # Validate inputs
        if not isinstance(entities, list):
            raise ValueError(f"entities must be a list, got {type(entities).__name__}")
        if not isinstance(chunks, list):
            raise ValueError(f"chunks must be a list, got {type(chunks).__name__}")
        if not entities:
            raise ValueError("entities list cannot be empty")
        if not chunks:
            raise ValueError("chunks list cannot be empty")

        # Build entity -> chunks mapping
        entity_to_chunks = defaultdict(list)
        invalid_chunk_count = 0
        total_evidence_count = 0

        for entity in entities:
            for evidence in entity.get('evidence', []):
                total_evidence_count += 1
                chunk_id = evidence.get('chunk_id')
                if chunk_id and chunk_id != "unknown":
                    entity_to_chunks[entity['canonical_term']].append(chunk_id)
                else:
                    invalid_chunk_count += 1

        # Warn if significant chunk_id issues
        if total_evidence_count > 0:
            invalid_pct = (invalid_chunk_count / total_evidence_count) * 100
            if invalid_pct > 10:
                logger.warning(
                    f"⚠️ {invalid_chunk_count}/{total_evidence_count} evidence items "
                    f"({invalid_pct:.1f}%) have invalid chunk_ids! "
                    f"This will significantly reduce relationship detection."
                )
                logger.warning("💡 Consider running chunk_id_repair.py to fix entities.json")
            elif invalid_chunk_count > 0:
                logger.info(
                    f"Note: {invalid_chunk_count}/{total_evidence_count} evidence items "
                    f"({invalid_pct:.1f}%) have invalid chunk_ids (skipped)"
                )

        # Build chunk index for quick lookup
        chunk_index = {c['chunk_id']: c for c in chunks}

        # Find co-occurrences
        co_occurrences = []
        entity_terms = list(entity_to_chunks.keys())

        for i, term_a in enumerate(entity_terms):
            for term_b in entity_terms[i+1:]:
                # Find common chunks
                chunks_a = set(entity_to_chunks[term_a])
                chunks_b = set(entity_to_chunks[term_b])
                common_chunks = chunks_a & chunks_b

                if common_chunks:
                    # They appear in same chunk
                    for chunk_id in common_chunks:
                        chunk = chunk_index.get(chunk_id)
                        if chunk:
                            co_occurrences.append({
                                'term_a': term_a,
                                'term_b': term_b,
                                'chunk_id': chunk_id,
                                'context': chunk['text']
                            })

        self.stats['co_occurrences_found'] = len(co_occurrences)

        return co_occurrences

    def generate_classification_prompt(
        self,
        co_occurrence: dict[str, Any],
        entities_dict: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Generate prompt for Claude Code to classify relationship.

        Args:
            co_occurrence: Dict with term_a, term_b, context
            entities_dict: Dict mapping term -> entity data

        Returns:
            {
                'prompt': str,
                'metadata': {...}
            }
        """
        # Validate inputs
        if not isinstance(co_occurrence, dict):
            raise ValueError(f"co_occurrence must be a dict, got {type(co_occurrence).__name__}")
        if not isinstance(entities_dict, dict):
            raise ValueError(f"entities_dict must be a dict, got {type(entities_dict).__name__}")
        if 'term_a' not in co_occurrence or 'term_b' not in co_occurrence:
            raise ValueError("co_occurrence must contain 'term_a' and 'term_b'")

        term_a = co_occurrence['term_a']
        term_b = co_occurrence['term_b']

        entity_a = entities_dict.get(term_a, {})
        entity_b = entities_dict.get(term_b, {})

        # Get first definition for each
        def_a = entity_a.get('definitions', [''])[0] if entity_a.get('definitions') else ''
        def_b = entity_b.get('definitions', [''])[0] if entity_b.get('definitions') else ''

        prompt = self.CLASSIFICATION_PROMPT.format(
            term_a=term_a,
            definition_a=def_a,
            term_b=term_b,
            definition_b=def_b,
            context=co_occurrence['context'][:500]  # Limit context length
        )

        return {
            'prompt': prompt,
            'metadata': {
                'term_a': term_a,
                'term_b': term_b,
                'chunk_id': co_occurrence['chunk_id'],
                'task': 'relationship_classification'
            }
        }

    def parse_classification_response(
        self,
        response_text: str,
        metadata: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Parse Claude's relationship classification.

        Returns:
            Relationship dict or None if invalid
        """
        # Validate inputs
        if not isinstance(response_text, str):
            raise ValueError(f"response_text must be a string, got {type(response_text).__name__}")
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata must be a dict, got {type(metadata).__name__}")
        if 'term_a' not in metadata or 'term_b' not in metadata:
            raise ValueError("metadata must contain 'term_a' and 'term_b'")

        try:
            import re

            # Extract JSON
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(response_text)

            rel_type = data.get('relationship_type', 'RELATED')
            direction = data.get('direction', 'A->B')

            # Determine source and target based on direction
            if direction == 'A->B':
                source = metadata['term_a']
                target = metadata['term_b']
            elif direction == 'B->A':
                source = metadata['term_b']
                target = metadata['term_a']
            else:  # bidirectional
                source = metadata['term_a']
                target = metadata['term_b']
                rel_type = 'RELATED'  # Bidirectional usually means general relationship

            relationship = {
                'source': source,
                'target': target,
                'type': rel_type,
                'strength': data.get('strength', 0.5),
                'explanation': data.get('explanation', ''),
                'confidence': data.get('confidence', 0.5),
                'evidence': [{
                    'chunk_id': metadata['chunk_id']
                }]
            }

            self.stats['relationships_extracted'] += 1
            self.stats['by_type'][rel_type] += 1

            return relationship

        except Exception as e:
            logger.error(f"Error parsing relationship response: {e}")
            return None

    def create_classification_batch(
        self,
        entities: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        output_path: Path
    ) -> Path:
        """
        Create batch file with relationship classification prompts.

        Args:
            entities: List of entity dicts
            chunks: List of chunk dicts
            output_path: Where to save batch file

        Returns:
            Path to batch file
        """
        # Validate inputs
        if not isinstance(entities, list):
            raise ValueError(f"entities must be a list, got {type(entities).__name__}")
        if not isinstance(chunks, list):
            raise ValueError(f"chunks must be a list, got {type(chunks).__name__}")
        if not entities:
            raise ValueError("entities list cannot be empty")
        if not chunks:
            raise ValueError("chunks list cannot be empty")
        if not isinstance(output_path, Path):
            output_path = Path(output_path)

        # Find co-occurrences
        logger.info(f"Finding co-occurrences in {len(chunks)} chunks...")
        co_occurrences = self.find_co_occurrences(entities, chunks)

        logger.info(f"Found {len(co_occurrences)} entity co-occurrences")

        # Create entity lookup dict
        entities_dict = {e['canonical_term']: e for e in entities}

        # Generate prompts
        batch_data: dict[str, Any] = {
            'task': 'relationship_classification',
            'total_pairs': len(co_occurrences),
            'prompts': []
        }

        for co_occ in co_occurrences:
            prompt_data = self.generate_classification_prompt(co_occ, entities_dict)
            batch_data['prompts'].append(prompt_data)

        output_path = Path(output_path)
        output_path.write_text(json.dumps(batch_data, indent=2), encoding='utf-8')

        logger.info(f"✓ Created batch file: {output_path}")
        logger.info(f"  Total relationship classification prompts: {len(co_occurrences)}")

        return output_path

    def parse_classification_batch(
        self,
        responses_file: Path,
        output_path: Path
    ) -> list[dict[str, Any]]:
        """
        Parse Claude's relationship classifications.

        Args:
            responses_file: File with Claude's responses
            output_path: Where to save relationships

        Returns:
            List of relationship dicts
        """
        # Validate inputs
        if not isinstance(responses_file, Path):
            responses_file = Path(responses_file)
        if not isinstance(output_path, Path):
            output_path = Path(output_path)
        if not responses_file.exists():
            raise FileNotFoundError(f"Responses file not found: {responses_file}")

        responses_data = json.loads(responses_file.read_text(encoding='utf-8'))

        relationships = []
        errors = []

        for i, response in enumerate(responses_data.get('responses', [])):
            try:
                metadata = response['metadata']
                relationship = self.parse_classification_response(
                    response['response_text'],
                    metadata
                )
                if relationship:
                    relationships.append(relationship)
            except Exception as e:
                errors.append(f"Response {i}: {str(e)}")

        # Save relationships
        output_path = Path(output_path)
        output_data = {
            'num_relationships': len(relationships),
            'stats': dict(self.stats),
            'relationships': relationships
        }
        output_path.write_text(json.dumps(output_data, indent=2), encoding='utf-8')

        logger.info(f"✓ Parsed {len(relationships)} relationships")
        logger.info("  By type:")
        for rel_type, count in sorted(self.stats['by_type'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"    {rel_type}: {count}")

        if errors:
            logger.warning(f"\n⚠ {len(errors)} parsing errors")

        return relationships


# =========================================================================
# CLI
# =========================================================================

def main() -> int:
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract relationships between entities (MCP version)'
    )
    parser.add_argument('command', choices=['create-batch', 'parse-responses'],
                       help='Command to execute')
    parser.add_argument('--entities', type=str,
                       help='Path to entities JSON file')
    parser.add_argument('--chunks', type=str,
                       help='Path to chunks JSON file')
    parser.add_argument('--responses', type=str,
                       help='Path to responses JSON file')
    parser.add_argument('--output', type=str, default='relationships.json',
                       help='Output file')

    args = parser.parse_args()

    extractor = RelationshipExtractor()

    if args.command == 'create-batch':
        if not args.entities or not args.chunks:
            logger.error("Error: --entities and --chunks required")
            return 1

        # Load data
        entities_data = json.loads(Path(args.entities).read_text(encoding='utf-8'))
        chunks_data = json.loads(Path(args.chunks).read_text(encoding='utf-8'))

        entities = entities_data.get('entities', [])
        chunks = chunks_data.get('chunks', [])

        print(f"Loaded {len(entities)} entities and {len(chunks)} chunks")

        # Create batch
        batch_file = extractor.create_classification_batch(
            entities, chunks, Path(args.output)
        )

        print("\nNext steps:")
        print(f"1. Use Claude Code to process: {batch_file}")
        print("2. Claude Code will classify each relationship")
        print("3. Run: python relationship_extractor_mcp.py parse-responses --responses <file>")

    elif args.command == 'parse-responses':
        if not args.responses:
            logger.error("Error: --responses required")
            return 1

        relationships = extractor.parse_classification_batch(
            Path(args.responses),
            Path(args.output)
        )

        print(f"\nRelationships saved to: {args.output}")

    return 0


if __name__ == "__main__":
    main()
