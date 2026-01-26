#!/usr/bin/env python3
"""
Type Existing Relationships Based on Entity Categories

Upgrades generic "RELATED" or "CO_OCCURS" relationships to semantic types
based on entity category pairs (method, theory, phenomenon, etc.).

This is different from infer_relationships_tfidf.py which creates NEW
relationships. This script TYPES existing co-occurrence relationships.

Usage:
    python3 type_relationships.py entities.json relationships.json

Output:
    - relationships.json (updated with types)
    - relationships_backup_before_typing.json (backup)
"""

import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


# Relationship type inference based on category pairs
CATEGORY_RELATIONSHIPS = {
    # Method relationships
    ('method', 'concept'): 'APPLIES_TO',
    ('method', 'extracted'): 'APPLIES_TO',
    ('method', 'problem'): 'ADDRESSES',
    ('method', 'phenomenon'): 'STUDIES',
    ('method', 'metric'): 'USES',
    ('method', 'method'): 'VARIANT_OF',
    ('method', 'theory'): 'IMPLEMENTS',
    ('method', 'tool'): 'USES',
    ('method', 'principle'): 'GUIDED_BY',
    ('method', 'bias'): 'MITIGATES',

    # Theory relationships
    ('theory', 'concept'): 'DEFINES',
    ('theory', 'extracted'): 'DEFINES',
    ('theory', 'phenomenon'): 'EXPLAINS',
    ('theory', 'method'): 'GROUNDS',
    ('theory', 'principle'): 'CONTAINS',
    ('theory', 'theory'): 'RELATED_TO',

    # Concept relationships
    ('concept', 'concept'): 'RELATED_TO',
    ('concept', 'extracted'): 'RELATED_TO',
    ('extracted', 'extracted'): 'RELATED_TO',
    ('concept', 'phenomenon'): 'MANIFESTS_AS',
    ('concept', 'metric'): 'QUANTIFIED_BY',

    # Phenomenon relationships
    ('phenomenon', 'phenomenon'): 'RELATED_TO',
    ('phenomenon', 'concept'): 'INVOLVES',
    ('phenomenon', 'extracted'): 'INVOLVES',
    ('phenomenon', 'method'): 'STUDIED_BY',
    ('phenomenon', 'metric'): 'MEASURED_BY',

    # Principle relationships
    ('principle', 'method'): 'GUIDES',
    ('principle', 'concept'): 'GOVERNS',
    ('principle', 'extracted'): 'GOVERNS',
    ('principle', 'principle'): 'RELATED_TO',

    # Metric relationships
    ('metric', 'concept'): 'MEASURES',
    ('metric', 'extracted'): 'MEASURES',
    ('metric', 'phenomenon'): 'QUANTIFIES',
    ('metric', 'method'): 'EVALUATES',

    # Problem relationships
    ('problem', 'method'): 'SOLVED_BY',
    ('problem', 'concept'): 'INVOLVES',
    ('problem', 'phenomenon'): 'CAUSED_BY',

    # Tool relationships
    ('tool', 'method'): 'IMPLEMENTS',
    ('tool', 'concept'): 'OPERATES_ON',
    ('tool', 'extracted'): 'OPERATES_ON',

    # Other category relationships
    ('bias', 'phenomenon'): 'AFFECTS',
    ('bias', 'method'): 'AFFECTS',
    ('assessment', 'concept'): 'EVALUATES',
    ('assessment', 'extracted'): 'EVALUATES',
    ('cognitive_process', 'phenomenon'): 'PRODUCES',
    ('feature', 'concept'): 'CHARACTERIZES',
    ('feature', 'extracted'): 'CHARACTERIZES',
}


def load_json(file_path: Path) -> dict:
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(file_path: Path, data: dict) -> None:
    """Save JSON file with pretty printing"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def infer_relationship_type(
    source_cat: str,
    target_cat: str
) -> str:
    """
    Infer relationship type from entity categories.

    Tries both (source, target) and (target, source) combinations.
    Returns 'RELATED_TO' if no specific mapping found.
    """
    # Try direct mapping
    key1 = (source_cat, target_cat)
    if key1 in CATEGORY_RELATIONSHIPS:
        return CATEGORY_RELATIONSHIPS[key1]

    # Try reverse mapping
    key2 = (target_cat, source_cat)
    if key2 in CATEGORY_RELATIONSHIPS:
        return CATEGORY_RELATIONSHIPS[key2]

    # Default
    return 'RELATED_TO'


def type_relationships(
    entities_file: Path,
    relationships_file: Path,
    backup: bool = True
) -> Tuple[int, int, Dict[str, int]]:
    """
    Type relationships based on entity categories.

    Args:
        entities_file: Path to entities.json
        relationships_file: Path to relationships.json
        backup: Whether to create backup before modifying

    Returns:
        Tuple of (total_relationships, upgraded_count, type_distribution)
    """
    # Load data
    print(f"Loading {entities_file}...")
    entities_data = load_json(entities_file)

    # Handle both list and dict formats
    if isinstance(entities_data, dict):
        entities = entities_data.get('entities', entities_data)
    else:
        entities = entities_data

    print(f"Loading {relationships_file}...")
    rels_data = load_json(relationships_file)

    # Handle both list and dict formats
    if isinstance(rels_data, dict):
        relationships = rels_data.get('relationships', rels_data)
    else:
        relationships = rels_data
        rels_data = {'relationships': relationships}

    print(f"  Loaded {len(entities)} entities")
    print(f"  Loaded {len(relationships)} relationships\n")

    # Build category lookup
    print("Building entity category index...")
    entity_category = {}
    for entity in entities:
        term = entity.get('canonical_term') or entity.get('term') or entity.get('id')
        category = entity.get('category', 'concept')
        entity_category[term] = category

    # Backup original file
    if backup:
        backup_path = relationships_file.parent / f"{relationships_file.stem}_backup_before_typing{relationships_file.suffix}"
        print(f"Creating backup: {backup_path}")
        shutil.copy(relationships_file, backup_path)

    # Type relationships
    print("Typing relationships...\n")
    upgraded = 0

    for rel in relationships:
        source = rel.get('source')
        target = rel.get('target')

        if not source or not target:
            continue

        source_cat = entity_category.get(source, 'concept')
        target_cat = entity_category.get(target, 'concept')

        # Infer type
        new_type = infer_relationship_type(source_cat, target_cat)

        # Update if changed
        old_type = rel.get('type', 'RELATED')
        if old_type != new_type:
            rel['type'] = new_type
            upgraded += 1

    # Count types
    type_counts: Dict[str, int] = {}
    for rel in relationships:
        rel_type = rel.get('type', 'RELATED')
        type_counts[rel_type] = type_counts.get(rel_type, 0) + 1

    # Save updated relationships
    print(f"Saving updated relationships to {relationships_file}...")
    save_json(relationships_file, rels_data)

    return len(relationships), upgraded, type_counts


def main():
    parser = argparse.ArgumentParser(
        description='Type existing relationships based on entity categories'
    )
    parser.add_argument(
        'entities_file',
        type=Path,
        help='Path to entities.json file'
    )
    parser.add_argument(
        'relationships_file',
        type=Path,
        help='Path to relationships.json file'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup file'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.entities_file.exists():
        print(f"❌ Error: {args.entities_file} not found")
        return 1

    if not args.relationships_file.exists():
        print(f"❌ Error: {args.relationships_file} not found")
        return 1

    # Type relationships
    print("=" * 70)
    print("RELATIONSHIP TYPING")
    print("=" * 70)
    print()

    total, upgraded, type_counts = type_relationships(
        args.entities_file,
        args.relationships_file,
        backup=not args.no_backup
    )

    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Total relationships: {total}")
    print(f"Types upgraded:      {upgraded} ({100 * upgraded / total:.1f}%)")
    print(f"Unique types:        {len(type_counts)}")
    print()
    print("Type distribution:")
    print("-" * 70)

    for rel_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        print(f"  {rel_type:<20} {count:>6} ({pct:>5.1f}%)")

    print()
    print("=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    exit(main())
