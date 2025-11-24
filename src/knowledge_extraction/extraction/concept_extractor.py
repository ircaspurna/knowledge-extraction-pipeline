#!/usr/bin/env python3
"""
MCP-Native Concept Extractor - Returns prompts for Claude Code

This version doesn't call the Anthropic API directly. Instead, it generates
prompts that Claude Code executes, saving money and following MCP patterns.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ExtractedConcept:
    """Concept extracted from text with full provenance"""
    term: str
    definition: str
    category: str
    importance: str
    justification: str
    quote: str
    chunk_id: str
    source_file: str
    page: int
    confidence: float = 0.0
    validation_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'term': self.term,
            'definition': self.definition,
            'category': self.category,
            'importance': self.importance,
            'justification': self.justification,
            'quote': self.quote,
            'chunk_id': self.chunk_id,
            'source_file': self.source_file,
            'page': self.page,
            'confidence': self.confidence,
            'validation_notes': self.validation_notes
        }


class ConceptExtractorMCP:
    """
    Generate extraction prompts for Claude Code to execute.

    This version returns prompts instead of calling APIs directly.
    Claude Code handles the actual LLM calls.
    """

    # Fallback prompts if config file not found
    FALLBACK_EXTRACTION_PROMPT = """You are a research assistant extracting key concepts from academic texts.

Extract important concepts from this passage. For EACH concept, provide:

1. **Term**: The exact phrase from the text (2-5 words preferred)
2. **Definition**: One clear sentence defining the concept
3. **Category**: Choose one: theory | method | phenomenon | tool | principle | bias | heuristic
4. **Importance**: Choose one: critical | high | medium | low
5. **Justification**: One sentence explaining why this importance level
6. **Quote**: Exact quote from passage supporting this concept (max 100 words)

**Important**:
- Only extract concepts explicitly discussed in the passage
- Use exact terminology from the text
- Quotes must be VERBATIM from the passage
- Focus on concepts, not examples or anecdotes
- Aim for 3-7 concepts per passage

**Output Format** (JSON):
```json
{{
  "concepts": [
    {{
      "term": "Loss Aversion",
      "definition": "The tendency for losses to loom larger than equivalent gains.",
      "category": "bias",
      "importance": "critical",
      "justification": "Fundamental principle explaining numerous decision-making patterns.",
      "quote": "losses loom larger than gains in people's evaluations"
    }}
  ]
}}
```

**Passage**:
{passage}

Extract concepts:"""

    FALLBACK_VALIDATION_PROMPT = """Validate this concept extraction:

**Concept**: {term}
**Definition**: {definition}
**Quote**: "{quote}"
**Original Passage**: {passage}

Check:
1. Is the quote actually from the passage? (VERBATIM check)
2. Does the definition accurately describe the term?
3. Is the quote sufficient evidence for this concept?

**Output Format** (JSON):
```json
{{
  "quote_valid": true/false,
  "definition_accurate": true/false,
  "evidence_sufficient": true/false,
  "issues": ["list any problems"],
  "confidence": 0.0-1.0
}}
```

Validate:"""

    def __init__(self, max_concepts_per_chunk: int | None = None, config_path: str | None = None, domain: str = 'psychology') -> None:
        """
        Initialize concept extractor with domain-specific configuration.

        Args:
            max_concepts_per_chunk: Maximum concepts to extract per chunk (None = use domain default)
            config_path: Path to prompts.yaml config file
            domain: Domain for extraction (psychology, computer_science, medicine, etc.)
        """
        self.domain = domain

        # Load domain configuration first
        self.domain_config = self._load_domain_config(domain)

        # Use domain's max_concepts_per_chunk if not specified
        if max_concepts_per_chunk is None:
            max_concepts_per_chunk = self.domain_config['extraction_params'].get('max_concepts_per_chunk', 10)

        self.max_concepts_per_chunk = max_concepts_per_chunk

        # Load prompts from YAML config
        if config_path is None:
            # Default to config/prompts.yaml in same directory as this script
            config_path_obj: Path = Path(__file__).parent / "config" / "prompts.yaml"
        else:
            config_path_obj = Path(config_path)

        self.config = self._load_config(config_path_obj)

        # Inject domain-specific content into prompts
        self._inject_domain_content()

        # Extract prompt templates (now domain-customized)
        extraction_config = self.config['concept_extraction']
        self.EXTRACTION_PROMPT = extraction_config['user_template']
        self.extraction_filters = extraction_config.get('filters', {})

        validation_config = self.config['concept_validation']
        self.VALIDATION_PROMPT = validation_config['user_template']

        # Track extraction stats
        self.stats: dict[str, Any] = {
            'chunks_processed': 0,
            'concepts_extracted': 0,
            'concepts_validated': 0,
            'validation_failures': 0,
            'avg_confidence': 0.0,
            'domain': domain
        }

        num_categories = len(self.domain_config['categories']['core']) + len(self.domain_config['categories']['domain_specific'])
        logger.info(f"ConceptExtractor initialized with domain: {self.domain_config['name']}")
        logger.debug(f"Max concepts/chunk: {self.max_concepts_per_chunk}, Categories: {num_categories}")

    def _load_config(self, config_path: Path) -> dict[str, Any]:
        """Load prompts configuration from YAML"""
        try:
            with open(config_path, encoding='utf-8') as f:
                return yaml.safe_load(f)  # type: ignore[no-any-return]
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using fallback prompts")
            return self._get_fallback_config()
        except Exception as e:
            logger.warning(f"Error loading config: {e}, using fallback prompts")
            return self._get_fallback_config()

    def _get_fallback_config(self) -> dict[str, Any]:
        """Return fallback config if YAML loading fails"""
        return {
            'concept_extraction': {
                'user_template': self.FALLBACK_EXTRACTION_PROMPT,
                'filters': {
                    'min_word_count': 20,
                    'skip_patterns': []  # Page markers no longer added to text
                }
            },
            'concept_validation': {
                'user_template': self.FALLBACK_VALIDATION_PROMPT
            }
        }

    def _load_domain_config(self, domain: str) -> dict[str, Any]:
        """
        Load domain configuration from domains.yaml

        Args:
            domain: Domain name or alias

        Returns:
            Domain configuration dict
        """
        try:
            domains_file = Path(__file__).parent / "config" / "domains.yaml"
            with open(domains_file, encoding='utf-8') as f:
                domains_data = yaml.safe_load(f)

            # Handle aliases
            if domain in domains_data.get('aliases', {}):
                original_domain = domain
                domain = domains_data['aliases'][domain]
                logger.debug(f"Domain alias '{original_domain}' → '{domain}'")

            # Get domain config
            if domain not in domains_data.get('domains', {}):
                default_domain = domains_data.get('default_domain', 'psychology')
                logger.warning(f"Domain '{domain}' not found, using default: '{default_domain}'")
                domain = default_domain

            return domains_data['domains'][domain]  # type: ignore[no-any-return]

        except FileNotFoundError:
            logger.warning("domains.yaml not found, using fallback psychology config")
            return self._get_fallback_domain_config()
        except Exception as e:
            logger.warning(f"Error loading domain config: {e}, using fallback")
            return self._get_fallback_domain_config()

    def _get_fallback_domain_config(self) -> dict[str, Any]:
        """Return fallback domain config (psychology)"""
        return {
            'name': 'Psychology & Cognitive Science',
            'description': 'Fallback domain configuration',
            'categories': {
                'core': ['theory', 'method', 'phenomenon', 'principle', 'metric', 'tool'],
                'domain_specific': ['bias', 'heuristic', 'cognitive_process']
            },
            'extraction_params': {
                'require_definition': True,
                'require_explanation': True,
                'extract_mentioned_only': False,
                'max_concepts_per_chunk': 10,
                'min_explanation_words': 15
            },
            'examples': []
        }

    def _inject_domain_content(self) -> None:
        """Inject domain-specific content into prompt templates"""
        if 'concept_extraction' not in self.config:
            return

        user_template = self.config['concept_extraction']['user_template']

        # Build category string
        categories = self.domain_config['categories']
        all_categories = categories['core'] + categories['domain_specific']
        category_str = ' | '.join(all_categories)

        # Build domain-specific examples
        examples_str = self._format_domain_examples()

        # Build domain-specific rules
        rules_str = self._format_domain_rules()

        # Inject into template
        user_template = user_template.replace('{domain_categories}', category_str)
        user_template = user_template.replace('{domain_examples}', examples_str)
        user_template = user_template.replace('{domain_rules}', rules_str)

        self.config['concept_extraction']['user_template'] = user_template

    def _format_domain_examples(self) -> str:
        """Format domain-specific examples for prompt injection"""
        examples = self.domain_config.get('examples', [])
        if not examples:
            return "    (No domain-specific examples available)"

        formatted = []
        for i, ex in enumerate(examples, 1):
            formatted.append(f"""
    Example {i} ({self.domain_config['name']}):
    Passage: "{ex.get('passage', '')[:150]}..."

    ✓ Extract:
    {{
      "term": "{ex.get('term', '')}",
      "definition": "{ex.get('definition', '')}",
      "category": "{ex.get('category', '')}",
      "importance": "high",
      "justification": "Key {ex.get('category', 'concept')} in {self.domain_config['name']}.",
      "quote": "..."
    }}""")

        return '\n'.join(formatted)

    def _format_domain_rules(self) -> str:
        """Format domain-specific extraction rules"""
        params = self.domain_config['extraction_params']

        rules = []

        # Rule about requiring definition
        if params.get('require_definition', True):
            rules.append("✓ Only extract concepts that are EXPLAINED or DEFINED")
        else:
            rules.append("✓ Extract technical terms even if not defined in THIS passage")

        # Rule about requiring explanation
        if params.get('require_explanation', True):
            rules.append("✗ Skip mere mentions without explanation")
        else:
            rules.append("✓ Extract known terms that are just used or mentioned")

        # Rule about examples
        rules.append("✗ Skip examples that illustrate concepts (extract the concept, not the example)")

        # Rule about methodology
        rules.append("✗ Skip methodology/procedural steps unless they're novel methods")

        # Rule about citations
        rules.append("✗ Skip citations to other works")

        # Additional domain-specific rules
        if params.get('extract_abbreviations'):
            rules.append("✓ Extract abbreviations and acronyms")

        if params.get('extract_acronyms'):
            rules.append("✓ Pay attention to acronyms and their expansions")

        return '\n    '.join(rules)

    def should_process_chunk(self, chunk_text: str) -> bool:
        """
        Determine if chunk is substantive enough to process.

        Args:
            chunk_text: Chunk text to evaluate

        Returns:
            True if chunk should be processed, False to skip
        """
        # Check minimum word count
        min_words = self.extraction_filters.get('min_word_count', 20)
        word_count = len(chunk_text.split())

        if word_count < min_words:
            return False

        # Check skip patterns
        skip_patterns = self.extraction_filters.get('skip_patterns', [])
        for pattern in skip_patterns:
            if re.match(pattern, chunk_text.strip()):
                return False

        return True

    def generate_extraction_prompt(
        self,
        chunk_text: str,
        chunk_id: str,
        source_file: str,
        page: int
    ) -> dict[str, Any]:
        """
        Generate extraction prompt for Claude Code to execute.

        Returns:
            {
                'prompt': str,
                'metadata': {
                    'chunk_id': str,
                    'source_file': str,
                    'page': int,
                    'task': 'extraction'
                }
            }
        """
        prompt = self.EXTRACTION_PROMPT.format(passage=chunk_text)

        return {
            'prompt': prompt,
            'metadata': {
                'chunk_id': chunk_id,
                'source_file': source_file,
                'page': page,
                'task': 'extraction'
            }
        }

    def parse_extraction_response(
        self,
        response_text: str,
        chunk_id: str,
        source_file: str,
        page: int
    ) -> list[ExtractedConcept]:
        """
        Parse Claude's response into ExtractedConcept objects.
        
        Args:
            response_text: Claude's JSON response
            chunk_id: Chunk identifier
            source_file: Source filename
            page: Page number
        
        Returns:
            List of validated concepts
        """
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                concepts_data = json.loads(json_match.group(1))
            else:
                # Try parsing entire response as JSON
                concepts_data = json.loads(response_text)

            # Create ExtractedConcept objects
            concepts = []
            for concept_dict in concepts_data.get('concepts', [])[:self.max_concepts_per_chunk]:
                concept = ExtractedConcept(
                    term=concept_dict.get('term', ''),
                    definition=concept_dict.get('definition', ''),
                    category=concept_dict.get('category', 'general'),
                    importance=concept_dict.get('importance', 'medium'),
                    justification=concept_dict.get('justification', ''),
                    quote=concept_dict.get('quote', ''),
                    chunk_id=chunk_id,
                    source_file=source_file,
                    page=page
                )
                concepts.append(concept)

            self.stats['chunks_processed'] += 1
            self.stats['concepts_extracted'] += len(concepts)

            return concepts

        except Exception as e:
            logger.error(f"Error parsing extraction response for {chunk_id}: {e}")
            return []

    def generate_validation_prompt(
        self,
        concept: ExtractedConcept,
        chunk_text: str
    ) -> dict[str, Any]:
        """
        Generate validation prompt for Claude Code.
        
        Returns:
            {
                'prompt': str,
                'metadata': {
                    'concept_term': str,
                    'chunk_id': str,
                    'task': 'validation'
                }
            }
        """
        prompt = self.VALIDATION_PROMPT.format(
            term=concept.term,
            definition=concept.definition,
            quote=concept.quote,
            passage=chunk_text
        )

        return {
            'prompt': prompt,
            'metadata': {
                'concept_term': concept.term,
                'chunk_id': concept.chunk_id,
                'task': 'validation'
            }
        }

    def parse_validation_response(
        self,
        response_text: str,
        concept: ExtractedConcept
    ) -> ExtractedConcept:
        """
        Parse validation response and update concept.
        
        Args:
            response_text: Claude's validation JSON
            concept: Concept to update
        
        Returns:
            Updated concept with confidence and validation notes
        """
        try:
            # Extract JSON
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                validation_data = json.loads(json_match.group(1))
            else:
                validation_data = json.loads(response_text)

            # Update concept
            quote_valid = validation_data.get('quote_valid', False)
            definition_accurate = validation_data.get('definition_accurate', False)
            evidence_sufficient = validation_data.get('evidence_sufficient', False)
            issues = validation_data.get('issues', [])

            # Calculate confidence
            confidence = validation_data.get('confidence', 0.0)

            # Adjust confidence based on validation checks
            if not quote_valid:
                confidence *= 0.5
                issues.append("Quote not found in passage")
            if not definition_accurate:
                confidence *= 0.7
                issues.append("Definition doesn't match term")
            if not evidence_sufficient:
                confidence *= 0.8
                issues.append("Insufficient evidence")

            concept.confidence = confidence
            concept.validation_notes = issues

            if confidence < 0.5:
                self.stats['validation_failures'] += 1
            else:
                self.stats['concepts_validated'] += 1

            return concept

        except Exception as e:
            logger.error(f"Error parsing validation response for {concept.term}: {e}")
            concept.confidence = 0.3
            concept.validation_notes = [f"Validation error: {str(e)}"]
            return concept

    def get_stats(self) -> dict[str, Any]:
        """Get extraction statistics"""
        if self.stats['concepts_extracted'] > 0:
            self.stats['avg_confidence'] = (
                self.stats['concepts_validated'] / self.stats['concepts_extracted']
            )
        return self.stats

    def save_concepts(
        self,
        concepts: list[ExtractedConcept],
        output_path: Path
    ) -> None:
        """Save concepts to JSON"""
        output_path = Path(output_path)

        data = {
            'num_concepts': len(concepts),
            'extraction_stats': self.get_stats(),
            'concepts': [c.to_dict() for c in concepts]
        }

        output_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        logger.info(f"Saved {len(concepts)} concepts to: {output_path}")


# =========================================================================
# Batch Processing Helper
# =========================================================================

def create_batch_extraction_file(
    chunks: list[dict[str, Any]],
    output_path: Path,
    filter_non_substantive: bool = True
) -> Path:
    """
    Create a batch file with all extraction prompts.
    Claude Code can process this sequentially.

    Args:
        chunks: List of chunk dicts
        output_path: Where to save batch file
        filter_non_substantive: Skip chunks with headers, page numbers, etc.

    Returns:
        Path to batch file
    """
    # Validate inputs
    if not isinstance(chunks, list):
        raise ValueError(f"chunks must be a list, got {type(chunks).__name__}")

    if not chunks:
        raise ValueError("chunks list cannot be empty")

    # Validate chunk structure
    required_fields = ['text', 'chunk_id']
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk {i} must be a dict, got {type(chunk).__name__}")

        missing = [f for f in required_fields if f not in chunk]
        if missing:
            raise ValueError(f"Chunk {i} missing required fields: {missing}")

    if not output_path:
        raise ValueError("output_path cannot be None or empty")

    extractor = ConceptExtractorMCP()

    batch_data: dict[str, Any] = {
        'task': 'concept_extraction_batch',
        'total_chunks': len(chunks),
        'prompts': []
    }

    filtered_count: int = 0

    for chunk in chunks:
        chunk_text = chunk['text']

        # Filter non-substantive chunks if enabled
        if filter_non_substantive and not extractor.should_process_chunk(chunk_text):
            filtered_count += 1
            continue

        prompt_data = extractor.generate_extraction_prompt(
            chunk_text=chunk_text,
            chunk_id=chunk['chunk_id'],
            source_file=chunk.get('source_file', 'unknown'),
            page=chunk.get('page', 1)
        )
        batch_data['prompts'].append(prompt_data)

    # Update total to reflect filtered count
    batch_data['substantive_chunks'] = len(batch_data['prompts'])
    batch_data['filtered_chunks'] = filtered_count

    output_path = Path(output_path)
    output_path.write_text(json.dumps(batch_data, indent=2), encoding='utf-8')

    logger.info(f"Created batch extraction file: {output_path}")
    logger.info(f"Total chunks: {len(chunks)}, Substantive: {len(batch_data['prompts'])}")
    if filtered_count > 0:
        logger.debug(f"Filtered out: {filtered_count} ({filtered_count/len(chunks)*100:.1f}%)")

    return output_path


# =========================================================================
# CLI
# =========================================================================

def main() -> None:
    """Command-line interface for generating extraction prompts"""
    import argparse

    # Configure logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    parser = argparse.ArgumentParser(
        description='Generate concept extraction prompts for Claude Code'
    )
    parser.add_argument('input', type=str, help='Input JSON file with chunks')
    parser.add_argument('--output', type=str, default='extraction_batch.json',
                       help='Output batch file for Claude Code')
    parser.add_argument('--sample', type=int, default=None,
                       help='Only create prompts for first N chunks (for testing)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load chunks
    input_path = Path(args.input)
    chunks_data = json.loads(input_path.read_text(encoding='utf-8'))
    chunks = chunks_data.get('chunks', [])

    if args.sample:
        chunks = chunks[:args.sample]
        print(f"Using first {args.sample} chunks for testing")

    print(f"Loaded {len(chunks)} chunks from {input_path.name}")

    # Create batch file
    batch_file = create_batch_extraction_file(chunks, Path(args.output))

    print("\nNext steps:")
    print(f"1. Use Claude Code to process: {batch_file}")
    print("2. Claude Code will execute each prompt sequentially")
    print("3. Collect responses into concepts.json")
    print("\nExample Claude Code command:")
    print(f"  mcp use knowledge_extraction extract_concepts {batch_file}")


if __name__ == "__main__":
    main()
