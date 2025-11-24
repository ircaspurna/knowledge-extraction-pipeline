# Custom Domain Example

This example shows how to configure the pipeline for a specific domain (e.g., medical, legal, engineering).

## What This Example Shows

- Creating domain-specific extraction prompts
- Customizing concept categories
- Using domain-specific terminology
- Adjusting extraction parameters

## Files

- `run.py` - Example using custom domain configuration
- `custom_prompts.yaml` - Domain-specific prompts
- `sample_paper.pdf` - Place your domain-specific PDF here

## Quick Start

```bash
# 1. Edit custom_prompts.yaml for your domain
# 2. Add a sample PDF
# 3. Run the extractor

python run.py
```

## Customizing for Your Domain

### 1. Define Domain Categories

Edit `custom_prompts.yaml`:

```yaml
# Example: Medical domain
categories:
  - disease
  - treatment
  - symptom
  - biomarker
  - drug
  - procedure

# Example: Legal domain
# categories:
#   - statute
#   - precedent
#   - principle
#   - procedure
#   - regulation
#   - right

# Example: Engineering domain
# categories:
#   - method
#   - component
#   - material
#   - process
#   - standard
#   - metric
```

### 2. Customize Extraction Prompt

```yaml
extraction_prompt: |
  You are a medical research assistant extracting key concepts.

  Extract medical concepts from this passage. For EACH concept:

  1. **Term**: Medical terminology (prefer standard names)
  2. **Definition**: Clinical definition
  3. **Category**: disease | treatment | symptom | biomarker | drug | procedure
  4. **Importance**: critical | high | medium | low
  5. **Clinical Relevance**: Why this matters clinically
  6. **Evidence**: Exact quote from passage

  **Domain-Specific Rules**:
  - Use ICD-10 codes where applicable
  - Prioritize FDA-approved treatments
  - Note contraindications
  - Flag experimental treatments

  **Passage**:
  {passage}

  Extract concepts:
```

### 3. Add Domain-Specific Validation

```yaml
validation_rules:
  - Check against medical terminology database
  - Verify drug names against FDA database
  - Validate ICD-10 codes
  - Flag outdated treatments
```

## Domain Examples

### Medical Research

```python
from knowledge_extraction.extraction import ConceptExtractorMCP

extractor = ConceptExtractorMCP(
    config_path="custom_prompts_medical.yaml"
)
```

**Custom categories:**
- Disease, Treatment, Symptom, Biomarker, Drug, Procedure

**Special handling:**
- ICD-10 code extraction
- Drug interaction detection
- Clinical trial phase annotation

### Legal Documents

```python
extractor = ConceptExtractorMCP(
    config_path="custom_prompts_legal.yaml"
)
```

**Custom categories:**
- Statute, Precedent, Principle, Procedure, Regulation, Right

**Special handling:**
- Case citation extraction
- Jurisdiction tagging
- Date-sensitive precedents

### Engineering Specifications

```python
extractor = ConceptExtractorMCP(
    config_path="custom_prompts_engineering.yaml"
)
```

**Custom categories:**
- Method, Component, Material, Process, Standard, Metric

**Special handling:**
- Standard reference extraction (ISO, IEEE, etc.)
- Measurement unit normalization
- Specification requirement linking

## Advanced Configuration

### Custom Importance Criteria

Define domain-specific importance:

```yaml
importance_criteria:
  critical:
    - FDA-approved treatments
    - Level 1 evidence
    - First-line therapies
  high:
    - Clinical guidelines
    - Phase 3 trials
  medium:
    - Observational studies
    - Phase 2 trials
  low:
    - Case reports
    - Preclinical studies
```

### Domain-Specific Aliases

```yaml
known_aliases:
  # Medical
  MI: Myocardial Infarction
  CHF: Congestive Heart Failure
  COPD: Chronic Obstructive Pulmonary Disease

  # Legal
  USC: United States Code
  CFR: Code of Federal Regulations

  # Engineering
  FEA: Finite Element Analysis
  CAD: Computer-Aided Design
```

### Entity Resolution Tuning

```yaml
entity_resolution:
  # Medical: High precision (avoid merging different conditions)
  similarity_threshold: 0.95

  # Legal: Medium precision (merge similar precedents)
  # similarity_threshold: 0.90

  # Engineering: Lower precision (merge equivalent methods)
  # similarity_threshold: 0.85
```

## Example Output

Medical domain extraction:

```json
{
  "term": "Hypertension",
  "definition": "Chronic elevation of blood pressure above 140/90 mmHg",
  "category": "disease",
  "importance": "critical",
  "icd10_code": "I10",
  "clinical_relevance": "Major risk factor for cardiovascular disease",
  "evidence": "Hypertension affects 30% of adults..."
}
```

## Tips

- **Start Specific**: Begin with narrow domain focus
- **Iterate Prompts**: Test and refine extraction quality
- **Use Standards**: Leverage domain-specific ontologies
- **Validate Output**: Check against domain experts
- **Document Decisions**: Track why categories were chosen

## Resources

- **Medical**: UMLS, MeSH, SNOMED CT
- **Legal**: Bluebook citation, Legal Information Institute
- **Engineering**: ISO standards, IEEE databases
