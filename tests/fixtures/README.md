# Test Fixtures

This directory contains real academic papers used for testing the knowledge extraction pipeline.

## Contents

### PDF Files

1. **Brodersen_2015_CausalImpact_Inferring_Causal_Effects.pdf** (94KB)
   - **Authors**: Kay H. Brodersen et al.
   - **Year**: 2015
   - **Topic**: Causal inference methodology
   - **Use**: Small PDF for basic processing tests

2. **Gigerenzer_2009_Why_Heuristics_Work.pdf** (169KB)
   - **Authors**: Gerd Gigerenzer, Henry Brighton
   - **Year**: 2009
   - **Topic**: Decision-making heuristics
   - **Use**: Medium-sized PDF for text extraction tests

3. **Chen_2020_CausalML_Python_Package_Causal.pdf** (190KB)
   - **Authors**: Huigang Chen et al.
   - **Year**: 2020
   - **Topic**: Machine learning for causal inference
   - **Use**: Technical paper with code examples

## Purpose

These PDFs are used to test:
- Document processing (PDF parsing)
- Text extraction accuracy
- Metadata extraction
- Semantic chunking
- Vector embedding
- Knowledge graph construction

## Licensing

These are publicly available academic papers used for testing purposes only.

- Papers are used under fair use for software testing
- Not redistributed for consumption
- Only used to verify extraction functionality

## Size

Total size: ~450KB (suitable for version control)

## Adding New Fixtures

When adding new test PDFs:
1. Choose small papers (<500KB)
2. Use properly named files (Author_Year_Title.pdf)
3. Select diverse topics
4. Document in this README
5. Verify licensing allows testing use
