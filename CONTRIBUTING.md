# Contributing to Knowledge Extraction Pipeline

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

### 1. Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/knowledge-extraction-pipeline.git
   cd knowledge-extraction-pipeline
   ```

### 2. Set Up Development Environment

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e .

# Install development dependencies
make install-dev
# Or manually: pip install -r requirements-dev.txt && pre-commit install
```

### 3. Verify Your Setup

Run these commands to ensure everything works:

```bash
# Run tests (should pass)
make test

# Run code quality checks (should pass)
make lint

# Format code (should complete without errors)
make format
```

If all three pass, you're ready to contribute! ✅

### 4. Create a Branch

Create a feature branch for your changes:
```bash
git checkout -b feature/your-feature-name
# Or for bug fixes: git checkout -b fix/bug-description
```

## Development Workflow

### Code Style

We use:
- **black** for code formatting
- **isort** for import sorting
- **ruff** for linting
- **mypy** for type checking

Run before committing:
```bash
make format  # Format code
make lint    # Check code quality
```

### Testing

All new features should include tests:

```bash
# Run all tests
make test

# Run with coverage report
make test-cov

# Run specific test file
pytest tests/test_smoke.py -v

# Run specific test
pytest tests/test_smoke.py::test_package_imports -v
```

**Test Guidelines:**
- Write tests for all new functionality
- Aim for >60% code coverage (checked in CI)
- Include both unit tests and integration tests
- Use descriptive test names: `test_entity_resolver_merges_duplicates()`
- Add docstrings to complex tests explaining what they verify

### Type Hints

All new code should include type hints:
```python
def process_document(file_path: Path) -> DocumentResult:
    """Process a document and return results."""
    ...
```

## Pull Request Process

1. Update documentation for any new features
2. Add tests for new functionality
3. Ensure all tests pass: `pytest tests/ -v`
4. Ensure code quality checks pass: `make lint`
5. Update CHANGELOG.md with your changes
6. Submit PR with clear description of changes

## Commit Messages

Use clear, descriptive commit messages:
- `feat: Add semantic similarity graph building`
- `fix: Resolve entity resolution bug with special characters`
- `docs: Update API reference for GraphBuilder`
- `test: Add integration tests for full pipeline`

## Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Full error traceback
- Minimal reproducible example

## Feature Requests

We welcome feature requests! Please:
- Check existing issues first
- Clearly describe the use case
- Explain expected behavior
- Provide examples if possible

## Code of Conduct

Be respectful and inclusive. We're all here to build something useful together.

## Local Development & Testing

### Testing the Full Pipeline

To test the complete extraction pipeline locally:

```bash
# 1. Process a sample PDF
python scripts/process_pdf.py examples/simple_extraction/sample.pdf --output ./test_output/

# 2. The pipeline will generate extraction_batch.json
# You can review this file to see the prompts

# 3. For testing, you can create a mock response file
# or use Claude Code to process the batch (see README)

# 4. Verify output structure
ls ./test_output/
# Should contain: chunks.json, extraction_batch.json, etc.
```

### Pre-commit Hooks

Pre-commit hooks run automatically before each commit:

```bash
# Install hooks (done automatically by make install-dev)
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Skip hooks for a specific commit (not recommended)
git commit --no-verify -m "WIP: debugging"
```

The hooks will:
- ✅ Format code with black
- ✅ Sort imports with isort
- ✅ Check types with mypy --strict
- ✅ Fix trailing whitespace
- ✅ Validate YAML/JSON files

## Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'knowledge_extraction'"**
- Solution: Install package in development mode: `pip install -e .`

**"pre-commit: command not found"**
- Solution: Install dev dependencies: `pip install -r requirements-dev.txt`

**"mypy --strict fails on my code"**
- Add type hints to all functions
- Use `from typing import List, Dict, Optional` for complex types
- Check `pyproject.toml` for mypy configuration

**"Tests fail with ChromaDB errors"**
- Tests use temporary ChromaDB instances
- Ensure you have write permissions in temp directory
- Try: `rm -rf /tmp/test_chroma_*`

**"CI/CD workflows fail but tests pass locally"**
- Ensure you've committed all changes
- Check that code passes `make lint` (not just `make test`)
- Verify Python 3.11+ compatibility

## Questions?

- **General questions:** Open a GitHub Discussion
- **Bug reports:** Open a GitHub Issue (use issue template)
- **Feature ideas:** Open a GitHub Issue with `enhancement` label
- **Pull requests:** Use the PR template and link to related issues
