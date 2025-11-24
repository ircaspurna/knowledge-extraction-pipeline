# Contributing to Knowledge Extraction Pipeline

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/knowledge-extraction-pipeline.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Set up development environment: `pip install -r requirements-dev.txt`
5. Install pre-commit hooks: `pre-commit install`

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
pytest tests/ -v
pytest tests/ --cov  # With coverage
```

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

## Questions?

Open a GitHub Discussion or issue.
