"""Smoke tests to ensure basic functionality works."""

from pathlib import Path

import pytest


def test_config_files_exist() -> None:
    """Test that configuration files exist."""
    package_dir = Path(__file__).parent.parent

    config_files = [
        "config/prompts.yaml",
        "config/domains.yaml",
    ]

    for config_file in config_files:
        config_path = package_dir / config_file
        assert config_path.exists(), f"Config file not found: {config_path}"
        assert config_path.stat().st_size > 0, f"Config file is empty: {config_path}"


def test_documentation_exists() -> None:
    """Test that main documentation files exist."""
    package_dir = Path(__file__).parent.parent

    docs = [
        "README.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "CHANGELOG.md",
    ]

    for doc in docs:
        doc_path = package_dir / doc
        assert doc_path.exists(), f"Documentation not found: {doc_path}"
        assert doc_path.stat().st_size > 100, f"Documentation seems incomplete: {doc_path}"


def test_package_structure() -> None:
    """Test that package has correct structure."""
    package_dir = Path(__file__).parent.parent

    expected_dirs = [
        "src/knowledge_extraction",
        "src/knowledge_extraction/core",
        "src/knowledge_extraction/extraction",
        "src/knowledge_extraction/mcp",
        "src/knowledge_extraction/visualization",
        "scripts",
        "tests",
        "config",
        "docs",
        "examples",
    ]

    for dir_path in expected_dirs:
        full_path = package_dir / dir_path
        assert full_path.exists(), f"Expected directory not found: {full_path}"
        assert full_path.is_dir(), f"Expected directory is not a directory: {full_path}"


def test_core_modules_importable() -> None:
    """Test that core modules can be imported."""
    try:
        from knowledge_extraction.core import DocumentProcessor, SemanticChunker, VectorStore, GraphBuilder
        assert DocumentProcessor is not None
        assert SemanticChunker is not None
        assert VectorStore is not None
        assert GraphBuilder is not None
    except ImportError as e:
        pytest.fail(f"Could not import core modules: {e}")


def test_extraction_modules_importable() -> None:
    """Test that extraction modules can be imported."""
    try:
        from knowledge_extraction.extraction import ConceptExtractorMCP, EntityResolverMCP
        assert ConceptExtractorMCP is not None
        assert EntityResolverMCP is not None
    except ImportError as e:
        pytest.fail(f"Could not import extraction modules: {e}")


def test_scripts_exist() -> None:
    """Test that all user-facing scripts exist."""
    package_dir = Path(__file__).parent.parent

    expected_scripts = [
        "scripts/process_pdf.py",
        "scripts/batch_process.py",
        "scripts/build_graph.py",
        "scripts/search.py",
        "scripts/import_neo4j.py",
    ]

    for script_path in expected_scripts:
        full_path = package_dir / script_path
        assert full_path.exists(), f"Script not found: {full_path}"
        assert full_path.stat().st_size > 0, f"Script is empty: {full_path}"


def test_example_exists() -> None:
    """Test that at least one example exists."""
    package_dir = Path(__file__).parent.parent
    examples_dir = package_dir / "examples"

    assert examples_dir.exists(), "Examples directory not found"

    # Check that simple_extraction example exists
    simple_example = examples_dir / "simple_extraction"
    assert simple_example.exists(), "simple_extraction example not found"
    assert (simple_example / "README.md").exists(), "simple_extraction README not found"
    assert (simple_example / "run.py").exists(), "simple_extraction run.py not found"


def test_pyproject_toml_valid() -> None:
    """Test that pyproject.toml exists and has required fields."""
    package_dir = Path(__file__).parent.parent
    pyproject_file = package_dir / "pyproject.toml"

    assert pyproject_file.exists(), "pyproject.toml not found"

    content = pyproject_file.read_text()

    # Check for required fields
    assert 'name = "knowledge-extraction-pipeline"' in content
    assert 'version' in content
    assert 'dependencies' in content
    assert 'requires-python' in content


def test_pre_commit_config_exists() -> None:
    """Test that pre-commit configuration exists."""
    package_dir = Path(__file__).parent.parent
    pre_commit_file = package_dir / ".pre-commit-config.yaml"

    assert pre_commit_file.exists(), "Pre-commit config not found"

    content = pre_commit_file.read_text()
    assert "mypy" in content, "Pre-commit should include mypy"
    assert "ruff" in content or "black" in content, "Pre-commit should include linter"


def test_github_templates_exist() -> None:
    """Test that GitHub templates are configured."""
    package_dir = Path(__file__).parent.parent
    github_dir = package_dir / ".github"

    assert github_dir.exists(), "GitHub directory not found"

    # Check for issue templates
    issue_template_dir = github_dir / "ISSUE_TEMPLATE"
    assert issue_template_dir.exists(), "Issue template directory not found"

    # Check for PR template
    pr_template = github_dir / "PULL_REQUEST_TEMPLATE.md"
    assert pr_template.exists(), "PR template not found"
