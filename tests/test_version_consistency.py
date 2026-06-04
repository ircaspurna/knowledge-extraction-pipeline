"""Guard against version drift across packaging metadata.

The package version is declared in several places that must agree. Past releases
(v4.0.3, v4.0.4) bumped ``pyproject.toml`` + ``CHANGELOG.md`` but silently left
``__init__.py`` and ``CITATION.cff`` stale — so an installed package reported the
wrong ``__version__``. This test makes such a mismatch fail CI instead of shipping.

Treat ``pyproject.toml`` as the single source of truth; every other declaration
must match it.
"""

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def _pyproject_version() -> str:
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def _init_version() -> str:
    text = (REPO_ROOT / "src" / "knowledge_extraction" / "__init__.py").read_text()
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.M)
    assert m, "__version__ not found in src/knowledge_extraction/__init__.py"
    return m.group(1)


def _citation_version() -> str:
    text = (REPO_ROOT / "CITATION.cff").read_text()
    m = re.search(r"^version:\s*([^\s#]+)", text, re.M)
    assert m, "version not found in CITATION.cff"
    return m.group(1).strip().strip('"').strip("'")


def _changelog_latest_version() -> str:
    text = (REPO_ROOT / "CHANGELOG.md").read_text()
    m = re.search(r"^##\s*\[([0-9]+\.[0-9]+\.[0-9]+)\]", text, re.M)
    assert m, "no [x.y.z] release heading found in CHANGELOG.md"
    return m.group(1)


def test_all_versions_match_pyproject() -> None:
    source = _pyproject_version()
    declared = {
        "pyproject.toml": source,
        "src/knowledge_extraction/__init__.py": _init_version(),
        "CITATION.cff": _citation_version(),
        "CHANGELOG.md (latest entry)": _changelog_latest_version(),
    }
    mismatches = {k: v for k, v in declared.items() if v != source}
    assert not mismatches, (
        f"Version drift detected. pyproject.toml declares {source!r}, but: "
        + ", ".join(f"{k} = {v!r}" for k, v in mismatches.items())
        + ". Update all declarations to match pyproject.toml."
    )
