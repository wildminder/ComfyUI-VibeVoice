"""NTH-002: enforce version parity between pyproject.toml and the README changelog.

``pyproject.toml`` is the single source of truth for the released version. The
top entry of the README Changelog must declare the same version, so users and
packagers get a consistent release signal. This CI test fails if they drift.
"""

import os
import re

import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PYPROJECT_PATH = os.path.join(ROOT_DIR, "pyproject.toml")
CHANGELOG_PATH = os.path.join(ROOT_DIR, "CHANGELOG.md")


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_pyproject_version(text: str) -> str:
    # Match: version = "2.0.0"
    m = re.search(r'^\s*version\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not m:
        raise AssertionError("Could not find `version = \"...\"` in pyproject.toml")
    return m.group(1).strip()


def _parse_top_changelog_version(readme: str) -> str:
    # The first <summary><strong>vX.Y.Z ...</strong></summary> in the changelog
    # is the most recent release. Strip the leading 'v'.
    m = re.search(r"<summary>.*?<strong>\s*v([0-9]+\.[0-9]+\.[0-9]+)", readme, re.IGNORECASE | re.DOTALL)
    if not m:
        raise AssertionError("Could not find a changelog version (vX.Y.Z) in CHANGELOG.md")
    return m.group(1).strip()


class TestVersionSync:
    """pyproject.toml version must match the top README changelog version."""

    def test_pyproject_version_present(self):
        version = _parse_pyproject_version(_read(PYPROJECT_PATH))
        assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version), f"Bad version: {version}"

    def test_readme_changelog_version_present(self):
        version = _parse_top_changelog_version(_read(CHANGELOG_PATH))
        assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version), f"Bad version: {version}"

    def test_versions_match(self):
        pyproject_version = _parse_pyproject_version(_read(PYPROJECT_PATH))
        readme_version = _parse_top_changelog_version(_read(CHANGELOG_PATH))
        assert pyproject_version == readme_version, (
            f"Version mismatch: pyproject.toml={pyproject_version} "
            f"CHANGELOG={readme_version}. Keep pyproject.toml as the "
            f"source of truth and update the CHANGELOG top entry."
        )
