"""The version has exactly one source; these go red if a second one returns.

The single source is ``loopward.__version__``. ``pyproject.toml`` derives it at
build time (``dynamic = ["version"]`` + ``[tool.setuptools.dynamic]``). Two ways
the single-source invariant can rot, one test each:

  1. Someone re-adds a static ``version = "..."`` under ``[project]`` in
     pyproject, reintroducing the duplicate literal that this change removed.
  2. The built/installed distribution's version drifts from the source literal
     (a stale build, or a hand-edited PKG-INFO).
"""

from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

import pytest

import loopward

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_pyproject_declares_version_dynamic_not_static() -> None:
    """pyproject must derive the version, never carry its own literal."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data["project"]

    assert "version" not in project, (
        "pyproject [project] has a static `version` again — the number is now "
        "duplicated. Remove it; the single source is loopward.__version__."
    )
    assert "version" in project.get("dynamic", []), (
        "pyproject must list `version` under [project].dynamic so it is derived "
        "from loopward.__version__."
    )


def test_installed_metadata_matches_source_literal() -> None:
    """The shipped distribution's version must equal the source literal."""
    try:
        installed = importlib.metadata.version("loopward")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        pytest.skip("loopward is not installed; metadata comparison needs a build")

    assert installed == loopward.__version__, (
        f"installed metadata {installed!r} != source {loopward.__version__!r} — "
        "the build is stale or the version diverged. Reinstall/rebuild."
    )
