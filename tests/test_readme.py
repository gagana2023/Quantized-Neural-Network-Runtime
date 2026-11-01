"""The README results section must equal what tools.make_readme_tables builds from results/."""

from __future__ import annotations

from tools import make_readme_tables as mrt


def test_readme_results_section_is_current() -> None:
    readme = mrt.README.read_text(encoding="utf-8")
    assert mrt.splice(readme, mrt.build()) == readme


def test_tables_are_generated_from_results_only() -> None:
    tables = mrt.build()
    for label in mrt.MODELS.values():
        assert label in tables
    assert "MODELLED" in tables
