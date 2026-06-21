"""Cross-substrate parity + determinism guard (ARCHITECTURE.md Appendix B.8).

The architecture fixes a hard invariant: the edge TypeScript detector
(``edge/src/detector.ts``) and the Python detector
(``flarecast/detect/{primitives,cusum}.py``) must produce the **same** onset
decisions on the same input sequence with the same ``alpha``/``k``/``h``.

We lock that contract with a committed *golden vector*
(``tests/golden/cusum_golden.json``) generated once from the Python
reference pipeline (see ``tests/golden/generate_cusum_golden.py``). This module
is the **Python half** of the parity check: it loads the golden and asserts that
re-running the reference EMA+EWMV+CUSUM pipeline reproduces every stored array
*exactly*. That gives us two guarantees at once:

1. **Determinism / regression guard.** Any change to the frozen-foundation
   CUSUM/EMA/EWMV math (intentional or accidental) breaks this test, so the
   cross-substrate contract cannot drift silently.
2. **The TS side is held to the same file.** ``edge/test/parity.mjs`` (run by
   ``cd edge && npm test``) loads this *same* ``cusum_golden.json`` and asserts
   its JS mirror of the pipeline reproduces the identical statistic/onset
   arrays. Python and TypeScript are therefore validated against one shared
   source of truth -- the definition of the parity invariant.

These tests are import-safe and pure-stdlib (no numpy/torch/network), so they
run in the minimal offline sandbox and in CI.
"""

from __future__ import annotations

import json
import os

import pytest
from flarecast.constants import CUSUM_H, CUSUM_K_SLACK

from tests.golden.generate_cusum_golden import (
    GOLDEN_PATH,
    PARITY_H,
    PARITY_K,
    build_input_vector,
    run_pipeline,
)

# Path to the TS mirror, documented here so the relationship is discoverable.
_EDGE_PARITY_MJS = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "edge", "test", "parity.mjs"
)


@pytest.fixture(scope="module")
def golden() -> dict:
    """Load the committed golden parity vector."""
    assert os.path.exists(GOLDEN_PATH), (
        f"missing golden vector {GOLDEN_PATH}; regenerate with "
        "`python tests/golden/generate_cusum_golden.py`"
    )
    with open(GOLDEN_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def test_golden_is_well_formed(golden: dict) -> None:
    """The golden document has the expected shape and self-consistent lengths."""
    assert golden["n"] == len(golden["input"])
    exp = golden["expected"]
    n = golden["n"]
    for key in ("statistic", "baseline", "sigma", "in_event"):
        assert len(exp[key]) == n, f"{key} length mismatch"
    assert len(exp["onset_indices"]) == len(exp["onset_times"])
    # Every onset index is a valid position into the series.
    assert all(0 <= i < n for i in exp["onset_indices"])
    # Onset indices are strictly increasing (one onset per flare, in order).
    idx = exp["onset_indices"]
    assert idx == sorted(idx)
    assert len(set(idx)) == len(idx)


def test_golden_params_match_constants(golden: dict) -> None:
    """The golden was generated with the canonical CUSUM constants (k, h)."""
    assert golden["params"]["k_slack"] == CUSUM_K_SLACK == PARITY_K
    assert golden["params"]["h"] == CUSUM_H == PARITY_H


def test_input_vector_is_deterministic(golden: dict) -> None:
    """Rebuilding the fixed input vector reproduces the committed input exactly.

    The generator uses a seeded ``random.Random`` and pure-stdlib arithmetic, so
    the input is reproducible byte-for-byte; this guards the *input* half of the
    contract (a changed input would silently invalidate the golden).
    """
    rebuilt = build_input_vector()
    assert rebuilt == golden["input"]


def test_python_reproduces_golden_exactly(golden: dict) -> None:
    """Re-running the reference pipeline reproduces every stored array exactly.

    This is the core determinism / regression guard: the EMA+EWMV+CUSUM math in
    the frozen foundation must yield the committed statistic, baseline, sigma,
    in_event, and onset arrays bit-for-bit (after the generator's rounding).
    """
    result = run_pipeline(golden["input"])
    exp = golden["expected"]

    assert result["onset_indices"] == exp["onset_indices"]
    assert result["onset_times"] == exp["onset_times"]
    assert result["in_event"] == exp["in_event"]
    # The rounded numeric arrays must match exactly (generator rounds to 9 dp).
    assert result["statistic"] == exp["statistic"]
    assert result["baseline"] == exp["baseline"]
    assert result["sigma"] == exp["sigma"]


def test_onset_time_precedes_detection(golden: dict) -> None:
    """The MLE onset time is at or before the alarm index (the CUSUM payoff).

    CUSUM reports the last reset-to-zero before the alarm as the maximum-
    likelihood change point, so the reported onset time must never be *after* the
    sample on which the alarm fired (ARCHITECTURE.md Section 4.4). With unit
    time steps the onset time is the change-point index.
    """
    exp = golden["expected"]
    for idx, t_onset in zip(exp["onset_indices"], exp["onset_times"], strict=True):
        # time == float(index) in this fixture, so t_onset <= idx.
        assert t_onset <= float(idx)
        # ...and the change point is not absurdly early (within this flare).
        assert float(idx) - t_onset < 60.0


def test_detected_three_flares(golden: dict) -> None:
    """The fixture injects three flares; the pipeline finds exactly three onsets.

    A sanity check that the contract vector actually exercises onset detection
    (not a degenerate all-quiet / all-firing series).
    """
    assert len(golden["expected"]["onset_indices"]) == 3


def test_edge_parity_harness_exists() -> None:
    """The TS/JS half of the parity check is present and references the golden.

    Documents (and lightly enforces) that ``edge/ npm test`` runs
    ``edge/test/parity.mjs`` against this same ``cusum_golden.json``. The TS
    execution itself is exercised in the ``edge`` CI job (Node), kept separate
    from the Python suite so a missing Node toolchain never blocks pytest.
    """
    assert os.path.exists(_EDGE_PARITY_MJS), (
        "edge/test/parity.mjs (the TypeScript parity harness) is missing"
    )
    with open(_EDGE_PARITY_MJS, encoding="utf-8") as fh:
        src = fh.read()
    assert "cusum_golden.json" in src, (
        "edge parity harness must load the shared golden vector"
    )
