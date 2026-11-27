"""Named pipeline steps, in dependency order. Used by the notebook and by ``python -m painpred``.

Every step is a function ``step(project, overwrite)``. Slow, deterministic steps skip work whose
outputs already exist unless ``overwrite=True``.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable

from . import hoeppli, massuni, overlap, peaks, signatures, supplementary
from .config import Project
from .figures import fig1, fig2, fig3, fig4, fig5, fig6, figs1, figs2, supp_baseline
from .utils import PipelineError


def _supplement(project: Project, overwrite: bool) -> None:
    result = supplementary.load_saved(project) if project.public else supplementary.run(project)
    supp_baseline.run(project, result)


# Steps that need the participants' images; they are skipped when working from the shared data.
NEEDS_IMAGES = {"hoeppli_mask", "massuni", "lasso_pcr"}


STEPS: dict[str, Callable[[Project, bool], object]] = {
    "hoeppli_mask": lambda p, o: hoeppli.build_mask(p, force=o),
    "figure1": lambda p, o: fig1.run(p),
    "massuni": lambda p, o: massuni.run_all(p, overwrite=o),
    "figure2_peaks": lambda p, o: peaks.run(p),
    "figure2_overlap": lambda p, o: overlap.run_reference_overlap(p),
    "figure2": lambda p, o: fig2.run(p, overwrite=o),
    "figure3_overlap": lambda p, o: (overlap.count_overlap(p), overlap.count4_regions(p)),
    "figure3": lambda p, o: fig3.run(p),
    "lasso_pcr": lambda p, o: signatures.run_lasso_pcr(p, overwrite=o),
    "figure4": lambda p, o: fig4.run(p),
    "figure5": lambda p, o: fig5.run(p),
    "figure6": lambda p, o: fig6.run(p),
    "figure_s1": lambda p, o: figs1.run(p),
    "figure_s2": lambda p, o: figs2.run(p),
    "supplement": _supplement,
}
__all__ = ["NEEDS_IMAGES", "STEPS", "run_steps"]


def run_steps(project: Project, steps=None, overwrite: bool = False, stop_on_error: bool = True) -> dict[str, str]:
    """Run ``steps`` (default: all) in order. Returns {step: 'ok' | error message}."""
    names = list(STEPS) if steps is None else list(steps)
    unknown = [n for n in names if n not in STEPS]
    if unknown:
        raise PipelineError(f"Unknown step(s) {unknown}. Available: {list(STEPS)}")
    project.make_output_dirs()
    status = {}
    for name in names:
        if project.public and name in NEEDS_IMAGES:
            status[name] = "skipped (needs subject images; results are part of the shared data)"
            print(f"\n[{name}] skipped: needs subject images, the results are already in data/derivatives.")
            continue
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        start = time.time()
        try:
            STEPS[name](project, overwrite)
            status[name] = "ok"
            print(f"[{name}] done in {time.time() - start:.0f} s")
        except PipelineError as exc:  # expected, user-fixable problems: message only
            status[name] = f"FAILED: {exc}"
            print(f"\n[{name}] FAILED\n{exc}")
            if stop_on_error:
                raise
        except Exception:  # unexpected: keep the traceback
            status[name] = "FAILED (unexpected error)"
            print(f"\n[{name}] unexpected error:\n{traceback.format_exc()}")
            if stop_on_error:
                raise
    return status
