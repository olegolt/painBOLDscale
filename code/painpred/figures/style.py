"""Shared plotting style and small plotting helpers."""

from __future__ import annotations

import numpy as np
from scipy import stats

ARM_COLOR = "#3B6670"
LEG_COLOR = "#C45A4B"
OVERALL_COLOR = "#6D6D6D"
POINT_COLOR = "#000000"
SAVE_DPI = 600

REGRESSION = dict(point_size=28, point_alpha=0.80, line_width=2.2, ci_alpha=0.18)
SIZES = dict(axis_label=16, tick=13.5, stat=13.0, panel_label=20, condition_title=15.5, block_title=18)


def site_color(site: str) -> str:
    return ARM_COLOR if site == "arm" else LEG_COLOR


def apply_style(font_size: float = 13) -> None:
    """Global matplotlib defaults (Arial-like font if available; editable text in PDF/SVG)."""
    import matplotlib as mpl
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    font = next((n for n in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans") if n in available), "DejaVu Sans")
    mpl.rcParams.update({
        "font.family": font, "font.size": font_size, "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.facecolor": "white", "axes.edgecolor": "#4A4A4A", "axes.linewidth": 1.30,
        "xtick.color": "#333333", "ytick.color": "#333333", "text.color": "#111111",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False})


def style_axis(ax, tick_size: float = SIZES["tick"]) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for name in ("bottom", "left"):
        ax.spines[name].set_linewidth(1.30)
    ax.tick_params(axis="both", labelsize=tick_size, width=1.15, length=4.2, pad=3)


def add_panel_label(ax, label: str, x: float = -0.075, y: float = 1.13, size: float = SIZES["panel_label"]) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top", fontsize=size, fontweight="bold", clip_on=False)


def padded_limits(values, fraction: float = 0.07) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    lower, upper = float(values.min()), float(values.max())
    span = upper - lower if upper > lower else max(abs(upper), 1.0)
    return lower - fraction * span, upper + fraction * span


def regression_band(x, y, confidence: float = 0.95, n_points: int = 300):
    """OLS line and pointwise confidence band: (x_line, y_line, lower, upper); band is None if df <= 0."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    fit = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), n_points)
    y_line = fit.intercept + fit.slope * x_line
    n, sxx, df = len(x), float(np.sum((x - x.mean()) ** 2)), len(x) - 2
    if df <= 0 or sxx <= 0:
        return x_line, y_line, None, None
    variance = float(np.sum((y - (fit.intercept + fit.slope * x)) ** 2) / df)
    half = stats.t.ppf(0.5 + confidence / 2, df) * np.sqrt(variance * (1.0 / n + (x_line - x.mean()) ** 2 / sxx))
    return x_line, y_line, y_line - half, y_line + half


def add_ols_line_and_ci(ax, x, y, color: str, line_z: int = 4, band_z: int = 2) -> None:
    x_line, y_line, lower, upper = regression_band(x, y)
    ax.plot(x_line, y_line, color=color, linewidth=REGRESSION["line_width"], zorder=line_z)
    if lower is not None:
        ax.fill_between(x_line, lower, upper, color=color, alpha=REGRESSION["ci_alpha"], linewidth=0, zorder=band_z)


def scatter_with_fit(ax, x, y, color: str, line_z: int = 4, band_z: int = 2) -> None:
    ax.scatter(x, y, s=REGRESSION["point_size"], color=POINT_COLOR, alpha=REGRESSION["point_alpha"],
               edgecolors="none", zorder=3, clip_on=False)
    add_ols_line_and_ci(ax, x, y, color, line_z, band_z)


def rating_axis(ax, y) -> None:
    """VAS y-axis: limits on a 5-point grid, tick step chosen from the displayed span."""
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    span = max(y_max - y_min, 1.0)
    lower = max(0.0, 5.0 * np.floor(max(0.0, y_min - max(2.0, 0.08 * span)) / 5.0))
    target = max(y_max * 1.05 if y_max > 0 else y_max + 0.05 * span, y_max + 0.05 * span)
    upper = 5.0 * np.ceil(target / 5.0)
    step = 5.0 if upper - lower <= 30 else 10.0 if upper - lower <= 60 else 20.0
    ax.set_ylim(lower, upper)
    ax.set_yticks(np.arange(step * np.ceil(lower / step), step * np.floor(upper / step) + 0.5 * step, step))


def save_figure(fig, files, dpi: int = SAVE_DPI, **kwargs) -> None:
    """Save to every path in ``files`` (format from the extension)."""
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, facecolor="white", edgecolor="white", **kwargs)
        print(f"Saved {path}")
