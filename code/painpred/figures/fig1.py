"""Figure 1: pain ratings, within-participant rating variability and pre-experiment calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Project
from ..utils import DataError, add_subject_key, canonical_subject_id, drop_excluded, read_table, require_columns
from .style import ARM_COLOR, LEG_COLOR, OVERALL_COLOR, apply_style, save_figure

TEMPERATURES, SITES, N_TRIALS = (46, 48), ("arm", "leg"), 12
CALIBRATION_LEVELS = (50, 60, 70, 80)
CALIBRATION_IQR = 1.5
ARM_GROUP, LEG_GROUP, OVERALL_GROUP = "#234A53", "#963B32", "#363636"
PAIN_THRESHOLD = 50.0
X_TEMPERATURE = {46: 0.0, 48: 1.0}
X_SUMMARY = {"arm": 0.0, "leg": 1.0, "overall": 2.0}
RATING_YLIM = (0.0, 104.0)
SIZE = dict(title=17, label=16, tick=14, panel=21, point=26, point_alpha=0.72, line_alpha=0.17, line_width=0.72,
            marker=11.0, errorbar=2.3, cap=5.8)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def within_participant_sd(trials: pd.DataFrame) -> pd.DataFrame:
    """SD (ddof=1) across the 12 repetitions of every condition, per participant."""
    rows = []
    for _, row in trials.iterrows():
        record = {"subject_key": row["subject_key"]}
        for temperature in TEMPERATURES:
            for site in SITES:
                prefix = f"pain_{temperature}_{site}_rep"
                columns = sorted(c for c in trials.columns if c.startswith(prefix))
                if len(columns) != N_TRIALS:
                    raise DataError(f"Trial-rating table: expected {N_TRIALS} columns starting with '{prefix}', found {len(columns)}.")
                values = pd.to_numeric(row[columns], errors="coerce").to_numpy(float)
                values = values[np.isfinite(values)]
                record[f"n_trials_pain_{temperature}_{site}"] = len(values)
                record[f"sd_pain_{temperature}_{site}"] = float(np.std(values, ddof=1)) if len(values) >= 2 else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def behaviour_table(project: Project) -> pd.DataFrame:
    """Complete-case table of mean ratings and within-participant SDs (nothing is written)."""
    means = read_table(project.mean_ratings, "Mean-rating table", sep="\t")
    columns = [f"pain_{t}_{s}" for s in SITES for t in TEMPERATURES]
    require_columns(means, ["sub_id", *columns], "Mean-rating table")
    means = drop_excluded(add_subject_key(means, "Mean-rating table", "sub_id"), project.excluded)
    trials = read_table(project.trial_ratings, "Trial-rating table")
    require_columns(trials, ["sub_num"], "Trial-rating table")
    trials = drop_excluded(add_subject_key(trials, "Trial-rating table", "sub_num"), project.excluded)
    merged = means[["subject_key", *columns]].merge(within_participant_sd(trials), on="subject_key", how="inner", validate="one_to_one")
    merged["arm_mean_across_temperatures"] = merged[["pain_46_arm", "pain_48_arm"]].mean(axis=1, skipna=False)
    merged["leg_mean_across_temperatures"] = merged[["pain_46_leg", "pain_48_leg"]].mean(axis=1, skipna=False)
    merged["overall_mean_across_conditions"] = merged[columns].mean(axis=1, skipna=False)
    required = [*columns, *[f"sd_{c}" for c in columns], "arm_mean_across_temperatures", "leg_mean_across_temperatures",
                "overall_mean_across_conditions"]
    data = merged[np.isfinite(merged[required].to_numpy(float)).all(axis=1)].sort_values("subject_key").reset_index(drop=True)
    if data.empty:
        raise DataError("No participant has complete ratings and trial data; check the input tables.")
    return data


def prepare_behaviour(project: Project) -> pd.DataFrame:
    """Figure 1 behavioural sample plus its group summary, both written to disk."""
    data = behaviour_table(project)
    columns = [f"pain_{t}_{s}" for s in SITES for t in TEMPERATURES]
    measures = {f"mean_{c[5:]}": c for c in columns} | {f"within_subject_sd_{c[5:]}": f"sd_{c}" for c in columns} | {
        k: k for k in ("arm_mean_across_temperatures", "leg_mean_across_temperatures", "overall_mean_across_conditions")}
    summary = pd.DataFrame([{"measure": name, "n_subjects": len(data), "group_mean": data[c].mean(), "group_sd": data[c].std(ddof=1),
                             "group_sem": data[c].std(ddof=1) / np.sqrt(len(data)), "median": data[c].median(),
                             "iqr": data[c].quantile(0.75) - data[c].quantile(0.25)} for name, c in measures.items()])
    data.to_csv(project.f1 / "subjects.csv", index=False)
    summary.to_csv(project.f1 / "summary.csv", index=False)
    print(f"Figure 1: {len(data)} participants with complete behavioural data")
    return data


def prepare_calibration(project: Project, subject_keys) -> pd.DataFrame:
    """Calibration temperatures (mean per participant x target VAS) after Tukey outlier removal per VAS level."""
    table = read_table(project.calibration, "Calibration table")
    require_columns(table, ["subject_id", "target_vas", "estimated_temperature_c"], "Calibration table")
    table["subject_key"] = table["subject_id"].map(canonical_subject_id)
    table = drop_excluded(table, project.excluded)
    for column in ("target_vas", "estimated_temperature_c"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table[table["subject_key"].isin(set(subject_keys.astype(str))) & table["target_vas"].isin(CALIBRATION_LEVELS)
                  & np.isfinite(table["estimated_temperature_c"])]
    if table.empty:
        raise DataError("No calibration rows match the behavioural sample.")
    per_cell = table.groupby(["subject_key", "target_vas"], as_index=False).agg(
        estimated_temperature_c=("estimated_temperature_c", "mean"), n_source_rows=("estimated_temperature_c", "size"))
    if (per_cell["n_source_rows"] > 1).any():
        print(f"Note: duplicated calibration rows averaged for {int((per_cell['n_source_rows'] > 1).sum())} participant x target pairs.")
    flagged = []
    for _, group in per_cell.groupby("target_vas"):
        group = group.copy()
        q1, q3 = group["estimated_temperature_c"].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = (q1 - CALIBRATION_IQR * iqr, q3 + CALIBRATION_IQR * iqr) if iqr > 0 else (q1, q3)
        group["is_outlier"] = ((group["estimated_temperature_c"] < lower) | (group["estimated_temperature_c"] > upper)) if iqr > 0 else False
        group[["q1_temperature_c", "q3_temperature_c", "iqr_temperature_c", "lower_outlier_fence_c", "upper_outlier_fence_c"]] = q1, q3, iqr, lower, upper
        flagged.append(group)
    flagged = pd.concat(flagged, ignore_index=True).sort_values(["subject_key", "target_vas"]).reset_index(drop=True)
    clean, outliers = flagged[~flagged["is_outlier"].astype(bool)], flagged[flagged["is_outlier"].astype(bool)]
    counts = clean.groupby("target_vas")["subject_key"].nunique().reindex(CALIBRATION_LEVELS, fill_value=0)
    if (counts < 2).any():
        raise DataError(f"Fewer than two calibration observations remain after outlier removal at VAS {list(counts[counts < 2].index)}.")
    summary = pd.DataFrame([{
        "target_vas": v, "n_before_outlier_removal": int((flagged["target_vas"] == v).sum()),
        "n_outliers_removed": int(flagged.loc[flagged["target_vas"] == v, "is_outlier"].sum()),
        "n_after_outlier_removal": int((clean["target_vas"] == v).sum()),
        "group_mean_temperature_c": float(clean.loc[clean["target_vas"] == v, "estimated_temperature_c"].mean()),
        "group_sd_temperature_c": float(clean.loc[clean["target_vas"] == v, "estimated_temperature_c"].std(ddof=1)),
        "group_sem_temperature_c": float(clean.loc[clean["target_vas"] == v, "estimated_temperature_c"].std(ddof=1) / np.sqrt((clean["target_vas"] == v).sum())),
        "median_temperature_c": float(clean.loc[clean["target_vas"] == v, "estimated_temperature_c"].median()),
        **{c: float(flagged.loc[flagged["target_vas"] == v, c].iloc[0]) for c in
           ("q1_temperature_c", "q3_temperature_c", "lower_outlier_fence_c", "upper_outlier_fence_c")},
        "outlier_rule": f"Tukey {CALIBRATION_IQR:g} x IQR"} for v in CALIBRATION_LEVELS])
    clean.to_csv(project.f1 / "calibration_subjects.csv", index=False)
    summary.to_csv(project.f1 / "calibration_summary.csv", index=False)
    outliers.to_csv(project.f1 / "calibration_outliers.csv", index=False)
    print(f"Calibration: {len(clean)} observations kept, {len(outliers)} outliers removed (Tukey {CALIBRATION_IQR:g} x IQR per VAS level)")
    return clean


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def subject_jitter(keys: pd.Series, width: float) -> np.ndarray:
    """Per-participant offset (same at 46 and 48 °C, so paired lines stay coherent)."""
    unique = sorted(keys.astype(str).unique())
    offsets = dict(zip(unique, np.random.default_rng(20260803).uniform(-width, width, len(unique)), strict=True))
    return keys.astype(str).map(offsets).to_numpy(float)


def style_axis(ax, xlim, xticks, xticklabels) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for name in ("bottom", "left"):
        ax.spines[name].set_linewidth(1.30)
    ax.tick_params(axis="both", labelsize=SIZE["tick"], width=1.15, length=4.2, pad=3)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)


def mean_and_sd(ax, x, values, color) -> None:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    ax.errorbar(x, values.mean(), yerr=values.std(ddof=1), fmt="D", markersize=SIZE["marker"], markerfacecolor=color,
                markeredgecolor="black", markeredgewidth=1.0, ecolor="black", elinewidth=SIZE["errorbar"],
                capsize=SIZE["cap"], capthick=SIZE["errorbar"], zorder=10, clip_on=False)


def points(ax, x, y, color) -> None:
    ax.scatter(x, y, s=SIZE["point"], color=color, alpha=SIZE["point_alpha"], edgecolors="none", zorder=3, clip_on=False)


def threshold_line(ax) -> None:
    ax.axhline(PAIN_THRESHOLD, color="#5A5A5A", linewidth=1.15, alpha=0.72, linestyle="--", zorder=0).set_dashes((4.0, 3.0))


def paired_panel(ax, data, col46, col48, color, group_color, title, ylim=None, threshold=False) -> None:
    wide = data[["subject_key", col46, col48]].dropna()
    if threshold:
        threshold_line(ax)
    jitter = subject_jitter(wide["subject_key"], 0.042)
    x46, x48 = X_TEMPERATURE[46] + jitter, X_TEMPERATURE[48] + jitter
    y46, y48 = wide[col46].to_numpy(float), wide[col48].to_numpy(float)
    for xs, ys in zip(zip(x46, x48), zip(y46, y48), strict=True):
        ax.plot(xs, ys, color=color, alpha=SIZE["line_alpha"], linewidth=SIZE["line_width"], zorder=1, solid_capstyle="round")
    points(ax, x46, y46, color)
    points(ax, x48, y48, color)
    mean_and_sd(ax, X_TEMPERATURE[46], y46, group_color)
    mean_and_sd(ax, X_TEMPERATURE[48], y48, group_color)
    ax.set_title(title, loc="left", fontsize=SIZE["title"], fontweight="bold", pad=7)
    style_axis(ax, (-0.20, 1.20), [0.0, 1.0], ["46 °C", "48 °C"])
    if ylim:
        ax.set_ylim(*ylim)


def summary_panel(ax, data) -> None:
    threshold_line(ax)
    for column, x, seed, color, group in (("arm_mean_across_temperatures", X_SUMMARY["arm"], 20260811, ARM_COLOR, ARM_GROUP),
                                          ("leg_mean_across_temperatures", X_SUMMARY["leg"], 20260812, LEG_COLOR, LEG_GROUP),
                                          ("overall_mean_across_conditions", X_SUMMARY["overall"], 20260813, OVERALL_COLOR, OVERALL_GROUP)):
        values = data[column].to_numpy(float)
        points(ax, x + np.random.default_rng(seed).uniform(-0.075, 0.075, len(values)), values, color)
        mean_and_sd(ax, x, values, group)
    ax.set_ylabel("Mean pain rating (VAS)", fontsize=SIZE["label"], labelpad=7)
    ax.set_ylim(*RATING_YLIM)
    style_axis(ax, (-0.42, 2.42), list(X_SUMMARY.values()), ["Arm", "Leg", "Overall"])


def calibration_panel(ax, calibration) -> None:
    wide = (calibration.pivot(index="subject_key", columns="target_vas", values="estimated_temperature_c")
            .reindex(columns=list(CALIBRATION_LEVELS)).sort_index())
    wide = wide[wide.notna().any(axis=1)]
    jitter = subject_jitter(pd.Series(wide.index.astype(str), index=wide.index), 0.65)
    base = np.asarray(CALIBRATION_LEVELS, float)
    for i, (_, row) in enumerate(wide.iterrows()):
        y, x = row.to_numpy(float), base + jitter[i]
        ax.plot(x, y, color=ARM_COLOR, alpha=SIZE["line_alpha"], linewidth=SIZE["line_width"], zorder=1, solid_capstyle="round")
        finite = np.isfinite(y)
        points(ax, x[finite], y[finite], ARM_COLOR)
    lows, highs = [], []
    for level in CALIBRATION_LEVELS:
        v = wide[level].dropna().to_numpy(float)
        lows.append(v.mean() - v.std(ddof=1))
        highs.append(v.mean() + v.std(ddof=1))
        mean_and_sd(ax, float(level), v, ARM_GROUP)
    finite_all = wide.to_numpy(float)[np.isfinite(wide.to_numpy(float))]
    low, high = min(finite_all.min(), *lows), max(finite_all.max(), *highs)
    pad = max(0.75, 0.08 * (high - low))
    ax.set_ylim(np.floor(low - pad), np.ceil(high + pad))
    ax.set_xlabel("Target pain rating (VAS)", fontsize=SIZE["label"], labelpad=8)
    ax.set_ylabel("Estimated temperature\n(°C)", fontsize=SIZE["label"], labelpad=10)
    style_axis(ax, (47.0, 83.0), list(CALIBRATION_LEVELS), [str(v) for v in CALIBRATION_LEVELS])


def make_figure(data: pd.DataFrame, calibration: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12.0, 10.0), facecolor="white")
    ax_arm = fig.add_axes([0.105, 0.710, 0.385, 0.250])
    ax_leg = fig.add_axes([0.555, 0.710, 0.385, 0.250], sharey=ax_arm)
    sd_arm = fig.add_axes([0.105, 0.395, 0.385, 0.250])
    sd_leg = fig.add_axes([0.555, 0.395, 0.385, 0.250], sharey=sd_arm)
    ax_summary = fig.add_axes([0.105, 0.065, 0.335, 0.225])
    ax_calibration = fig.add_axes([0.550, 0.065, 0.400, 0.225])
    paired_panel(ax_arm, data, "pain_46_arm", "pain_48_arm", ARM_COLOR, ARM_GROUP, "Arm", RATING_YLIM, True)
    paired_panel(ax_leg, data, "pain_46_leg", "pain_48_leg", LEG_COLOR, LEG_GROUP, "Leg", RATING_YLIM, True)
    paired_panel(sd_arm, data, "sd_pain_46_arm", "sd_pain_48_arm", ARM_COLOR, ARM_GROUP, "")
    paired_panel(sd_leg, data, "sd_pain_46_leg", "sd_pain_48_leg", LEG_COLOR, LEG_GROUP, "")
    sd_top = max(30.0, np.ceil(np.nanmax(data[[f"sd_pain_{t}_{s}" for t in TEMPERATURES for s in SITES]].to_numpy(float)) / 5.0) * 5.0)
    sd_arm.set_ylim(0.0, sd_top)
    sd_leg.set_ylim(0.0, sd_top)
    summary_panel(ax_summary, data)
    calibration_panel(ax_calibration, calibration)
    ax_arm.set_ylabel("Mean pain rating (VAS)", fontsize=SIZE["label"], labelpad=10)
    sd_arm.set_ylabel("Within-participant SD\nof pain ratings (VAS)", fontsize=SIZE["label"], labelpad=8)
    ax_leg.tick_params(labelleft=False)
    sd_leg.tick_params(labelleft=False)
    for letter, x, y in (("b", 0.010, 0.982), ("c", 0.010, 0.662), ("d", 0.010, 0.315), ("e", 0.455, 0.315)):
        fig.text(x, y, letter, fontsize=SIZE["panel"], fontweight="bold", ha="left", va="top")
    return fig


def run(project: Project) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    project.require("figure1")
    project.make_output_dirs()
    apply_style(14)
    data = prepare_behaviour(project)
    calibration = prepare_calibration(project, data["subject_key"])
    fig = make_figure(data, calibration)
    save_figure(fig, project.figure_files("Figure1"), bbox_inches=None)
    plt.close(fig)
    return data
