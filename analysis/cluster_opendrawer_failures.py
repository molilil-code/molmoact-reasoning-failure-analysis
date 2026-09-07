"""Exploratory failure-only clustering for OpenDrawer.

This is Path A in the analysis plan: retrospective failure-mode
characterization, not online prediction.  The clustering input contains only
episode-level aggregates of reasoning features from failure episodes.  Labels,
task-state fields, post-action environment fields, diagnostics, and onset
timestamps are excluded from the clustering input.

The small OpenDrawer sample (8 failures) makes this an exploratory analysis.
The script therefore reports cluster quality and leave-one-episode-out
stability, and uses a two-cluster solution only when every cluster has at least
two episodes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


DEFAULT_STEP_FEATURES = Path("data/formal_features/step_features.csv")
DEFAULT_ONSETS = Path("data/opendrawer_analysis/opendrawer_onsets.csv")
DEFAULT_OUTPUT_DIR = Path("data/opendrawer_analysis")


# These are intentionally limited to interpretable instability signals.  The
# signs are all oriented as "larger = more instability" before aggregation.
FEATURE_FAMILIES: dict[str, list[str]] = {
    "depth_instability": [
        "feat_depth_hamming_prev",
        "feat_depth_change_ratio_prev",
        "feat_depth_change_ratio_rollmean",
    ],
    "trace_instability": [
        "feat_trace_endpoint_drift_prev",
        "feat_trace_shape_drift_prev",
        "feat_trace_direction_change_prev_rad",
        "feat_trace_shape_drift_rollmean",
    ],
    "action_instability": [
        "feat_model_action_jump_prev_l2",
        "feat_model_translation_jump_prev_l2",
        "feat_model_rotation_jump_prev_l2",
        "feat_model_translation_direction_flip",
    ],
}

AGG_STATS = ("mean", "p90")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-features", type=Path, default=DEFAULT_STEP_FEATURES)
    parser.add_argument("--onsets", type=Path, default=DEFAULT_ONSETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def percentile90(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(np.nanpercentile(values, 90)) if len(values) else np.nan


def aggregate_failures(step_features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    required = {"task", "episode_key", "condition_id", "step_id", "label_episode_success"}
    missing = required.difference(step_features.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = step_features.loc[
        step_features["task"].eq("open_drawer")
        & step_features["label_episode_success"].eq(0)
    ].copy()
    if rows.empty:
        raise ValueError("No OpenDrawer failure episodes found")

    available: dict[str, list[str]] = {}
    for family, columns in FEATURE_FAMILIES.items():
        present = [column for column in columns if column in rows.columns]
        if not present:
            raise ValueError(f"No columns available for feature family {family}")
        available[family] = present

    output: list[dict[str, object]] = []
    for episode_key, episode in rows.groupby("episode_key", sort=True):
        record: dict[str, object] = {
            "task": "open_drawer",
            "episode_key": episode_key,
            "condition_id": int(episode["condition_id"].iloc[0]),
            "num_steps": int(len(episode)),
        }
        for family, columns in available.items():
            for column in columns:
                numeric = pd.to_numeric(episode[column], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
                record[f"{family}__{column}__mean"] = float(numeric.mean())
                record[f"{family}__{column}__p90"] = percentile90(numeric)
        output.append(record)

    return pd.DataFrame(output), available


def family_scores(
    aggregates: pd.DataFrame, available: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return standardized family scores and the exact clustering inputs.

    First average the feature aggregates within each family, then standardize
    the family-level mean and p90 columns across failure episodes.  This keeps
    one unusually noisy raw feature from dominating an entire family and makes
    the resulting three-dimensional representation easy to interpret.
    """

    feature_columns: list[str] = []
    for family, columns in available.items():
        for column in columns:
            feature_columns.extend(
                [
                    f"{family}__{column}__mean",
                    f"{family}__{column}__p90",
                ]
            )

    raw_family = pd.DataFrame(index=aggregates.index)
    for family, columns in available.items():
        for stat in AGG_STATS:
            cols = [f"{family}__{column}__{stat}" for column in columns]
            raw_family[f"{family}__{stat}"] = (
                aggregates[cols]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .mean(axis=1)
            )
    raw_family = raw_family.fillna(raw_family.median()).fillna(0.0)

    # Standardize each family-level aggregate across failure episodes.  A
    # constant aggregate contributes zero rather than an artificial direction.
    scaled_family = pd.DataFrame(
        StandardScaler().fit_transform(raw_family),
        columns=raw_family.columns,
        index=aggregates.index,
    )
    scores = pd.DataFrame(index=aggregates.index)
    for family in available:
        family_columns = [f"{family}__{stat}" for stat in AGG_STATS]
        scores[f"{family}_score"] = scaled_family[family_columns].mean(axis=1)

    return scores, scaled_family, feature_columns


def mode_name(centroid: pd.Series) -> str:
    values = {
        "depth": float(centroid["depth_instability_score"]),
        "trace": float(centroid["trace_instability_score"]),
        "action": float(centroid["action_instability_score"]),
    }
    ordered = sorted(values, key=values.get, reverse=True)
    top, second, third = ordered
    if values[top] - values[second] >= 0.25:
        return f"{top}_dominant_instability"
    if values[top] + values[second] > values[third] + 0.25:
        return f"{top}_{second}_mixed_instability"
    return "mixed_instability"


def cluster_quality(scores: pd.DataFrame, random_state: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    x = scores.to_numpy(dtype=float)
    n = len(scores)
    for k in range(2, min(4, n - 1) + 1):
        model = KMeans(n_clusters=k, n_init=100, random_state=random_state)
        labels = model.fit_predict(x)
        counts = np.bincount(labels, minlength=k)
        rows.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(x, labels)) if len(set(labels)) > 1 else np.nan,
                "min_cluster_size": int(counts.min()),
                "cluster_sizes": ";".join(str(int(count)) for count in counts),
                "eligible_min_size_2": bool(counts.min() >= 2),
            }
        )
    return pd.DataFrame(rows)


def fit_two_cluster_solution(scores: pd.DataFrame, random_state: int) -> tuple[np.ndarray, pd.DataFrame, float, float]:
    x = scores.to_numpy(dtype=float)
    model = KMeans(n_clusters=2, n_init=100, random_state=random_state)
    labels = model.fit_predict(x)
    counts = np.bincount(labels, minlength=2)
    if counts.min() < 2:
        raise ValueError(
            "The requested K=2 solution has a singleton cluster; "
            "the failure sample is too small for a defensible partition."
        )
    centroids = pd.DataFrame(model.cluster_centers_, columns=scores.columns)

    # Make cluster IDs deterministic: cluster 0 is the one with more
    # trace-dominant behavior, if such a distinction exists.
    contrast = centroids["trace_instability_score"] - centroids[
        ["depth_instability_score", "action_instability_score"]
    ].mean(axis=1)
    order = np.argsort(-contrast.to_numpy())
    remap = {int(old): int(new) for new, old in enumerate(order)}
    labels = np.array([remap[int(label)] for label in labels], dtype=int)
    centroids = centroids.iloc[order].reset_index(drop=True)
    silhouette = float(silhouette_score(x, labels))

    # Leave-one-episode-out ARI, comparing each re-fit to the baseline labels
    # on the retained episodes.  This is more interpretable than a bootstrap
    # with only eight episodes.
    loo_ari: list[float] = []
    for held_out in range(len(scores)):
        keep = np.arange(len(scores)) != held_out
        if len(np.unique(labels[keep])) < 2:
            continue
        loo_model = KMeans(n_clusters=2, n_init=100, random_state=random_state)
        loo_labels = loo_model.fit_predict(x[keep])
        loo_ari.append(float(adjusted_rand_score(labels[keep], loo_labels)))
    stability = float(np.mean(loo_ari)) if loo_ari else np.nan
    return labels, centroids, silhouette, stability


def plot_clusters(assignments: pd.DataFrame, output_path: Path) -> None:
    """Save a compact 2-D view of the three family scores.

    The vertical axis is the average of Depth and Action instability.  This is
    a display only; the clustering itself uses all three family scores.
    """

    import matplotlib.pyplot as plt

    plot = assignments.copy()
    plot["depth_action_mean"] = plot[
        ["depth_instability_score", "action_instability_score"]
    ].mean(axis=1)
    colors = {
        "trace_dominant_instability": "#4C78A8",
        "depth_action_mixed_instability": "#F58518",
        "mixed_instability": "#54A24B",
    }
    fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=160)
    for mode, group in plot.groupby("failure_mode", sort=True):
        ax.scatter(
            group["trace_instability_score"],
            group["depth_action_mean"],
            s=70,
            alpha=0.9,
            label=mode,
            color=colors.get(mode, "#777777"),
            edgecolor="white",
            linewidth=0.7,
        )
        for row in group.itertuples(index=False):
            ax.annotate(
                f"C{int(row.condition_id):02d}",
                (row.trace_instability_score, row.depth_action_mean),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    ax.axhline(0, color="#bbbbbb", linewidth=0.8)
    ax.axvline(0, color="#bbbbbb", linewidth=0.8)
    ax.set_xlabel("Trace instability score")
    ax.set_ylabel("Mean of Depth and Action instability scores")
    ax.set_title("OpenDrawer failure-only modes (exploratory, n=8)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    assignments: pd.DataFrame,
    centroids: pd.DataFrame,
    quality: pd.DataFrame,
    silhouette: float,
    stability: float,
    feature_columns: Iterable[str],
) -> None:
    lines = [
        "# OpenDrawer failure-only clustering",
        "",
        "This is a retrospective failure-mode analysis (Path A), not an online predictor.",
        "Only the eight OpenDrawer failure episodes are clustered.",
        "The clustering input contains episode-level mean and p90 aggregates of",
        "Depth, Trace, and Action instability features. Success labels, task-state",
        "fields, post-action fields, diagnostics, and onset timestamps are excluded",
        "from the clustering input.",
        "",
        "## Sample and method",
        "",
        f"- Failure episodes: {len(assignments)}",
        "- Primary solution: K=2 K-means on three standardized family scores",
        f"- K=2 silhouette: {silhouette:.3f}",
        f"- Leave-one-episode-out ARI: {stability:.3f}",
        "- Quality interpretation: values this small sample should be treated as exploratory, not confirmatory.",
        "",
        "## Candidate K values",
        "",
        "| K | Silhouette | Cluster sizes | Eligible (all sizes >= 2) |",
        "|---:|---:|---|:---:|",
    ]
    for row in quality.itertuples(index=False):
        lines.append(
            f"| {int(row.k)} | {row.silhouette:.3f} | {row.cluster_sizes} | "
            f"{'yes' if row.eligible_min_size_2 else 'no'} |"
        )

    lines += ["", "## Failure modes", "", "| Mode | Episodes | Proportion |", "|---|---:|---:|"]
    counts = assignments["failure_mode"].value_counts().sort_index()
    for mode, count in counts.items():
        lines.append(f"| `{mode}` | {int(count)} | {count / len(assignments):.1%} |")

    lines += [
        "",
        "## Cluster centroids (standardized family scores)",
        "",
        "| Cluster | Failure mode | Depth | Trace | Action |",
        "|---:|---|---:|---:|---:|",
    ]
    for cluster_id, centroid in centroids.iterrows():
        mode = assignments.loc[assignments["cluster_k2"].eq(cluster_id), "failure_mode"].iloc[0]
        lines.append(
            f"| {cluster_id} | `{mode}` | {centroid['depth_instability_score']:.2f} | "
            f"{centroid['trace_instability_score']:.2f} | {centroid['action_instability_score']:.2f} |"
        )

    lines += [
        "",
        "## Episode assignments",
        "",
        "| Episode | Cluster | Mode | Depth | Trace | Action | t_onset (post-hoc) |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in assignments.sort_values(["cluster_k2", "episode_key"]).itertuples(index=False):
        onset = "" if pd.isna(row.automatic_t_onset) else str(int(row.automatic_t_onset))
        lines.append(
            f"| `{row.episode_key}` | {row.cluster_k2} | `{row.failure_mode}` | "
            f"{row.depth_instability_score:.2f} | {row.trace_instability_score:.2f} | "
            f"{row.action_instability_score:.2f} | {onset} |"
        )

    lines += [
        "",
        "## Interpretation guardrail",
        "",
        "The mode names are descriptive labels assigned from cluster centroids; they are not environment ground truth.",
        "The clustering uses full-episode aggregates, so it cannot establish that a mode occurs before failure onset.",
        "Use the result to choose episodes for qualitative review and to generate hypotheses for a future temporal analysis.",
        "",
        "Clustering columns:",
        "",
    ]
    lines.extend(f"- `{column}`" for column in feature_columns)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    step_features = pd.read_csv(args.step_features)
    aggregates, available = aggregate_failures(step_features)
    scores, scaled, feature_columns = family_scores(aggregates, available)
    quality = cluster_quality(scores, args.random_state)
    labels, centroids, silhouette, stability = fit_two_cluster_solution(scores, args.random_state)

    assignments = aggregates[["task", "episode_key", "condition_id", "num_steps"]].copy()
    assignments = pd.concat([assignments, scores], axis=1)
    assignments["cluster_k2"] = labels
    mode_by_cluster = {cluster_id: mode_name(row) for cluster_id, row in centroids.iterrows()}
    assignments["failure_mode"] = assignments["cluster_k2"].map(mode_by_cluster)

    if args.onsets.exists():
        onsets = pd.read_csv(args.onsets)
        onsets = onsets[["episode_key", "automatic_t_onset", "automatic_onset_frac"]]
        assignments = assignments.merge(onsets, on="episode_key", how="left")
    else:
        assignments["automatic_t_onset"] = np.nan
        assignments["automatic_onset_frac"] = np.nan

    assignments.to_csv(args.output_dir / "failure_mode_cluster_assignments.csv", index=False)
    aggregates.to_csv(args.output_dir / "failure_mode_cluster_episode_aggregates.csv", index=False)
    scores_with_ids = pd.concat(
        [aggregates[["episode_key", "condition_id"]], scores], axis=1
    )
    scores_with_ids.to_csv(args.output_dir / "failure_mode_cluster_family_scores.csv", index=False)
    centroids.assign(cluster_k2=centroids.index).to_csv(
        args.output_dir / "failure_mode_cluster_centroids.csv", index=False
    )
    quality.to_csv(args.output_dir / "failure_mode_cluster_quality.csv", index=False)
    write_report(
        args.output_dir / "failure_mode_clustering_report.md",
        assignments,
        centroids,
        quality,
        silhouette,
        stability,
        feature_columns,
    )
    plot_clusters(assignments, args.output_dir / "failure_mode_cluster_map.png")

    print(f"failure episodes: {len(assignments)}")
    print(f"K=2 silhouette: {silhouette:.3f}")
    print(f"leave-one-episode-out ARI: {stability:.3f}")
    print(assignments[["episode_key", "cluster_k2", "failure_mode"]].to_string(index=False))


if __name__ == "__main__":
    main()
