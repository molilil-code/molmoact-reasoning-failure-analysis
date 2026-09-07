"""Calculate episode-level peak interval statistics for depth-change ratio.

No onset timestamps are used.  The raw feature uses every interior local
maximum. Optional prominence-filtered columns are included to show whether
small-amplitude wiggles are driving the raw result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


DEFAULT_INPUT = Path("data/formal_features/step_features_trace_depth.csv")
DEFAULT_OUTPUT = Path("data/opendrawer_analysis/opendrawer_peak_interval_episode.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def peak_interval(values: np.ndarray, prominence: float | None) -> tuple[float, int]:
    peaks, _ = find_peaks(values, prominence=prominence)
    intervals = np.diff(peaks)
    if len(intervals) == 0:
        return np.nan, int(len(peaks))
    return float(np.median(intervals)), int(len(peaks))


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input)
    data = data.loc[data["task"].eq("open_drawer")].copy()
    data["feat_depth_change_ratio_prev"] = pd.to_numeric(
        data["feat_depth_change_ratio_prev"], errors="coerce"
    )
    rows: list[dict[str, object]] = []
    for episode_key, episode in data.groupby("episode_key", sort=True):
        values = episode["feat_depth_change_ratio_prev"].dropna().to_numpy(float)
        raw_interval, raw_peaks = peak_interval(values, prominence=None)
        p05_interval, p05_peaks = peak_interval(values, prominence=0.05)
        p10_interval, p10_peaks = peak_interval(values, prominence=0.10)
        rows.append(
            {
                "task": "open_drawer",
                "condition_id": int(episode["condition_id"].iloc[0]),
                "episode_key": episode_key,
                "episode_success": int(episode["label_episode_success"].iloc[0]),
                "num_valid_steps": int(len(values)),
                "raw_num_peaks": raw_peaks,
                "peak_interval_median": raw_interval,
                "prominence_0p05_num_peaks": p05_peaks,
                "peak_interval_median_prominence_0p05": p05_interval,
                "prominence_0p10_num_peaks": p10_peaks,
                "peak_interval_median_prominence_0p10": p10_interval,
            }
        )
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(args.output)


if __name__ == "__main__":
    main()
