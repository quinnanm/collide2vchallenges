#!/usr/bin/env python
"""
Candidate-multiplicity comparison plot: number of candidates/event, log-y,
4 series overlaid -- PF vs PUPPI (weight > 0.5), L1T vs FullReco. Reads the
raw (no selection/cuts applied) candidate collections produced by
convert_data.py against configs/raw_candidates_ttbar_semileptonic.yaml
(candidate_selection.pt: none -- see that config's comments).

Series:
  L1T PF cands            -- L1T_PFPart, unfiltered candidate count
  L1T Puppi cands, w>0.5  -- L1T_PUPPIPart, count with puppi_weight > 0.5
  FullReco PF cands       -- FullReco_PFPart, unfiltered candidate count
  FullReco Puppi, w>0.5   -- FullReco_PUPPIPart, count with puppi_weight > 0.5

Supports combining multiple sample directories into one dataset (e.g. the
three ttbar final states mixed to a branching-ratio-weighted "inclusive
ttbar"), each optionally capped at a fixed number of events -- since
preprocessing uses whole-file granularity (overshoot allowed, never
truncates mid-file), a small requested target_events can still yield ~1
file's worth (~9-10k events); the exact desired proportions are enforced
here at load time via --max-events, not at preprocessing time.

Usage
-----
# single sample
python plot_candidate_multiplicity.py \
    --data-dir /mnt/temp-data/raw_candidates_ttbar_semileptonic/tt0123j_5f_ckm_LO_MLM_semiLeptonic \
    --out-path-prefix /tmp/candidate_multiplicity_ttbar_semileptonic

# multiple samples combined, each capped to an exact event count
python plot_candidate_multiplicity.py \
    --data-dir .../hadronic .../semiLeptonic .../leptonic \
    --max-events 4570 4380 1050 \
    --out-path-prefix /tmp/candidate_multiplicity_ttbar_inclusive
"""
import argparse
import glob

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_sample(data_dir: str, max_events: int = None) -> ak.Array:
    frags = sorted(glob.glob(f"{data_dir}/*.parquet"))
    if not frags:
        raise FileNotFoundError(f"no .parquet fragments found under {data_dir}")
    arrays = [ak.from_parquet(f) for f in frags]
    arr = arrays[0] if len(arrays) == 1 else ak.concatenate(arrays, axis=0)
    if max_events is not None:
        if len(arr) < max_events:
            raise ValueError(f"{data_dir}: only {len(arr)} events available, need {max_events}")
        arr = arr[:max_events]
    return arr


def load_all_fragments(data_dir: str) -> ak.Array:
    """Back-compat single-dir loader (used by tests/smoke test)."""
    return load_sample(data_dir)


def load_combined(data_dirs: list, max_events_list: list = None) -> ak.Array:
    if max_events_list is None:
        max_events_list = [None] * len(data_dirs)
    samples = []
    for data_dir, max_events in zip(data_dirs, max_events_list):
        arr = load_sample(data_dir, max_events)
        print(f"Loaded {len(arr)} events from {data_dir}" + (f" (capped at {max_events})" if max_events else ""))
        samples.append(arr)
    return samples[0] if len(samples) == 1 else ak.concatenate(samples, axis=0)


def build_series(arr: ak.Array) -> dict:
    return {
        "L1T PF cands": ak.to_numpy(ak.num(arr["L1T_PFPart"]["pt"], axis=1)),
        "L1T Puppi cands, w>0.5": ak.to_numpy(ak.sum(arr["L1T_PUPPIPart"]["puppi_weight"] > 0.5, axis=1)),
        "FullReco PF cands": ak.to_numpy(ak.num(arr["FullReco_PFPart"]["pt"], axis=1)),
        "FullReco Puppi cands, w>0.5": ak.to_numpy(ak.sum(arr["FullReco_PUPPIPart"]["puppi_weight"] > 0.5, axis=1)),
    }


def plot_series(series: dict, title: str, out_path: str, n_bins: int = 80, log_x: bool = False) -> None:
    max_val = max(arr.max() for arr in series.values())
    if log_x:
        # Linear bins would look wrong (bunched at the low end) once the
        # x-axis itself is log-scaled -- need log-spaced bin edges instead.
        # log(0) is undefined, so the lower edge is the smallest ACTUAL
        # value across every series (clipped to >=1 defensively), not 0.
        min_val = max(1, min(arr[arr > 0].min() if (arr > 0).any() else 1 for arr in series.values()))
        bins = np.logspace(np.log10(min_val), np.log10(max_val + 1), n_bins + 1)
    else:
        bins = np.linspace(0, max_val + 1, n_bins + 1)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#3b6fa0", "#e07b39", "#4a9c5f", "#b23b6f"]
    for (name, arr), color in zip(series.items(), colors):
        ax.hist(arr, bins=bins, histtype="step", linewidth=1.8, color=color,
                label=f"{name} (mean={arr.mean():.0f}, median={np.median(arr):.0f})")

    ax.set_yscale("log")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("candidates / event")
    ax.set_ylabel("events")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, nargs="+",
                         help="One or more directories of *.parquet fragments to read and combine.")
    parser.add_argument("--max-events", type=int, nargs="+", default=None,
                         help="Cap events taken from each --data-dir, same order (enforces an exact mixture "
                              "when combining samples). A single value applies to every --data-dir; otherwise "
                              "must have one value per --data-dir.")
    parser.add_argument("--out-path-prefix", default="candidate_multiplicity",
                         help="Both '<prefix>_linear.png' and '<prefix>_logx.png' are written.")
    parser.add_argument("--title", default="candidate multiplicity (raw, no selection applied)")
    parser.add_argument("--n-bins", type=int, default=80)
    args = parser.parse_args()

    max_events_list = args.max_events
    if max_events_list is not None and len(max_events_list) == 1 and len(args.data_dir) > 1:
        max_events_list = max_events_list * len(args.data_dir)
    if max_events_list is not None and len(max_events_list) != len(args.data_dir):
        parser.error(f"--max-events has {len(max_events_list)} values but there are {len(args.data_dir)} --data-dir")

    arr = load_combined(args.data_dir, max_events_list)
    print(f"Combined total: {len(arr)} events")
    series = build_series(arr)
    for name, values in series.items():
        print(f"{name}: n={len(values)} mean={values.mean():.1f} median={np.median(values):.1f} "
              f"min={values.min()} max={values.max()}")

    plot_series(series, args.title, f"{args.out_path_prefix}_linear.png", args.n_bins, log_x=False)
    plot_series(series, args.title, f"{args.out_path_prefix}_logx.png", args.n_bins, log_x=True)
