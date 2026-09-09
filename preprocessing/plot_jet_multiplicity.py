#!/usr/bin/env python
"""
AK4 jet-multiplicity comparison plot: number of jets/event, log-y, 4 series
overlaid -- non-Puppi vs Puppi AK4, L1T vs FullReco. Reads the raw (no cap/
truncation applied) jet collections produced by convert_data.py against
configs/raw_jets_minbias.yaml / raw_jets_ttbar_inclusive.yaml
(collections.<name>: null -- every jet kept, in original per-event order).

Series:
  L1T AK4 (non-Puppi)      -- L1T_JetAK4, unfiltered jet count
  L1T Puppi AK4            -- L1T_JetPuppiAK4, unfiltered jet count
  FullReco AK4 (non-Puppi) -- FullReco_JetAK4, unfiltered jet count
  FullReco Puppi AK4       -- FullReco_JetPuppiAK4, unfiltered jet count

CAVEAT: the underlying preprocessing run's empty-axis filter drops any
event where ALL FOUR jet collections are simultaneously empty (0 jets
everywhere) -- see raw_jets_minbias.yaml's comment. This means the true
zero-jet bin of each individual series may be under-represented, especially
for minbias. Check the preprocessing job log's total_empty_axis_dropped
before treating the low end of these distributions as unbiased.

Supports combining multiple sample directories into one dataset (e.g. the
three ttbar final states mixed to a branching-ratio-weighted "inclusive
ttbar"), each optionally capped at a fixed number of events -- see
plot_candidate_multiplicity.py's docstring for the same --max-events
rationale (whole-file-granularity overshoot at preprocessing time).

Usage
-----
# single sample
python plot_jet_multiplicity.py \
    --data-dir /mnt/temp-data/raw_jets_minbias/minbias \
    --max-events 10000 \
    --out-path-prefix /tmp/jet_multiplicity_minbias

# multiple samples combined, each capped to an exact event count
python plot_jet_multiplicity.py \
    --data-dir .../hadronic .../semiLeptonic .../leptonic \
    --max-events 4570 4380 1050 \
    --out-path-prefix /tmp/jet_multiplicity_ttbar_inclusive
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
        "L1T AK4 (non-Puppi)": ak.to_numpy(ak.num(arr["L1T_JetAK4"]["PT"], axis=1)),
        "L1T Puppi AK4": ak.to_numpy(ak.num(arr["L1T_JetPuppiAK4"]["PT"], axis=1)),
        "FullReco AK4 (non-Puppi)": ak.to_numpy(ak.num(arr["FullReco_JetAK4"]["PT"], axis=1)),
        "FullReco Puppi AK4": ak.to_numpy(ak.num(arr["FullReco_JetPuppiAK4"]["PT"], axis=1)),
    }


def plot_series(series: dict, title: str, out_path: str, n_bins: int = 40, log_x: bool = False) -> None:
    max_val = max(arr.max() for arr in series.values())
    if log_x:
        min_val = max(1, min(arr[arr > 0].min() if (arr > 0).any() else 1 for arr in series.values()))
        bins = np.logspace(np.log10(min_val), np.log10(max_val + 1), n_bins + 1)
    else:
        bins = np.arange(0, max_val + 2) - 0.5  # integer-centered bins -- jet counts are small integers

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#3b6fa0", "#e07b39", "#4a9c5f", "#b23b6f"]
    for (name, arr), color in zip(series.items(), colors):
        ax.hist(arr, bins=bins, histtype="step", linewidth=1.8, color=color,
                label=f"{name} (mean={arr.mean():.1f}, median={np.median(arr):.0f})")

    ax.set_yscale("log")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("AK4 jets / event")
    ax.set_ylabel("events")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, nargs="+",
                         help="One or more directories of *.parquet fragments to read and combine.")
    parser.add_argument("--max-events", type=int, nargs="+", default=None,
                         help="Cap events taken from each --data-dir, same order. A single value applies to "
                              "every --data-dir; otherwise must have one value per --data-dir.")
    parser.add_argument("--out-path-prefix", default="jet_multiplicity",
                         help="Both '<prefix>_linear.png' and '<prefix>_logx.png' are written.")
    parser.add_argument("--title", default="AK4 jet multiplicity (raw, uncapped)")
    parser.add_argument("--n-bins", type=int, default=40)
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
        n_zero = int((values == 0).sum())
        print(f"{name}: n={len(values)} mean={values.mean():.2f} median={np.median(values):.0f} "
              f"min={values.min()} max={values.max()} n_zero={n_zero}")

    plot_series(series, args.title, f"{args.out_path_prefix}_linear.png", args.n_bins, log_x=False)
    plot_series(series, args.title, f"{args.out_path_prefix}_logx.png", args.n_bins, log_x=True)
