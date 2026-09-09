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

Usage
-----
python plot_candidate_multiplicity.py \
    --data-dir /mnt/temp-data/raw_candidates_ttbar_semileptonic/tt0123j_5f_ckm_LO_MLM_semiLeptonic \
    --out-path /tmp/candidate_multiplicity_ttbar_semileptonic.png
"""
import argparse
import glob

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_all_fragments(data_dir: str) -> ak.Array:
    frags = sorted(glob.glob(f"{data_dir}/*.parquet"))
    if not frags:
        raise FileNotFoundError(f"no .parquet fragments found under {data_dir}")
    arrays = [ak.from_parquet(f) for f in frags]
    return arrays[0] if len(arrays) == 1 else ak.concatenate(arrays, axis=0)


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
    parser.add_argument("--data-dir", required=True,
                         help="Directory of *.parquet fragments (one sample's worth) to read.")
    parser.add_argument("--out-path", default="candidate_multiplicity.png")
    parser.add_argument("--title", default="tt0123j_5f_ckm_LO_MLM_semiLeptonic: candidate multiplicity "
                                             "(raw, no selection applied)")
    parser.add_argument("--n-bins", type=int, default=80)
    parser.add_argument("--log-x", action="store_true",
                         help="Log-scale the x-axis too (bins become log-spaced instead of linear).")
    args = parser.parse_args()

    arr = load_all_fragments(args.data_dir)
    print(f"Loaded {len(arr)} events from {args.data_dir}")
    series = build_series(arr)
    for name, values in series.items():
        print(f"{name}: n={len(values)} mean={values.mean():.1f} median={np.median(values):.1f} "
              f"min={values.min()} max={values.max()}")

    plot_series(series, args.title, args.out_path, args.n_bins, log_x=args.log_x)
