#!/usr/bin/env python
"""
One-off study: for every process directory in the source EOS dataset (not
scoped to any one challenge's sample list), using 1 real file per process,
compares two L1T_PUPPIPart candidate-selection scenarios -- same region
geometry (90-region PFL1, cap=18/region), same |eta|<=3 cut, same 1 GeV
floor, differing only in which pT drives the floor/rank:

  puppi cut: selection_pt="weighted" -- PuppiW*pt (puppi_pt) >= 1 GeV
  raw cut:   selection_pt="raw"      -- raw pt >= 1 GeV

and reports, per process, the number-of-surviving-candidates-per-event
distribution (mean/median/quartiles) for each scenario, plus a sanity check
that no surviving candidate actually violates its own cut (PuppiW<=0 or
puppi_pt<1 for the puppi cut; raw_pt<1 for the raw cut) -- both cuts should
report zero violations if gather_and_select_puppi_candidates's floor logic
is doing what its docstring says.

Reuses gather_and_select_puppi_candidates (the same selection code real
conversion runs use) rather than reimplementing the region/floor logic --
see that function's docstring in converters.py for the full mechanics.

Usage
-----
python candidate_selection_study.py
python candidate_selection_study.py --report-path study.txt --npz-path study.npz --version-scan-limit 50
"""
import argparse
import logging
import os

import awkward as ak
import numpy as np

from converters import (
    PUPPI_CAND_RAW_FIELDS,
    _discover_sample_files,
    _file_dataset_version,
    gather_and_select_puppi_candidates,
)
from config import join_remote

logger = logging.getLogger("candidate_selection_study")

DEFAULT_SAMPLE_DIR = "/eos/project/f/foundational-model-dataset/samples/production_final"
DEFAULT_REDIR = "root://eosproject-f.cern.ch/"
DEFAULT_DATASET_VERSION = "collide2v_v1.0"

ETA_CUT = [{"field": "Eta", "op": ">=", "value": -3.0}, {"field": "Eta", "op": "<=", "value": 3.0}]
NEUTRAL_CUT = [{"field": "Charge", "op": "==", "value": 0}]
# The output pt/pt_weighted fields gather_and_select_puppi_candidates returns
# are float16-downcast for storage (selection itself runs at float64) -- a
# candidate that passed at, say, pt_weighted=1.0000001 can round to slightly
# under 1.0 in float16, so the post-hoc violation check needs a tolerance
# matching float16's ~1e-3 relative precision, not a bit-exact >=1.0 test.
FLOAT16_TOL = 2e-3


def list_processes(sample_dir: str, redir: str) -> list:
    """Every top-level (process) subdirectory under sample_dir. Mirrors
    list_remote_files's own detail=False + hidden-entry-filtering style,
    just without the .parquet filter (we want directory names here, not
    files) -- deliberately doesn't filter by fsspec's 'type' key, so a
    misclassified entry just ends up with zero discoverable files downstream
    and gets skipped there, rather than silently vanishing from this list."""
    import fsspec

    url = join_remote(redir, sample_dir) if redir else sample_dir
    fs, fpath = fsspec.core.url_to_fs(url)
    names = fs.ls(fpath, detail=False)
    return sorted(os.path.basename(n.rstrip("/")) for n in names if not os.path.basename(n).startswith("."))


def first_matching_file(sample_dir: str, redir: str, sample: str, dataset_version: str,
                         version_scan_limit: int):
    """First file (in discovery order) whose dataset_version matches, or
    None if none of the first `version_scan_limit` candidates match (cheap,
    metadata-only checks -- see _file_dataset_version)."""
    candidates = _discover_sample_files(sample_dir, redir, sample, explicit_files=None)
    for fname in candidates[:version_scan_limit]:
        sample_path = f"{sample_dir}/{sample}/{fname}"
        src = join_remote(redir, sample_path) if redir else sample_path
        if _file_dataset_version(src) == dataset_version.encode():
            return src
    return None


def ncands_stats(ncands: np.ndarray) -> dict:
    return {
        "n_events": len(ncands),
        "mean": float(np.mean(ncands)),
        "median": float(np.median(ncands)),
        "min": int(np.min(ncands)),
        "p25": float(np.percentile(ncands, 25)),
        "p75": float(np.percentile(ncands, 75)),
        "max": int(np.max(ncands)),
    }


def run_one_process(src: str, neutral_only: bool = False) -> dict:
    raw_columns = [f"L1T_PUPPIPart_{f}" for f in PUPPI_CAND_RAW_FIELDS]
    arr = ak.from_parquet(src, columns=raw_columns)

    # neutral_only restricts the CANDIDATE POOL to Charge==0 before the
    # region/floor selection runs (object_selection is applied before
    # ranking/capping -- see gather_and_select_puppi_candidates's docstring),
    # so neutrals compete only against other neutrals for each region's 18
    # slots -- not "select from everyone, then filter to neutral survivors".
    object_selection = ETA_CUT + NEUTRAL_CUT if neutral_only else ETA_CUT

    puppi = gather_and_select_puppi_candidates(
        arr, selection_pt="weighted", mode="region", cap=18, floor_gev=1.0, object_selection=object_selection,
    )
    raw = gather_and_select_puppi_candidates(
        arr, selection_pt="raw", mode="region", cap=18, floor_gev=1.0, object_selection=object_selection,
    )

    puppi_ncands = ak.to_numpy(ak.num(puppi["pt"], axis=1))
    raw_ncands = ak.to_numpy(ak.num(raw["pt"], axis=1))

    puppi_flat_w = ak.to_numpy(ak.flatten(puppi["puppi_weight"], axis=1))
    puppi_flat_ptw = ak.to_numpy(ak.flatten(puppi["pt_weighted"], axis=1))
    raw_flat_pt = ak.to_numpy(ak.flatten(raw["pt"], axis=1))

    puppi_violations = int(np.sum((puppi_flat_w <= 0) | (puppi_flat_ptw < 1.0 - FLOAT16_TOL)))
    raw_violations = int(np.sum(raw_flat_pt < 1.0 - FLOAT16_TOL))

    return {
        "n_events_in_file": len(arr),
        "puppi": ncands_stats(puppi_ncands),
        "raw": ncands_stats(raw_ncands),
        "puppi_violations": puppi_violations,
        "raw_violations": raw_violations,
        "puppi_ncands_raw_array": puppi_ncands,
        "raw_ncands_raw_array": raw_ncands,
    }


def build_report(results: dict, brief: bool = False, neutral_only: bool = False) -> str:
    if brief:
        header = f"{'process':<45} {'n_ev':>6} | {'puppi mean':>10} {'median':>7} | {'raw mean':>9} {'median':>7}"
    else:
        header = (
            f"{'process':<45} {'n_ev':>6} | "
            f"{'puppi mean':>10} {'median':>7} {'p25':>6} {'p75':>6} {'min':>4} {'max':>4} {'viol':>5} | "
            f"{'raw mean':>9} {'median':>7} {'p25':>6} {'p75':>6} {'min':>4} {'max':>4} {'viol':>5}"
        )
    lines = [
        "Candidate-selection study: L1T_PUPPIPart, region geometry, cap=18/region, |eta|<=3, floor=1 GeV"
        + (", Charge==0 (neutral) candidates only" if neutral_only else ""),
        "  puppi cut: floor/rank driven by PuppiW*pt (weighted pT)",
        "  raw cut:   floor/rank driven by raw pT",
        "1 file/process, real EOS data."
        + ("" if brief else " 'viol' = surviving candidates that actually violate their own cut "
                             "(PuppiW<=0 or puppi_pt<1 for puppi; raw_pt<1 for raw) -- should be 0, every row."),
        "",
        header,
        "-" * len(header),
    ]
    for name, r in results.items():
        if r is None:
            lines.append(f"{name:<45}  ** no version-matched file found -- skipped **")
            continue
        p, w = r["puppi"], r["raw"]
        if brief:
            lines.append(
                f"{name:<45} {r['n_events_in_file']:>6} | "
                f"{p['mean']:>10.2f} {p['median']:>7.1f} | "
                f"{w['mean']:>9.2f} {w['median']:>7.1f}"
            )
        else:
            lines.append(
                f"{name:<45} {r['n_events_in_file']:>6} | "
                f"{p['mean']:>10.2f} {p['median']:>7.1f} {p['p25']:>6.1f} {p['p75']:>6.1f} "
                f"{p['min']:>4} {p['max']:>4} {r['puppi_violations']:>5} | "
                f"{w['mean']:>9.2f} {w['median']:>7.1f} {w['p25']:>6.1f} {w['p75']:>6.1f} "
                f"{w['min']:>4} {w['max']:>4} {r['raw_violations']:>5}"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-dir", default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--redir", default=DEFAULT_REDIR)
    parser.add_argument("--dataset-version", default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--version-scan-limit", type=int, default=50,
                         help="Files to check dataset_version on (cheap, metadata-only) before giving up "
                              "on a process (default: 50).")
    parser.add_argument("--report-path", default="candidate_selection_study.txt")
    parser.add_argument("--npz-path", default="candidate_selection_study.npz",
                         help="Per-process, per-cut raw ncands-per-event arrays, for follow-up plotting.")
    parser.add_argument("--neutral-only", action="store_true",
                         help="Restrict the candidate pool to Charge==0 before selection (neutrals compete "
                              "only against other neutrals for each region's cap -- see run_one_process).")
    parser.add_argument("--brief", action="store_true",
                         help="Report just mean/median per process (drop p25/p75/min/max/viol columns). "
                              "Violations are still checked and logged as warnings if any are found.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    processes = list_processes(args.sample_dir, args.redir)
    logger.info(f"Found {len(processes)} process directories.")

    results = {}
    npz_arrays = {}
    for proc in processes:
        try:
            src = first_matching_file(args.sample_dir, args.redir, proc, args.dataset_version,
                                       args.version_scan_limit)
        except Exception as e:
            logger.warning(f"{proc}: directory listing/version-scan failed -- skipping. {e}")
            results[proc] = None
            continue
        if src is None:
            logger.warning(f"{proc}: no dataset_version={args.dataset_version!r} file found in first "
                            f"{args.version_scan_limit} candidates -- skipping.")
            results[proc] = None
            continue
        logger.info(f"{proc}: measuring {src}")
        try:
            r = run_one_process(src, neutral_only=args.neutral_only)
        except Exception as e:
            logger.warning(f"{proc}: failed to process {src} -- skipping. {e}")
            results[proc] = None
            continue
        if r["puppi_violations"] or r["raw_violations"]:
            logger.warning(f"{proc}: {r['puppi_violations']} puppi / {r['raw_violations']} raw violation(s) "
                            f"found -- selection logic may not be airtight for this process.")
        npz_arrays[f"{proc}__puppi_ncands"] = r.pop("puppi_ncands_raw_array")
        npz_arrays[f"{proc}__raw_ncands"] = r.pop("raw_ncands_raw_array")
        results[proc] = r
        logger.info(f"{proc}: puppi mean={r['puppi']['mean']:.2f} median={r['puppi']['median']:.1f}  "
                    f"raw mean={r['raw']['mean']:.2f} median={r['raw']['median']:.1f}")

    report = build_report(results, brief=args.brief, neutral_only=args.neutral_only)
    with open(args.report_path, "w") as fh:
        fh.write(report)
    logger.info(f"Wrote report to {args.report_path}")

    np.savez_compressed(args.npz_path, **npz_arrays)
    logger.info(f"Wrote per-event ncands arrays to {args.npz_path}")

    print(report)
