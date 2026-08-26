#!/usr/bin/env python
"""Predict the total on-disk parquet size (and final event count) a
`convert_data.py` run against a given `dataconfig.yml` will produce, WITHOUT
running the full production job.

This is a light real-data dry run, not a pure static calculation: per-event
byte size and event-survival rates (event_selection, the empty-axis filter,
real per-collection object multiplicities) depend on the actual physics
content of the sample and can't be guessed accurately from config numbers
alone (e.g. L1T_PUPPIPart's region-based selection rarely comes close to
saturating its nominal per-region cap in practice). So for each sample, this
script:
  1. Reads a small number of that sample's real EOS files (in the same
     dataset_version-filtered discovery order production uses).
  2. Runs them through the EXACT SAME collections/candidate_selection/
     event_selection pipeline convert_collide2v_regionized uses (via the
     shared helpers in converters.py -- _resolve_collections_cfg,
     _resolve_candidate_selection_cfg, _build_record_for_file -- so this can
     never silently drift from what a real conversion would actually keep).
  3. Writes the sampled, converted events to a real (compressed) parquet
     file and measures its actual on-disk size, to get real bytes/event.
  4. Extrapolates to the sample's configured scope (target_events/max_files/
     "every file"), also checking real file-inventory limits (e.g. flags a
     target_events that exceeds what the sample's directory actually has).

Writes a `size_estimate.txt` report next to the config file (override with
--report-path) -- full/train/eval totals, then a breakdown of every output
directory a real conversion would create (events, predicted parquet fragment
count, approx size/file, total size) -- and prints the same report to
stdout.

Needs the same EOS/xrootd access (grid proxy) a real conversion does.

Usage
-----
python estimate_output_size.py --config ../C1_HH4b/dataconfig.yml
python estimate_output_size.py --config ../C1_HH4b/dataconfig.yml --sample-files 10 --version-scan-limit 2000
"""
import argparse
import datetime
import logging
import os
import tempfile
from pathlib import Path

import awkward as ak
import numpy as np

from config import DataConfig, join_remote
from converters import (
    _build_record_for_file,
    _discover_sample_files,
    _event_selection_columns,
    _file_dataset_version,
    _read_parquet_tolerant,
    _resolve_candidate_selection_cfg,
    _resolve_collections_cfg,
    _resolve_sample_entry,
)

logger = logging.getLogger("collide2vpreproc.estimate_output_size")


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < 1024.0:
            return f"{n:,.2f} {unit}"
        n /= 1024.0
    return f"{n:,.2f} EB"


def _sample_measurement_basis(sample: str, m: dict) -> str:
    if m["pinned"]:
        return f"{sample}: pinned files -- exact (every listed file measured)"
    basis = "exact" if m["exact_match_count"] else f"~{m['estimated_total_matching_files']:.0f} est."
    return (f"{sample}: measured {m['n_data_files_read']}/{m['total_files_in_dir']} files "
            f"({m['n_matched']}/{m['n_scanned']} version-matched, {basis} total-match count)")


def measure_sample(sample_dir: str, redir: str, dataset_version: str, max_events_per_file,
                    coll_cfg: dict, cs_cfg: dict, se: dict, sample_files_target: int,
                    version_scan_limit: int) -> dict:
    """Read+convert up to `sample_files_target` real (dataset_version-
    matching) files for this sample -- ALL of them if `files:` is pinned,
    since that list is already small/fixed and gives an exact answer -- plus
    a version-only scan (no data read) up to `version_scan_limit` files to
    estimate the sample directory's overall dataset_version match rate.
    Returns a measurement dict (see keys below); values are None where
    nothing could be measured (e.g. every sampled event was dropped)."""
    sample = se["sample"]
    pinned = se["explicit_files"] is not None
    candidate_file_names = _discover_sample_files(sample_dir, redir, sample, se["explicit_files"])
    total_files_in_dir = len(candidate_file_names)
    required_columns = list(dict.fromkeys(coll_cfg["candidate_columns"]
                                           + _event_selection_columns(se["event_selection"])))

    def _resolve_src(fname, sample_path=f"{sample_dir}/{sample}"):
        return join_remote(redir, f"{sample_path}/{fname}") if redir else os.path.join(sample_dir, sample, fname)

    scan_limit = total_files_in_dir if pinned else min(version_scan_limit, total_files_in_dir)
    data_target = total_files_in_dir if pinned else sample_files_target

    records = []
    n_scanned = n_matched = n_data_files_read = n_raw_events = n_kept_events = 0
    for i, fname in enumerate(candidate_file_names):
        if i >= scan_limit:
            break
        src = _resolve_src(fname)
        version = _file_dataset_version(src)
        n_scanned += 1
        if version != dataset_version.encode():
            if pinned:
                raise ValueError(f"[{sample}] pinned file {fname!r} has dataset_version={version!r} != "
                                  f"{dataset_version!r} -- pinned file lists must all match exactly.")
            continue
        n_matched += 1
        if n_data_files_read < data_target:
            n_data_files_read += 1
            arr = _read_parquet_tolerant(src, required_columns, coll_cfg["other_columns"], max_events_per_file)
            if len(arr) == 0:
                continue
            record, n_events_read, _, _ = _build_record_for_file(
                arr, want_candidates=coll_cfg["want_candidates"], candidate_selection_pt=cs_cfg["pt"],
                candidate_mode=cs_cfg["mode"], candidate_cap=coll_cfg["candidate_cap"],
                floor_gev=cs_cfg["floor_gev"], candidate_object_selection=coll_cfg["candidate_object_selection"],
                candidate_drop_fields=coll_cfg["candidate_drop_fields"], realistic_pid=cs_cfg["realistic_pid"],
                candidate_total_cap=coll_cfg["candidate_total_cap"],
                other_collections_cfg=coll_cfg["other_collections_cfg"],
                event_selection=se["event_selection"], label=se["label"], source_file_idx=n_data_files_read - 1,
                candidate_collection=coll_cfg["candidate_collection"])
            n_raw_events += n_events_read
            if record is not None:
                records.append(record)
                n_kept_events += len(record)

    bytes_size = 0
    if records:
        combined = records[0] if len(records) == 1 else ak.concatenate(records, axis=0)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True) as tmp:
            ak.to_parquet(combined, tmp.name)
            bytes_size = os.path.getsize(tmp.name)

    exact_match_count = pinned or (n_scanned >= total_files_in_dir)
    match_rate = (n_matched / n_scanned) if n_scanned else 0.0
    estimated_total_matching_files = n_matched if exact_match_count else match_rate * total_files_in_dir

    bytes_per_event = (bytes_size / n_kept_events) if n_kept_events else None
    events_per_matched_file = (n_kept_events / n_data_files_read) if n_data_files_read else None

    return {
        "total_files_in_dir": total_files_in_dir, "n_scanned": n_scanned, "n_matched": n_matched,
        "exact_match_count": exact_match_count, "estimated_total_matching_files": estimated_total_matching_files,
        "n_data_files_read": n_data_files_read, "pinned": pinned,
        "n_raw_events": n_raw_events, "n_kept_events": n_kept_events,
        "bytes_per_event": bytes_per_event, "events_per_matched_file": events_per_matched_file,
    }


def predict_sample(se: dict, m: dict) -> dict:
    """Extrapolate one sample's measurement to its configured scope
    (target_events/max_files/every-file). Returns {predicted_events, exact,
    predicted_bytes, shortfall_events (None if achievable/unknown)}."""
    epf = m["events_per_matched_file"]
    if m["pinned"]:
        return {"predicted_events": m["n_kept_events"], "exact": True,
                "predicted_bytes": m["bytes_per_event"] * m["n_kept_events"] if m["bytes_per_event"] else 0,
                "shortfall_events": None}

    if epf is None:
        return {"predicted_events": 0, "exact": False, "predicted_bytes": 0,
                "shortfall_events": None}

    if se["target_events"] is not None:
        target = se["target_events"]
        # whole-file-granularity overshoot -- production keeps whole files
        # until the running total is >= target, so round up to the next
        # file's worth using this sample's own measured events/file.
        files_needed = int(np.ceil(target / epf)) if epf > 0 else 0
        predicted_events = files_needed * epf
        achievable_events = m["estimated_total_matching_files"] * epf
        shortfall = target - achievable_events if achievable_events < target else None
        if shortfall is not None:
            predicted_events = achievable_events
        return {"predicted_events": predicted_events, "exact": False,
                "predicted_bytes": predicted_events * m["bytes_per_event"],
                "shortfall_events": shortfall}

    if se["max_files"] and se["max_files"] > 0:
        predicted_events = epf * se["max_files"]
        return {"predicted_events": predicted_events, "exact": False,
                "predicted_bytes": predicted_events * m["bytes_per_event"], "shortfall_events": None}

    predicted_events = epf * m["estimated_total_matching_files"]
    return {"predicted_events": predicted_events, "exact": False,
            "predicted_bytes": predicted_events * m["bytes_per_event"], "shortfall_events": None}


def build_report(config_path: str, out_base: Path, rows: list, split_cfg: dict, flush_every_events: int) -> str:
    """Build the human-readable size-estimate report: full/train/eval totals,
    then a breakdown of every output directory a real conversion would
    create (matching convert_collide2v_regionized's own layout -- see
    docs/challenge_dataconfig.md's "Output layout" section) -- events,
    predicted fragment-file count (via the same flush_every_events chunking
    production uses), approx size/file, and total size -- plus each
    sample's measurement basis and any shortfall warnings."""
    train_frac = split_cfg.get("train_frac", 0.9) if split_cfg else None
    full_events = sum(p["predicted_events"] for _, _, p in rows)
    full_bytes = sum(p["predicted_bytes"] for _, _, p in rows)

    lines = [
        f"Dataset size estimate for {out_base}",
        f"(generated {datetime.datetime.now().isoformat(timespec='seconds')}, from {config_path})",
        "",
        f"full dataset total # events: {full_events:,.0f}",
    ]
    if split_cfg:
        train_events = full_events * train_frac
        eval_events = full_events - train_events
        train_bytes = full_bytes * train_frac
        eval_bytes = full_bytes - train_bytes
        lines += [
            f"train total # events: {train_events:,.0f}",
            f"eval total # events: {eval_events:,.0f}",
            "",
            f"full dataset size: {_human_bytes(full_bytes)}",
            f"train size: {_human_bytes(train_bytes)}",
            f"eval size: {_human_bytes(eval_bytes)}",
        ]
    else:
        lines += ["", f"full dataset size: {_human_bytes(full_bytes)}"]

    lines += ["", "-" * 78]
    for se, m, p in rows:
        sample = se["sample"]
        if split_cfg:
            splits = [("train", p["predicted_events"] * train_frac, p["predicted_bytes"] * train_frac),
                      ("eval", p["predicted_events"] * (1 - train_frac), p["predicted_bytes"] * (1 - train_frac))]
        else:
            splits = [(None, p["predicted_events"], p["predicted_bytes"])]

        for split_name, events, size_bytes in splits:
            dir_path = (out_base / split_name / sample) if split_name else (out_base / sample)
            n_fragments = int(np.ceil(events / flush_every_events)) if events > 0 else 0
            size_per_file = (size_bytes / n_fragments) if n_fragments else 0
            lines += [
                str(dir_path),
                f"  total # events: {events:,.0f}",
                f"  # parquet files: {n_fragments}",
                f"  approx size per file: {_human_bytes(size_per_file)}",
                f"  total size: {_human_bytes(size_bytes)}",
                "",
            ]
    lines.append("-" * 78)

    lines += ["", "Per-sample measurement basis / warnings:"]
    for se, m, p in rows:
        lines.append(f"  {_sample_measurement_basis(se['sample'], m)}")
        if p["shortfall_events"] is not None:
            lines.append(f"    ** target_events NOT ACHIEVABLE -- directory only supports "
                         f"~{p['predicted_events']:,.0f}, short by ~{p['shortfall_events']:,.0f} **")
        if m["events_per_matched_file"] is None:
            lines.append("    ** every sampled event was dropped -- can't measure a rate, size is 0 **")

    lines += [
        "",
        "Note: sizes are extrapolated from a small real-data sample per process -- treat as an estimate,",
        "not an exact figure. Re-run with --sample-files/--version-scan-limit higher for tighter estimates.",
    ]
    return "\n".join(lines)


def estimate_output_size(cfg: DataConfig, config_path: str, sample_files_target: int = 5,
                          version_scan_limit: int = 500, report_path=None) -> str:
    """Runs the per-sample measurement/prediction, writes the report to
    `report_path` (default: `size_estimate.txt` next to the config file) and
    prints it to stdout. Returns the report path actually written to."""
    sample_dir = cfg.get_sample_dir().rstrip("/")
    redir = cfg.dp("redir", "")
    dataset_version = cfg.dp("dataset_version", "collide2v_v1.0")
    max_events_per_file = cfg.dp("max_events_per_file", -1)
    default_max_files = cfg.dp("max_files_per_sample", -1)
    out_base = Path(cfg.dp("out_path", "./")).expanduser()
    flush_every_events = cfg.dp("flush_every_events", 2_000_000)
    samples = cfg.dp("samples", None)
    if not samples:
        raise ValueError("data_processing.samples (list of EOS subdirectory names) is required.")

    cs_cfg = _resolve_candidate_selection_cfg(cfg)
    coll_cfg = _resolve_collections_cfg(cfg)
    split_cfg = cfg.dp("split", None)

    rows = []
    for entry in samples:
        se = _resolve_sample_entry(entry, default_max_files)
        sample = se["sample"]
        logger.info(f"[{sample}] sampling up to {sample_files_target} file(s) for measurement...")
        m = measure_sample(sample_dir, redir, dataset_version, max_events_per_file, coll_cfg, cs_cfg, se,
                            sample_files_target, version_scan_limit)
        p = predict_sample(se, m)
        rows.append((se, m, p))

    report = build_report(config_path, out_base, rows, split_cfg, flush_every_events)
    print()
    print(report)

    if report_path is None:
        report_path = Path(config_path).resolve().parent / "size_estimate.txt"
    with open(report_path, "w") as fh:
        fh.write(report + "\n")
    logger.info(f"Wrote size estimate report to {report_path}")
    return str(report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a data_processing YAML config.")
    parser.add_argument("--sample-files", type=int, default=5,
                         help="Real files to actually READ (full column data) per sample for the byte-size "
                              "measurement (default: 5). Ignored (all files used) for a `files:`-pinned sample.")
    parser.add_argument("--version-scan-limit", type=int, default=500,
                         help="Files to check dataset_version on (cheap metadata-only reads, no column data) "
                              "per sample, to estimate the directory's overall version-match rate for "
                              "extrapolation (default: 500). Set higher for a tighter estimate on samples with "
                              "many files; ignored (every file checked) for a `files:`-pinned sample.")
    parser.add_argument("--report-path", default=None,
                         help="Where to write the size-estimate report (default: size_estimate.txt next to "
                              "--config, e.g. C1_HH4b/size_estimate.txt).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cfg = DataConfig(args.config)
    estimate_output_size(cfg, args.config, sample_files_target=args.sample_files,
                          version_scan_limit=args.version_scan_limit, report_path=args.report_path)
