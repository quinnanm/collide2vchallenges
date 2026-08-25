"""Tests for estimate_output_size.py against local synthetic parquet (no
network) -- covers the three sample-scope branches (target_events, max_files,
pinned files:) plus the "target not achievable" shortfall detection.
Run with:
    python -m pytest tests/test_estimate_output_size.py -v
"""
import os

import awkward as ak
import numpy as np
import pyarrow.parquet as pq

from config import DataConfig
from estimate_output_size import estimate_output_size, measure_sample, predict_sample
from converters import _resolve_candidate_selection_cfg, _resolve_collections_cfg, _resolve_sample_entry


def _write_pq(arr, path, version=b"collide2v_v1.0"):
    table = ak.to_arrow_table(arr)
    meta = dict(table.schema.metadata or {})
    meta[b"dataset_version"] = version
    pq.write_table(table.replace_schema_metadata(meta), path)


def _write_jetak4_sample(sample_dir, n_files, events_per_file, seed, bad_version_files=0):
    rng = np.random.default_rng(seed)
    os.makedirs(sample_dir, exist_ok=True)
    for i in range(n_files):
        n_jet = rng.integers(1, 6, size=events_per_file)
        jpt = [rng.uniform(0.5, 60, n).tolist() for n in n_jet]
        jeta = [rng.uniform(-2, 2, n).tolist() for n in n_jet]
        jphi = [rng.uniform(-3, 3, n).tolist() for n in n_jet]

        def const(v, dt=float):
            return [[dt(v)] * n for n in n_jet]

        arr = ak.Array({
            "L1T_JetAK4_PT": jpt, "L1T_JetAK4_Eta": jeta, "L1T_JetAK4_Phi": jphi,
            "L1T_JetAK4_Mass": const(5.0), "L1T_JetAK4_Charge": const(0, int),
            "L1T_JetAK4_Flavor": const(1, int), "L1T_JetAK4_BTag": const(0, int),
            "L1T_JetAK4_BTagPhys": const(0, int), "L1T_JetAK4_NCharged": const(1, int),
            "L1T_JetAK4_NNeutrals": const(1, int),
            "L1T_JetAK4_Constituents": [[[0] for _ in range(n)] for n in n_jet],
        })
        version = b"OLD_VERSION" if i < bad_version_files else b"collide2v_v1.0"
        _write_pq(arr, os.path.join(sample_dir, f"file_{i}.parquet"), version=version)


def _base_config(sample_dir, samples):
    return {
        "ds_name": "estimator_test",
        "data_processing": {
            "sample_dir": str(sample_dir),
            "redir": "",
            "out_path": "/unused",  # DataConfig requires out_path to be present-ish but estimator never writes there
            "dataset_version": "collide2v_v1.0",
            "collections": {"L1T_JetAK4": 5},
            "samples": samples,
        },
    }


def _write_config(tmp_path, config):
    import yaml
    path = tmp_path / "dataconfig.yml"
    path.write_text(yaml.dump(config))
    return DataConfig(str(path))


def test_measure_and_predict_target_events(tmp_path):
    sample_dir = tmp_path / "eos"
    _write_jetak4_sample(str(sample_dir / "TestTargetEvents"), n_files=10, events_per_file=30, seed=1,
                         bad_version_files=2)
    cfg = _write_config(tmp_path, _base_config(sample_dir, [
        {"name": "TestTargetEvents", "label": 0, "target_events": 150},
    ]))
    cs_cfg = _resolve_candidate_selection_cfg(cfg)
    coll_cfg = _resolve_collections_cfg(cfg)
    se = _resolve_sample_entry(cfg.dp("samples")[0], cfg.dp("max_files_per_sample", -1))

    m = measure_sample(cfg.get_sample_dir(), "", "collide2v_v1.0", -1, coll_cfg, cs_cfg, se,
                        sample_files_target=3, version_scan_limit=20)
    assert m["n_scanned"] == 10  # only 10 files exist, well under the scan limit -- exact
    assert m["n_matched"] == 8   # 2 bad-version files correctly excluded
    assert m["exact_match_count"] is True
    assert m["n_data_files_read"] == 3
    assert m["events_per_matched_file"] == 30
    assert m["bytes_per_event"] > 0

    p = predict_sample(se, m)
    assert p["predicted_events"] == 150  # 150 / 30-per-file = exactly 5 files, no overshoot in this fixture
    assert p["predicted_bytes"] > 0
    assert p["shortfall_events"] is None


def test_measure_and_predict_max_files(tmp_path):
    sample_dir = tmp_path / "eos"
    _write_jetak4_sample(str(sample_dir / "TestMaxFiles"), n_files=6, events_per_file=25, seed=2)
    cfg = _write_config(tmp_path, _base_config(sample_dir, [
        {"name": "TestMaxFiles", "label": 1, "max_files": 3},
    ]))
    cs_cfg = _resolve_candidate_selection_cfg(cfg)
    coll_cfg = _resolve_collections_cfg(cfg)
    se = _resolve_sample_entry(cfg.dp("samples")[0], cfg.dp("max_files_per_sample", -1))

    m = measure_sample(cfg.get_sample_dir(), "", "collide2v_v1.0", -1, coll_cfg, cs_cfg, se,
                        sample_files_target=3, version_scan_limit=20)
    p = predict_sample(se, m)
    assert p["predicted_events"] == 75  # 25/file * max_files=3
    assert p["shortfall_events"] is None


def test_measure_and_predict_pinned_files_is_exact(tmp_path):
    sample_dir = tmp_path / "eos"
    _write_jetak4_sample(str(sample_dir / "TestPinned"), n_files=3, events_per_file=20, seed=3)
    cfg = _write_config(tmp_path, _base_config(sample_dir, [
        {"name": "TestPinned", "label": 2, "files": ["file_0.parquet", "file_1.parquet"]},
    ]))
    cs_cfg = _resolve_candidate_selection_cfg(cfg)
    coll_cfg = _resolve_collections_cfg(cfg)
    se = _resolve_sample_entry(cfg.dp("samples")[0], cfg.dp("max_files_per_sample", -1))

    # sample_files_target is deliberately smaller than the pinned list --
    # pinned samples always measure every listed file regardless.
    m = measure_sample(cfg.get_sample_dir(), "", "collide2v_v1.0", -1, coll_cfg, cs_cfg, se,
                        sample_files_target=1, version_scan_limit=20)
    assert m["pinned"] is True
    assert m["n_data_files_read"] == 2
    assert m["n_kept_events"] == 40  # both pinned files, 20 events each, exact

    p = predict_sample(se, m)
    assert p["exact"] is True
    assert p["predicted_events"] == 40
    assert p["shortfall_events"] is None


def test_predict_flags_unreachable_target_events(tmp_path):
    sample_dir = tmp_path / "eos"
    _write_jetak4_sample(str(sample_dir / "TestUnreachable"), n_files=2, events_per_file=15, seed=4)
    cfg = _write_config(tmp_path, _base_config(sample_dir, [
        {"name": "TestUnreachable", "label": 3, "target_events": 100000},
    ]))
    cs_cfg = _resolve_candidate_selection_cfg(cfg)
    coll_cfg = _resolve_collections_cfg(cfg)
    se = _resolve_sample_entry(cfg.dp("samples")[0], cfg.dp("max_files_per_sample", -1))

    m = measure_sample(cfg.get_sample_dir(), "", "collide2v_v1.0", -1, coll_cfg, cs_cfg, se,
                        sample_files_target=3, version_scan_limit=20)
    p = predict_sample(se, m)
    assert p["shortfall_events"] is not None
    assert p["predicted_events"] == 30  # capped at what the 2 real files actually provide (15/file)


def test_estimate_output_size_runs_end_to_end_and_writes_report(tmp_path, capsys):
    sample_dir = tmp_path / "eos"
    _write_jetak4_sample(str(sample_dir / "SampleA"), n_files=4, events_per_file=10, seed=5)
    _write_jetak4_sample(str(sample_dir / "SampleB"), n_files=4, events_per_file=10, seed=6)
    cfg = _write_config(tmp_path, _base_config(sample_dir, [
        {"name": "SampleA", "label": 0, "max_files": 2},
        {"name": "SampleB", "label": 1, "max_files": 2},
    ]))
    report_path = estimate_output_size(cfg, str(cfg.path), sample_files_target=2, version_scan_limit=10)
    out = capsys.readouterr().out
    assert "SampleA" in out and "SampleB" in out
    assert "full dataset total # events" in out

    assert os.path.exists(report_path)
    # default report location: alongside the config file
    assert os.path.dirname(os.path.abspath(report_path)) == os.path.dirname(os.path.abspath(str(cfg.path)))
    with open(report_path) as fh:
        report_contents = fh.read()
    assert "SampleA" in report_contents
    assert str(cfg.dp("out_path")) in report_contents  # per-directory breakdown references the real out_path


def test_estimate_output_size_split_report_includes_train_eval_directories(tmp_path):
    sample_dir = tmp_path / "eos"
    _write_jetak4_sample(str(sample_dir / "SampleA"), n_files=4, events_per_file=10, seed=7)
    config = _base_config(sample_dir, [{"name": "SampleA", "label": 0, "max_files": 2}])
    config["data_processing"]["split"] = {"train_frac": 0.8, "eval_frac": 0.2, "seed": 1}
    cfg = _write_config(tmp_path, config)

    report_path = estimate_output_size(cfg, str(cfg.path), sample_files_target=2, version_scan_limit=10)
    with open(report_path) as fh:
        report_contents = fh.read()

    out_base = cfg.dp("out_path")
    assert f"{out_base}/train/SampleA" in report_contents.replace(os.sep, "/")
    assert f"{out_base}/eval/SampleA" in report_contents.replace(os.sep, "/")
    assert "train total # events" in report_contents
    assert "eval total # events" in report_contents