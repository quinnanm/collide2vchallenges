"""End-to-end synthetic-data test of the region-based candidate/collection
gathering functions in converters.py (no network -- hand-built awkward
arrays shaped like the real EOS schema). Copied and adapted from
aidascoutrepo's tests/test_region_converter.py, since converters.py's Stage
1 logic here is a byte-for-byte extraction of that same code. Run with:
    python -m pytest tests/test_converters.py -v
"""
import awkward as ak
import numpy as np
import pytest

from converters import (
    CANDIDATE_PT_FLOOR_GEV,
    COLLECTION_REGISTRY,
    _drop_events_with_empty_axis,
    _empty_placeholder,
    _evaluate_event_selection_cut,
    _placeholder_spec_for,
    convert_collide2v_regionized,
    diagnose_puppi_selection,
    gather_and_select_puppi_candidates,
    gather_collection,
    gather_other_l1t_collections,
)
from config import DataConfig
from regionize import select_top_n_per_event


def _puppipart_array():
    """2 events x 5 candidates, all in distinct/valid regions. Event 0: 3
    candidates above the pT floor (weighted), 2 below. Event 1: all 5 below
    the floor (tests the zero-survivors case)."""
    # columns: pt, eta, phi, pid, charge, puppiw
    ev0 = [
        (10.0, 0.1, 0.0, 211, 1, 1.0),    # pt_weighted=10.0, survives
        (5.0, 0.1, 1.5, 13, -1, 0.5),      # pt_weighted=2.5, survives
        (8.0, -2.9, -3.0, 22, 0, 0.01),    # pt_weighted=0.08, fails floor
        (3.0, 0.1, -1.0, 11, 1, 1.0),      # pt_weighted=3.0, survives
        (0.0, 0.0, 0.0, 0, 0, 0.0),        # padding-like row, fails floor
    ]
    ev1 = [(1.0, 0.1, p, 211, 1, 0.001) for p in [0.0, 0.5, 1.0, -0.5, -1.0]]  # all fail floor

    events = [ev0, ev1]
    pt = [[c[0] for c in ev] for ev in events]
    eta = [[c[1] for c in ev] for ev in events]
    phi = [[c[2] for c in ev] for ev in events]
    pid = [[c[3] for c in ev] for ev in events]
    charge = [[c[4] for c in ev] for ev in events]
    puppiw = [[c[5] for c in ev] for ev in events]
    zeros = [[0.0] * 5, [0.0] * 5]

    return ak.Array({
        "L1T_PUPPIPart_PT": pt, "L1T_PUPPIPart_Eta": eta, "L1T_PUPPIPart_Phi": phi,
        "L1T_PUPPIPart_PID": pid, "L1T_PUPPIPart_Charge": charge, "L1T_PUPPIPart_PuppiW": puppiw,
        "L1T_PUPPIPart_E": zeros, "L1T_PUPPIPart_Mass": zeros,
        "L1T_PUPPIPart_D0": zeros, "L1T_PUPPIPart_DZ": zeros,
        "L1T_PUPPIPart_ErrorD0": [[1.0] * 5, [1.0] * 5], "L1T_PUPPIPart_ErrorDZ": zeros,
        "L1T_PUPPIPart_IsPU": [[0] * 5, [0] * 5], "L1T_PUPPIPart_IsRecoPU": [[0] * 5, [0] * 5],
        "L1T_PUPPIPart_fUniqueID": [[0] * 5, [0] * 5],
    })


def _ragged_puppipart_array():
    """3 events with a genuinely different number of raw candidates each (5,
    2, 4) -- reproduces the production crash where ak.to_numpy on the raw 2D
    field assumed a regular (same-length-per-event) array and threw
    "cannot convert to RegularArray because subarray lengths are not
    regular" the first time a real file had non-uniform per-event counts."""
    pt = [[10.0, 5.0, 8.0, 3.0, 2.0], [6.0, 4.0], [9.0, 1.0, 7.0, 3.0]]
    eta = [[0.1, 0.1, -0.2, 0.1, 0.3], [0.1, 0.2], [0.1, 0.1, 0.1, 0.1]]
    phi = [[0.0, 1.5, -3.0, -1.0, 2.0], [0.0, 1.0], [0.0, 1.0, 2.0, -1.0]]
    pid = [[211, 13, 22, 11, 0], [211, 211], [211, 211, 211, 211]]
    charge = [[1, -1, 0, 1, 0], [1, -1], [1, 1, 1, 1]]
    puppiw = [[1.0, 0.5, 1.0, 1.0, 1.0], [1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]

    def const(counts, value):
        return [[value] * n for n in counts]

    counts = [5, 2, 4]
    return ak.Array({
        "L1T_PUPPIPart_PT": pt, "L1T_PUPPIPart_Eta": eta, "L1T_PUPPIPart_Phi": phi,
        "L1T_PUPPIPart_PID": pid, "L1T_PUPPIPart_Charge": charge, "L1T_PUPPIPart_PuppiW": puppiw,
        "L1T_PUPPIPart_E": const(counts, 0.0), "L1T_PUPPIPart_Mass": const(counts, 0.0),
        "L1T_PUPPIPart_D0": const(counts, 0.0), "L1T_PUPPIPart_DZ": const(counts, 0.0),
        "L1T_PUPPIPart_ErrorD0": const(counts, 1.0), "L1T_PUPPIPart_ErrorDZ": const(counts, 0.0),
        "L1T_PUPPIPart_IsPU": const(counts, 0), "L1T_PUPPIPart_IsRecoPU": const(counts, 0),
        "L1T_PUPPIPart_fUniqueID": const(counts, 0),
    })


def test_gather_and_select_handles_ragged_per_event_candidate_counts():
    arr = _ragged_puppipart_array()
    out = gather_and_select_puppi_candidates(arr)
    assert len(out) == 3
    # event 1 (only 2 raw candidates, both above floor) keeps both
    assert len(out["pt"][1]) == 2


def test_diagnose_puppi_selection_handles_ragged_per_event_candidate_counts():
    arr = _ragged_puppipart_array()
    diag = diagnose_puppi_selection(arr)
    assert diag["n_events"] == 3
    total = diag["out_of_acceptance"] + diag["below_floor"] + diag["rank_truncated"] + diag["kept"]
    assert total == 5 + 2 + 4


def test_candidate_pt_floor_and_ordering():
    arr = _puppipart_array()
    out = gather_and_select_puppi_candidates(arr)

    assert len(out) == 2  # 2 events
    assert ak.sum(ak.num(out["pt"], axis=1)) == 3  # 3 survivors total (event0 only)
    n_per_event = ak.to_numpy(ak.num(out["pt"], axis=1))
    assert n_per_event[0] == 3
    assert n_per_event[1] == 0  # every event-1 candidate fails the floor

    ev0_pt = ak.to_numpy(out["pt"][0])
    assert list(ev0_pt) == sorted(ev0_pt, reverse=True)  # sorted by raw pT descending
    np.testing.assert_allclose(sorted(ev0_pt, reverse=True), [10.0, 5.0, 3.0], atol=1e-2)


def test_candidate_weighted_pt_matches_pt_times_weight():
    arr = _puppipart_array()
    out = gather_and_select_puppi_candidates(arr)
    pt = ak.to_numpy(out["pt"][0]).astype(np.float32)
    w = ak.to_numpy(out["puppi_weight"][0]).astype(np.float32)
    pt_w = ak.to_numpy(out["pt_weighted"][0]).astype(np.float32)
    np.testing.assert_allclose(pt_w, pt * w, atol=1e-1)  # float16 roundtrip tolerance


def test_candidate_pdgid_bucketing():
    arr = _puppipart_array()
    out = gather_and_select_puppi_candidates(arr)
    pdgid = set(ak.to_numpy(ak.flatten(out["pdgId"])).tolist())
    # Unsigned -- charge lives in its own separate `charge` column, not
    # folded into pdgId's sign (deliberately different from
    # gather_pfcands_collide's convention). ev0 survivors: pid 211, 13, 11.
    assert pdgid == {211, 13, 11}
    assert all(p >= 0 for p in pdgid)


def test_candidate_dtypes():
    arr = _puppipart_array()
    out = gather_and_select_puppi_candidates(arr)
    assert out["pt"].layout.content.dtype == np.float16
    assert out["pdgId"].layout.content.dtype == np.int16
    assert out["charge"].layout.content.dtype == np.int8
    assert out["is_reco_pu"].layout.content.dtype == np.int8  # verified 0/1-only
    assert out["funique_id"].layout.content.dtype == np.int32  # verified: real values reach ~140k, exceeds int16


def test_candidate_pt_floor_is_exclusive_boundary_sane():
    # Sanity check the floor constant itself hasn't drifted from the design
    # (flat 1 GeV everywhere, no region-dependent thresholds).
    assert CANDIDATE_PT_FLOOR_GEV == 1.0


def _other_collections_array():
    return ak.Array({
        "L1T_Electron_Charge": [[1], []], "L1T_Electron_D0": [[0.1], []],
        "L1T_Electron_DZ": [[0.1], []], "L1T_Electron_Eta": [[0.5], []],
        "L1T_Electron_Phi": [[0.5], []], "L1T_Electron_PT": [[20.0], []],
        "L1T_Electron_ErrorD0": [[0.01], []], "L1T_Electron_ErrorDZ": [[0.01], []],
        "L1T_Electron_EhadOverEem": [[0.1], []], "L1T_Electron_IsolationVar": [[0.1], []],
        "L1T_Electron_IsolationVarRhoCorr": [[0.1], []],
        "L1T_MuonTight_Charge": [[], [-1]], "L1T_MuonTight_D0": [[], [0.2]],
        "L1T_MuonTight_DZ": [[], [0.2]], "L1T_MuonTight_Eta": [[], [-0.3]],
        "L1T_MuonTight_Phi": [[], [1.0]], "L1T_MuonTight_PT": [[], [15.0]],
        "L1T_MuonTight_ErrorD0": [[], [0.01]], "L1T_MuonTight_ErrorDZ": [[], [0.01]],
        "L1T_MuonTight_IsolationVar": [[], [0.1]], "L1T_MuonTight_IsolationVarRhoCorr": [[], [0.1]],
        "L1T_PhotonTight_Eta": [[], []], "L1T_PhotonTight_Phi": [[], []],
        "L1T_PhotonTight_PT": [[], []], "L1T_PhotonTight_EhadOverEem": [[], []],
        "L1T_PhotonTight_IsolationVar": [[], []], "L1T_PhotonTight_IsolationVarRhoCorr": [[], []],
        "L1T_JetAK4_Eta": [[0.1, 0.2], []], "L1T_JetAK4_Phi": [[0.1, 0.2], []],
        "L1T_JetAK4_PT": [[50.0, 30.0], []], "L1T_JetAK4_Mass": [[10.0, 5.0], []],
        "L1T_JetAK4_Charge": [[0, 0], []], "L1T_JetAK4_Flavor": [[5, 21], []],
        "L1T_JetAK4_BTag": [[1, 0], []], "L1T_JetAK4_BTagPhys": [[1, 0], []],
        "L1T_JetAK4_NCharged": [[12, 8], []], "L1T_JetAK4_NNeutrals": [[6, 4], []],
        "L1T_JetAK4_Constituents": [[[1, 2, 3], [4, 5]], []],  # no ConstituentsIdx -- real schema lacks it for JetAK4
        "L1T_JetAK8_Eta": [[], []], "L1T_JetAK8_Phi": [[], []], "L1T_JetAK8_PT": [[], []],
        "L1T_JetAK8_Mass": [[], []], "L1T_JetAK8_Charge": [[], []], "L1T_JetAK8_Flavor": [[], []],
        "L1T_JetAK8_BTag": [[], []], "L1T_JetAK8_BTagPhys": [[], []],
        "L1T_JetAK8_NCharged": [[], []], "L1T_JetAK8_NNeutrals": [[], []],
        "L1T_JetAK8_Constituents": [[], []], "L1T_JetAK8_ConstituentsIdx": [[], []],
        "L1T_JetPuppiAK4_Eta": [[], []], "L1T_JetPuppiAK4_Phi": [[], []], "L1T_JetPuppiAK4_PT": [[], []],
        "L1T_JetPuppiAK4_Mass": [[], []], "L1T_JetPuppiAK4_Charge": [[], []], "L1T_JetPuppiAK4_Flavor": [[], []],
        "L1T_JetPuppiAK4_BTag": [[], []], "L1T_JetPuppiAK4_BTagPhys": [[], []],
        "L1T_JetPuppiAK4_NCharged": [[], []], "L1T_JetPuppiAK4_NNeutrals": [[], []],
        "L1T_JetPuppiAK4_Constituents": [[], []], "L1T_JetPuppiAK4_ConstituentsIdx": [[], []],
        "L1T_JetPuppiAK8_Eta": [[], []], "L1T_JetPuppiAK8_Phi": [[], []], "L1T_JetPuppiAK8_PT": [[], []],
        "L1T_JetPuppiAK8_Mass": [[], []], "L1T_JetPuppiAK8_Charge": [[], []], "L1T_JetPuppiAK8_Flavor": [[], []],
        "L1T_JetPuppiAK8_BTag": [[], []], "L1T_JetPuppiAK8_BTagPhys": [[], []],
        "L1T_JetPuppiAK8_NCharged": [[], []], "L1T_JetPuppiAK8_NNeutrals": [[], []],
        "L1T_JetPuppiAK8_Constituents": [[], []], "L1T_JetPuppiAK8_ConstituentsIdx": [[], []],
        "L1T_MET_MET": [[100.0], [50.0]], "L1T_MET_Eta": [[0.0], [0.0]], "L1T_MET_Phi": [[1.0], [-1.0]],
        "L1T_PUPPIMET_MET": [[90.0], [45.0]], "L1T_PUPPIMET_Eta": [[0.0], [0.0]], "L1T_PUPPIMET_Phi": [[1.0], [-1.0]],
        "L1T_Rho_Rho": [[1.0] * 5, [2.0] * 5],
        "L1T_ScalarHT_HT": [[200.0], [150.0]],
    })


def test_other_collections_passthrough_and_precision():
    arr = _other_collections_array()
    out = gather_other_l1t_collections(arr)

    assert set(out.keys()) == {
        "L1T_Electron", "L1T_MuonTight", "L1T_PhotonTight", "L1T_JetAK4", "L1T_JetAK8",
        "L1T_JetPuppiAK4", "L1T_JetPuppiAK8", "L1T_MET", "L1T_PUPPIMET", "L1T_Rho", "L1T_ScalarHT",
    }
    np.testing.assert_allclose(ak.to_numpy(out["L1T_Electron"]["PT"][0]), [20.0], atol=1e-1)
    assert out["L1T_JetAK4"]["PT"].layout.content.dtype == np.float16
    assert out["L1T_JetAK4"]["Charge"].layout.content.dtype == np.int8
    assert out["L1T_JetAK4"]["Flavor"].layout.content.dtype == np.int8  # verified max 21
    assert out["L1T_JetAK4"]["NCharged"].layout.content.dtype == np.int8  # verified max 54
    assert out["L1T_JetAK4"]["NNeutrals"].layout.content.dtype == np.int16  # verified max 427 (>int8)
    # Constituents (jagged reference list): left at native dtype, not downcast
    constituents_flat = ak.flatten(ak.flatten(out["L1T_JetAK4"]["Constituents"]))
    assert list(ak.to_numpy(constituents_flat)) == [1, 2, 3, 4, 5]


def test_diagnose_puppi_selection_reasons_sum_to_total():
    arr = _puppipart_array()
    diag = diagnose_puppi_selection(arr)
    total = diag["n_events"] * 5  # 5 candidates/event in the fixture
    assert diag["n_events"] == 2
    assert diag["out_of_acceptance"] + diag["below_floor"] + diag["rank_truncated"] + diag["kept"] == total


def test_diagnose_puppi_selection_matches_known_fixture_breakdown():
    # ev0: 3 survive the floor, 2 don't (cand2 pt_weighted=0.08, cand4=0);
    # ev1: all 5 fail the floor (pt_weighted=0.001 each). Nothing is out of
    # acceptance or rank-truncated in this small fixture (well under the
    # 18/region cap).
    arr = _puppipart_array()
    diag = diagnose_puppi_selection(arr)
    assert diag["out_of_acceptance"] == 0
    assert diag["rank_truncated"] == 0
    assert diag["below_floor"] == 7  # 2 (ev0) + 5 (ev1)
    assert diag["kept"] == 3  # ev0 only
    assert diag["region_truncated_counts"].sum() == 0


def test_placeholder_spec_for_constituents_fields():
    # Depth-2 (doubly-nested), specific dtypes -- the two fields that skip
    # _downcast()'s normalization, so getting this right actually matters.
    assert _placeholder_spec_for("L1T_JetAK8_Constituents") == (2, np.uint32)
    assert _placeholder_spec_for("L1T_JetAK8_ConstituentsIdx") == (2, np.int16)
    assert _placeholder_spec_for("L1T_JetPuppiAK4_Constituents") == (2, np.uint32)
    # Everything else: depth-1, dtype doesn't matter (normalized by _downcast later).
    assert _placeholder_spec_for("L1T_JetAK4_PT") == (1, np.float64)
    assert _placeholder_spec_for("L1T_Electron_Charge") == (1, np.float64)


def test_empty_placeholder_shapes_and_emptiness():
    depth1 = _empty_placeholder(3, 1, np.float32)
    assert len(depth1) == 3
    assert ak.sum(ak.num(depth1, axis=1)) == 0  # every event's list is empty

    depth2 = _empty_placeholder(3, 2, np.uint32)
    assert len(depth2) == 3
    assert ak.sum(ak.num(depth2, axis=1)) == 0  # zero "jets" per event
    assert depth2.layout.minmax_depth == (3, 3)  # event -> jet -> constituent-index = 3 levels of Content nesting


def test_empty_placeholder_concatenates_with_real_doubly_nested_data():
    # This is the exact scenario that broke in production: file A has real
    # (non-empty) Constituents for some jets, file B is missing the column
    # entirely and gets the placeholder -- concatenating the two used to
    # produce a type mismatch (dense_union) that pyarrow can't write to
    # parquet. Confirms the fix: same depth/dtype, so concatenation and a
    # parquet round-trip both work.
    real = ak.Array({"Constituents": [[[1, 2, 3], [4, 5]], [[6]]]})["Constituents"]
    real = ak.values_astype(real, np.uint32)
    placeholder = _empty_placeholder(2, 2, np.uint32)

    combined = ak.concatenate([real, placeholder], axis=0)
    assert len(combined) == 4
    assert list(ak.to_numpy(ak.flatten(ak.flatten(combined)))) == [1, 2, 3, 4, 5, 6]

    import tempfile
    import os as _os
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _os.path.join(tmpdir, "test.parquet")
        ak.to_parquet(ak.Array({"Constituents": combined}), path)  # must not raise ArrowNotImplementedError


# --------------------------------------------------------------------------
# New config-driven-system tests: select_top_n_per_event, generic
# variable_object collection truncation, a generalized empty-axis filter, and
# a small synthetic end-to-end conversion covering label override,
# target_events overshoot, split, and flush_every_events fragmenting.
# --------------------------------------------------------------------------

def test_select_top_n_per_event_basic_ranking_and_cap():
    # 2 events: event 0 has 4 candidates (values 10,30,20,5), event 1 has 1
    # (value 1). Cap=2 -> event 0 keeps the two highest (30, 20); event 1
    # keeps its only candidate.
    event_idx = np.array([0, 0, 0, 0, 1])
    values = np.array([10.0, 30.0, 20.0, 5.0, 1.0])
    above_floor = np.ones(5, dtype=bool)
    survive = select_top_n_per_event(event_idx, values, above_floor, n_events=2, cap=2)
    assert list(values[survive]) == [30.0, 20.0, 1.0]


def test_select_top_n_per_event_respects_above_floor_mask():
    event_idx = np.array([0, 0, 0])
    values = np.array([10.0, 5.0, 1.0])
    above_floor = np.array([True, False, True])  # the pt=5.0 candidate fails an external floor
    survive = select_top_n_per_event(event_idx, values, above_floor, n_events=1, cap=10)
    assert list(survive) == [True, False, True]


def test_select_top_n_per_event_empty_input():
    survive = select_top_n_per_event(np.array([], dtype=np.int64), np.array([]), np.array([], dtype=bool),
                                      n_events=3, cap=5)
    assert len(survive) == 0


def _jetak4_array_for_truncation():
    """2 events, 4 jets each -- event 0's PTs are all distinct (easy to check
    top-2 survivors); event 1 has a tie to confirm ties don't crash the
    ranking (order between tied candidates is otherwise unspecified)."""
    pt = [[10.0, 40.0, 20.0, 30.0], [5.0, 5.0, 5.0, 5.0]]
    n = 4

    def const(v):
        return [[v] * n, [v] * n]

    return ak.Array({
        "L1T_JetAK4_PT": pt, "L1T_JetAK4_Eta": const(0.1), "L1T_JetAK4_Phi": const(0.1),
        "L1T_JetAK4_Mass": const(5.0), "L1T_JetAK4_Charge": const(0), "L1T_JetAK4_Flavor": const(1),
        "L1T_JetAK4_BTag": const(0), "L1T_JetAK4_BTagPhys": const(0),
        "L1T_JetAK4_NCharged": const(1), "L1T_JetAK4_NNeutrals": const(1),
        "L1T_JetAK4_Constituents": [[[0], [0], [0], [0]], [[0], [0], [0], [0]]],
    })


def test_gather_collection_truncates_variable_object_to_top_n_by_rank_field():
    arr = _jetak4_array_for_truncation()
    out = gather_collection(arr, "L1T_JetAK4", COLLECTION_REGISTRY["L1T_JetAK4"], cap=2, n_events=2)
    n_per_event = ak.to_numpy(ak.num(out["PT"], axis=1))
    assert list(n_per_event) == [2, 2]
    # event 0: top-2 by PT descending = [40.0, 30.0]
    np.testing.assert_allclose(ak.to_numpy(out["PT"][0]), [40.0, 30.0], atol=1e-1)


def test_gather_collection_no_cap_preserves_original_order():
    arr = _jetak4_array_for_truncation()
    out = gather_collection(arr, "L1T_JetAK4", COLLECTION_REGISTRY["L1T_JetAK4"], cap=None, n_events=2)
    # No cap -> no reordering, exactly the source order/count.
    np.testing.assert_allclose(ak.to_numpy(out["PT"][0]), [10.0, 40.0, 20.0, 30.0], atol=1e-1)


def test_drop_events_with_empty_axis_generalizes_to_requested_collections():
    # Only L1T_JetAK4 requested (no candidates, no other object collections) --
    # an event with zero jets should be dropped even though the original
    # hardcoded 7-collection check would never have looked at JetAK4 alone.
    others = {
        "L1T_JetAK4": ak.Array({"Eta": [[0.1], [], [0.2, 0.3]]}),  # event 1 has zero jets
    }
    keep_mask, n_dropped = _drop_events_with_empty_axis(None, others, n_events=3, want_candidates=False)
    assert list(keep_mask) == [True, False, True]
    assert n_dropped == 1


def test_drop_events_with_empty_axis_no_variable_object_collections_drops_nothing():
    # Only a fixed_scalar collection requested (e.g. just L1T_MET) -- no
    # object axis to check at all, so nothing is dropped on that basis.
    keep_mask, n_dropped = _drop_events_with_empty_axis(None, {}, n_events=3, want_candidates=False)
    assert list(keep_mask) == [True, True, True]
    assert n_dropped == 0


def _write_parquet_with_dataset_version(arr: ak.Array, path: str, version: bytes = b"collide2v_v1.0") -> None:
    import pyarrow.parquet as pq
    table = ak.to_arrow_table(arr)
    meta = dict(table.schema.metadata or {})
    meta[b"dataset_version"] = version
    pq.write_table(table.replace_schema_metadata(meta), path)


def _write_synthetic_jetak4_sample(sample_dir: str, n_files: int, events_per_file: int, seed: int = 0) -> None:
    import os as _os
    rng = np.random.default_rng(seed)
    _os.makedirs(sample_dir, exist_ok=True)
    for i in range(n_files):
        n_jets = rng.integers(1, 4, size=events_per_file)  # 1-3 jets/event, never 0 -- nothing should be
        # dropped by the empty-axis filter in this fixture.
        pt = [rng.uniform(10, 100, n).tolist() for n in n_jets]
        eta = [rng.uniform(-2, 2, n).tolist() for n in n_jets]
        phi = [rng.uniform(-3, 3, n).tolist() for n in n_jets]

        def const(value, dtype=float):
            return [[dtype(value)] * n for n in n_jets]

        arr = ak.Array({
            "L1T_JetAK4_PT": pt, "L1T_JetAK4_Eta": eta, "L1T_JetAK4_Phi": phi,
            "L1T_JetAK4_Mass": const(5.0), "L1T_JetAK4_Charge": const(0, int),
            "L1T_JetAK4_Flavor": const(1, int), "L1T_JetAK4_BTag": const(0, int),
            "L1T_JetAK4_BTagPhys": const(0, int), "L1T_JetAK4_NCharged": const(1, int),
            "L1T_JetAK4_NNeutrals": const(1, int),
            "L1T_JetAK4_Constituents": [[[0] for _ in range(n)] for n in n_jets],
        })
        _write_parquet_with_dataset_version(arr, _os.path.join(sample_dir, f"file_{i}.parquet"))


def test_convert_collide2v_regionized_label_override_target_events_split_and_flush(tmp_path):
    import yaml

    sample_dir = tmp_path / "eos"
    out_dir = tmp_path / "out"
    _write_synthetic_jetak4_sample(str(sample_dir / "TestSample"), n_files=10, events_per_file=50, seed=42)

    config = {
        "ds_name": "test_ds",
        "data_processing": {
            "sample_dir": str(sample_dir),
            "redir": "",
            "out_path": str(out_dir),
            "dataset_version": "collide2v_v1.0",
            "collections": {"L1T_JetAK4": 2},
            "samples": [{"name": "TestSample", "label": 7, "target_events": 250}],
            "split": {"train_frac": 0.8, "eval_frac": 0.2, "seed": 123},
            "flush_every_events": 100,
        },
    }
    config_path = tmp_path / "dataconfig.yml"
    config_path.write_text(yaml.dump(config))

    cfg = DataConfig(str(config_path))
    convert_collide2v_regionized(cfg, overwrite=False)

    # New output layout: train/<sample>/ and eval/<sample>/ under out_path,
    # NOT <sample>/train and <sample>/eval.
    train_frags = sorted((out_dir / "train" / "TestSample").glob("*.parquet"))
    eval_frags = sorted((out_dir / "eval" / "TestSample").glob("*.parquet"))
    assert len(train_frags) >= 2  # flush_every_events=100 with ~200 train events -> multiple fragments
    assert len(eval_frags) >= 1

    train = ak.concatenate([ak.from_parquet(str(p)) for p in train_frags], axis=0) if len(train_frags) > 1 \
        else ak.from_parquet(str(train_frags[0]))
    ev = ak.concatenate([ak.from_parquet(str(p)) for p in eval_frags], axis=0) if len(eval_frags) > 1 \
        else ak.from_parquet(str(eval_frags[0]))

    total = len(train) + len(ev)
    # target_events=250, whole-file granularity (50 events/file) -- lands
    # exactly on a file boundary here, so no overshoot in this fixture.
    assert total == 250
    train_ratio = len(train) / total
    assert 0.6 < train_ratio < 0.95  # roughly 0.8, not exact given per-event randomness

    assert set(ak.to_numpy(train["label"]).tolist()) == {7}
    assert set(ak.to_numpy(ev["label"]).tolist()) == {7}

    # cap=2 enforced on every event, in both splits
    assert ak.max(ak.num(train["L1T_JetAK4"]["PT"], axis=1)) <= 2
    assert ak.max(ak.num(ev["L1T_JetAK4"]["PT"], axis=1)) <= 2

    # source_files.txt is shared (top-level, not nested under either split)
    with open(out_dir / "TestSample_source_files.txt") as fh:
        source_files = fh.read().splitlines()
    assert len(source_files) == 5  # 250 events / 50 events-per-file


def test_convert_collide2v_regionized_overwrite_semantics(tmp_path):
    import yaml

    sample_dir = tmp_path / "eos"
    out_dir = tmp_path / "out"
    _write_synthetic_jetak4_sample(str(sample_dir / "TestSample"), n_files=2, events_per_file=20, seed=7)

    config = {
        "ds_name": "test_ds",
        "data_processing": {
            "sample_dir": str(sample_dir),
            "redir": "",
            "out_path": str(out_dir),
            "dataset_version": "collide2v_v1.0",
            "collections": {"L1T_JetAK4": None},
            "samples": [{"name": "TestSample", "label": 0}],
        },
    }
    config_path = tmp_path / "dataconfig.yml"
    config_path.write_text(yaml.dump(config))
    cfg = DataConfig(str(config_path))

    convert_collide2v_regionized(cfg, overwrite=False)
    with pytest.raises(FileExistsError):
        convert_collide2v_regionized(cfg, overwrite=False)
    convert_collide2v_regionized(cfg, overwrite=True)  # must not raise


# --------------------------------------------------------------------------
# event_selection (per-event, reduce modes) and object_selection/drop_fields/
# realistic_pid (per-collection) tests.
# --------------------------------------------------------------------------

def _scalarht_array(values):
    return ak.Array({"L1T_ScalarHT_HT": [[v] for v in values]})


def test_event_selection_scalar_reduce():
    arr = _scalarht_array([100.0, 300.0, 250.0])
    cut = {"collection": "L1T_ScalarHT", "field": "HT", "op": ">", "value": 240}
    result = _evaluate_event_selection_cut(arr, cut, n_events=3)
    assert list(result) == [False, True, True]


def test_event_selection_scalar_reduce_rejects_multi_valued_field():
    # L1T_Rho has 5 entries/event -- 'scalar' reduce should refuse it rather
    # than silently picking one value.
    arr = ak.Array({"L1T_Rho_Rho": [[1.0] * 5, [2.0] * 5]})
    cut = {"collection": "L1T_Rho", "field": "Rho", "op": ">", "value": 1.0}
    with pytest.raises(ValueError, match="single-valued"):
        _evaluate_event_selection_cut(arr, cut, n_events=2)


def _jet_pt_array():
    # 3 events: [], [10, 50], [20, 20, 5]
    return ak.Array({"L1T_JetAK4_PT": [[], [10.0, 50.0], [20.0, 20.0, 5.0]]})


def test_event_selection_count_reduce():
    arr = _jet_pt_array()
    cut = {"collection": "L1T_JetAK4", "field": "PT", "reduce": "count", "op": ">=", "value": 2}
    assert list(_evaluate_event_selection_cut(arr, cut, n_events=3)) == [False, True, True]


def test_event_selection_any_reduce():
    arr = _jet_pt_array()
    cut = {"collection": "L1T_JetAK4", "field": "PT", "reduce": "any", "op": ">", "value": 30}
    assert list(_evaluate_event_selection_cut(arr, cut, n_events=3)) == [False, True, False]


def test_event_selection_all_reduce_vacuously_true_for_empty_event():
    arr = _jet_pt_array()
    cut = {"collection": "L1T_JetAK4", "field": "PT", "reduce": "all", "op": ">", "value": 10}
    # event 0 has zero jets -> vacuously True; event 1 (10, 50) both >10? no, 10 is not >10 -> False;
    # event 2 (20, 20, 5) has a 5.0, not >10 -> False
    assert list(_evaluate_event_selection_cut(arr, cut, n_events=3)) == [True, False, False]


def test_event_selection_max_min_reduce():
    arr = _jet_pt_array()
    max_cut = {"collection": "L1T_JetAK4", "field": "PT", "reduce": "max", "op": ">", "value": 40}
    min_cut = {"collection": "L1T_JetAK4", "field": "PT", "reduce": "min", "op": ">", "value": 15}
    assert list(_evaluate_event_selection_cut(arr, max_cut, n_events=3)) == [False, True, False]
    assert list(_evaluate_event_selection_cut(arr, min_cut, n_events=3)) == [False, False, False]


def test_event_selection_leading_reduce():
    arr = _jet_pt_array()
    # "leading" = highest-PT jet (this collection's own rank_field) per event.
    cut = {"collection": "L1T_JetAK4", "field": "PT", "reduce": "leading", "op": ">", "value": 30}
    assert list(_evaluate_event_selection_cut(arr, cut, n_events=3)) == [False, True, False]


def test_gather_collection_object_selection_and_drop_fields():
    arr = _jetak4_array_for_truncation()  # PTs: event0 [10,40,20,30], event1 [5,5,5,5]
    out = gather_collection(arr, "L1T_JetAK4", COLLECTION_REGISTRY["L1T_JetAK4"], cap=None, n_events=2,
                             object_selection=[{"field": "PT", "op": ">", "value": 15}],
                             drop_fields={"NNeutrals"})
    assert "NNeutrals" not in out.fields
    np.testing.assert_allclose(ak.to_numpy(out["PT"][0]), [40.0, 20.0, 30.0], atol=1e-1)  # 10.0 dropped
    assert ak.num(out["PT"], axis=1)[1] == 0  # event1: all PTs are 5.0, none survive >15


def test_gather_and_select_puppi_candidates_object_selection_drop_fields_realistic_pid():
    arr = _puppipart_array()  # ev0 survivors (floor): pt 10/5/3, pid 211/13/11, charge 1/-1/1
    out = gather_and_select_puppi_candidates(
        arr, object_selection=[{"field": "Charge", "op": "!=", "value": -1}],
        drop_fields={"is_pu", "is_reco_pu"}, realistic_pid=False)
    assert "is_pu" not in out.fields and "is_reco_pu" not in out.fields
    # the charge=-1 candidate (pt=5.0, pid=13) is excluded by object_selection
    ev0_pt = sorted(ak.to_numpy(out["pt"][0]).tolist(), reverse=True)
    assert ev0_pt == pytest.approx([10.0, 3.0], abs=1e-1)
    # realistic_pid=False -> raw signed PID preserved (211/11 here, both already
    # "known" bucket values so this fixture doesn't distinguish it from the
    # collapsed scheme numerically, but dtype should be int32, not int16)
    assert out["pdgId"].layout.content.dtype == np.int32


def test_convert_collide2v_regionized_event_selection_drops_samples_failing_cut(tmp_path):
    import yaml

    sample_dir = tmp_path / "eos"
    out_dir = tmp_path / "out"
    _write_synthetic_jetak4_sample(str(sample_dir / "TestSample"), n_files=3, events_per_file=30, seed=99)

    config = {
        "ds_name": "test_ds",
        "data_processing": {
            "sample_dir": str(sample_dir),
            "redir": "",
            "out_path": str(out_dir),
            "dataset_version": "collide2v_v1.0",
            "collections": {"L1T_JetAK4": None},
            "samples": [{
                "name": "TestSample", "label": 0,
                # every synthetic event has 1-3 jets with PT in [10,100) --
                # requiring >=3 jets is a real, partial cut on this fixture.
                "event_selection": [{"collection": "L1T_JetAK4", "field": "PT", "reduce": "count",
                                      "op": ">=", "value": 3}],
            }],
        },
    }
    config_path = tmp_path / "dataconfig.yml"
    config_path.write_text(yaml.dump(config))
    cfg = DataConfig(str(config_path))
    convert_collide2v_regionized(cfg, overwrite=False)

    out_frag = out_dir / "TestSample" / "TestSample_00000.parquet"
    result = ak.from_parquet(str(out_frag))
    assert 0 < len(result) < 90  # some, but not all, of the 90 synthetic events pass the >=3-jets cut
    assert ak.all(ak.num(result["L1T_JetAK4"]["PT"], axis=1) >= 3)


def _write_synthetic_puppipart_sample(sample_dir: str, n_files: int, events_per_file: int, seed: int = 0,
                                       prefix: str = "L1T_PUPPIPart") -> None:
    import os as _os
    rng = np.random.default_rng(seed)
    _os.makedirs(sample_dir, exist_ok=True)
    for i in range(n_files):
        n_cands = rng.integers(5, 10, size=events_per_file)
        pt = [rng.uniform(0.5, 50, n).tolist() for n in n_cands]
        eta = [rng.uniform(-2.9, 2.9, n).tolist() for n in n_cands]
        phi = [rng.uniform(-3.0, 3.0, n).tolist() for n in n_cands]
        charge = [rng.choice([-1, 0, 1], n).tolist() for n in n_cands]
        pid = [rng.choice([0, 11, 13, 22, 211], n).tolist() for n in n_cands]

        def const(value, dtype=float):
            return [[dtype(value)] * n for n in n_cands]

        arr = ak.Array({
            f"{prefix}_PT": pt, f"{prefix}_Eta": eta, f"{prefix}_Phi": phi,
            f"{prefix}_PID": pid, f"{prefix}_Charge": charge,
            f"{prefix}_E": const(1.0), f"{prefix}_Mass": const(0.0),
            f"{prefix}_D0": const(0.0), f"{prefix}_DZ": const(0.0),
            f"{prefix}_ErrorD0": const(1.0), f"{prefix}_ErrorDZ": const(1.0),
            f"{prefix}_IsPU": const(0, int), f"{prefix}_IsRecoPU": const(0, int),
            f"{prefix}_PuppiW": const(1.0), f"{prefix}_fUniqueID": const(0, int),
        })
        _write_parquet_with_dataset_version(arr, _os.path.join(sample_dir, f"file_{i}.parquet"))


def test_convert_collide2v_regionized_puppipart_object_selection_and_drop_fields(tmp_path):
    # Regression test: collections.L1T_PUPPIPart.drop_fields must validate
    # against OUTPUT field names (pt, is_pu, ...), not the raw registry
    # field names (PT, IsPU, ...) -- a real bug caught by hand-validating
    # configs/example_challenge_dataconfig.yaml before this test existed.
    import yaml

    sample_dir = tmp_path / "eos"
    out_dir = tmp_path / "out"
    _write_synthetic_puppipart_sample(str(sample_dir / "TestSample"), n_files=2, events_per_file=20, seed=13)

    config = {
        "ds_name": "test_ds",
        "data_processing": {
            "sample_dir": str(sample_dir),
            "redir": "",
            "out_path": str(out_dir),
            "dataset_version": "collide2v_v1.0",
            "collections": {
                "L1T_PUPPIPart": {
                    "cap": 18,
                    "object_selection": [{"field": "PT", "op": ">=", "value": 1.0}],
                    "drop_fields": ["is_pu", "is_reco_pu"],
                },
            },
            "samples": [{"name": "TestSample", "label": 0}],
        },
    }
    config_path = tmp_path / "dataconfig.yml"
    config_path.write_text(yaml.dump(config))
    cfg = DataConfig(str(config_path))
    convert_collide2v_regionized(cfg, overwrite=False)  # must not raise

    out_frag = out_dir / "TestSample" / "TestSample_00000.parquet"
    result = ak.from_parquet(str(out_frag))
    assert "is_pu" not in result["L1T_PUPPIPart"].fields
    assert "is_reco_pu" not in result["L1T_PUPPIPart"].fields
    # object_selection PT>=1.0 enforced -- every surviving candidate's raw pt >= 1.0
    assert ak.all(ak.flatten(result["L1T_PUPPIPart"]["pt"]) >= 1.0)


def _many_candidates_array(pts):
    """1 event with len(pts) candidates, all in-acceptance/no-op fields
    otherwise -- for total_cap tests, where the point is purely "how many
    survive and which ones", not the floor/region logic itself."""
    n = len(pts)
    zeros = [[0.0] * n]
    return ak.Array({
        "L1T_PUPPIPart_PT": [pts], "L1T_PUPPIPart_Eta": [[0.1] * n], "L1T_PUPPIPart_Phi": [[0.0] * n],
        "L1T_PUPPIPart_PID": [[211] * n], "L1T_PUPPIPart_Charge": [[1] * n],
        "L1T_PUPPIPart_E": zeros, "L1T_PUPPIPart_Mass": zeros,
        "L1T_PUPPIPart_D0": zeros, "L1T_PUPPIPart_DZ": zeros,
        "L1T_PUPPIPart_ErrorD0": [[1.0] * n], "L1T_PUPPIPart_ErrorDZ": zeros,
        "L1T_PUPPIPart_IsPU": [[0] * n], "L1T_PUPPIPart_IsRecoPU": [[0] * n],
        "L1T_PUPPIPart_PuppiW": [[1.0] * n], "L1T_PUPPIPart_fUniqueID": [[0] * n],
    })


def test_gather_and_select_puppi_candidates_total_cap_truncates_after_region_selection():
    arr = _many_candidates_array([50.0, 10.0, 90.0, 5.0, 30.0, 70.0, 20.0, 60.0])
    # selection_pt="none" keeps all 8 (region/floor selection is bypassed
    # entirely in that mode) -- total_cap should then trim to the top 3 by
    # raw pT regardless.
    out = gather_and_select_puppi_candidates(arr, selection_pt="none", total_cap=3)
    assert ak.num(out["pt"], axis=1)[0] == 3
    np.testing.assert_allclose(ak.to_numpy(out["pt"][0]), [90.0, 70.0, 60.0], atol=1e-1)


def test_gather_and_select_puppi_candidates_total_cap_noop_when_fewer_survive():
    arr = _many_candidates_array([50.0, 10.0, 90.0])
    out = gather_and_select_puppi_candidates(arr, selection_pt="none", total_cap=500)
    assert ak.num(out["pt"], axis=1)[0] == 3  # total_cap=500 never binds here


def test_collections_total_cap_only_supported_for_puppipart(tmp_path):
    import yaml

    sample_dir = tmp_path / "eos"
    out_dir = tmp_path / "out"
    _write_synthetic_jetak4_sample(str(sample_dir / "TestSample"), n_files=1, events_per_file=5, seed=1)

    config = {
        "ds_name": "test_ds",
        "data_processing": {
            "sample_dir": str(sample_dir), "redir": "", "out_path": str(out_dir),
            "dataset_version": "collide2v_v1.0",
            "collections": {"L1T_JetAK4": {"cap": 5, "total_cap": 100}},
            "samples": [{"name": "TestSample", "label": 0}],
        },
    }
    config_path = tmp_path / "dataconfig.yml"
    config_path.write_text(yaml.dump(config))
    cfg = DataConfig(str(config_path))
    with pytest.raises(ValueError, match="total_cap is only supported for a candidate-kind collection"):
        convert_collide2v_regionized(cfg, overwrite=False)


def test_convert_collide2v_regionized_fullreco_puppipart_as_candidate_collection(tmp_path):
    # FullReco_PUPPIPart shares L1T_PUPPIPart's raw fields and registry
    # "candidate" kind -- region mode + total_cap must work identically,
    # and the output column must be named FullReco_PUPPIPart, not
    # L1T_PUPPIPart (a real bug this test guards against: the per-file
    # record builder used to hardcode the output key name).
    import yaml

    sample_dir = tmp_path / "eos"
    out_dir = tmp_path / "out"
    _write_synthetic_puppipart_sample(str(sample_dir / "TestSample"), n_files=1, events_per_file=10, seed=21,
                                       prefix="FullReco_PUPPIPart")

    config = {
        "ds_name": "test_ds",
        "data_processing": {
            "sample_dir": str(sample_dir), "redir": "", "out_path": str(out_dir),
            "dataset_version": "collide2v_v1.0",
            "collections": {"FullReco_PUPPIPart": {"cap": 18, "total_cap": 3}},
            "samples": [{"name": "TestSample", "label": 0}],
        },
    }
    config_path = tmp_path / "dataconfig.yml"
    config_path.write_text(yaml.dump(config))
    cfg = DataConfig(str(config_path))
    convert_collide2v_regionized(cfg, overwrite=False)  # must not raise

    out_frag = out_dir / "TestSample" / "TestSample_00000.parquet"
    result = ak.from_parquet(str(out_frag))
    assert "FullReco_PUPPIPart" in result.fields
    assert "L1T_PUPPIPart" not in result.fields
    assert ak.max(ak.num(result["FullReco_PUPPIPart"]["pt"], axis=1)) <= 3  # total_cap enforced
