"""End-to-end synthetic-data test of the region-based candidate/collection
gathering functions in converters.py (no network -- hand-built awkward
arrays shaped like the real EOS schema). Copied and adapted from
aidascoutrepo's tests/test_region_converter.py, since converters.py's Stage
1 logic here is a byte-for-byte extraction of that same code. Run with:
    python -m pytest tests/test_converters.py -v
"""
import awkward as ak
import numpy as np

from converters import (
    CANDIDATE_PT_FLOOR_GEV,
    _empty_placeholder,
    _placeholder_spec_for,
    diagnose_puppi_selection,
    gather_and_select_puppi_candidates,
    gather_other_l1t_collections,
)


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
