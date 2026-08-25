"""
Stage 1 of AIDA-Scout's data pipeline: EOS foundational-model-dataset (raw
CMS Phase-2 L1T trigger-emulation parquet) -> per-sample regionized parquet
(real 90-region PFL1 candidate-selection geometry, one fixed integer label
per sample). See docs/central_dataset_preprocessing.md for the full schema
and docs/eos_dataset_schema.md for the raw EOS layout this reads from.

This is a standalone extraction of ONLY Stage 1 from AIDA-Scout's full
two-stage pipeline (github.com/AIDA-Scout/aidascoutrepo,
src/aida_scout/data/converters.py, commit c145ce6) -- Stage 2 (that parquet
-> {'pf','label','obj'} training tensors) is deliberately not included here,
so this package has no torch dependency at all: awkward/numpy/pyarrow (via
fsspec-xrootd) for the actual conversion, PyYAML for config loading. See
README.md for setup and provenance.
"""
import glob
import logging
import os
from pathlib import Path

import awkward as ak
import numpy as np

from config import DataConfig, join_remote
from constants import EPS
from regionize import (
    AE_ELIGIBLE_COLLECTIONS,
    AE_OBJ_COLLECTIONS,
    CANDIDATE_PT_FLOOR_GEV,
    CANDIDATES_PER_REGION,
    ETA_EDGES,
    N_REGIONS,
    OTHER,
    PHI_BINS,
    assign_region,
    label_for_sample,
    select_top_per_region,
)

logger = logging.getLogger("collide2vpreproc.converters")

PUPPI_CAND_RAW_FIELDS = [
    "PT", "Eta", "Phi", "PID", "Charge", "E", "Mass",
    "D0", "DZ", "ErrorD0", "ErrorDZ", "IsPU", "IsRecoPU", "PuppiW", "fUniqueID",
]

# Which raw L1T_PUPPIPart field feeds the tensor-gatherer's "pt" column --
# "weighted" (pt x puppi_weight, the region-selection criterion) is the
# default per the AE/contrastive training migration; "raw" is a conversion-
# time alternative (data_processing.pf_pt_mode), not a --pf_columns ablation,
# since whichever mode is chosen must physically be tensor column 0 (the
# padding sentinel PFCandsDataset/ContrastiveModel rely on).
PF_PT_MODES = {"weighted": "pt_weighted", "raw": "pt"}

# Ordered pf-tensor feature columns after "pt" (see gather_pfcands_regionized).
# is_pu/is_reco_pu are MC-truth pileup flags -- kept available for diagnostics/
# explicit ablation via --pf_columns, but excluded from PFPreProcessor's
# default feature set (a model trained on them would exploit information no
# real trigger has online). funique_id (no physics meaning) isn't gathered at
# all -- pure noise, not worth even offering as an ablation.
PF_FEATURE_NAMES_BASE = ["eta", "phi", "dxy", "dxysig", "pdgId", "charge",
                         "puppi_weight", "e", "mass", "dz", "error_dz", "is_pu", "is_reco_pu"]
PF_TRUTH_ONLY_FIELDS = {"is_pu", "is_reco_pu"}

# Every other L1T_* collection this central dataset keeps, and the raw
# fields read from each -- see docs/eos_dataset_schema.md. Kept close to
# their raw form (no selection/truncation, no weighting -- that's specific
# to the candidate collection above), just precision-downcast.
OTHER_L1T_COLLECTIONS = {
    "L1T_Electron": ["Charge", "D0", "DZ", "Eta", "Phi", "PT", "ErrorD0", "ErrorDZ",
                      "EhadOverEem", "IsolationVar", "IsolationVarRhoCorr"],
    "L1T_MuonTight": ["Charge", "D0", "DZ", "Eta", "Phi", "PT", "ErrorD0", "ErrorDZ",
                       "IsolationVar", "IsolationVarRhoCorr"],
    "L1T_PhotonTight": ["Eta", "Phi", "PT", "EhadOverEem", "IsolationVar", "IsolationVarRhoCorr"],
    # L1T_JetAK4 has no ConstituentsIdx field in the real production schema
    # (confirmed against actual data) -- despite docs/eos_dataset_schema.md
    # documenting JetAK8/JetPuppiAK4/JetPuppiAK8 as "same fields as JetAK4",
    # those three DO have it; JetAK4 alone is missing it. Asymmetric on
    # purpose, not a copy-paste omission. Kept (not dropped) despite not
    # being used as a model feature today -- may be useful for future
    # trainings; see _read_parquet_tolerant for how per-file absence of
    # these specifically-doubly-nested fields is handled.
    "L1T_JetAK4": ["Eta", "Phi", "PT", "Mass", "Charge", "Flavor", "BTag", "BTagPhys",
                    "NCharged", "NNeutrals", "Constituents"],
    "L1T_JetAK8": ["Eta", "Phi", "PT", "Mass", "Charge", "Flavor", "BTag", "BTagPhys",
                    "NCharged", "NNeutrals", "Constituents", "ConstituentsIdx"],
    "L1T_JetPuppiAK4": ["Eta", "Phi", "PT", "Mass", "Charge", "Flavor", "BTag", "BTagPhys",
                         "NCharged", "NNeutrals", "Constituents", "ConstituentsIdx"],
    "L1T_JetPuppiAK8": ["Eta", "Phi", "PT", "Mass", "Charge", "Flavor", "BTag", "BTagPhys",
                         "NCharged", "NNeutrals", "Constituents", "ConstituentsIdx"],
    "L1T_MET": ["MET", "Eta", "Phi"],
    "L1T_PUPPIMET": ["MET", "Eta", "Phi"],
    "L1T_Rho": ["Rho"],
    "L1T_ScalarHT": ["HT"],
}

_FLOAT_L1T_FIELDS = {"PT", "Eta", "Phi", "D0", "DZ", "ErrorD0", "ErrorDZ", "Mass",
                      "EhadOverEem", "IsolationVar", "IsolationVarRhoCorr", "MET", "Rho", "HT"}

# Integer fields verified (via a real-data min/max/percentile check across
# DY, tttt_incl, and QCD_HT50toInf) to overflow the default int8 -- kept at
# the smallest width that actually covers the observed range. Flavor
# (max 21) and NCharged (max 54 observed, tttt JetAK8) fit int8 and use the
# default; NNeutrals reached 427 (tttt JetAK8), so it needs int16.
_INT_FIELD_DTYPE_OVERRIDES = {"NNeutrals": np.int16}


def _downcast(arr: ak.Array, is_float: bool, field_name: str = "") -> ak.Array:
    if is_float:
        return ak.values_astype(arr, np.float16)
    if field_name in _INT_FIELD_DTYPE_OVERRIDES:
        return ak.values_astype(arr, _INT_FIELD_DTYPE_OVERRIDES[field_name])
    return ak.values_astype(arr, np.int8)


def gather_other_l1t_collections(arr: ak.Array) -> dict:
    """Every non-candidate L1T_* collection this central dataset keeps, kept
    close to raw (no selection, no weighting), just precision-downcast. Jet
    Constituents/ConstituentsIdx (jagged index lists) are left at their
    native dtype -- downcasting a reference/index field without checking its
    real range is exactly the mistake this session already caught once
    (pdgId int8 overflow), and neither is used as a model feature today."""
    out = {}
    for collection, fields in OTHER_L1T_COLLECTIONS.items():
        collection_out = {}
        for f in fields:
            raw = arr[f"{collection}_{f}"]
            if f in ("Constituents", "ConstituentsIdx"):
                collection_out[f] = raw
            else:
                collection_out[f] = _downcast(raw, is_float=f in _FLOAT_L1T_FIELDS, field_name=f)
        out[collection] = ak.zip(collection_out, depth_limit=1)
    return out


def _realistic_pdgid(raw_pid: np.ndarray) -> np.ndarray:
    """Same 5-bucket collapse as gather_pfcands_collide's realistic-pid
    branch, but UNSIGNED -- this pipeline's finalized design keeps charge as
    its own separate `charge` column rather than folding its sign into
    pdgId (a deliberate reversal of gather_pfcands_collide's own signed
    convention, decided earlier this session; do not reuse that function's
    sign-by-charge logic here). Reimplemented standalone (numpy, not
    awkward -- this pipeline is already working in flat numpy arrays by the
    time this is called) rather than shared, since that function's own
    candidate-gathering logic (aux objects, is_pf, include_aux) doesn't
    apply here at all: no PFPart, no aux-lepton appending -- L1T_Electron/
    MuonTight/PhotonTight are their own separate output collections in this
    pipeline instead of being merged into the candidate list."""
    abs_pid = np.abs(raw_pid)
    known = (abs_pid == 0) | (abs_pid == 11) | (abs_pid == 13) | (abs_pid == 22)
    return np.where(known, abs_pid, 211)


def gather_and_select_puppi_candidates(arr: ak.Array, selection_pt: str = "weighted") -> ak.Array:
    """The core of the new design: read raw L1T_PUPPIPart fields, compute
    the realistic pdgId + PUPPI-weighted pT, select up to
    CANDIDATES_PER_REGION candidates per (event, region) across the 90 PFL1
    regions by weighted pT with a flat CANDIDATE_PT_FLOOR_GEV floor applied
    before truncation, then return the survivors flattened per event and
    sorted by RAW pT descending (see regionize.py's module docstring for why
    weighted pT drives selection but raw pT drives presentation order).

    `selection_pt` (default "weighted", matching the design above) picks
    which pT the CANDIDATE_PT_FLOOR_GEV floor and the per-region top-N rank
    are computed against: "weighted" (`pt_raw * puppi_weight`, the default)
    or "raw" (`pt_raw` alone, ignoring PUPPI weight entirely for selection
    purposes -- an ablation of the design rationale above, since raw-pT
    selection no longer protects the fixed per-region budget from
    high-raw-pT/low-weight pileup candidates the way weighted selection
    does). Only affects WHICH candidates survive and their rank order within
    a region; `pt_weighted` is still computed and stored as its own output
    field either way, and the final flattened order stays raw-pT-descending
    regardless (that's presentation order, a separate concern from
    selection).

    "none" is a third option: no region assignment, no pT floor, no
    per-region cap at all -- every candidate with `pt_raw > 0` is kept (see
    the padding-sentinel reasoning below; confirmed on real data this
    session that literally every raw candidate already has `pt_raw > 0` --
    L1T_PUPPIPart has zero true zero-padding at the raw level -- so this
    keeps ALL 1000 raw candidates/event, unfiltered). This is the
    "original"/no-cuts baseline for comparing against the region-based
    design's own selection choices, not a production mode -- ~500-1000
    candidates/event vs. the design's own ~15-20, so datasets built this
    way are much larger per event (confirmed this session: ~12x the
    per-event bytes of the "weighted" design, ~1.8x of "raw").

    Returns a ragged (per-event variable-length) awkward Record array with
    fields: pt, eta, phi, dxy, dxysig, pdgId, charge, pt_weighted,
    puppi_weight, e, mass, dz, error_dz, is_pu, is_reco_pu, funique_id.
    """
    if selection_pt not in ("weighted", "raw", "none"):
        raise ValueError(f"selection_pt must be 'weighted', 'raw', or 'none', got {selection_pt!r}")
    prefix = "L1T_PUPPIPart"
    n_events = len(arr)

    # Per-event candidate counts are NOT uniform within a file (confirmed by
    # a production crash: ak.to_numpy on the raw 2D field assumes a regular
    # array and throws "subarray lengths are not regular" the first time a
    # file actually has ragged per-event counts) -- flatten explicitly
    # rather than reshaping a to_numpy'd rectangular array.
    counts_per_event = ak.to_numpy(ak.num(arr[f"{prefix}_PT"], axis=1)).astype(np.int64)
    event_idx = np.repeat(np.arange(n_events), counts_per_event)

    def _flat(field_name: str, dtype) -> np.ndarray:
        return ak.to_numpy(ak.flatten(arr[f"{prefix}_{field_name}"], axis=1)).astype(dtype)

    pt_raw_flat = _flat("PT", np.float64)
    eta_flat = _flat("Eta", np.float64)
    phi_flat = _flat("Phi", np.float64)
    puppi_w_flat = _flat("PuppiW", np.float64)
    charge_flat = _flat("Charge", np.int64)
    raw_pid_flat = _flat("PID", np.int64)
    e_flat = _flat("E", np.float64)
    mass_flat = _flat("Mass", np.float64)
    d0_flat = _flat("D0", np.float64)
    dz_flat = _flat("DZ", np.float64)
    error_d0_flat = _flat("ErrorD0", np.float64)
    error_dz_flat = _flat("ErrorDZ", np.float64)
    is_pu_flat = _flat("IsPU", np.int64)
    is_reco_pu_flat = _flat("IsRecoPU", np.int64)
    funique_id_flat = _flat("fUniqueID", np.int64)

    pt_weighted_flat = pt_raw_flat * puppi_w_flat
    dxy_flat = d0_flat
    dxysig_flat = d0_flat / (error_d0_flat + EPS)
    pdgid_flat = _realistic_pdgid(raw_pid_flat)

    if selection_pt == "none":
        # pt_raw_flat > 0 is the ONLY requirement in this mode -- same
        # padding-sentinel reasoning as the comment below, just with no
        # region/floor/rank selection on top of it.
        keep_idx = np.where(pt_raw_flat > 0)[0]
    else:
        selection_pt_flat = pt_weighted_flat if selection_pt == "weighted" else pt_raw_flat
        region_id = assign_region(eta_flat, phi_flat)
        # pt_raw_flat > 0 is enforced explicitly (not just inferred from
        # CANDIDATE_PT_FLOOR_GEV being positive), independent of
        # selection_pt/CANDIDATE_PT_FLOOR_GEV: the stored `pt` column (raw pT,
        # always column 0 -- see PUPPI_CAND_OUTPUT_COLUMNS) is the exact
        # padding-slot sentinel every downstream consumer checks
        # (ContrastiveModel._make_mask: `x[..., 0] == 0`). A real (non-padded)
        # candidate with pt_raw exactly 0 would be silently treated as padding.
        above_floor = (selection_pt_flat >= CANDIDATE_PT_FLOOR_GEV) & (pt_raw_flat > 0)
        survive = select_top_per_region(event_idx, region_id, selection_pt_flat, above_floor,
                                         n_events=n_events, cap=CANDIDATES_PER_REGION)
        keep_idx = np.where(survive)[0]

    ev_kept = event_idx[keep_idx]
    pt_raw_kept = pt_raw_flat[keep_idx]
    order = np.lexsort((-pt_raw_kept, ev_kept))  # per-event, raw-pT-descending
    ev_sorted = ev_kept[order]
    counts = np.bincount(ev_sorted, minlength=n_events)

    def field(values_flat):
        return ak.unflatten(values_flat[keep_idx][order], counts)

    fields = {
        "pt": _downcast(field(pt_raw_flat), is_float=True),
        "eta": _downcast(field(eta_flat), is_float=True),
        "phi": _downcast(field(phi_flat), is_float=True),
        "dxy": _downcast(field(dxy_flat), is_float=True),
        "dxysig": _downcast(field(dxysig_flat), is_float=True),
        "pdgId": ak.values_astype(field(pdgid_flat), np.int16),
        "charge": ak.values_astype(field(charge_flat), np.int8),
        "pt_weighted": _downcast(field(pt_weighted_flat), is_float=True),
        "puppi_weight": _downcast(field(puppi_w_flat), is_float=True),
        "e": _downcast(field(e_flat), is_float=True),
        "mass": _downcast(field(mass_flat), is_float=True),
        "dz": _downcast(field(dz_flat), is_float=True),
        "error_dz": _downcast(field(error_dz_flat), is_float=True),
        "is_pu": ak.values_astype(field(is_pu_flat), np.int8),
        # IsRecoPU verified 0/1-only (DY/tttt_incl/QCD_HT50toInf, 5M
        # candidates each) -- same int8 flag as is_pu. fUniqueID verified as
        # a per-file running index reaching ~140k within a single file (well
        # past int16's 32767 max), so it stays int32.
        "is_reco_pu": ak.values_astype(field(is_reco_pu_flat), np.int8),
        "funique_id": ak.values_astype(field(funique_id_flat), np.int32),
    }
    return ak.zip(fields, depth_limit=1)


# The 7 genuinely variable-count L1T_* collections gather_other_l1t_collections
# stores -- used by _drop_events_with_empty_axis below to decide whether an
# event has any real L1T object content at all. Deliberately excludes
# L1T_MET/L1T_PUPPIMET/L1T_Rho/L1T_ScalarHT: those are fixed-size scalars (1
# or 5 slots) always "populated" regardless of real per-event activity, so
# including them would make the populated check trivially true for almost
# every event and defeat the point of this filter.
_L1T_VARIABLE_COUNT_COLLECTIONS = (
    "L1T_Electron", "L1T_MuonTight", "L1T_PhotonTight",
    "L1T_JetAK4", "L1T_JetAK8", "L1T_JetPuppiAK4", "L1T_JetPuppiAK8",
)


def _drop_events_with_empty_axis(cands: ak.Array, others: dict, n_events: int) -> tuple:
    """Boolean keep-mask (True = keep) over n_events, dropping any event
    where EITHER axis this dataset is built for ends up with nothing:
    L1T_PUPPIPart has zero surviving candidates (after region selection/
    floor/per-region cap), OR every one of the 7 genuinely variable-count
    L1T object collections (see _L1T_VARIABLE_COUNT_COLLECTIONS above) is
    empty -- an event only survives if it has real content on BOTH.

    Addresses a real problem, not just a diagnostic curiosity. Empty
    candidates: the axis-2 contrastive encoder's CLS-token mask
    (aida_scout.models.contrastive.ContrastiveModel._make_mask) leaves
    every candidate slot masked for a zero-candidate event, so its
    embedding collapses to the same fixed, uninformative constant
    regardless of what actually happened in that event. Empty L1T objects:
    the axis-1 AE's fixed 23-slot input would be entirely padding for such
    an event, an equally degenerate (and equally uninformative) case on the
    other axis. Confirmed on real data this session (one real QCD_HT50toInf/
    minbias file each): ~0.66%/38.94% of events have populated L1T but zero
    candidates (MinBias is soft/pileup-dominated, so its candidates are
    systematically more likely to fall below CANDIDATE_PT_FLOOR_GEV or be
    PUPPI-weighted near zero); a further ~0.01%/0.01% have candidates but
    zero L1T objects, or nothing on either axis at all -- rare, but real,
    not purely hypothetical. Any of these fractions, left in, risks
    correlating with the axis-1 nuisance bin for a spurious
    (padding-artifact) reason rather than real physics -- exactly the kind
    of confound NURD's decorrelation exists to fight, not accidentally
    manufacture.

    Returns (keep_mask, n_dropped).
    """
    n_cands = ak.to_numpy(ak.num(cands["pt"], axis=1))
    l1t_populated = np.zeros(n_events, dtype=bool)
    for name in _L1T_VARIABLE_COUNT_COLLECTIONS:
        # others[name] is ak.zip(..., depth_limit=1) -- a RECORD of
        # independently-jagged per-field columns (e.g. {PT: var*float16,
        # Eta: var*float16, ...}), not a jagged list of per-object records.
        # ak.num on the whole record returns a record of per-field counts
        # (all identical, since every field in one collection shares the
        # same per-event object count), which ak.to_numpy can't compare
        # against a scalar -- so count one representative field ("PT",
        # present in all 7 collections here) instead of the whole record.
        l1t_populated |= ak.to_numpy(ak.num(others[name]["PT"], axis=1)) > 0
    drop = (n_cands == 0) | ~l1t_populated
    return ~drop, int(drop.sum())


def diagnose_puppi_selection(arr: ak.Array, selection_pt: str = "weighted") -> dict:
    """Per-candidate accounting of WHY each L1T_PUPPIPart candidate was or
    wasn't kept by gather_and_select_puppi_candidates -- not called by the
    production path, purely for reporting how much (and where) the
    region-based selection actually cuts. Recomputes region/floor/survive
    independently rather than sharing state with
    gather_and_select_puppi_candidates (cheap vectorized numpy either way --
    xrootd I/O dominates per-file wall time, not this, confirmed empirically
    this session) so it can categorize every candidate, not just survivors.

    `selection_pt` -- see gather_and_select_puppi_candidates's identical
    parameter; must match whatever this conversion run actually used, or
    this accounting won't reflect the real selection.

    Returns raw (not per-event-mean) counts, summable across files/events:
      n_events, out_of_acceptance, below_floor, rank_truncated, kept
        (the first four should sum to n_events * n_candidates_per_event)
      region_truncated_counts: length-N_REGIONS array, how many candidates
        were rank-truncated out of each region -- shows WHICH regions are
        actually doing the cutting, not just how many candidates overall.
    """
    if selection_pt not in ("weighted", "raw"):
        raise ValueError(f"selection_pt must be 'weighted' or 'raw', got {selection_pt!r}")
    prefix = "L1T_PUPPIPart"
    n_events = len(arr)

    counts_per_event = ak.to_numpy(ak.num(arr[f"{prefix}_PT"], axis=1)).astype(np.int64)
    event_idx = np.repeat(np.arange(n_events), counts_per_event)

    pt_raw_flat = ak.to_numpy(ak.flatten(arr[f"{prefix}_PT"], axis=1)).astype(np.float64)
    eta_flat = ak.to_numpy(ak.flatten(arr[f"{prefix}_Eta"], axis=1)).astype(np.float64)
    phi_flat = ak.to_numpy(ak.flatten(arr[f"{prefix}_Phi"], axis=1)).astype(np.float64)
    puppi_w_flat = ak.to_numpy(ak.flatten(arr[f"{prefix}_PuppiW"], axis=1)).astype(np.float64)
    pt_weighted_flat = pt_raw_flat * puppi_w_flat
    selection_pt_flat = pt_weighted_flat if selection_pt == "weighted" else pt_raw_flat

    region_id = assign_region(eta_flat, phi_flat)
    # pt_raw_flat > 0 mirrors gather_and_select_puppi_candidates's identical
    # guard -- see that function's comment for why (the padding-slot
    # sentinel every downstream consumer checks). Kept in the same
    # "below_floor" accounting bucket below rather than a separate category:
    # a pt_raw==0 candidate is never a real, meaningful survivor either way.
    above_floor = (selection_pt_flat >= CANDIDATE_PT_FLOOR_GEV) & (pt_raw_flat > 0)
    survive = select_top_per_region(event_idx, region_id, selection_pt_flat, above_floor,
                                     n_events=n_events, cap=CANDIDATES_PER_REGION)

    out_of_acceptance = region_id < 0
    below_floor = (~out_of_acceptance) & (~above_floor)
    rank_truncated = (~out_of_acceptance) & above_floor & (~survive)

    return {
        "n_events": n_events,
        "out_of_acceptance": int(out_of_acceptance.sum()),
        "below_floor": int(below_floor.sum()),
        "rank_truncated": int(rank_truncated.sum()),
        "kept": int(survive.sum()),
        "region_truncated_counts": np.bincount(region_id[rank_truncated], minlength=N_REGIONS),
    }


def log_diagnostics_summary(sample: str, diag: dict, top_n_regions: int = 5) -> None:
    """Log a diagnose_puppi_selection() accumulation (summed across every
    file processed for `sample`) as a per-event breakdown."""
    ne = diag["n_events"]
    if ne == 0:
        return
    logger.info(
        f"[{sample}] candidate cuts/event (mean over {ne} events): "
        f"out_of_acceptance={diag['out_of_acceptance'] / ne:.1f}, "
        f"below_floor={diag['below_floor'] / ne:.1f}, "
        f"rank_truncated={diag['rank_truncated'] / ne:.1f}, "
        f"kept={diag['kept'] / ne:.1f}"
    )
    top_regions = np.argsort(diag["region_truncated_counts"])[::-1][:top_n_regions]
    for r in top_regions:
        cnt = diag["region_truncated_counts"][r]
        if cnt == 0:
            continue
        eta_bin, phi_bin = int(r) // PHI_BINS, int(r) % PHI_BINS
        logger.info(
            f"[{sample}]   region {r} (eta in [{ETA_EDGES[eta_bin]:.1f}, {ETA_EDGES[eta_bin + 1]:.1f}), "
            f"phi_bin={phi_bin}/{PHI_BINS}): {cnt} rank-truncated ({cnt / ne:.2f}/event)"
        )


def list_remote_files(dir_url: str) -> list:
    """List every file in a remote (xrootd) directory via fsspec-xrootd --
    returns bare filenames (fsspec-xrootd's own `.ls()` behavior), not full
    paths/URLs; callers join these back onto the directory themselves (see
    convert_collide2v_regionized). Needed because hand-enumerating files
    isn't practical at this dataset's scale (some samples have 10,000+
    files).

    Filters out non-`.parquet` entries -- confirmed on real data that EOS's
    directory listing can include CephFS system-versioning pseudo-files
    (e.g. `.sys.v#.<name>.parquet`, seen for `HH_bbgammagamma`) that aren't
    real data and make `ak.from_parquet` fail outright if read as-is."""
    import fsspec

    fs, fpath = fsspec.core.url_to_fs(dir_url)
    names = fs.ls(fpath, detail=False)
    return sorted(n for n in names if n.endswith(".parquet") and not os.path.basename(n).startswith("."))


def _placeholder_spec_for(column_name: str) -> tuple:
    """(nesting depth, leaf dtype) a missing column's empty placeholder
    needs, so it matches the real field's structure closely enough to
    concatenate with files that do have it. Everything is singly-jagged
    (depth 1) except Constituents/ConstituentsIdx, which are a per-jet list
    of index lists (depth 2) -- checked by suffix, not full column name, so
    this covers every jet collection (JetAK4/JetAK8/JetPuppiAK4/
    JetPuppiAK8) uniformly. Every other field gets its final dtype from
    _downcast() regardless of what's used here (see gather_other_l1t_collections),
    so getting the placeholder's leaf dtype exactly right only matters for
    these two, which skip that normalization."""
    if column_name.endswith("_ConstituentsIdx"):
        return 2, np.int16
    if column_name.endswith("_Constituents"):
        return 2, np.uint32
    return 1, np.float64


def _empty_placeholder(n_events: int, depth: int, dtype) -> ak.Array:
    """All-empty ragged array: `n_events` top-level entries, each with zero
    sub-entries at every level down to `depth`, leaf type `dtype`. depth=1
    -> list<dtype> (an empty list per event); depth=2 -> list<list<dtype>>
    (an empty list-of-lists per event, e.g. "zero jets" for Constituents)."""
    content = ak.Array(np.array([], dtype=dtype))
    for _ in range(depth - 1):
        content = ak.unflatten(content, np.array([], dtype=np.int64))
    return ak.unflatten(content, np.zeros(n_events, dtype=np.int64))


def _read_parquet_tolerant(src: str, candidate_columns: list, other_columns: list, max_events) -> ak.Array:
    """Like ak.from_parquet(src, columns=candidate_columns+other_columns),
    but tolerant of `other_columns` entries that don't exist in this
    particular file's schema. Confirmed on real data: a field like
    L1T_JetAK8_Constituents can be present in some files of a sample and
    entirely absent from others -- most plausibly because the writer drops
    a (heavier, doubly-nested) field rather than writing it all-empty when
    every event in that file has zero objects in that collection. Missing
    `other_columns` fields are filled with a correctly-typed empty
    placeholder instead of failing outright -- correctly-typed matters:
    Constituents/ConstituentsIdx are doubly-nested (a per-jet list of
    constituent-index lists), not singly-jagged like every other "other"
    field here, and a depth/dtype-mismatched placeholder still breaks
    downstream when files are concatenated (awkward/pyarrow represent the
    mismatch as a dense_union type, which pyarrow's parquet writer can't
    serialize -- hit exactly this before adding the depth/dtype handling
    below).

    `candidate_columns` (the core L1T_PUPPIPart fields this whole pipeline
    is built on) are NOT given this tolerance -- if one of those is ever
    missing, that's a real problem worth failing loudly on, not silently
    padding around.
    """
    import fsspec
    import pyarrow.parquet as pq

    fs, fpath = fsspec.core.url_to_fs(src)
    with fs.open(fpath, "rb") as f:
        schema_names = set(pq.read_schema(f).names)

    missing_candidates = [c for c in candidate_columns if c not in schema_names]
    if missing_candidates:
        raise ValueError(f"{src}: missing required L1T_PUPPIPart column(s): {missing_candidates}")

    available_other = [c for c in other_columns if c in schema_names]
    missing_other = [c for c in other_columns if c not in schema_names]

    arr = ak.from_parquet(src, columns=candidate_columns + available_other)
    if max_events is not None and max_events > 0:
        arr = arr[:max_events]
    if missing_other:
        n_events = len(arr)
        logger.warning(f"{os.path.basename(src)}: {len(missing_other)} column(s) absent from this file's "
                        f"schema (collection has zero objects in every event here) -- filled empty: {missing_other}")
        fields = {f: arr[f] for f in arr.fields}
        for m in missing_other:
            depth, dtype = _placeholder_spec_for(m)
            fields[m] = _empty_placeholder(n_events, depth, dtype)
        arr = ak.zip(fields, depth_limit=1)
    return arr


def _file_dataset_version(src: str):
    """Reads a parquet file's own `dataset_version` custom key-value
    metadata entry (bytes, e.g. b"collide2v_v1.0"), or None if absent.

    Per the dataset's own producer (session communication): early/testing
    files have NO custom metadata block at all (not just a missing key
    within one) -- `.metadata` itself is None for those, which `.get()`
    on `None` would raise on, hence the `or {}` guard. Only reads the
    file's footer metadata (this call, like _read_parquet_tolerant's own
    schema check above, never touches row-group data), so it's cheap
    relative to actually reading a file's columns."""
    import fsspec
    import pyarrow.parquet as pq

    fs, fpath = fsspec.core.url_to_fs(src)
    with fs.open(fpath, "rb") as f:
        meta = pq.read_metadata(f).metadata
    return (meta or {}).get(b"dataset_version")


def _select_files_by_dataset_version(file_names: list, resolve_src, required_version: str,
                                      max_files: int, sample: str) -> tuple:
    """Scans `file_names` IN ORDER, checking each one's dataset_version
    metadata (see _file_dataset_version), keeping only files whose version
    matches `required_version` exactly -- stops early once `max_files`
    matching files have been found (max_files <= 0 = scan the entire list,
    keeping every match). `resolve_src(fname)` builds that file's full
    xrootd/local path (mirrors how the main loop below resolves `src`).

    This makes max_files mean "N usable (correct-version) files", not "N
    files, some possibly wrong-version and silently reducing the actual
    file count below what was requested" -- the version check has to
    happen before truncation, not after, for max_files to mean what it
    says.

    Returns (kept_file_names, n_skipped_wrong_version).
    """
    required = required_version.encode()
    kept, n_skipped = [], 0
    for fname in file_names:
        if max_files > 0 and len(kept) >= max_files:
            break
        version = _file_dataset_version(resolve_src(fname))
        if version == required:
            kept.append(fname)
        else:
            n_skipped += 1
            logger.info(f"[{sample}] {fname}: dataset_version={version!r} != {required_version!r} -- skipping file.")
    return kept, n_skipped


def convert_collide2v_regionized(cfg: DataConfig, overwrite: bool = False) -> None:
    """EOS foundational-model-dataset -> per-sample parquet training files,
    implementing the central-dataset preprocessing design worked out this
    session (see docs/central_dataset_preprocessing.md and regionize.py):
    parquet output, training-file-only (no Gen_*/FullReco_*/Vertex_*/
    Event_*), every L1T_* collection except L1T_PFPart, PUPPI candidates
    selected via the real 90-region PFL1 geometry, one fixed integer label
    per sample, source_file/source_row provenance.

    Config (`data_processing:`):
      sample_dir:  EOS base path (e.g. .../production_final)
      redir:       xrootd redirector, e.g. "root://eosproject-f.cern.ch/"
                   (empty = local glob, matching convert_collide2v's own
                   local/remote convention)
      out_path:    local output base directory
      samples:     list of EOS subdirectory names to convert -- either
                   plain strings (e.g. tttt_incl), using max_files_per_sample
                   below, or {name: <dir>, max_files: <N>} dicts for a
                   per-sample override (e.g. different background classes
                   needing different file counts to hit target event
                   counts). NOT the old {label: [files]} scheme; label is
                   derived automatically from the directory name
                   (regionize.label_for_sample). {name: <dir>, files: [...]}
                   pins an EXACT filename list instead of auto-discovering +
                   max_files-truncating -- for when multiple dataset
                   variants (e.g. different candidate_selection_pt configs)
                   need to read the exact same source files for a fair
                   comparison. Still runs the dataset_version check, but a
                   mismatch is a hard error (not a silent skip) -- pinning
                   is only meaningful if the pinned list is honored exactly.
      max_files_per_sample: -1 = discover and convert every file in the
                   sample's directory (default); fallback for any `samples`
                   entry that doesn't set its own max_files
      max_events_per_file:  -1 = every event (default)
      candidate_selection_pt: "weighted" (default), "raw", or "none" -- see
                   gather_and_select_puppi_candidates's identical parameter.
                   Which pT the CANDIDATE_PT_FLOOR_GEV floor and per-region
                   top-N rank are computed against; "raw" is an ablation of
                   the design's normal weighted-pT selection (see that
                   function's docstring for why the two differ). "none"
                   skips region assignment/floor/rank entirely -- every
                   candidate with pt_raw>0 is kept (in practice this is
                   EVERY raw candidate, ~1000/event, since real data has no
                   true zero-padding at the raw level) -- a no-cuts
                   baseline, not a production mode; incompatible with
                   report_diagnostics (raises if both are set). Regardless
                   of mode, `pt`/`pt_weighted` are both still stored as
                   separate output columns -- this only changes which
                   candidates are selected in the first place, not what
                   gets recorded about the ones that are.
      report_diagnostics: false (default) -- if true, also computes and
                   logs (per sample, after that sample's files are all
                   processed) a candidate-cut breakdown: how many
                   candidates/event are excluded for being out of the
                   region geometry's eta acceptance vs. below the pT floor
                   vs. rank-truncated within their region, plus which
                   regions are doing the most rank-truncating. Recomputes
                   the selection logic a second time (see
                   diagnose_puppi_selection) -- cheap relative to the
                   per-file xrootd read, not free, so off by default.
      dataset_version: "collide2v_v1.0" (default) -- REQUIRED file-level
                   `dataset_version` parquet metadata value (see
                   _file_dataset_version/_select_files_by_dataset_version).
                   The dataset's own producer confirmed this EOS dataset
                   mixes early/testing-stage files (no dataset_version
                   metadata at all -- `None`) with final production files
                   (`dataset_version=b"collide2v_v1.0"`); only files whose
                   version matches exactly are ever read. A config value,
                   not a hardcoded constant, since the producer expects
                   this string to change for a future production version.
                   Checked BEFORE max_files truncation, so max_files means
                   "N usable files", not "N files, some possibly
                   wrong-version".

    Unconditional (not config-gated): every event missing real content on
    EITHER axis -- zero surviving PUPPI candidates, or every L1T object
    collection empty, or both (see _drop_events_with_empty_axis) -- is
    dropped before being written, a real data-quality fix, not an ablation
    knob. Confirmed on real data: ~0.66%/38.94% of QCD/MinBias events have
    populated L1T but zero candidates; a further ~0.01%/0.01% have
    candidates but zero L1T objects, or nothing on either axis. Logged
    per-file and as a per-sample total regardless of report_diagnostics.

    One output file per sample: <out_path>/<sample>/<sample>.parquet, plus
    <out_path>/<sample>/<sample>_source_files.txt (the filename list
    source_file indexes into), mirroring the original EOS directory layout.
    """
    sample_dir = cfg.get_sample_dir().rstrip("/")
    redir = cfg.dp("redir", "")
    out_base = Path(cfg.dp("out_path", "./")).expanduser()
    samples = cfg.dp("samples", None)
    if not samples:
        raise ValueError("data_processing.samples (list of EOS subdirectory names) is required.")
    default_max_files = cfg.dp("max_files_per_sample", -1)
    max_events_per_file = cfg.dp("max_events_per_file", -1)
    candidate_selection_pt = cfg.dp("candidate_selection_pt", "weighted")
    if candidate_selection_pt not in ("weighted", "raw", "none"):
        raise ValueError(f"data_processing.candidate_selection_pt must be 'weighted', 'raw', or 'none', "
                          f"got {candidate_selection_pt!r}")
    report_diagnostics = cfg.dp("report_diagnostics", False)
    if report_diagnostics and candidate_selection_pt == "none":
        raise ValueError("data_processing.report_diagnostics doesn't apply to candidate_selection_pt: "
                          "none -- there's no region/floor/rank selection to report on in that mode.")
    dataset_version = cfg.dp("dataset_version", "collide2v_v1.0")

    candidate_columns = [f"L1T_PUPPIPart_{f}" for f in PUPPI_CAND_RAW_FIELDS]
    other_columns = [f"{c}_{f}" for c, cf in OTHER_L1T_COLLECTIONS.items() for f in cf]
    columns = candidate_columns + other_columns

    for entry in samples:
        if isinstance(entry, dict):
            sample = entry["name"]
            max_files = entry.get("max_files", default_max_files)
            explicit_files = entry.get("files")
        else:
            sample, max_files, explicit_files = entry, default_max_files, None
        label = label_for_sample(sample)
        sample_path = f"{sample_dir}/{sample}"

        def _resolve_src(fname, sample_path=sample_path):
            return join_remote(redir, f"{sample_path}/{fname}") if redir else os.path.join(sample_dir, sample, fname)

        if explicit_files is not None:
            # Pinned file list (e.g. multiple dataset variants that must
            # read the EXACT same source files for a fair comparison) --
            # still runs the dataset_version check (a pinned file's version
            # could go stale, or the list could have a typo), but a mismatch
            # here is a hard error, not a silent skip: the whole point of
            # pinning is determinism, so silently dropping one would produce
            # a variant that quietly no longer matches its siblings.
            file_names, n_wrong_version = _select_files_by_dataset_version(
                explicit_files, _resolve_src, dataset_version, -1, sample)
            if n_wrong_version:
                raise ValueError(f"[{sample}] {n_wrong_version} of the explicitly pinned `files:` "
                                  f"don't have dataset_version={dataset_version!r} -- pinned file lists "
                                  f"must all match exactly, not silently drop mismatches.")
        else:
            if redir:
                all_file_names = list_remote_files(join_remote(redir, sample_path))
            else:
                all_file_names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(sample_dir, sample, "*.parquet")))
            file_names, n_wrong_version = _select_files_by_dataset_version(
                all_file_names, _resolve_src, dataset_version, max_files, sample)
            if n_wrong_version:
                logger.info(f"[{sample}] skipped {n_wrong_version} file(s) with the wrong dataset_version "
                            f"(required {dataset_version!r}).")
        if not file_names:
            logger.warning(f"No files found for sample {sample!r} in {sample_path} with dataset_version="
                            f"{dataset_version!r} -- skipping.")
            continue

        out_dir = out_base / sample
        out_fname = out_dir / f"{sample}.parquet"
        if out_fname.exists() and not overwrite:
            raise FileExistsError(f"{out_fname} already exists. Use --overwrite.")

        per_file_records = []
        diag_totals = {"n_events": 0, "out_of_acceptance": 0, "below_floor": 0, "rank_truncated": 0,
                        "kept": 0, "region_truncated_counts": np.zeros(N_REGIONS, dtype=np.int64)}
        total_empty_axis_dropped = 0
        for source_file_idx, fname in enumerate(file_names):
            src = _resolve_src(fname)
            arr = _read_parquet_tolerant(src, candidate_columns, other_columns, max_events_per_file)
            n_events_read = len(arr)
            if n_events_read == 0:
                continue

            cands = gather_and_select_puppi_candidates(arr, selection_pt=candidate_selection_pt)
            others = gather_other_l1t_collections(arr)

            if report_diagnostics:
                d = diagnose_puppi_selection(arr, selection_pt=candidate_selection_pt)
                for k in ("n_events", "out_of_acceptance", "below_floor", "rank_truncated", "kept"):
                    diag_totals[k] += d[k]
                diag_totals["region_truncated_counts"] += d["region_truncated_counts"]

            # Drop events missing real content on either axis -- zero
            # surviving PUPPI candidates, every L1T object collection empty,
            # or both -- see _drop_events_with_empty_axis's docstring.
            # source_row (below) must stay the ORIGINAL row index into the
            # source file (not a post-drop renumbering), so provenance
            # still resolves correctly.
            keep_mask, n_dropped = _drop_events_with_empty_axis(cands, others, n_events_read)
            total_empty_axis_dropped += n_dropped
            source_rows = np.arange(n_events_read)
            if n_dropped:
                cands = cands[keep_mask]
                others = {name: coll[keep_mask] for name, coll in others.items()}
                source_rows = source_rows[keep_mask]
            n_events = len(source_rows)
            if n_events == 0:
                logger.warning(f"[{sample}] {fname}: all {n_events_read} events dropped by the "
                                f"empty-axis filter -- skipping file.")
                continue

            record_fields = {"L1T_PUPPIPart": cands, **others}
            record_fields["label"] = ak.values_astype(ak.Array(np.full(n_events, label)), np.int8)
            record_fields["source_file"] = ak.values_astype(ak.Array(np.full(n_events, source_file_idx)), np.int32)
            record_fields["source_row"] = ak.values_astype(ak.Array(source_rows), np.int16)

            per_file_records.append(ak.zip(record_fields, depth_limit=1))
            logger.info(f"[{sample}] {fname}: {n_events} events processed, {n_dropped} dropped "
                        f"(empty candidates and/or empty L1T collections) "
                        f"(file {source_file_idx + 1}/{len(file_names)})")

        if not per_file_records:
            logger.warning(f"No events converted for sample {sample!r} -- skipping output.")
            continue

        full = per_file_records[0] if len(per_file_records) == 1 else ak.concatenate(per_file_records, axis=0)
        os.makedirs(out_dir, exist_ok=True)
        ak.to_parquet(full, os.fspath(out_fname))
        # Filename list saved alongside so source_file's integer index can be
        # resolved back to an actual path later.
        with open(out_dir / f"{sample}_source_files.txt", "w") as fh:
            fh.write("\n".join(file_names))
        logger.info(f"[{sample}] wrote {len(full)} events, {len(file_names)} source files "
                    f"({n_wrong_version} file(s) skipped for wrong dataset_version), label={label} -> {out_fname} "
                    f"({total_empty_axis_dropped} events dropped total: empty candidates and/or empty L1T collections)")
        if report_diagnostics:
            log_diagnostics_summary(sample, diag_totals)
