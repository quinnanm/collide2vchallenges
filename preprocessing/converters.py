"""Stage 1 of AIDA-Scout's data pipeline: EOS foundational-model-dataset (raw
CMS Phase-2 L1T trigger-emulation parquet) -> per-sample regionized parquet
(real 90-region PFL1 candidate-selection geometry, one fixed integer label
per sample). See docs/central_dataset_preprocessing.md for the full schema
and docs/eos_dataset_schema.md for the raw EOS layout this reads from, and
docs/challenge_dataconfig.md for the config-driven per-challenge system
(collections/candidate_selection/split/flush_every_events/event_selection)
built on top of this module's original fixed recipe.

This is a standalone extraction of ONLY Stage 1 from AIDA-Scout's full
two-stage pipeline (github.com/AIDA-Scout/aidascoutrepo,
src/aida_scout/data/converters.py, commit c145ce6) -- Stage 2 (that parquet
-> {'pf','label','obj'} training tensors) is deliberately not included here,
so this package has no torch dependency at all: awkward/numpy/pyarrow (via
fsspec-xrootd) for the actual conversion, PyYAML for config loading. See
README.md for setup and provenance.
"""
import ctypes
import gc
import glob
import logging
import operator
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
    select_top_n_per_event,
    select_top_per_region,
)

logger = logging.getLogger("collide2vpreproc.converters")

PUPPI_CAND_RAW_FIELDS = [
    "PT", "Eta", "Phi", "PID", "Charge", "E", "Mass",
    "D0", "DZ", "ErrorD0", "ErrorDZ", "IsPU", "IsRecoPU", "PuppiW", "fUniqueID",
]

# gather_and_select_puppi_candidates's OUTPUT field names (derived/renamed
# from the raw fields above -- e.g. PID -> pdgId, IsPU -> is_pu, plus
# dxy/dxysig/pt_weighted which don't correspond 1:1 to any single raw field).
# Used to validate collections.L1T_PUPPIPart.drop_fields, which (unlike
# object_selection, which reads the RAW fields above) refers to these output
# names -- e.g. drop_fields: [is_pu, is_reco_pu].
PUPPI_CAND_OUTPUT_FIELDS = {"pt", "eta", "phi", "dxy", "dxysig", "pdgId", "charge", "pt_weighted",
                            "puppi_weight", "e", "mass", "dz", "error_dz", "is_pu", "is_reco_pu", "funique_id"}

# Which raw L1T_PUPPIPart field feeds the tensor-gatherer's "pt" column --
# "weighted" (pt x puppi_weight, the region-selection criterion) is the
# default per the AE/contrastive training migration; "raw" is a conversion-
# time alternative (candidate_selection.pt), not a --pf_columns ablation,
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

# --------------------------------------------------------------------------
# Collection registry: every EOS collection this module knows how to read,
# keyed by its raw column-name prefix (raw column = f"{key}_{field}" --
# confirmed for every L1T_*/FullReco_* entry below against real data; the
# Vertex_*/Event_* entries assume the same "<prefix>_<field>" convention by
# analogy with the documented L1T_* naming in docs/eos_dataset_schema.md, NOT
# independently confirmed against a real file -- verify with a schema read
# (e.g. pyarrow.parquet.read_schema) before trusting a conversion that
# requests one of those two).
#
# kind:
#   "candidate"       -- L1T_PUPPIPart, FullReco_PUPPIPart, FullReco_PFPart --
#                         zero or more of these may be requested at once,
#                         each independently gathered/capped, but they all
#                         share the SAME candidate_selection: settings (mode/
#                         pt/floor_gev/realistic_pid -- a single global
#                         block, not per-collection). Special-cased region/
#                         flat_topn selection logic, see
#                         gather_and_select_puppi_candidates.
#   "fixed_scalar"     -- always fully populated (MET/Rho/ScalarHT-style
#                         event-level scalars, or Event_* MC metadata); a
#                         configured cap/object_selection is ignored for these
#                         (only one "object" -- the event itself -- exists).
#   "variable_object"  -- a per-event jagged collection; a configured cap
#                         truncates to the top-N by `rank_field` descending
#                         (see gather_collection).
# --------------------------------------------------------------------------
_ELECTRON_FIELDS = ["Charge", "D0", "DZ", "Eta", "Phi", "PT", "ErrorD0", "ErrorDZ",
                     "EhadOverEem", "IsolationVar", "IsolationVarRhoCorr"]
_MUON_FIELDS = ["Charge", "D0", "DZ", "Eta", "Phi", "PT", "ErrorD0", "ErrorDZ",
                "IsolationVar", "IsolationVarRhoCorr"]
_PHOTON_FIELDS = ["Eta", "Phi", "PT", "EhadOverEem", "IsolationVar", "IsolationVarRhoCorr"]
# L1T_JetAK4 has no ConstituentsIdx field in the real production schema
# (confirmed against actual data) -- despite docs/eos_dataset_schema.md
# documenting JetAK8/JetPuppiAK4/JetPuppiAK8 as "same fields as JetAK4",
# those three DO have it; JetAK4 alone is missing it. FullReco_JetAK4, by
# contrast, DOES have ConstituentsIdx (per
# docs/central_dataset_preprocessing.md Sec.3 -- all four FullReco_Jet*
# collections share the same field list, unlike their L1T_* counterparts).
# Asymmetric on purpose, not a copy-paste omission.
_JET_FIELDS_NO_CIDX = ["Eta", "Phi", "PT", "Mass", "Charge", "Flavor", "BTag", "BTagPhys",
                       "NCharged", "NNeutrals", "Constituents"]
_JET_FIELDS_WITH_CIDX = _JET_FIELDS_NO_CIDX + ["ConstituentsIdx"]

COLLECTION_REGISTRY = {
    "L1T_PUPPIPart": {"kind": "candidate", "fields": PUPPI_CAND_RAW_FIELDS},

    "L1T_Electron": {"kind": "variable_object", "fields": _ELECTRON_FIELDS, "rank_field": "PT"},
    "L1T_MuonTight": {"kind": "variable_object", "fields": _MUON_FIELDS, "rank_field": "PT"},
    "L1T_PhotonTight": {"kind": "variable_object", "fields": _PHOTON_FIELDS, "rank_field": "PT"},
    "L1T_JetAK4": {"kind": "variable_object", "fields": _JET_FIELDS_NO_CIDX, "rank_field": "PT"},
    "L1T_JetAK8": {"kind": "variable_object", "fields": _JET_FIELDS_WITH_CIDX, "rank_field": "PT"},
    "L1T_JetPuppiAK4": {"kind": "variable_object", "fields": _JET_FIELDS_WITH_CIDX, "rank_field": "PT"},
    "L1T_JetPuppiAK8": {"kind": "variable_object", "fields": _JET_FIELDS_WITH_CIDX, "rank_field": "PT"},
    "L1T_MET": {"kind": "fixed_scalar", "fields": ["MET", "Eta", "Phi"]},
    "L1T_PUPPIMET": {"kind": "fixed_scalar", "fields": ["MET", "Eta", "Phi"]},
    "L1T_Rho": {"kind": "fixed_scalar", "fields": ["Rho"]},
    "L1T_ScalarHT": {"kind": "fixed_scalar", "fields": ["HT"]},

    # FullReco_* mirrors L1T_* field-for-field (see docs/eos_dataset_schema.md).
    # FullReco_PUPPIPart/FullReco_PFPart share L1T_PUPPIPart's exact raw
    # field set, so either (or both at once -- see
    # gather_and_select_puppi_candidates's `prefix` parameter and
    # convert_collide2v_regionized's per-candidate-collection loop) can use
    # the SAME region-geometry candidate design (candidate_selection:
    # mode/pt/floor_gev/realistic_pid apply identically to every requested
    # candidate-kind collection -- it's a single global block, not
    # per-collection; collections.<name>.total_cap IS still per-collection).
    # L1T_PFPart deliberately does NOT get this treatment (stays
    # variable_object below) -- only the FullReco_* pair does, since
    # FullReco_PFPart vs. FullReco_PUPPIPart is a real requested comparison
    # (confirmed redundant/identical in practice, per README.md), not just
    # an omission.
    "FullReco_PUPPIPart": {"kind": "candidate", "fields": PUPPI_CAND_RAW_FIELDS},
    "FullReco_PFPart": {"kind": "candidate", "fields": PUPPI_CAND_RAW_FIELDS},
    "FullReco_Electron": {"kind": "variable_object", "fields": _ELECTRON_FIELDS, "rank_field": "PT"},
    "FullReco_MuonTight": {"kind": "variable_object", "fields": _MUON_FIELDS, "rank_field": "PT"},
    "FullReco_PhotonTight": {"kind": "variable_object", "fields": _PHOTON_FIELDS, "rank_field": "PT"},
    "FullReco_JetAK4": {"kind": "variable_object", "fields": _JET_FIELDS_WITH_CIDX, "rank_field": "PT"},
    "FullReco_JetAK8": {"kind": "variable_object", "fields": _JET_FIELDS_WITH_CIDX, "rank_field": "PT"},
    "FullReco_JetPuppiAK4": {"kind": "variable_object", "fields": _JET_FIELDS_WITH_CIDX, "rank_field": "PT"},
    "FullReco_JetPuppiAK8": {"kind": "variable_object", "fields": _JET_FIELDS_WITH_CIDX, "rank_field": "PT"},
    "FullReco_MET": {"kind": "fixed_scalar", "fields": ["MET", "Eta", "Phi"]},
    "FullReco_PUPPIMET": {"kind": "fixed_scalar", "fields": ["MET", "Eta", "Phi"]},
    "FullReco_Rho": {"kind": "fixed_scalar", "fields": ["Rho"]},
    "FullReco_ScalarHT": {"kind": "fixed_scalar", "fields": ["HT"]},

    # Generator-level truth. NOT safe as a training feature (leaks truth-level
    # info no real trigger has) -- useful for offline validation/labeling
    # only. See docs/eos_dataset_schema.md's Gen_* section for the caveat.
    "Gen_Part": {"kind": "variable_object",
                 "fields": ["PID", "Status", "PT", "Eta", "Phi", "Mass", "M1", "M2", "D1", "D2", "IsPU"],
                 "rank_field": "PT"},
    "Gen_JetAK4": {"kind": "variable_object", "fields": ["PT", "Eta", "Phi", "Mass"], "rank_field": "PT"},
    "Gen_JetAK8": {"kind": "variable_object", "fields": ["PT", "Eta", "Phi", "Mass"], "rank_field": "PT"},
    "Gen_MissingET": {"kind": "fixed_scalar", "fields": ["MET", "Eta", "Phi"]},

    # Reconstructed vertices (primary + pileup) -- no PT field, so ranked (if
    # capped) by SumPT2 descending instead, the natural "hardness"/primary-
    # vertex-likelihood proxy for this collection.
    "Vertex": {"kind": "variable_object",
               "fields": ["Index", "X", "Y", "Z", "T", "NDF", "SumPT2", "Constituents"],
               "rank_field": "SumPT2"},

    # Per-event MC generation metadata -- every field here is an event-level
    # scalar (docs/eos_dataset_schema.md: "size 1 for every field").
    "Event": {"kind": "fixed_scalar",
              "fields": ["Number", "ProcessID", "Weight", "CrossSection", "CrossSectionError", "Scale",
                         "AlphaQCD", "AlphaQED", "ID1", "ID2", "X1", "X2", "PDF1", "PDF2", "ScalePDF"]},
}

# Today's implicit default set (used whenever a config omits `collections:`
# entirely) -- every L1T_* collection except L1T_PFPart (redundant with
# unweighted L1T_PUPPIPart, confirmed on real data), L1T_PUPPIPart at the
# original fixed 18/region, every "other" L1T_* collection uncapped (kept in
# full, exactly like today's gather_other_l1t_collections default).
OTHER_L1T_COLLECTIONS = {name: spec["fields"] for name, spec in COLLECTION_REGISTRY.items()
                          if name.startswith("L1T_") and name != "L1T_PUPPIPart"}
DEFAULT_COLLECTIONS_CFG = {"L1T_PUPPIPart": CANDIDATES_PER_REGION, **{name: None for name in OTHER_L1T_COLLECTIONS}}

_FLOAT_FIELDS = {"PT", "Eta", "Phi", "D0", "DZ", "ErrorD0", "ErrorDZ", "Mass",
                 "EhadOverEem", "IsolationVar", "IsolationVarRhoCorr", "MET", "Rho", "HT",
                 # Vertex_*/Event_* additions -- physics quantities in a similar dynamic
                 # range to the L1T_*/FullReco_* fields above, downcast the same way.
                 "X", "Y", "Z", "T", "SumPT2",
                 "Weight", "CrossSection", "CrossSectionError", "Scale", "AlphaQCD", "AlphaQED",
                 "X1", "X2", "PDF1", "PDF2", "ScalePDF"}

# Integer fields verified (via a real-data min/max/percentile check across
# DY, tttt_incl, and QCD_HT50toInf) to overflow the default int8 -- kept at
# the smallest width that actually covers the observed range. Flavor
# (max 21) and NCharged (max 54 observed, tttt JetAK8) fit int8 and use the
# default; NNeutrals reached 427 (tttt JetAK8), so it needs int16.
_INT_FIELD_DTYPE_OVERRIDES = {"NNeutrals": np.int16}

# Integer field names verified safe for the default int8 downcast (every
# non-float field gather_collection/_downcast has ever actually produced this
# way, across L1T_*/FullReco_*'s Electron/Muon/PhotonTight/Jet* collections).
# A field name NOT in this set (e.g. every genuinely new Gen_*/Vertex_*/
# Event_* integer field: PID, Status, M1/M2/D1/D2, Index, NDF, Number,
# ProcessID, ID1/ID2, ...) keeps its native dtype instead of being guessed
# into int8 -- downcasting an unverified-range field is exactly the mistake
# this codebase has already caught once (a real pdgId int8 overflow) and
# guards against elsewhere (Constituents/ConstituentsIdx, NNeutrals above).
_VERIFIED_INT8_FIELDS = {"Charge", "Flavor", "BTag", "BTagPhys", "NCharged"}

# Comparison operators available to event_selection/object_selection cuts.
_COMPARISON_OPS = {
    ">": operator.gt, ">=": operator.ge, "<": operator.lt,
    "<=": operator.le, "==": operator.eq, "!=": operator.ne,
}
_VALID_EVENT_REDUCE_MODES = {"scalar", "count", "any", "all", "leading", "max", "min"}


def _downcast(arr: ak.Array, is_float: bool, field_name: str = "") -> ak.Array:
    if is_float:
        return ak.values_astype(arr, np.float16)
    if field_name in _INT_FIELD_DTYPE_OVERRIDES:
        return ak.values_astype(arr, _INT_FIELD_DTYPE_OVERRIDES[field_name])
    if field_name in _VERIFIED_INT8_FIELDS:
        return ak.values_astype(arr, np.int8)
    return arr  # unverified integer field -- keep native dtype rather than guess a width


def _normalize_collection_entry(entry):
    """A `collections:` value can be a plain cap (int, or null for no cap --
    the common case) or a dict `{cap, object_selection, drop_fields, total_cap}`
    for finer per-collection control. Always returns (cap, object_selection_list,
    drop_fields_set, total_cap). `total_cap` only applies to a candidate-kind
    collection (L1T_PUPPIPart/FullReco_PUPPIPart/FullReco_PFPart -- see
    gather_and_select_puppi_candidates) -- validated elsewhere, not here."""
    if entry is None or isinstance(entry, int):
        return entry, [], set(), None
    if isinstance(entry, dict):
        return (entry.get("cap"), list(entry.get("object_selection", []) or []),
                set(entry.get("drop_fields", []) or []), entry.get("total_cap"))
    raise ValueError(f"data_processing.collections: entry must be a cap (int/null) or a dict with "
                      f"cap/object_selection/drop_fields/total_cap, got {entry!r}")


def _validate_object_selection_cuts(collection: str, spec: dict, cuts: list) -> None:
    if cuts and spec["kind"] == "fixed_scalar":
        raise ValueError(f"data_processing.collections.{collection}: object_selection isn't supported for a "
                          f"fixed_scalar collection (always exactly one value/event -- see event_selection instead).")
    valid_fields = set(spec["fields"])
    for cut in cuts:
        if not {"field", "op", "value"} <= cut.keys():
            raise ValueError(f"data_processing.collections.{collection}.object_selection: each cut needs "
                              f"'field', 'op', and 'value' -- got {cut!r}")
        if cut["op"] not in _COMPARISON_OPS:
            raise ValueError(f"data_processing.collections.{collection}.object_selection: unsupported op "
                              f"{cut['op']!r} -- must be one of {sorted(_COMPARISON_OPS)}")
        if cut["field"] not in valid_fields:
            raise ValueError(f"data_processing.collections.{collection}.object_selection: unknown field "
                              f"{cut['field']!r} -- must be one of {sorted(valid_fields)}")


def _validate_drop_fields(collection: str, spec: dict, drop_fields: set) -> None:
    valid_fields = set(spec["fields"])
    unknown = drop_fields - valid_fields
    if unknown:
        raise ValueError(f"data_processing.collections.{collection}.drop_fields: unknown field(s) "
                          f"{sorted(unknown)} -- must be a subset of {sorted(valid_fields)}")
    if drop_fields == valid_fields:
        raise ValueError(f"data_processing.collections.{collection}.drop_fields removed every field -- "
                          f"nothing left to store for this collection.")


def _evaluate_object_selection_cut(arr: ak.Array, collection: str, cut: dict, n_flat: int) -> np.ndarray:
    """Per-OBJECT boolean (flat, one entry per candidate/jet/lepton/... across
    all events in this file) -- unlike event_selection, no aggregation: this
    decides which individual objects within a collection survive, evaluated
    on that collection's own RAW field (`f"{collection}_{cut['field']}"`,
    read regardless of whether that field ends up in the output -- see
    drop_fields), before ranking/capping."""
    op = _COMPARISON_OPS[cut["op"]]
    raw_flat = ak.to_numpy(ak.flatten(arr[f"{collection}_{cut['field']}"], axis=1)).astype(np.float64)
    if len(raw_flat) != n_flat:
        raise ValueError(f"data_processing.collections.{collection}.object_selection: field {cut['field']!r} "
                          f"has a different per-event object count than this collection's own objects.")
    return op(raw_flat, cut["value"])


def _rebuild_object_fields(arr: ak.Array, name: str, out_fields: list, final_idx: np.ndarray,
                            counts: np.ndarray) -> ak.Array:
    collection_out = {}
    for f in out_fields:
        raw = arr[f"{name}_{f}"]
        if f in ("Constituents", "ConstituentsIdx"):
            flat = ak.flatten(raw, axis=1)
            collection_out[f] = ak.unflatten(flat[final_idx], counts)
        else:
            flat = ak.to_numpy(ak.flatten(raw, axis=1))
            downcast_flat = _downcast(ak.Array(flat[final_idx]), is_float=f in _FLOAT_FIELDS, field_name=f)
            collection_out[f] = ak.unflatten(downcast_flat, counts)
    return ak.zip(collection_out, depth_limit=1)


def gather_collection(arr: ak.Array, name: str, spec: dict, cap, n_events: int,
                       object_selection: list = None, drop_fields: set = None) -> ak.Array:
    """Read one COLLECTION_REGISTRY collection ('fixed_scalar' or
    'variable_object' kind -- L1T_PUPPIPart ('candidate' kind) is NOT handled
    here, see gather_and_select_puppi_candidates) from `arr`, downcast, and
    -- for a capped variable_object collection -- truncate to the top-`cap`
    objects/event by `spec['rank_field']` descending (same top-N-by-value
    convention used everywhere else in this module).

    `object_selection` (variable_object only): a list of {field, op, value}
    cuts (ALL must pass -- AND semantics), evaluated on this collection's own
    RAW fields, applied to each individual object BEFORE ranking/capping --
    e.g. drop individual jets below a PT threshold, independent of (and
    composable with) the cap. `drop_fields`: field names to omit from the
    output entirely (read regardless, in case object_selection needs one of
    them -- see convert_collide2v_regionized's column-reading).

    `cap=None` with no `object_selection` keeps every object in its original
    per-event order -- reproduces this module's pre-config-system behavior
    exactly (no reordering, no truncation, no filtering) when nothing asks
    for either.
    """
    object_selection = object_selection or []
    drop_fields = drop_fields or set()
    all_fields = spec["fields"]
    out_fields = [f for f in all_fields if f not in drop_fields]
    if not out_fields:
        raise ValueError(f"data_processing.collections.{name}.drop_fields removed every field -- "
                          f"nothing left to store for this collection.")

    if spec["kind"] == "fixed_scalar":
        collection_out = {}
        for f in out_fields:
            raw = arr[f"{name}_{f}"]
            if f in ("Constituents", "ConstituentsIdx"):
                collection_out[f] = raw
            else:
                collection_out[f] = _downcast(raw, is_float=f in _FLOAT_FIELDS, field_name=f)
        return ak.zip(collection_out, depth_limit=1)

    rep_field = spec.get("rank_field", all_fields[0])
    counts_per_event = ak.to_numpy(ak.num(arr[f"{name}_{rep_field}"], axis=1)).astype(np.int64)
    event_idx = np.repeat(np.arange(n_events), counts_per_event)
    n_flat = len(event_idx)

    keep = np.ones(n_flat, dtype=bool)
    for cut in object_selection:
        keep &= _evaluate_object_selection_cut(arr, name, cut, n_flat)

    if cap is None:
        # No cap -- still apply object_selection (if any), but preserve
        # original relative order among survivors (no rank-based reordering,
        # matching this module's pre-config-system passthrough behavior when
        # object_selection is also empty).
        keep_idx = np.where(keep)[0]
        counts = np.bincount(event_idx[keep_idx], minlength=n_events)
        return _rebuild_object_fields(arr, name, out_fields, keep_idx, counts)

    # A real cap: rank the object_selection-surviving objects by rank_field
    # descending, keep the top `cap` -- this also reorders the survivors by
    # rank_field descending (consistent with every other capped/ranked
    # collection in this module, e.g. L1T_PUPPIPart's own presentation order).
    rank_field = spec["rank_field"]
    rank_values = ak.to_numpy(ak.flatten(arr[f"{name}_{rank_field}"], axis=1)).astype(np.float64)
    survive = select_top_n_per_event(event_idx, rank_values, keep, n_events, cap)
    keep_idx = np.where(survive)[0]
    order = np.lexsort((-rank_values[keep_idx], event_idx[keep_idx]))
    final_idx = keep_idx[order]
    counts = np.bincount(event_idx[final_idx], minlength=n_events)
    return _rebuild_object_fields(arr, name, out_fields, final_idx, counts)


def gather_other_l1t_collections(arr: ak.Array, collections_cfg: dict = None) -> dict:
    """Every requested non-candidate collection, gathered via gather_collection.
    `collections_cfg` maps name -> a plain cap or a full {cap, object_selection,
    drop_fields} dict (see _normalize_collection_entry); defaults to every
    L1T_* collection except L1T_PFPart, uncapped -- this module's original
    (pre-config-system) behavior."""
    if collections_cfg is None:
        collections_cfg = {name: None for name in OTHER_L1T_COLLECTIONS}
    n_events = len(arr)
    out = {}
    for name, entry in collections_cfg.items():
        cap, object_selection, drop_fields, _total_cap = _normalize_collection_entry(entry)
        out[name] = gather_collection(arr, name, COLLECTION_REGISTRY[name], cap, n_events,
                                       object_selection=object_selection, drop_fields=drop_fields)
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


def gather_and_select_puppi_candidates(arr: ak.Array, selection_pt: str = "weighted", mode: str = "region",
                                        cap: int = CANDIDATES_PER_REGION,
                                        floor_gev: float = CANDIDATE_PT_FLOOR_GEV,
                                        object_selection: list = None, drop_fields: set = None,
                                        realistic_pid: bool = True, total_cap: int = None,
                                        prefix: str = "L1T_PUPPIPart") -> ak.Array:
    """The core of the new design: read raw L1T_PUPPIPart (or, via `prefix`,
    FullReco_PUPPIPart -- the only other COLLECTION_REGISTRY entry with
    kind="candidate", sharing the exact same raw field set) fields, compute
    the realistic pdgId + PUPPI-weighted pT, select candidates by `selection_pt`
    with a flat `floor_gev` floor applied before truncation, then return the
    survivors flattened per event and sorted by RAW pT descending (see
    regionize.py's module docstring for why weighted pT drives selection but
    raw pT drives presentation order).

    `mode` picks the selection geometry: "region" (default) replays the
    original design -- the real 90-region PFL1 geometry, up to `cap`
    candidates kept per (event, region); "flat_topn" skips the region
    geometry entirely and keeps the top `cap` candidates for the WHOLE event
    (still subject to the same floor), for challenges that don't need the
    per-region budget. `cap`'s meaning therefore depends on `mode`:
    candidates-per-region under "region" (original default: CANDIDATES_PER_REGION,
    i.e. 18), or a flat per-event total under "flat_topn".

    `selection_pt` (default "weighted", matching the design above) picks
    which pT the `floor_gev` floor and the top-N rank are computed against:
    "weighted" (`pt_raw * puppi_weight`, the default) or "raw" (`pt_raw`
    alone, ignoring PUPPI weight entirely for selection purposes -- an
    ablation of the design rationale above, since raw-pT selection no longer
    protects the fixed per-region budget from high-raw-pT/low-weight pileup
    candidates the way weighted selection does). Only affects WHICH
    candidates survive and their rank order within a region/event;
    `pt_weighted` is still computed and stored as its own output field either
    way, and the final flattened order stays raw-pT-descending regardless
    (that's presentation order, a separate concern from selection).

    "none" is a third option: no region assignment, no pT floor, no
    per-region/per-event cap at all -- every candidate with `pt_raw > 0` is
    kept (see the padding-sentinel reasoning below; confirmed on real data
    this session that literally every raw candidate already has `pt_raw > 0`
    -- L1T_PUPPIPart has zero true zero-padding at the raw level -- so this
    keeps ALL 1000 raw candidates/event, unfiltered). This is the
    "original"/no-cuts baseline for comparing against the region-based
    design's own selection choices, not a production mode -- ~500-1000
    candidates/event vs. the design's own ~15-20, so datasets built this
    way are much larger per event (confirmed this session: ~12x the
    per-event bytes of the "weighted" design, ~1.8x of "raw"). `mode` is
    irrelevant when `selection_pt` is "none". `object_selection` (see below)
    still applies in "none" mode.

    `object_selection`: a list of {field, op, value} cuts (ALL must pass),
    evaluated on RAW L1T_PUPPIPart fields (PT, Eta, Phi, PID, Charge, E,
    Mass, D0, DZ, ErrorD0, ErrorDZ, IsPU, IsRecoPU, PuppiW, fUniqueID) --
    e.g. {field: PT, op: '>=', value: 1.0}. Applied BEFORE the floor/region/
    flat_topn selection above (a candidate failing object_selection is never
    considered, same as failing the floor). Independent of whichever pT
    variant drives `selection_pt`'s own ranking.

    `drop_fields`: output field names (pt, eta, phi, dxy, dxysig, pdgId,
    charge, pt_weighted, puppi_weight, e, mass, dz, error_dz, is_pu,
    is_reco_pu, funique_id) to omit entirely from the returned collection --
    e.g. {"is_pu", "is_reco_pu"} to drop the MC-truth pileup flags.

    `realistic_pid`: True (default) collapses PID the way this pipeline
    always has -- the unsigned 5-bucket scheme via _realistic_pdgid (int16
    output). False stores the raw PID value unmodified (int32, matching the
    raw schema's own width) -- e.g. for a dataset that deliberately wants
    generator-level particle ID instead of the sanitized realistic scheme;
    note this reintroduces the truth-leakage _realistic_pdgid exists to
    avoid, so this is an explicit opt-out, not a recommended default.

    `total_cap`: an optional secondary ceiling on the FLATTENED per-event
    candidate count, applied AFTER the primary selection above (region/
    flat_topn/none) -- ranks whatever survived that primary selection by raw
    pT descending and keeps at most `total_cap`. Mainly meaningful under
    `mode="region"`, where the primary selection's own `cap` is a PER-REGION
    limit (so a busy event's flattened total can still run up to
    N_REGIONS * cap, e.g. up to 1620 for the default cap=18) -- `total_cap`
    lets you keep the region geometry's per-region selection logic while
    still bounding the whole event's candidate count with one flat number
    (e.g. total_cap=500). A no-op if the primary selection already keeps
    fewer than `total_cap` candidates for every event.

    Returns a ragged (per-event variable-length) awkward Record array with
    fields: pt, eta, phi, dxy, dxysig, pdgId, charge, pt_weighted,
    puppi_weight, e, mass, dz, error_dz, is_pu, is_reco_pu, funique_id (minus
    any in `drop_fields`).
    """
    if selection_pt not in ("weighted", "raw", "none"):
        raise ValueError(f"selection_pt must be 'weighted', 'raw', or 'none', got {selection_pt!r}")
    if mode not in ("region", "flat_topn"):
        raise ValueError(f"mode must be 'region' or 'flat_topn', got {mode!r}")
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
    pdgid_flat = _realistic_pdgid(raw_pid_flat) if realistic_pid else raw_pid_flat

    obj_keep = np.ones(len(pt_raw_flat), dtype=bool)
    if object_selection:
        _raw_field_map = {"PT": pt_raw_flat, "Eta": eta_flat, "Phi": phi_flat, "PID": raw_pid_flat,
                          "Charge": charge_flat, "E": e_flat, "Mass": mass_flat, "D0": d0_flat, "DZ": dz_flat,
                          "ErrorD0": error_d0_flat, "ErrorDZ": error_dz_flat, "IsPU": is_pu_flat,
                          "IsRecoPU": is_reco_pu_flat, "PuppiW": puppi_w_flat, "fUniqueID": funique_id_flat}
        for cut in object_selection:
            field = cut["field"]
            if field not in _raw_field_map:
                raise ValueError(f"{prefix} object_selection: unknown field {field!r} -- must be one "
                                  f"of {sorted(_raw_field_map)}")
            obj_keep &= _COMPARISON_OPS[cut["op"]](_raw_field_map[field].astype(np.float64), cut["value"])

    if selection_pt == "none":
        # pt_raw_flat > 0 is the ONLY requirement in this mode -- same
        # padding-sentinel reasoning as the comment below, just with no
        # region/floor/rank selection on top of it.
        keep_idx = np.where((pt_raw_flat > 0) & obj_keep)[0]
    else:
        selection_pt_flat = pt_weighted_flat if selection_pt == "weighted" else pt_raw_flat
        # pt_raw_flat > 0 is enforced explicitly (not just inferred from
        # floor_gev being positive), independent of selection_pt/floor_gev:
        # the stored `pt` column (raw pT, always column 0 -- see
        # PUPPI_CAND_OUTPUT_COLUMNS) is the exact padding-slot sentinel every
        # downstream consumer checks (ContrastiveModel._make_mask:
        # `x[..., 0] == 0`). A real (non-padded) candidate with pt_raw
        # exactly 0 would be silently treated as padding.
        above_floor = (selection_pt_flat >= floor_gev) & (pt_raw_flat > 0) & obj_keep
        if mode == "region":
            region_id = assign_region(eta_flat, phi_flat)
            survive = select_top_per_region(event_idx, region_id, selection_pt_flat, above_floor,
                                             n_events=n_events, cap=cap)
        else:  # flat_topn
            survive = select_top_n_per_event(event_idx, selection_pt_flat, above_floor,
                                              n_events=n_events, cap=cap)
        keep_idx = np.where(survive)[0]

    if total_cap is not None and len(keep_idx) > 0:
        # Secondary flat ceiling on top of whatever the primary
        # region/flat_topn/none selection above already kept -- ranks by raw
        # pT descending (this collection's own presentation order) rather
        # than re-deriving selection_pt, since total_cap is meant as a
        # simple, mode-independent "cap the total" safety valve, not a
        # second selection criterion.
        survive_total = select_top_n_per_event(event_idx[keep_idx], pt_raw_flat[keep_idx],
                                                np.ones(len(keep_idx), dtype=bool), n_events, cap=total_cap)
        keep_idx = keep_idx[survive_total]

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
        "pdgId": ak.values_astype(field(pdgid_flat), np.int16 if realistic_pid else np.int32),
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
    if drop_fields:
        fields = {k: v for k, v in fields.items() if k not in drop_fields}
    return ak.zip(fields, depth_limit=1)


def _drop_events_with_empty_axis(cands_by_name: dict, others: dict, n_events: int) -> tuple:
    """Boolean keep-mask (True = keep) over n_events, dropping any event
    where EITHER axis this dataset is built for ends up with nothing: EVERY
    requested candidate-kind collection has zero surviving candidates (after
    region/flat_topn/floor/cap selection), OR every one of the requested
    variable_object collections is empty -- an event only survives if it has
    real content on BOTH axes that are actually in play for this run (a
    challenge that requests no candidate-kind collection at all only has the
    object axis to satisfy; one that requests no variable_object collections
    at all only has the candidate axis). When MULTIPLE candidate-kind
    collections are requested at once (e.g. FullReco_PFPart AND
    FullReco_PUPPIPart), the candidate axis only counts as empty if ALL of
    them are -- any one having content is enough, mirroring how the object
    axis already treats its own multiple collections.

    With today's default `collections:` (all 7 variable L1T_* collections +
    L1T_PUPPIPart), this reproduces the original hardcoded design exactly --
    see docs/central_dataset_preprocessing.md's "Event-level filter" section
    for the full rationale (real per-sample drop-rate numbers, why a
    zero-candidate or zero-object event is degenerate for axis-2/axis-1 of
    the downstream training tensors) that motivated this filter. Note this
    check runs on the ALREADY-gathered (post object_selection/cap)
    `cands_by_name`/`others`, so an object_selection cut that empties a
    collection out naturally triggers this same drop.
    """
    variable_object_names = [name for name in others if COLLECTION_REGISTRY[name]["kind"] == "variable_object"]
    if variable_object_names:
        object_populated = np.zeros(n_events, dtype=bool)
        for name in variable_object_names:
            # A drop_fields config could remove this collection's usual
            # rep_field -- fall back to whatever field IS still present.
            available = [f for f in COLLECTION_REGISTRY[name]["fields"] if f in others[name].fields]
            rep_field = available[0]
            object_populated |= ak.to_numpy(ak.num(others[name][rep_field], axis=1)) > 0
    else:
        object_populated = np.ones(n_events, dtype=bool)  # nothing to check on this axis -- don't drop for it

    if cands_by_name:
        candidate_populated = np.zeros(n_events, dtype=bool)
        for cands in cands_by_name.values():
            pt_field = "pt" if "pt" in cands.fields else cands.fields[0]
            candidate_populated |= ak.to_numpy(ak.num(cands[pt_field], axis=1)) > 0
        drop = ~candidate_populated | ~object_populated
    else:
        drop = ~object_populated
    return ~drop, int(drop.sum())


def diagnose_puppi_selection(arr: ak.Array, selection_pt: str = "weighted",
                              cap: int = CANDIDATES_PER_REGION,
                              floor_gev: float = CANDIDATE_PT_FLOOR_GEV,
                              prefix: str = "L1T_PUPPIPart") -> dict:
    """Per-candidate accounting of WHY each L1T_PUPPIPart candidate was or
    wasn't kept by gather_and_select_puppi_candidates's "region" mode -- not
    called by the production path, purely for reporting how much (and where)
    the region-based selection actually cuts. Recomputes region/floor/survive
    independently rather than sharing state with
    gather_and_select_puppi_candidates (cheap vectorized numpy either way --
    xrootd I/O dominates per-file wall time, not this, confirmed empirically
    this session) so it can categorize every candidate, not just survivors.

    Only meaningful for candidate_selection.mode: region (the region-based
    accounting below has no equivalent for flat_topn), and does NOT account
    for object_selection (a candidate object_selection drops are folded into
    "below_floor" if you're using this alongside object_selection -- this
    function predates that option). `selection_pt`/`cap`/`floor_gev` -- see
    gather_and_select_puppi_candidates's identical parameters; must match
    whatever this conversion run actually used, or this accounting won't
    reflect the real selection.

    Returns raw (not per-event-mean) counts, summable across files/events:
      n_events, out_of_acceptance, below_floor, rank_truncated, kept
        (the first four should sum to n_events * n_candidates_per_event)
      region_truncated_counts: length-N_REGIONS array, how many candidates
        were rank-truncated out of each region -- shows WHICH regions are
        actually doing the cutting, not just how many candidates overall.
    """
    if selection_pt not in ("weighted", "raw"):
        raise ValueError(f"selection_pt must be 'weighted' or 'raw', got {selection_pt!r}")
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
    above_floor = (selection_pt_flat >= floor_gev) & (pt_raw_flat > 0)
    survive = select_top_per_region(event_idx, region_id, selection_pt_flat, above_floor,
                                     n_events=n_events, cap=cap)

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


def _read_parquet_tolerant(src: str, required_columns: list, other_columns: list, max_events) -> ak.Array:
    """Like ak.from_parquet(src, columns=required_columns+other_columns),
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

    `required_columns` (the core L1T_PUPPIPart fields this whole pipeline is
    built on, when requested at all, plus any raw column an event_selection
    cut needs) are NOT given this tolerance -- if one of those is ever
    missing, that's a real problem worth failing loudly on, not silently
    padding around.
    """
    import fsspec
    import pyarrow.parquet as pq

    fs, fpath = fsspec.core.url_to_fs(src)
    with fs.open(fpath, "rb") as f:
        schema_names = set(pq.read_schema(f).names)

    missing_required = [c for c in required_columns if c not in schema_names]
    if missing_required:
        raise ValueError(f"{src}: missing required column(s): {missing_required}")

    available_other = [c for c in other_columns if c in schema_names]
    missing_other = [c for c in other_columns if c not in schema_names]

    arr = ak.from_parquet(src, columns=required_columns + available_other)
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


# --------------------------------------------------------------------------
# Per-sample event_selection: keep/drop a whole event based on a condition
# over one of its collections (as opposed to object_selection above, which
# keeps/drops individual objects WITHIN a collection but never the event
# itself). Evaluated on RAW fields, independent of whatever `collections:`
# requests for output -- a cut can reference a collection that isn't even
# being saved.
# --------------------------------------------------------------------------

def _validate_event_selection_cuts(event_selection: list) -> None:
    for cut in event_selection:
        if not {"collection", "op", "value"} <= cut.keys():
            raise ValueError(f"event_selection: each cut needs 'collection', 'op', and 'value' -- got {cut!r}")
        collection = cut["collection"]
        if collection not in COLLECTION_REGISTRY:
            raise ValueError(f"event_selection: unknown collection {collection!r}")
        if cut["op"] not in _COMPARISON_OPS:
            raise ValueError(f"event_selection: unsupported op {cut['op']!r} -- must be one of "
                              f"{sorted(_COMPARISON_OPS)}")
        spec = COLLECTION_REGISTRY[collection]
        reduce = cut.get("reduce")
        if reduce is not None and reduce not in _VALID_EVENT_REDUCE_MODES:
            raise ValueError(f"event_selection: unsupported reduce {reduce!r} -- must be one of "
                              f"{sorted(_VALID_EVENT_REDUCE_MODES)}")
        if spec["kind"] != "fixed_scalar" and reduce is None:
            raise ValueError(f"event_selection: {collection} is a variable-count collection -- an explicit "
                              f"`reduce` (count/any/all/leading/max/min) is required.")
        if reduce == "scalar" and spec["kind"] != "fixed_scalar":
            raise ValueError(f"event_selection: reduce: scalar only applies to a fixed_scalar collection "
                              f"(got {collection}, kind={spec['kind']!r}).")


def _event_selection_columns(event_selection: list) -> list:
    """Raw column names an event_selection needs read (regardless of
    `collections:`), including the collection's own rank_field for a
    'leading' reduce."""
    cols = []
    for cut in event_selection:
        collection = cut["collection"]
        spec = COLLECTION_REGISTRY[collection]
        field = cut.get("field", spec["fields"][0])
        cols.append(f"{collection}_{field}")
        if cut.get("reduce") == "leading":
            rank_field = spec.get("rank_field") or ("PT" if collection == "L1T_PUPPIPart" else None)
            if rank_field:
                cols.append(f"{collection}_{rank_field}")
    return cols


def _evaluate_event_selection_cut(arr: ak.Array, cut: dict, n_events: int) -> np.ndarray:
    collection = cut["collection"]
    spec = COLLECTION_REGISTRY[collection]
    field = cut.get("field", spec["fields"][0])
    op = _COMPARISON_OPS[cut["op"]]
    value = cut["value"]
    raw = arr[f"{collection}_{field}"]
    reduce = cut.get("reduce", "scalar" if spec["kind"] == "fixed_scalar" else None)

    if reduce == "scalar":
        counts = ak.to_numpy(ak.num(raw, axis=1))
        if not np.all(counts == 1):
            raise ValueError(f"event_selection: {collection}.{field} isn't single-valued per event (counts "
                              f"range {counts.min()}-{counts.max()}) -- 'scalar' reduce needs exactly one "
                              f"value/event (e.g. L1T_Rho's 5 entries/event can't be used this way).")
        return op(ak.to_numpy(ak.flatten(raw, axis=1)).astype(np.float64), value)

    counts_per_event = ak.to_numpy(ak.num(raw, axis=1)).astype(np.int64)
    event_idx = np.repeat(np.arange(n_events), counts_per_event)
    values_flat = ak.to_numpy(ak.flatten(raw, axis=1)).astype(np.float64)

    if reduce == "count":
        return op(counts_per_event.astype(np.float64), value)
    if reduce == "any":
        satisfies = op(values_flat, value)
        result = np.zeros(n_events, dtype=bool)
        np.logical_or.at(result, event_idx, satisfies)
        return result
    if reduce == "all":
        fails = ~op(values_flat, value)
        any_fail = np.zeros(n_events, dtype=bool)
        np.logical_or.at(any_fail, event_idx, fails)
        return ~any_fail  # vacuously True for zero-object events
    if reduce in ("max", "min"):
        agg = np.full(n_events, -np.inf if reduce == "max" else np.inf)
        ufunc = np.maximum if reduce == "max" else np.minimum
        ufunc.at(agg, event_idx, values_flat)
        has_any = counts_per_event > 0
        result = np.zeros(n_events, dtype=bool)
        result[has_any] = op(agg[has_any], value)
        return result
    if reduce == "leading":
        rank_field = spec.get("rank_field") or ("PT" if collection == "L1T_PUPPIPart" else None)
        if rank_field is None:
            raise ValueError(f"event_selection: {collection} has no natural ordering -- 'leading' isn't "
                              f"defined for it.")
        rank_values_flat = ak.to_numpy(ak.flatten(arr[f"{collection}_{rank_field}"], axis=1)).astype(np.float64)
        above = np.ones(len(event_idx), dtype=bool)
        is_leading = select_top_n_per_event(event_idx, rank_values_flat, above, n_events, cap=1)
        leading_idx = np.where(is_leading)[0]
        result = np.zeros(n_events, dtype=bool)
        result[event_idx[leading_idx]] = op(values_flat[leading_idx], value)
        return result
    raise ValueError(f"event_selection: unsupported reduce {reduce!r}")


def _evaluate_event_selection(arr: ak.Array, event_selection: list, n_events: int) -> np.ndarray:
    mask = np.ones(n_events, dtype=bool)
    for cut in event_selection:
        mask &= _evaluate_event_selection_cut(arr, cut, n_events)
    return mask


def _prepare_output_dirs(split_dirs: dict, overwrite: bool) -> None:
    """Validate/clear each split's output directory before writing new
    fragments into it. Without --overwrite, any pre-existing fragment is a
    hard error (mirrors the original single-file FileExistsError check);
    with it, pre-existing fragments for that sample/split are removed first
    so a re-run with a different target size doesn't leave stale fragments
    from a previous run mixed in with the new ones."""
    for d in split_dirs.values():
        existing = sorted(d.glob("*.parquet")) if d.exists() else []
        if existing and not overwrite:
            raise FileExistsError(f"{d} already has parquet output ({len(existing)} fragment(s)). Use --overwrite.")
        for f in existing:
            f.unlink()
        d.mkdir(parents=True, exist_ok=True)


def _release_freed_memory() -> None:
    """Force a GC pass, then ask glibc's allocator to actually return freed
    arenas to the OS (malloc_trim) -- a real long-running conversion job's
    RSS grew far beyond any logical buffer size over many hours (e.g. a
    ~6GB theoretical flush-buffer peak vs. ~37GB observed via `kubectl top`),
    the classic signature of malloc fragmentation from many large numpy/
    pyarrow allocate/free cycles, not a Python reference leak -- buffers
    here ARE fully dereferenced each flush (see `buffers[key] = []` below).
    Silently a no-op on platforms without glibc's malloc_trim (e.g. macOS)."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _flush_buffer(buffers: dict, buffer_counts: dict, fragment_idx: dict, split_dirs: dict,
                   sample: str, key: str) -> None:
    if not buffers[key]:
        return
    chunk = buffers[key][0] if len(buffers[key]) == 1 else ak.concatenate(buffers[key], axis=0)
    out_fname = split_dirs[key] / f"{sample}_{fragment_idx[key]:05d}.parquet"
    ak.to_parquet(chunk, os.fspath(out_fname))
    fragment_idx[key] += 1
    buffers[key] = []
    buffer_counts[key] = 0
    del chunk
    _release_freed_memory()


def _append_and_maybe_flush(buffers: dict, buffer_counts: dict, fragment_idx: dict, split_dirs: dict,
                             sample: str, key: str, record: ak.Array, flush_every_events: int) -> None:
    if len(record) == 0:
        return
    buffers[key].append(record)
    buffer_counts[key] += len(record)
    if buffer_counts[key] >= flush_every_events:
        _flush_buffer(buffers, buffer_counts, fragment_idx, split_dirs, sample, key)


# --------------------------------------------------------------------------
# Config-resolution and per-file-processing helpers shared between
# convert_collide2v_regionized (below) and estimate_output_size.py -- pulled
# out so the size estimator runs the EXACT same collection/candidate/
# event_selection logic a real conversion would, instead of a
# separately-maintained (and possibly drifting) reimplementation.
# --------------------------------------------------------------------------

def _resolve_candidate_selection_cfg(cfg: DataConfig) -> dict:
    """Parse+validate `candidate_selection:`/`report_diagnostics:`. Returns
    {pt, mode, floor_gev, realistic_pid, report_diagnostics}."""
    candidate_selection_cfg = cfg.dp("candidate_selection", {}) or {}
    pt = candidate_selection_cfg.get("pt", cfg.dp("candidate_selection_pt", "weighted"))
    mode = candidate_selection_cfg.get("mode", "region")
    floor_gev = candidate_selection_cfg.get("floor_gev", CANDIDATE_PT_FLOOR_GEV)
    realistic_pid = candidate_selection_cfg.get("realistic_pid", True)
    if pt not in ("weighted", "raw", "none"):
        raise ValueError(f"data_processing.candidate_selection.pt must be 'weighted', 'raw', or 'none', got {pt!r}")
    if mode not in ("region", "flat_topn"):
        raise ValueError(f"data_processing.candidate_selection.mode must be 'region' or 'flat_topn', got {mode!r}")

    report_diagnostics = cfg.dp("report_diagnostics", False)
    if report_diagnostics and pt == "none":
        raise ValueError("data_processing.report_diagnostics doesn't apply to candidate_selection.pt: "
                          "none -- there's no region/floor/rank selection to report on in that mode.")
    if report_diagnostics and mode == "flat_topn":
        raise ValueError("data_processing.report_diagnostics only supports candidate_selection.mode: region -- "
                          "the region-based accounting doesn't apply to flat_topn.")
    return {"pt": pt, "mode": mode, "floor_gev": floor_gev, "realistic_pid": realistic_pid,
            "report_diagnostics": report_diagnostics}


def _resolve_collections_cfg(cfg: DataConfig) -> dict:
    """Parse+validate `collections:`. Returns {want_candidates,
    candidate_collections_cfg (name -> (cap, object_selection, drop_fields,
    total_cap), one entry per requested candidate-kind collection --
    L1T_PUPPIPart/FullReco_PUPPIPart/FullReco_PFPart, zero or more at once;
    candidate_selection: mode/pt/floor_gev/realistic_pid apply identically to
    ALL of them -- it's a single global block, not per-collection),
    other_collections_cfg (name -> (cap, object_selection, drop_fields,
    total_cap)), candidate_columns, other_columns (raw column-name lists for
    _read_parquet_tolerant)}."""
    collections_cfg_raw = cfg.dp("collections", None)
    if collections_cfg_raw is None:
        collections_cfg_raw = dict(DEFAULT_COLLECTIONS_CFG)
    if not collections_cfg_raw:
        raise ValueError("data_processing.collections is empty -- at least one collection must be requested.")
    unknown = [name for name in collections_cfg_raw if name not in COLLECTION_REGISTRY]
    if unknown:
        raise ValueError(f"data_processing.collections: unknown collection(s) {unknown} -- see "
                          f"COLLECTION_REGISTRY in converters.py for supported names.")

    collections_cfg = {}
    for name, entry in collections_cfg_raw.items():
        cap, object_selection, drop_fields, total_cap = _normalize_collection_entry(entry)
        spec = COLLECTION_REGISTRY[name]
        if total_cap is not None and spec["kind"] != "candidate":
            raise ValueError(f"data_processing.collections.{name}: total_cap is only supported for a "
                              f"candidate-kind collection (L1T_PUPPIPart/FullReco_PUPPIPart/FullReco_PFPart -- "
                              f"a secondary post-selection ceiling on the region-based candidate list) -- "
                              f"every other collection's own 'cap' is already the final per-event object limit.")
        # object_selection always refers to a collection's RAW field names
        # (spec["fields"]) -- true for candidate-kind collections too
        # (PUPPI_CAND_RAW_FIELDS is their registry `fields` entry).
        # drop_fields, however, refers to OUTPUT field names, which for a
        # candidate-kind collection differ from its raw fields (derived/
        # renamed -- see PUPPI_CAND_OUTPUT_FIELDS).
        _validate_object_selection_cuts(name, spec, object_selection)
        if spec["kind"] == "candidate":
            _validate_drop_fields(name, {"fields": PUPPI_CAND_OUTPUT_FIELDS}, drop_fields)
        else:
            _validate_drop_fields(name, spec, drop_fields)
        collections_cfg[name] = (cap, object_selection, drop_fields, total_cap)

    candidate_collections_cfg = {name: v for name, v in collections_cfg.items()
                                  if COLLECTION_REGISTRY[name]["kind"] == "candidate"}
    other_collections_cfg = {name: v for name, v in collections_cfg.items()
                              if name not in candidate_collections_cfg}
    want_candidates = bool(candidate_collections_cfg)

    candidate_columns = [f"{name}_{f}" for name in candidate_collections_cfg for f in PUPPI_CAND_RAW_FIELDS]
    other_columns = [f"{name}_{f}" for name in other_collections_cfg for f in COLLECTION_REGISTRY[name]["fields"]]

    return {
        "want_candidates": want_candidates, "candidate_collections_cfg": candidate_collections_cfg,
        "other_collections_cfg": other_collections_cfg,
        "candidate_columns": candidate_columns, "other_columns": other_columns,
    }


def _resolve_sample_entry(entry, default_max_files: int) -> dict:
    """Parse+validate one `samples:` list entry (plain string or dict).
    Returns {sample, max_files, explicit_files, target_events, label,
    event_selection}."""
    if isinstance(entry, dict):
        sample = entry["name"]
        max_files = entry.get("max_files", default_max_files)
        explicit_files = entry.get("files")
        target_events = entry.get("target_events")
        explicit_label = entry.get("label")
        event_selection = entry.get("event_selection") or []
    else:
        sample = entry
        max_files, explicit_files, target_events, explicit_label = default_max_files, None, None, None
        event_selection = []
    _validate_event_selection_cuts(event_selection)
    label = explicit_label if explicit_label is not None else label_for_sample(sample)
    return {"sample": sample, "max_files": max_files, "explicit_files": explicit_files,
            "target_events": target_events, "label": label, "event_selection": event_selection}


def _discover_sample_files(sample_dir: str, redir: str, sample: str, explicit_files) -> list:
    """The full candidate file-name list for one sample (before any
    dataset_version filtering/max_files truncation) -- the pinned `files:`
    list if given, else a full remote/local directory listing."""
    if explicit_files is not None:
        return list(explicit_files)
    sample_path = f"{sample_dir}/{sample}"
    if redir:
        return list_remote_files(join_remote(redir, sample_path))
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(sample_dir, sample, "*.parquet")))


def _build_record_for_file(arr: ak.Array, *, candidate_selection_pt: str,
                            candidate_mode: str, floor_gev: float, realistic_pid: bool,
                            candidate_collections_cfg: dict, other_collections_cfg: dict,
                            event_selection: list, label: int, source_file_idx: int) -> tuple:
    """Shared core of per-file processing: apply event_selection, gather
    every requested collection, apply the empty-axis filter, and build the
    final `record` (ak.Array, with label/source_file/source_row attached).
    Used by BOTH convert_collide2v_regionized's real per-sample loop and
    estimate_output_size.py's sampling, so the two can never silently drift
    apart in what actually survives to output.

    `candidate_collections_cfg`: name -> (cap, object_selection, drop_fields,
    total_cap), one entry per requested candidate-kind collection (zero or
    more -- e.g. FullReco_PFPart and FullReco_PUPPIPart at once).
    `candidate_selection_pt`/`candidate_mode`/`floor_gev`/`realistic_pid`
    apply identically to every one of them (a single global block).

    Returns (record_or_None, n_events_read, n_selection_dropped, n_axis_dropped)
    -- record is None if every event in this file was dropped.
    """
    n_events_read = len(arr)
    selection_mask = (_evaluate_event_selection(arr, event_selection, n_events_read)
                       if event_selection else np.ones(n_events_read, dtype=bool))
    n_selection_dropped = int((~selection_mask).sum())

    cands_by_name = {
        name: gather_and_select_puppi_candidates(
            arr, selection_pt=candidate_selection_pt, mode=candidate_mode, cap=cap, floor_gev=floor_gev,
            object_selection=object_selection, drop_fields=drop_fields, realistic_pid=realistic_pid,
            total_cap=total_cap, prefix=name)
        for name, (cap, object_selection, drop_fields, total_cap) in candidate_collections_cfg.items()
    }
    others = {name: gather_collection(arr, name, COLLECTION_REGISTRY[name], cap, n_events_read,
                                       object_selection=object_selection, drop_fields=drop_fields)
              for name, (cap, object_selection, drop_fields, _total_cap) in other_collections_cfg.items()}

    # Drop events failing event_selection, and/or missing real content on
    # either requested axis -- see _drop_events_with_empty_axis's docstring.
    # source_row (below) must stay the ORIGINAL row index into the source
    # file (not a post-drop renumbering), so provenance still resolves
    # correctly.
    axis_keep_mask, _ = _drop_events_with_empty_axis(cands_by_name, others, n_events_read)
    keep_mask = selection_mask & axis_keep_mask
    n_axis_dropped = int((selection_mask & ~axis_keep_mask).sum())

    source_rows = np.arange(n_events_read)
    n_dropped = n_events_read - int(keep_mask.sum())
    if n_dropped:
        cands_by_name = {name: c[keep_mask] for name, c in cands_by_name.items()}
        others = {name: coll[keep_mask] for name, coll in others.items()}
        source_rows = source_rows[keep_mask]
    n_events = len(source_rows)
    if n_events == 0:
        return None, n_events_read, n_selection_dropped, n_axis_dropped

    record_fields = dict(others)
    record_fields.update(cands_by_name)
    record_fields["label"] = ak.values_astype(ak.Array(np.full(n_events, label)), np.int8)
    record_fields["source_file"] = ak.values_astype(ak.Array(np.full(n_events, source_file_idx)), np.int32)
    record_fields["source_row"] = ak.values_astype(ak.Array(source_rows), np.int16)
    record = ak.zip(record_fields, depth_limit=1)
    return record, n_events_read, n_selection_dropped, n_axis_dropped


def convert_collide2v_regionized(cfg: DataConfig, overwrite: bool = False, resume: bool = False) -> None:
    """EOS foundational-model-dataset -> per-sample parquet training files.
    See docs/central_dataset_preprocessing.md for the original fixed-recipe
    design (region geometry, candidate selection, dataset_version filtering,
    the empty-axis event filter) this module started from, and
    docs/challenge_dataconfig.md for the full reference on every config key
    below -- every one of them is optional and defaults to that original
    recipe's exact behavior when omitted.

    Config (`data_processing:`):
      sample_dir / redir / out_path / dataset_version / max_files_per_sample /
      max_events_per_file / report_diagnostics: unchanged from the original
                   design -- see docs/challenge_dataconfig.md.
      samples:     list of EOS subdirectory names to convert (plain strings,
                   using max_files_per_sample, or {name, max_files/files/
                   target_events/label/event_selection} dicts for per-sample
                   overrides).
        target_events: alternative to max_files/files. Keeps reading files
                   (in dataset_version-filtered discovery order) until this
                   sample's running kept-event count is >= target_events
                   (whole-file granularity, overshoot allowed -- never splits
                   a file to hit the target exactly); logs a warning if the
                   sample's files run out first. Ignored for a `files:`-pinned
                   entry (pinning means "use exactly this list").
        label:     optional override -- explicit integer label for this
                   sample, taking priority over the default
                   regionize.label_for_sample(name) lookup. Challenges with
                   their own arbitrary process list (not the shared
                   regionize.SAMPLE_LABELS scheme) should set this explicitly
                   for every sample.
        event_selection: optional -- a list of {collection, field, op, value,
                   reduce} cuts (ALL must pass -- AND semantics), deciding
                   whether to keep this sample's events at all. `field`
                   defaults to the collection's first registry field;
                   `reduce` is required for a variable-count collection
                   (count/any/all/leading/max/min -- see
                   _evaluate_event_selection_cut) and defaults to "scalar"
                   (exactly one value/event) for a fixed_scalar collection.
                   Evaluated on RAW fields, independent of `collections:` --
                   e.g. `[{collection: FullReco_ScalarHT, field: HT,
                   op: '>', value: 240}]` keeps only events with
                   FullReco_ScalarHT.HT > 240, whether or not
                   FullReco_ScalarHT itself is in this run's `collections:`.
                   Composes with (is independent of) the unconditional
                   empty-axis filter below.
      collections: optional -- {collection_name: cap_or_null_or_dict, ...}
                   from COLLECTION_REGISTRY (L1T_*/FullReco_*/Gen_*/Vertex_*/
                   Event_*). Omit entirely for the original default (every
                   L1T_* collection except L1T_PFPart, L1T_PUPPIPart at
                   CANDIDATES_PER_REGION). A value is either a plain cap
                   (int, or null for no cap) or a dict for finer control:
                     cap: as above
                     object_selection: [{field, op, value}, ...] -- keeps
                       only individual objects (not whole events) passing
                       ALL cuts, evaluated on this collection's own RAW
                       fields, before ranking/capping (variable_object only).
                     drop_fields: [field, ...] -- output field names to
                       omit entirely (still read, in case object_selection
                       needs one of them).
                     total_cap: candidate-kind collections only
                       (L1T_PUPPIPart/FullReco_PUPPIPart/FullReco_PFPart --
                       zero or more may be requested at once, each with its
                       own total_cap) -- a secondary flat ceiling on the
                       FLATTENED per-event candidate count for THAT
                       collection, applied after the primary region/
                       flat_topn/none selection (see gather_and_select_puppi_candidates's
                       identical parameter for why this differs from `cap`
                       under mode: region).
                   A capped variable_object collection truncates its
                   (post-object_selection) survivors to the top-N/event by
                   its registry rank_field (PT, or SumPT2 for Vertex)
                   descending; a fixed_scalar collection's cap/object_selection
                   is ignored (drop_fields still applies).
      candidate_selection: optional -- {mode, pt, floor_gev, realistic_pid}.
                   A SINGLE global block applied identically to EVERY
                   candidate-kind collection requested in `collections:`
                   (zero, one, or several at once -- e.g. FullReco_PFPart
                   and FullReco_PUPPIPart together). `pt` ("weighted"/"raw"/
                   "none") falls back to the legacy top-level
                   `candidate_selection_pt` key, then "weighted", if omitted
                   -- see gather_and_select_puppi_candidates's identical
                   `selection_pt` parameter. `mode` ("region" default, or
                   "flat_topn"), `floor_gev` (default CANDIDATE_PT_FLOOR_GEV),
                   and `realistic_pid` (default True) -- see that function's
                   identically-named parameters.
      split:       optional -- {train_frac, eval_frac, seed}. Omit entirely
                   for the original single-output-per-sample behavior. When
                   set, every event is randomly assigned
                   (np.random.default_rng(seed), per sample) to train or
                   eval and written under separate `train/`/`eval/`
                   top-level directories.
      flush_every_events: optional (default 2_000_000) -- write accumulated
                   events as a new parquet fragment file once a split's (or
                   the whole sample's, if no split) buffered row count
                   reaches this threshold, instead of holding the entire
                   sample in memory before one final write. Bounds memory
                   regardless of how large `target_events` is; a sample
                   under this threshold still produces exactly one fragment.

    Unconditional (not config-gated): every event missing real content on
    EITHER requested axis (see _drop_events_with_empty_axis) is dropped
    before being written, a real data-quality fix, not an ablation knob.
    Logged per-file and as a per-sample total regardless of
    report_diagnostics.

    Output layout:
      no `split`:  <out_path>/<sample>/<sample>_NNNNN.parquet fragment(s),
                   plus <out_path>/<sample>/<sample>_source_files.txt.
      `split` set: <out_path>/train/<sample>/<sample>_NNNNN.parquet and
                   <out_path>/eval/<sample>/<sample>_NNNNN.parquet, plus one
                   shared <out_path>/<sample>_source_files.txt (both splits'
                   `source_file` values index into this same list).
    NOTE: the fragment-file layout (`_NNNNN` suffix) applies even in the
    fully-default/no-split case -- the one user-visible behavior change from
    before this config system existed, where a sample always produced
    exactly one file named `<sample>.parquet`.

    `resume`: skip a sample ENTIRELY (no re-discovery, no re-reading, no
    touching existing output) if its `<sample>_source_files.txt` already
    exists -- that file is written only as the very last step of a sample
    that finished successfully (see the end of the per-sample loop below),
    so its presence is a reliable "this sample is done" signal regardless
    of how many fragments were flushed along the way. A sample interrupted
    mid-run (crash, OOM, ...) never gets that file written, so it's treated
    as incomplete and re-run from scratch on the next attempt -- same as
    today, via `overwrite`'s existing wipe-and-redo behavior for that one
    sample (partial fragments from an interrupted run can't be trusted or
    cheaply resumed mid-sample, e.g. the train/eval split assignment is
    randomized per event and isn't reproducible file-by-file). Meant for a
    restart after a real crash (this session hit several -- OOM, a bad
    remote file) where re-doing every already-completed sample from sample
    #1 every time wastes real hours; `overwrite=True` alone does that.
    False by default -- existing behavior (always wipe-and-redo every
    sample) is unchanged unless a caller opts in.
    """
    sample_dir = cfg.get_sample_dir().rstrip("/")
    redir = cfg.dp("redir", "")
    out_base = Path(cfg.dp("out_path", "./")).expanduser()
    samples = cfg.dp("samples", None)
    if not samples:
        raise ValueError("data_processing.samples (list of EOS subdirectory names) is required.")
    default_max_files = cfg.dp("max_files_per_sample", -1)
    max_events_per_file = cfg.dp("max_events_per_file", -1)
    dataset_version = cfg.dp("dataset_version", "collide2v_v1.0")

    cs_cfg = _resolve_candidate_selection_cfg(cfg)
    candidate_selection_pt, candidate_mode = cs_cfg["pt"], cs_cfg["mode"]
    floor_gev, realistic_pid = cs_cfg["floor_gev"], cs_cfg["realistic_pid"]
    report_diagnostics = cs_cfg["report_diagnostics"]

    coll_cfg = _resolve_collections_cfg(cfg)
    candidate_collections_cfg = coll_cfg["candidate_collections_cfg"]
    other_collections_cfg = coll_cfg["other_collections_cfg"]
    candidate_columns = coll_cfg["candidate_columns"]
    other_columns = coll_cfg["other_columns"]
    if report_diagnostics and len(candidate_collections_cfg) != 1:
        raise ValueError("data_processing.report_diagnostics requires exactly one candidate-kind "
                          f"collection in `collections:` -- got {sorted(candidate_collections_cfg)}.")

    split_cfg = cfg.dp("split", None)
    train_frac = split_seed = None
    if split_cfg is not None:
        train_frac = split_cfg.get("train_frac", 0.9)
        eval_frac = split_cfg.get("eval_frac", 1.0 - train_frac)
        if abs(train_frac + eval_frac - 1.0) > 1e-6:
            raise ValueError(f"data_processing.split: train_frac ({train_frac}) + eval_frac ({eval_frac}) "
                              f"must sum to 1.0")
        split_seed = split_cfg.get("seed", 0)

    flush_every_events = cfg.dp("flush_every_events", 2_000_000)

    for entry in samples:
        se = _resolve_sample_entry(entry, default_max_files)
        sample, max_files, explicit_files = se["sample"], se["max_files"], se["explicit_files"]
        target_events, label, event_selection = se["target_events"], se["label"], se["event_selection"]

        source_files_path = (out_base / f"{sample}_source_files.txt" if split_cfg
                              else out_base / sample / f"{sample}_source_files.txt")
        if resume and source_files_path.exists():
            logger.info(f"[{sample}] resume: {source_files_path} already exists (sample finished on a prior "
                        f"run) -- skipping entirely, not touching existing output.")
            continue

        def _resolve_src(fname, sample_path=f"{sample_dir}/{sample}"):
            return join_remote(redir, f"{sample_path}/{fname}") if redir else os.path.join(sample_dir, sample, fname)

        candidate_file_names = _discover_sample_files(sample_dir, redir, sample, explicit_files)
        required_columns = list(dict.fromkeys(candidate_columns + _event_selection_columns(event_selection)))

        split_dirs = ({"train": out_base / "train" / sample, "eval": out_base / "eval" / sample} if split_cfg
                       else {"": out_base / sample})
        _prepare_output_dirs(split_dirs, overwrite)

        rng = np.random.default_rng(split_seed) if split_cfg else None
        buffers = {key: [] for key in split_dirs}
        buffer_counts = {key: 0 for key in split_dirs}
        fragment_idx = {key: 0 for key in split_dirs}

        used_file_names = []
        n_wrong_version = 0
        n_unreadable = 0
        total_kept_events = 0
        total_empty_axis_dropped = 0
        total_selection_dropped = 0
        diag_totals = {"n_events": 0, "out_of_acceptance": 0, "below_floor": 0, "rank_truncated": 0,
                        "kept": 0, "region_truncated_counts": np.zeros(N_REGIONS, dtype=np.int64)}

        for fname in candidate_file_names:
            if explicit_files is None and target_events is None and max_files > 0 and len(used_file_names) >= max_files:
                break

            src = _resolve_src(fname)
            try:
                version = _file_dataset_version(src)
            except OSError as e:
                if explicit_files is not None:
                    raise
                n_unreadable += 1
                logger.warning(f"[{sample}] {fname}: server error reading file (not this pipeline's bug -- "
                                f"e.g. a transient/corrupt remote file) -- skipping file. {e}")
                continue
            if version != dataset_version.encode():
                if explicit_files is not None:
                    raise ValueError(f"[{sample}] pinned file {fname!r} has dataset_version={version!r} != "
                                      f"{dataset_version!r} -- pinned file lists must all match exactly, not "
                                      f"silently drop mismatches.")
                n_wrong_version += 1
                logger.info(f"[{sample}] {fname}: dataset_version={version!r} != {dataset_version!r} -- skipping file.")
                continue

            try:
                arr = _read_parquet_tolerant(src, required_columns, other_columns, max_events_per_file)
            except OSError as e:
                if explicit_files is not None:
                    raise
                n_unreadable += 1
                logger.warning(f"[{sample}] {fname}: server error reading file (not this pipeline's bug -- "
                                f"e.g. a transient/corrupt remote file) -- skipping file. {e}")
                continue

            source_file_idx = len(used_file_names)
            used_file_names.append(fname)

            if len(arr) == 0:
                continue

            record, n_events_read, n_selection_dropped, n_axis_dropped = _build_record_for_file(
                arr, candidate_selection_pt=candidate_selection_pt,
                candidate_mode=candidate_mode, floor_gev=floor_gev,
                realistic_pid=realistic_pid, candidate_collections_cfg=candidate_collections_cfg,
                other_collections_cfg=other_collections_cfg, event_selection=event_selection,
                label=label, source_file_idx=source_file_idx)

            if report_diagnostics:
                # Validated single-candidate-collection above -- safe to
                # take the one entry here.
                (diag_prefix, (diag_cap, _, _, _)), = candidate_collections_cfg.items()
                d = diagnose_puppi_selection(arr, selection_pt=candidate_selection_pt, cap=diag_cap,
                                              floor_gev=floor_gev, prefix=diag_prefix)
                for k in ("n_events", "out_of_acceptance", "below_floor", "rank_truncated", "kept"):
                    diag_totals[k] += d[k]
                diag_totals["region_truncated_counts"] += d["region_truncated_counts"]

            total_selection_dropped += n_selection_dropped
            total_empty_axis_dropped += n_axis_dropped
            if record is None:
                logger.warning(f"[{sample}] {fname}: all {n_events_read} events dropped (event_selection "
                                f"and/or empty-axis filter) -- skipping file.")
                continue
            n_events = len(record)

            if split_cfg:
                assign_train = rng.random(n_events) < train_frac
                _append_and_maybe_flush(buffers, buffer_counts, fragment_idx, split_dirs, sample, "train",
                                         record[assign_train], flush_every_events)
                _append_and_maybe_flush(buffers, buffer_counts, fragment_idx, split_dirs, sample, "eval",
                                         record[~assign_train], flush_every_events)
            else:
                _append_and_maybe_flush(buffers, buffer_counts, fragment_idx, split_dirs, sample, "",
                                         record, flush_every_events)

            total_kept_events += n_events
            logger.info(f"[{sample}] {fname}: {n_events} events kept, {n_selection_dropped} dropped by "
                        f"event_selection, {n_axis_dropped} dropped by the empty-axis filter "
                        f"(source file {source_file_idx + 1})")

            if explicit_files is None and target_events is not None and total_kept_events >= target_events:
                break

        if not used_file_names or total_kept_events == 0:
            logger.warning(f"No events converted for sample {sample!r} -- skipping output.")
            continue
        if target_events is not None and total_kept_events < target_events:
            logger.warning(f"[{sample}] ran out of files before reaching target_events={target_events} "
                            f"(got {total_kept_events}).")

        for key in split_dirs:
            _flush_buffer(buffers, buffer_counts, fragment_idx, split_dirs, sample, key)

        with open(source_files_path, "w") as fh:
            fh.write("\n".join(used_file_names))
        n_fragments = sum(fragment_idx.values())
        logger.info(f"[{sample}] wrote {total_kept_events} events across {n_fragments} fragment(s), "
                    f"{len(used_file_names)} source files ({n_wrong_version} file(s) skipped for wrong "
                    f"dataset_version, {n_unreadable} file(s) skipped for a server/read error), "
                    f"label={label} -> {list(split_dirs.values())} "
                    f"({total_selection_dropped} dropped by event_selection, {total_empty_axis_dropped} "
                    f"dropped by the empty-axis filter)")
        if report_diagnostics:
            log_diagnostics_summary(sample, diag_totals)
