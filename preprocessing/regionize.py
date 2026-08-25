"""
PFL1 candidate regionization: the real CMS L1T Correlator geometry (90
regions: 9 equal-width phi bins x 10 non-uniform eta bins, confirmed against
real hardware -- see docs/central_dataset_preprocessing.md) plus the
per-region top-K-by-pT selection used to build the central preprocessed
dataset's PUPPI candidate collection (see convert_collide2v_regionized() in
converters.py).

Standalone extraction of AIDA-Scout's src/aida_scout/data/regionize.py
(github.com/AIDA-Scout/aidascoutrepo, commit c145ce6) -- identical except
that AE_SPLIT_SLOTS/take_ae_split_slots (the 2-AE DisCo baseline's TRAINING
object-tensor slot layout, the only torch-dependent piece of that file) are
dropped, since they're irrelevant to parquet preprocessing and their absence
means this package needs no torch dependency at all. Everything else below
is byte-for-byte identical to the source.

Design, resolved over a design conversation (session history) rather than
guessed:
  - Region assignment is purely geometric (eta, phi) -- a candidate outside
    the eta range covered by ETA_EDGES has no region at all (region_id -1),
    which is itself the eta acceptance boundary. No separate/tighter eta cut
    is applied on top of this.
  - Selection is done on PUPPI-weighted pT (pt_raw * PuppiW), not raw pT --
    ranking by raw pT within a region risks losing genuinely relevant
    candidates to a handful of high-raw-pT, near-zero-weight (pileup)
    candidates that fill the fixed per-region budget. Confirmed on real data
    (tttt_incl): raw-pT-only selection would silently drop ~1.8% of the
    candidates a weighted-pT selection keeps.
  - The pT floor (CANDIDATE_PT_FLOOR_GEV) is applied BEFORE ranking/
    truncating within each region, not after -- otherwise the fixed 18-slot
    budget can be wasted on sub-threshold candidates.
  - The final flattened candidate collection is presented sorted by RAW pT
    (not weighted pT): raw pT is the primary feature the model sees,
    weighted pT and the weight itself are additional columns, not a
    replacement for raw pT (see PUPPI_CAND_OUTPUT_COLUMNS in converters.py).
"""
from typing import Dict

import numpy as np

# 9 equal-width phi bins (each 2*pi/9 wide) x 10 non-uniform eta bins --
# confirmed real PFL1 Correlator regionization geometry.
PHI_BINS = 9
ETA_EDGES = np.array([-3.0, -2.5, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.5, 3.0])
N_ETA_BINS = len(ETA_EDGES) - 1  # 10
N_REGIONS = PHI_BINS * N_ETA_BINS  # 90

CANDIDATES_PER_REGION = 18  # matches real PFL1 hardware (not the earlier 16-candidate test)
CANDIDATE_PT_FLOOR_GEV = 1.0  # flat everywhere -- no region-dependent (Barrel/HGCal/HF) thresholds


def assign_region(eta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Flat numpy arrays in -> flat int64 region ids (0..89), or -1 for any
    candidate outside |eta| <= 3.0 (no region covers it -- this IS the eta
    acceptance, not a separate cut). `phi` is assumed in (-pi, pi].

    Acceptance is the CLOSED interval [-3.0, 3.0] -- both endpoints included
    (eta can be exactly +3.0 or -3.0). Every OTHER (internal) bin edge keeps
    its existing half-open [low, high) convention unchanged -- e.g. eta
    exactly -2.5 still falls in the bin starting at -2.5, not the one ending
    there -- this only closes the very top of the whole range, which
    `side="right"` alone would otherwise leave open (eta==-3.0, the very
    bottom, already falls in bin 0 correctly via that same rule; only the
    symmetric top case needed the explicit fix)."""
    eta_bin = np.searchsorted(ETA_EDGES, eta, side="right") - 1
    eta_bin = np.where(eta == ETA_EDGES[-1], N_ETA_BINS - 1, eta_bin)
    eta_ok = (eta_bin >= 0) & (eta_bin < N_ETA_BINS)
    phi_bin = np.clip(np.floor((phi + np.pi) / (2 * np.pi / PHI_BINS)).astype(np.int64), 0, PHI_BINS - 1)
    return np.where(eta_ok, eta_bin * PHI_BINS + phi_bin, -1)


def select_top_per_region(event_idx: np.ndarray, region_id: np.ndarray, pt_rank: np.ndarray,
                           above_floor: np.ndarray, n_events: int,
                           cap: int = CANDIDATES_PER_REGION) -> np.ndarray:
    """Per (event, region) group: keep candidates with above_floor=True,
    ranked by pt_rank descending, truncated to the top `cap`. All inputs are
    flat arrays of equal length (one row per candidate, across all events).

    Returns a boolean survive-mask of the same length as the inputs. Floor
    is applied before ranking/truncation (not after) so the fixed per-region
    budget is never spent on sub-threshold candidates.
    """
    keep = above_floor & (region_id >= 0)
    ev, reg, pt = event_idx[keep], region_id[keep], pt_rank[keep]
    if len(ev) == 0:
        return np.zeros(len(event_idx), dtype=bool)

    group_key = ev.astype(np.int64) * N_REGIONS + reg
    order = np.lexsort((-pt, group_key))
    group_sorted = group_key[order]
    change = np.empty(len(group_sorted), dtype=bool)
    change[0] = True
    change[1:] = group_sorted[1:] != group_sorted[:-1]
    group_start = np.where(change)[0]
    rank = np.arange(len(group_sorted)) - np.repeat(group_start, np.diff(np.append(group_start, len(group_sorted))))

    survives_sorted = rank < cap
    survives_in_kept_order = np.zeros(len(pt), dtype=bool)
    survives_in_kept_order[order] = survives_sorted

    out = np.zeros(len(event_idx), dtype=bool)
    out[np.where(keep)[0]] = survives_in_kept_order
    return out


# Canonical EOS directory -> fixed integer label, baked in at conversion
# time for every sample (backgrounds AND signals). Anything not listed here
# falls through to Other=5 -- deliberately only the ~10 canonical-class
# directories are enumerated explicitly, not all 53 (see docs/
# eos_dataset_schema.md's composition tables for the full directory list).
# WJets is WJetsToLNu specifically (leptonic) -- WJetsToQQ is a different
# sample and falls to Other, not WJets, unless you edit SAMPLE_LABELS below
# for your own project's class scheme.
QCD, MINBIAS, TT, WJETS, DY, OTHER = 0, 1, 2, 3, 4, 5

CLASS_NAMES: Dict[int, str] = {QCD: "QCD", MINBIAS: "MinBias", TT: "TT", WJETS: "WJets", DY: "DY", OTHER: "Other"}
N_BKG_CLASSES = 5

# Unsigned, 5-bucket realistic pdgId scheme this pipeline commits to (see
# converters._realistic_pdgid).
PDGID_BUCKETS = [0, 11, 13, 22, 211]

# Axis-1 AE object-collection layout this dataset's own AE_OBJ_SOURCE_COLUMNS
# defaults to -- only meaningful if you go on to build a training tensor
# stage of your own on top of this parquet output; harmless if you don't.
AE_OBJ_COLLECTIONS = {"L1T_JetAK4": 10, "L1T_MuonTight": 4, "L1T_Electron": 4, "L1T_PhotonTight": 4, "L1T_MET": 1}

# Every L1T_* collection this pipeline's object-view gatherer knows how to
# read -- same caveat as AE_OBJ_COLLECTIONS above.
AE_ELIGIBLE_COLLECTIONS = {
    "L1T_JetAK4", "L1T_JetAK8", "L1T_JetPuppiAK4", "L1T_JetPuppiAK8",
    "L1T_MuonTight", "L1T_Electron", "L1T_PhotonTight", "L1T_MET",
}

SAMPLE_LABELS: Dict[str, int] = {
    "QCD_HT50toInf": QCD,
    "QCD_HT50tobb": QCD,
    "minbias": MINBIAS,
    "tt0123j_5f_ckm_LO_MLM_hadronic": TT,
    "tt0123j_5f_ckm_LO_MLM_leptonic": TT,
    "tt0123j_5f_ckm_LO_MLM_semiLeptonic": TT,
    "WJetsToLNu_13TeV-madgraphMLM-pythia8": WJETS,
    "DYJetsToLL_13TeV-madgraphMLM-pythia8": DY,
}


def label_for_sample(eos_dirname: str) -> int:
    """EOS sample directory name -> fixed integer label (0-5). Matches on
    the directory name with any trailing `-<batch suffix>` stripped (the two
    extra `tt0123j_5f_ckm_LO_MLM_hadronic-<id>` single-file directories
    noted in docs/eos_dataset_schema.md fold into the same TT label as the
    main hadronic directory this way). Unrecognized names -> Other=5,
    covering every signal and every non-canonical background.

    Edit SAMPLE_LABELS above for your own project's set of canonical
    classes -- everything else in this file is independent of that choice.
    """
    base = eos_dirname.split("-")[0] if eos_dirname.startswith("tt0123j_5f_ckm_LO_MLM_hadronic-") else eos_dirname
    return SAMPLE_LABELS.get(base, OTHER)
