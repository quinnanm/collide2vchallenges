# EOS foundational-model-dataset: parquet schema reference

Source: `/eos/project/f/foundational-model-dataset/samples/production_final` (accessed via
`root://eosproject-f.cern.ch/` + grid proxy). 271 columns total, grouped into
five top-level branch prefixes: `L1T_*` (trigger-level candidates), `FullReco_*`
(offline/full reconstruction), `Gen_*` (generator-level truth), `Vertex_*`
(reconstructed vertices), `Event_*` (per-event MC metadata).

**How "size" was measured**: schema type inspected directly (fixed-size list vs.
variable-length list) via `pyarrow`. For `L1T_*` variable-length collections
(the ones the converter actually reads), the min/max per-event object count was
sampled from a physics sample **chosen to be expected-high** for that specific
object type, not an arbitrary/average sample — since the goal was to find the
practical maximum a collection can hold, not a typical value. Sample used per
collection is noted on its heading (`ZZ_leptonic` for leptons: Z→ℓℓ ×2 gives up
to 4 prompt leptons plus occasional extras; `tri_gamma` for photons: designed to
have 3 photons/event, which the scan confirms exactly; `QCD_HT50toInf` for jets:
highest raw jet multiplicity in the dataset; `tt̄→hadronic` for AK8: a
boosted-top candidate, needed since no AK8 jets appeared at all in a small
generic sample). **Means are not reported — they wouldn't mean anything for a
"what's the max capacity" question**, and are physics-sample-dependent besides.
For `FullReco_*`/`Gen_*`/`Vertex_*`/`Event_*` sections below, ranges are still
from the original single-file 500-event DY sample and have not been re-scanned
this way — treat those as illustrative, not maxima. **Fixed-size columns
(`L1T_PUPPIPart`/`L1T_PFPart` at 1000, `L1T_Rho` at 5, scalars at 1) are
structural** and don't need this treatment.

---

## Overall dataset composition: samples, events, size on EOS

`production_final` contains **53 top-level sample directories** (each a distinct
physics process/decay channel), split here into **29 backgrounds** (SM
processes with large cross-sections: QCD, min-bias, V+jets, diboson/triboson,
tt̄) and **24 signals** (rarer EW/Higgs/top-associated processes this dataset
targets: HH, VBF/ggF Higgs by decay channel, VH, ttH/ttW/ttZ, four-top). This
is the full available dataset — the `collide2vsubset` sample used in training
so far is a ~1.64M-event slice of just 4 of the 29 background directories
(DY, minbias, tt̄, WJetsToLNu).

**Methodology**: file count and total size are exact (summed directly from the
real EOS directory listing). Event counts are **estimated**: 5 files per
sample were opened and their real parquet row counts read (not the `NEVENT`
tag in the filename, which is a nominal target, not the actual count), then
averaged and multiplied by the sample's total file count. Sizes use decimal
GB/TB (1 TB = 10¹² bytes).

### Backgrounds (29 samples, ~83.4k files total, ~62.4k files/~123.0 TB/~582M events for this table)

| Signature type | EOS directory | Files | Est. events | Size |
|---|---|---:|---:|---:|
| DY (Drell-Yan) | `DYJetsToLL_13TeV-madgraphMLM-pythia8` | 2,348 | 23.5M | 4.59 TB |
| MinBias | `minbias` | 10,500 | 105.0M | 21.57 TB |
| QCD (HT50toInf) | `QCD_HT50toInf` | 13,126 | 107.6M | 21.74 TB |
| QCD (HT50tobb) | `QCD_HT50tobb` | 2,888 | 25.7M | 5.52 TB |
| W+jets → ℓν | `WJetsToLNu_13TeV-madgraphMLM-pythia8` | 2,625 | 26.2M | 5.49 TB |
| W+jets → qq̄ | `WJetsToQQ_13TeV-madgraphMLM-pythia8` | 2,625 | 26.2M | 5.55 TB |
| Z+jets → qq̄ | `ZJetsToQQ_13TeV-madgraphMLM-pythia8` | 2,625 | 26.2M | 5.55 TB |
| Z+jets → bb̄ | `ZJetsTobb_13TeV-madgraphMLM-pythia8` | 2,625 | 26.2M | 5.57 TB |
| Z+jets → cc̄ | `ZJetsTocc_13TeV-madgraphMLM-pythia8` | 2,625 | 26.2M | 5.57 TB |
| Z+jets → νν̄ | `ZJetsTovv_13TeV-madgraphMLM-pythia8` | 2,625 | 26.1M | 5.53 TB |
| WW → hadronic | `WW_hadronic` | 326 | 3.2M | 701.5 GB |
| WW → leptonic | `WW_leptonic` | 326 | 3.2M | 689.3 GB |
| WW → semileptonic | `WW_semileptonic` | 326 | 3.2M | 695.0 GB |
| WZ → hadronic | `WZ_hadronic` | 315 | 3.1M | 667.3 GB |
| WZ → leptonic | `WZ_leptonic` | 315 | 3.1M | 668.2 GB |
| WZ → semileptonic | `WZ_semileptonic` | 326 | 3.2M | 697.1 GB |
| ZZ → hadronic | `ZZ_hadronic` | 315 | 3.1M | 684.8 GB |
| ZZ → leptonic | `ZZ_leptonic` | 315 | 3.1M | 670.8 GB |
| ZZ → semileptonic | `ZZ_semileptonic` | 315 | 3.1M | 676.0 GB |
| VVV (triboson, incl.) | `VVV_incl` | 210 | 2.1M | 407.8 GB |
| γ + jets | `gamma` | 2,625 | 17.8M | 4.05 TB |
| γ + V | `gamma_V` | 1,050 | 7.7M | 2.08 TB |
| γγγ (tri-photon) | `tri_gamma` | 210 | 0.24M | 68.8 GB |
| Υ → leptons | `upsilon_to_leptons` | 2,625 | 26.25M | 5.64 TB |
| tt̄ → hadronic\* | `tt0123j_5f_ckm_LO_MLM_hadronic` (+2 extra single-file batches) | 2,730 | 26.8M | 6.04 TB |
| tt̄ → leptonic | `tt0123j_5f_ckm_LO_MLM_leptonic` | 2,730 | 26.8M | 5.88 TB |
| tt̄ → semileptonic | `tt0123j_5f_ckm_LO_MLM_semiLeptonic` | 2,730 | 26.8M | 5.96 TB |

\* `tt0123j_5f_ckm_LO_MLM_hadronic-10000-28002196` and `...-28002300` are two
extra single-file directories (1 file, ~9.8k events each) that appear to be
late-added/reprocessed batches of the same hadronic tt̄ sample — folded into
the row above (file/event/size totals include them) rather than listed
separately.

**Background totals**: ~62,401 files, ~123.0 TB, ~581.7M events.

### Signals (24 samples, ~21.0k files, ~42.0 TB, ~191.2M events)

| Signature type | EOS directory | Files | Est. events | Size |
|---|---|---:|---:|---:|
| HH → 4b | `HH_4b` | 210 | 2.10M | 467.3 GB |
| HH → bbWW | `HH_bbWW` | 231 | 2.07M | 459.1 GB |
| HH → bbZZ | `HH_bbZZ` | 231 | 2.07M | 461.1 GB |
| HH → bbγγ | `HH_bbgammagamma` | 210 | 2.10M | 458.4 GB |
| HH → bbττ | `HH_bbtautau` | 210 | 2.10M | 460.8 GB |
| VBF H → WW | `VBFHWW` | 1,313 | 10.29M | 2.22 TB |
| VBF H → ZZ | `VBFHZZ` | 1,313 | 10.22M | 2.23 TB |
| VBF H → bb̄ | `VBFHbb` | 1,208 | 10.49M | 2.27 TB |
| VBF H → cc̄ | `VBFHcc` | 1,418 | 10.84M | 2.34 TB |
| VBF H → γγ | `VBFHgammagamma` | 1,292 | 10.13M | 2.16 TB |
| VBF H → gg | `VBFHgluglu` | 1,155 | 10.72M | 2.34 TB |
| VBF H → ττ | `VBFHtautau` | 1,365 | 10.32M | 2.25 TB |
| VH (incl.) | `VH_incl` | 1,050 | 10.44M | 2.27 TB |
| ggF H → WW | `ggHWW` | 1,050 | 10.49M | 2.31 TB |
| ggF H → ZZ | `ggHZZ` | 1,050 | 10.49M | 2.33 TB |
| ggF H → bb̄ | `ggHbb` | 1,050 | 10.50M | 2.32 TB |
| ggF H → cc̄ | `ggHcc` | 1,050 | 10.49M | 2.31 TB |
| ggF H → γγ | `ggHgammagamma` | 1,050 | 10.49M | 2.25 TB |
| ggF H → gg | `ggHgluglu` | 1,050 | 10.49M | 2.34 TB |
| ggF H → ττ | `ggHtautau` | 1,050 | 10.49M | 2.29 TB |
| tt̄H (incl.) | `ttH_incl` | 1,103 | 10.74M | 2.44 TB |
| tt̄W (incl.) | `ttW_incl` | 578 | 5.45M | 1.21 TB |
| tt̄Z (incl.) | `ttZ_incl` | 579 | 5.59M | 1.26 TB |
| Four-top (tttt, incl.) | `tttt_incl` | 210 | 2.10M | 504.5 GB |

**Signal totals**: ~21,026 files, ~42.0 TB, ~191.2M events.

**Grand total (backgrounds + signals)**: ~83,427 files, ~165.0 TB, ~772.9M events.

---

## L1T_* — trigger-level candidates (what the converter currently reads)

### L1T_PUPPIPart — size: **1000 (fixed)**
| Variable | dtype |
|---|---|
| fUniqueID | uint32 |
| PID | int32 |
| Charge | int8 |
| PT | float16 |
| Eta | float16 |
| Phi | float16 |
| E | float16 |
| Mass | float32 |
| D0 | float32 |
| DZ | float32 |
| ErrorD0 | float32 |
| ErrorDZ | float32 |
| IsPU | int8 |
| IsRecoPU | int32 |
| PuppiW | float16 |

### L1T_PFPart — size: **1000 (fixed)**
Same 15 fields/dtypes as `L1T_PUPPIPart` above.

### L1T_Electron — size: variable, 0–5 observed (`ZZ_leptonic`, ~19.9k events scanned)
| Variable | dtype |
|---|---|
| Charge | int8 |
| D0 | float32 |
| DZ | float32 |
| Eta | float16 |
| Phi | float16 |
| PT | float16 |
| ErrorD0 | float32 |
| ErrorDZ | float32 |
| EhadOverEem | float32 |
| IsolationVar | float32 |
| IsolationVarRhoCorr | float32 |

### L1T_MuonTight — size: variable, 0–5 observed (`ZZ_leptonic`, ~19.9k events scanned)
| Variable | dtype |
|---|---|
| Charge | int8 |
| D0 | float32 |
| DZ | float32 |
| Eta | float16 |
| Phi | float16 |
| PT | float16 |
| ErrorD0 | float32 |
| ErrorDZ | float32 |
| IsolationVar | float32 |
| IsolationVarRhoCorr | float32 |

### L1T_PhotonTight — size: variable, 0–3 observed (`tri_gamma`, ~2.0k events scanned)
| Variable | dtype |
|---|---|
| Eta | float16 |
| Phi | float16 |
| PT | float16 |
| EhadOverEem | float32 |
| IsolationVar | float32 |
| IsolationVarRhoCorr | float32 |

### L1T_JetAK4 — size: variable, 1–25 observed (`QCD_HT50toInf`, ~16.4k events scanned)
| Variable | dtype |
|---|---|
| Eta | float16 |
| Phi | float16 |
| PT | float16 |
| Mass | float32 |
| Charge | int8 |
| Flavor | uint32 |
| BTag | int8 |
| BTagPhys | int8 |
| NCharged | int32 |
| NNeutrals | int32 |
| Constituents | list\<uint32\> (per-jet, jagged) |

**No `ConstituentsIdx` field** — confirmed absent from the real production
schema (checked directly against a real file's parquet column list), despite
JetAK8/JetPuppiAK4/JetPuppiAK8 below having it. Not a copy-paste omission.

### L1T_JetAK8 — size: variable, 0–5 observed (`tt0123j_5f_ckm_LO_MLM_hadronic`, ~19.6k events scanned)
Same fields as `L1T_JetAK4` **plus** `ConstituentsIdx` (list\<int16\>, per-jet,
jagged) — confirmed present here even though absent from JetAK4 itself. (The
earlier single small DY sample never saw a nonzero count for this collection
— a boosted-object sample was needed to find any.)

### L1T_JetPuppiAK4 — size: variable, 0–10 observed (`QCD_HT50toInf`, ~16.4k events scanned)
Same fields as `L1T_JetAK4` plus `ConstituentsIdx` (see `L1T_JetAK8` above).

### L1T_JetPuppiAK8 — size: variable, 0–5 observed (`tt0123j_5f_ckm_LO_MLM_hadronic`, ~19.6k events scanned)
Same fields as `L1T_JetAK4` plus `ConstituentsIdx` (see `L1T_JetAK8` above).

### L1T_MET — size: **1 (event-level scalar)**
| Variable | dtype |
|---|---|
| MET | float16 |
| Eta | float16 |
| Phi | float16 |

### L1T_PUPPIMET — size: **1 (event-level scalar)**
Same fields as `L1T_MET`.

### L1T_Rho — size: **5 (fixed)**
| Variable | dtype |
|---|---|
| Rho | float32 |

### L1T_ScalarHT — size: **1 (event-level scalar)**
| Variable | dtype |
|---|---|
| HT | float32 |

---

## FullReco_* — offline/full reconstruction (parallel objects, not L1T-emulated)

Same field names/dtypes per collection as the matching `L1T_*` collection above,
with one structurally important difference:

### FullReco_PUPPIPart / FullReco_PFPart — size: **variable, 4219–7381 observed (mean 5625.63)**
**Not pre-padded to a fixed size** the way `L1T_PUPPIPart`/`L1T_PFPart` are —
these carry the full, unpadded per-event candidate multiplicity (roughly
5–7x more candidates per event than the L1T-truncated-at-1000 collections).

The rows below use the **same targeted-sample scan** as the `L1T_*` section
above (same sample per collection, same event counts scanned) so the two are
directly comparable:

| Collection | L1T_\* max | FullReco_\* max | Sample used |
|---|---:|---:|---|
| Electron | 5 | **5** | `ZZ_leptonic`, ~19.9k events |
| MuonTight | 5 | **6** | `ZZ_leptonic`, ~19.9k events |
| PhotonTight | 3 | **2** | `tri_gamma`, ~2.0k events |
| JetAK4 | 25 | **69** | `QCD_HT50toInf`, ~16.4k events |
| JetAK8 | 5 | **6** | `tt0123j...hadronic`, ~19.6k events |
| JetPuppiAK4 | 10 | **12** | `QCD_HT50toInf`, ~16.4k events |
| JetPuppiAK8 | 5 | **5** | `tt0123j...hadronic`, ~19.6k events |

Full per-collection ranges: `FullReco_Electron` 0–5, `FullReco_MuonTight` 0–6,
`FullReco_PhotonTight` 0–2, `FullReco_JetAK4` 13–69, `FullReco_JetAK8` 0–6,
`FullReco_JetPuppiAK4` 0–12, `FullReco_JetPuppiAK8` 0–5.

**FullReco is not simply "fewer max events" than L1T** — for jets and muons it's
equal or noticeably *higher* (`FullReco_JetAK4` maxes at 69 vs. `L1T_JetAK4`'s
25 in the same QCD events — offline jet reconstruction has a lower effective pT
threshold / more inclusive clustering than the L1T emulation), consistent with
`FullReco_PUPPIPart`/`FullReco_PFPart` also carrying far more candidates than
their L1T-truncated-at-1000 counterparts. The one exception is
`PhotonTight` (L1T max 3, FullReco max 2 in the same tri-photon events) —
worth a second look if photon-reconstruction efficiency at L1T vs. offline
ever matters for a downstream analysis, since a priori you'd expect offline to
be equal-or-more inclusive there too.

### FullReco_MET / FullReco_PUPPIMET / FullReco_ScalarHT — 1 (event-level scalar)
### FullReco_Rho — 5 (fixed)

---

## Gen_* — generator-level truth (pre-detector-simulation)

### Gen_Part — size: variable, 58–2905 observed (mean 386.21)
The full generator particle record — genuine MC truth. **Do not use for
training** (would leak truth-level information no real trigger/detector can
access); useful only for offline validation/labeling.
| Variable | dtype |
|---|---|
| PID | int32 |
| Status | int8 |
| PT | float16 |
| Eta | float16 |
| Phi | float16 |
| Mass | float32 |
| M1, M2 | int32 (mother indices) |
| D1, D2 | int32 (daughter indices) |
| IsPU | int8 |

### Gen_JetAK4 — size: variable, 0–7 observed (mean 0.2)
| Variable | dtype |
|---|---|
| PT | float16 |
| Eta | float16 |
| Phi | float16 |
| Mass | float32 |

### Gen_JetAK8 — size: **0 in this sample**
Same fields as `Gen_JetAK4`.

### Gen_MissingET — size: **1 (event-level scalar)**
| Variable | dtype |
|---|---|
| MET | float16 |
| Eta | float16 |
| Phi | float16 |

---

## Vertex_* — reconstructed vertices (primary + pileup)

Size: variable, **154–253 observed (mean 201.24)** — consistent with a
high-pileup (~200 PU) production scenario. All 8 fields share this same
per-event count.

| Variable | dtype |
|---|---|
| Index | int32 |
| X, Y, Z, T | float32 |
| NDF | int32 |
| SumPT2 | double |
| Constituents | list\<uint32\> (per-vertex, jagged) |

---

## Event_* — per-event MC generation metadata

Size: **1 (event-level scalar)** for every field below.

| Variable | dtype |
|---|---|
| Number | int64 |
| ProcessID | int32 |
| Weight | float32 |
| CrossSection | float32 |
| CrossSectionError | float32 |
| Scale | float32 |
| AlphaQCD | float32 |
| AlphaQED | float32 |
| ID1, ID2 | int32 (incoming parton flavors) |
| X1, X2 | float32 (Bjorken-x) |
| PDF1, PDF2 | float32 |
| ScalePDF | float32 |

---

## Relevant to the current PID discussion

`L1T_PUPPIPart`/`L1T_PFPart` carry `PID` (raw, likely truth-level-leaky for
charged-hadron sub-flavor — see conversation) and an unused `Charge` field
(signed ±1/0, physically realistic, not currently read by `converters.py`).
No boolean `isChargedHadron`/`isNeutralHadron`/`isPhoton`/`isElectron`/`isMuon`
flags exist anywhere in the schema — `PID` (possibly collapsed) + `Charge` are
the only identity-bearing fields available for the generic candidate
collections.
