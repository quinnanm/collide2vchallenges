# Central preprocessed dataset: schema reference

This documents `/collide2vsubset/collide-1m-ptrawv1.0/` — the current production
dataset, built from `configs/data/convert_collide_1m_ptrawv1.yaml` /
`nrp/preprocess/preprocess_collide_1m_ptrawv1.yaml`. It replaces the earlier
`central_1m/` (whose raw preprocessing has since been deleted): that dataset
used weighted-pT candidate selection and included MinBias, both since found
to be problems (see "Dataset specs" and §1 below).

## Dataset specs

Backgrounds: QCD, TT (3 EOS sub-samples by decay channel), WJets, DY.
**No MinBias, no tttt_incl** — the EOS dataset's own producer flagged both
as unreliable/unphysical due to PUPPI-weight issues. Signals: 5 processes,
~50k events each — `tttt_incl` dropped from the usual 5-signal set (same
reliability warning) and replaced with `VBFHbb`.

| Sample | Class | Events |
|---|---|---:|
| `QCD_HT50toInf` | QCD (label 0) | 517,512 |
| `WJetsToLNu_13TeV-madgraphMLM-pythia8` | WJets (label 3) | 128,237 |
| `DYJetsToLL_13TeV-madgraphMLM-pythia8` | DY (label 4) | 129,876 |
| `tt0123j_5f_ckm_LO_MLM_hadronic` | TT (label 2) | 119,571 |
| `tt0123j_5f_ckm_LO_MLM_semiLeptonic` | TT (label 2) | 117,652 |
| `tt0123j_5f_ckm_LO_MLM_leptonic` | TT (label 2) | 29,426 |
| **Background total** | | **1,042,274** |
| `HH_bbgammagamma` | signal | 59,970 |
| `ggHbb` | signal | 59,966 |
| `VBFHtautau` | signal | 52,908 |
| `ttH_incl` | signal | 58,401 |
| `VBFHbb` | signal | 52,069 |

**Total on-disk size: 8.2 GB** (zstd-compressed parquet, Stage 1 only — no
Stage 2 `.pt` tensors built for this dataset yet).

Background composition is QCD:TT:WJets:DY = 4:2:1:1 by design (TT itself
split 46:44:10 hadronic:semiLeptonic:leptonic, by branching ratio) — counts
above are approximate (whole-file granularity) against those ratios, not
exact truncations. Every sample only draws from `dataset_version:
collide2v_v1.0` files (see §1), and every count is the real **post**-selection
number (after the region/floor selection and empty-axis filter below), not
a raw pre-filter count.

## 1. Preprocessing design

**Format & scope**: Parquet, one directory per sample (mirrors the EOS layout). `collide-1m-ptrawv1.0/` covers the 11 samples in "Dataset specs" above — a deliberately chosen subset, not all 53 EOS samples (the full set is documented for reference in `docs/eos_dataset_schema.md`, which this pipeline is capable of reading from but doesn't by default).

**File-level version filter**: per the dataset's own producer, this EOS dataset mixes early/testing-stage files (no `dataset_version` parquet metadata at all) with final production files (`dataset_version=b"collide2v_v1.0"`). Only files whose `dataset_version` matches `data_processing.dataset_version` (config key, default `"collide2v_v1.0"` — `collide-1m-ptrawv1.0` sets this explicitly too) are ever read — checked BEFORE `max_files`/`files:` truncation, so a file count or an explicit pinned list means "N usable files", not "N files, some possibly wrong-version". See `_file_dataset_version`/`_select_files_by_dataset_version` in `converters.py`.

**Collections**: every `L1T_*` collection except `L1T_PFPart` (redundant — identical to unweighted `L1T_PUPPIPart`, confirmed). No `Gen_*`/`FullReco_*`/`Vertex_*`/`Event_*`.

**Candidate processing (`L1T_PUPPIPart` only — single collection, no PF duplicate)**:

1. Compute `pt_weighted = pt_raw × PuppiW` (stored regardless of what drives selection — see step 3).
2. Assign each candidate to one of 90 geometry regions (9 phi × 10 eta bins) via eta/phi — this *is* the eta acceptance (|eta|<=3.0, closed interval — eta exactly ±3.0 is included), no separate eta cut.
3. Within each region, filter to `pt_raw >= 1 GeV` (`data_processing.candidate_selection_pt: raw` — this dataset's actual setting) AND `pt_raw > 0` always (the stored `pt` column, raw pT, is the exact padding-slot sentinel every downstream consumer checks, e.g. `ContrastiveModel._make_mask`'s `x[..., 0] == 0`, so a real candidate can never be allowed to have `pt_raw` exactly 0 — redundant with the floor here since it's positive, but enforced independently regardless), then rank survivors by `pt_raw` descending and keep the top 18. The pipeline's *other* selection mode, `candidate_selection_pt: weighted` (rank/floor on `pt_weighted` instead — the design `central_1m/` used, meant to protect the fixed per-region budget from high-raw-pT/low-weight pileup candidates), is still supported but is NOT what this dataset uses: real measurement this session found raw selection keeps a much larger, more inclusive candidate population (~500/event mean vs. weighted's ~17/event, one real QCD file) because most candidates have `puppi_weight` hard-zeroed by PUPPI, which weighted selection would silently discard. `pt`/`pt_weighted`/`puppi_weight` are always all three stored as separate columns regardless of which mode drove selection. A third mode, `candidate_selection_pt: none`, skips region/floor/rank selection entirely (every `pt_raw>0` candidate kept, ~1000/event) — a no-cuts baseline for comparison studies, not used here either.
4. Flatten the selected candidates across all 90 regions into one collection, **sorted by `pt_raw` descending** (raw pT is the primary presented feature) — this final ordering is the same regardless of which mode drove step 3.
5. Store three separate columns: **`pt_raw`**, **`puppi_weight`**, and **`pt_weighted`** (the already-computed product) — so the model gets both raw ingredients to learn its own combination from, and the pre-combined value is also available directly without needing to be recomputed downstream.

**Event-level filter (unconditional, not an ablation knob)**: an event only survives if it has real content on BOTH axes — at least one surviving `L1T_PUPPIPart` candidate after the above, AND at least one real object in at least one of the 7 genuinely variable-count `L1T_*` collections (`Electron`/`MuonTight`/`PhotonTight`/`JetAK4`/`JetAK8`/`JetPuppiAK4`/`JetPuppiAK8` — excludes the fixed-size scalars `MET`/`PUPPIMET`/`Rho`/`ScalarHT`, always "populated" regardless of real activity). Anything missing content on either axis (or both) is dropped before being written. Without this, the axis-2 contrastive encoder's CLS-token mask leaves every candidate slot masked for a zero-candidate event, collapsing its embedding to the same fixed constant regardless of what actually happened; a zero-L1T-object event is the equally degenerate case on axis-1's side (the AE's fixed 23-slot input would be entirely padding). Either risks correlating with the axis-1 nuisance bin for a spurious reason rather than real physics. See `_drop_events_with_empty_axis` in `converters.py`. `source_row` provenance is preserved correctly across the drop (original row index into the source file).

Under **weighted** selection this filter can bite hard — confirmed on real data (one QCD_HT50toInf/MinBias file each, `candidate_selection_pt: weighted`): ~0.66%/38.94% of events had populated L1T but zero candidates (MinBias, and DY too as it turned out, are soft/pileup-like enough that most candidates fall below the *weighted* floor or get PUPPI-weighted to exactly zero). Under **raw** selection — what `collide-1m-ptrawv1.0` actually uses — it's nearly inert: real per-sample drop rates from this dataset's own build were QCD 8/517,520 (0.0016%), WJets 14/128,251 (0.011%), DY 35/129,911 (0.027%), and exactly 0 for every TT sub-sample and every signal. Raw pT rarely zeroes out an entire event's candidate list the way the weighted floor does.

**Other variables**:

- `label`: fixed per-sample integer baked in at conversion — QCD=0, MinBias=1, TT=2, WJets=3, DY=4, Other=5.
- Provenance: `source_file` (int32) + `source_row` (int16) only — `event_id` dropped as redundant.

**PID**: unsigned int16, 5 buckets — 0=neutral hadron, 22=photon, 211=charged hadron (catch-all), 11=electron, 13=muon.

**Precision**: int8 for integers (exceptions: pdgId int16, source_file int32, source_row int16), float16 for floats.

**Scripts**:
- [`convert_data.py`](../convert_data.py) — CLI entrypoint (`--format collide2v_regionized`)
- [`src/aida_scout/data/converters.py`](../src/aida_scout/data/converters.py) — `convert_collide2v_regionized()` (orchestration), `gather_and_select_puppi_candidates()` (candidate selection), `gather_other_l1t_collections()` (everything else)
- [`src/aida_scout/data/regionize.py`](../src/aida_scout/data/regionize.py) — region geometry, per-region selection, sample→label mapping
- [`configs/data/convert_collide_1m_ptrawv1.yaml`](../configs/data/convert_collide_1m_ptrawv1.yaml) — this dataset's own config (`candidate_selection_pt: raw`, exact pinned source file list per sample, full composition-design rationale in its header comment)
- [`nrp/preprocess/preprocess_collide_1m_ptrawv1.yaml`](../nrp/preprocess/preprocess_collide_1m_ptrawv1.yaml) — the job manifest that actually built it

---

## 2. Training file schema

Every collection actually written to the training parquet file. "Cuts" is blank (`—`) where nothing is filtered at the variable level — the only per-candidate filtering in the whole file happens in `L1T_PUPPIPart` (region acceptance + the pT floor/rank selection described in §1); every other collection is kept close to raw, precision-downcast only.

### `L1T_PUPPIPart`

The only candidate collection (see §1) — up to 18 candidates per region × 90 regions, flattened and sorted by `pt` descending.

| Variable | Precision | Cuts | Definition |
|---|---|---|---|
| `pt` | float16 | >= 1 GeV within its region; top 18 per region by this value (`candidate_selection_pt: raw`) | Raw (unweighted) candidate pT. Primary feature; the stored collection is sorted by this, descending, and (in this dataset) is also the actual selection criterion for the whole collection (see §1). |
| `eta` | float16 | \|eta\| <= 3.0, closed interval (region acceptance — eta exactly ±3.0 is included) | Candidate pseudorapidity. Determines region assignment; candidates outside this range are never selected at all. |
| `phi` | float16 | — | Candidate azimuthal angle. Determines region assignment (9 phi bins). |
| `dxy` | float16 | — | Transverse impact parameter (raw `D0`). |
| `dxysig` | float16 | — | Impact-parameter significance, generated: `D0 / ErrorD0`. |
| `pdgId` | int16 | — | Generated: unsigned, 5-bucket realistic PID collapsed from the raw truth-level `PID` — 0=neutral hadron, 22=photon, 211=charged hadron (catch-all for any other nonzero species), 11=electron, 13=muon. |
| `charge` | int8 | — | Raw candidate charge (-1/0/+1). Kept separate from `pdgId` (not signed into it). |
| `pt_weighted` | float16 | — (would be the selection criterion under `candidate_selection_pt: weighted`, not this dataset's mode) | Generated: `pt × puppi_weight`. Always stored regardless of which mode drove selection (see §1). |
| `puppi_weight` | float16 | — | Raw PUPPI weight (0-1). |
| `e` | float16 | — | Candidate energy. |
| `mass` | float16 | — | Candidate mass. |
| `dz` | float16 | — | Longitudinal impact parameter (raw `DZ`). |
| `error_dz` | float16 | — | Uncertainty on `dz`. |
| `is_pu` | int8 | — | Raw pileup flag. |
| `is_reco_pu` | int8 | — | Raw `IsRecoPU`. Verified 0/1-only across 15M candidates (DY, tttt_incl, QCD_HT50toInf) — same flag type as `is_pu`. |
| `funique_id` | int32 | — | Raw `fUniqueID`, an internal per-candidate index with no physics meaning. Verified as a per-file running index reaching ~140k within a single 10k-event file — past int16's 32767 max, so kept at int32. |

### `L1T_Electron`

| Variable | Precision | Cuts | Definition |
|---|---|---|---|
| `Charge` | int8 | — | Electron charge (-1/0/+1). |
| `D0` | float16 | — | Transverse impact parameter. |
| `DZ` | float16 | — | Longitudinal impact parameter. |
| `Eta` | float16 | — | Pseudorapidity. |
| `Phi` | float16 | — | Azimuthal angle. |
| `PT` | float16 | — | Transverse momentum. |
| `ErrorD0` | float16 | — | Uncertainty on `D0`. |
| `ErrorDZ` | float16 | — | Uncertainty on `DZ`. |
| `EhadOverEem` | float16 | — | Hadronic-to-electromagnetic energy ratio. |
| `IsolationVar` | float16 | — | Isolation variable. |
| `IsolationVarRhoCorr` | float16 | — | Isolation variable, pileup (rho) corrected. |

### `L1T_MuonTight`

| Variable | Precision | Cuts | Definition |
|---|---|---|---|
| `Charge` | int8 | — | Muon charge (-1/0/+1). |
| `D0` | float16 | — | Transverse impact parameter. |
| `DZ` | float16 | — | Longitudinal impact parameter. |
| `Eta` | float16 | — | Pseudorapidity. |
| `Phi` | float16 | — | Azimuthal angle. |
| `PT` | float16 | — | Transverse momentum. |
| `ErrorD0` | float16 | — | Uncertainty on `D0`. |
| `ErrorDZ` | float16 | — | Uncertainty on `DZ`. |
| `IsolationVar` | float16 | — | Isolation variable. |
| `IsolationVarRhoCorr` | float16 | — | Isolation variable, pileup (rho) corrected. |

### `L1T_PhotonTight`

| Variable | Precision | Cuts | Definition |
|---|---|---|---|
| `Eta` | float16 | — | Pseudorapidity. |
| `Phi` | float16 | — | Azimuthal angle. |
| `PT` | float16 | — | Transverse momentum. |
| `EhadOverEem` | float16 | — | Hadronic-to-electromagnetic energy ratio. |
| `IsolationVar` | float16 | — | Isolation variable. |
| `IsolationVarRhoCorr` | float16 | — | Isolation variable, pileup (rho) corrected. |

### `L1T_JetAK4`

No `ConstituentsIdx` — confirmed absent from the real production schema (unlike `JetAK8`/`JetPuppiAK4`/`JetPuppiAK8` below, which have it).

| Variable | Precision | Cuts | Definition |
|---|---|---|---|
| `Eta` | float16 | — | Jet pseudorapidity. |
| `Phi` | float16 | — | Jet azimuthal angle. |
| `PT` | float16 | — | Jet transverse momentum. |
| `Mass` | float16 | — | Jet mass. |
| `Charge` | int8 | — | Jet charge. |
| `Flavor` | int8 | — | Parton flavor. Verified max 21 (DY/tttt_incl/QCD_HT50toInf, all 4 jet collections) — fits int8. |
| `BTag` | int8 | — | b-tag discriminant/flag. |
| `BTagPhys` | int8 | — | b-tag, physics working point. |
| `NCharged` | int8 | — | Charged constituent count. Verified max 54 (tttt_incl JetAK8) — fits int8. |
| `NNeutrals` | int16 | — | Neutral constituent count. Verified max 427 (tttt_incl JetAK8) — exceeds int8's 127 max, kept at int16. |
| `Constituents` | uint32 (native) | — | Per-jet jagged list of constituent indices. Not downcast — reference/index field, not a physics quantity. |

### `L1T_JetAK8`

| Variable | Precision | Cuts | Definition |
|---|---|---|---|
| `Eta` | float16 | — | Jet pseudorapidity. |
| `Phi` | float16 | — | Jet azimuthal angle. |
| `PT` | float16 | — | Jet transverse momentum. |
| `Mass` | float16 | — | Jet mass. |
| `Charge` | int8 | — | Jet charge. |
| `Flavor` | int8 | — | Parton flavor. Verified max 21 (DY/tttt_incl/QCD_HT50toInf, all 4 jet collections) — fits int8. |
| `BTag` | int8 | — | b-tag discriminant/flag. |
| `BTagPhys` | int8 | — | b-tag, physics working point. |
| `NCharged` | int8 | — | Charged constituent count. Verified max 54 (tttt_incl JetAK8) — fits int8. |
| `NNeutrals` | int16 | — | Neutral constituent count. Verified max 427 (tttt_incl JetAK8) — exceeds int8's 127 max, kept at int16. |
| `Constituents` | uint32 (native) | — | Per-jet jagged list of constituent indices. Not downcast. |
| `ConstituentsIdx` | int16 (native) | — | Per-jet jagged list of constituent indices (secondary index set). Not downcast. |

### `L1T_JetPuppiAK4`

Same fields as `L1T_JetAK8` above (including both `Constituents` and `ConstituentsIdx`).

### `L1T_JetPuppiAK8`

Same fields as `L1T_JetAK8` above (including both `Constituents` and `ConstituentsIdx`).

### `L1T_MET`

| Variable | Precision | Cuts | Definition |
|---|---|---|---|
| `MET` | float16 | — | Missing transverse energy magnitude. |
| `Eta` | float16 | — | MET pseudorapidity (event-level). |
| `Phi` | float16 | — | MET azimuthal angle. |

### `L1T_PUPPIMET`

Same fields as `L1T_MET` above, computed from PUPPI candidates.

### `L1T_Rho`

| Variable | Precision | Cuts | Definition |
|---|---|---|---|
| `Rho` | float16 | — | Pileup energy density (5 fixed values/event). |

### `L1T_ScalarHT`

| Variable | Precision | Cuts | Definition |
|---|---|---|---|
| `HT` | float16 | — | Scalar sum of jet transverse momenta (event-level). |

### Provenance & label

Per-event scalars, not nested in a collection.

| Variable | Precision | Cuts | Definition |
|---|---|---|---|
| `label` | int8 | — | Generated: fixed per-sample class, assigned from the EOS sample directory name at conversion time — QCD=0, MinBias=1, TT=2, WJets=3, DY=4, Other=5 (catch-all, covers every signal and every non-canonical background). |
| `source_file` | int32 | — | Generated: index into a per-sample saved filename list — which EOS parquet file this event came from. |
| `source_row` | int16 | — | Generated: row index within that source file. |

---

## 3. Unused collections (not included in the training file)

Listed for reference only — variable names as documented in `docs/eos_dataset_schema.md`, not re-verified against real data the way §2's collections have been (these are never read by the converter).

### `L1T_PFPart`

Excluded as redundant — identical raw `PT` to unweighted `L1T_PUPPIPart`, confirmed on real data.

fUniqueID, PID, Charge, PT, Eta, Phi, E, Mass, D0, DZ, ErrorD0, ErrorDZ, IsPU, IsRecoPU, PuppiW

### `Gen_Part`

PID, Status, PT, Eta, Phi, Mass, M1, M2, D1, D2, IsPU

### `Gen_JetAK4`

PT, Eta, Phi, Mass

### `Gen_JetAK8`

PT, Eta, Phi, Mass

### `Gen_MissingET`

MET, Eta, Phi

### `FullReco_PUPPIPart`

fUniqueID, PID, Charge, PT, Eta, Phi, E, Mass, D0, DZ, ErrorD0, ErrorDZ, IsPU, IsRecoPU, PuppiW

### `FullReco_PFPart`

fUniqueID, PID, Charge, PT, Eta, Phi, E, Mass, D0, DZ, ErrorD0, ErrorDZ, IsPU, IsRecoPU, PuppiW

### `FullReco_JetAK4` / `FullReco_JetAK8` / `FullReco_JetPuppiAK4` / `FullReco_JetPuppiAK8`

Eta, Phi, PT, Mass, Charge, Flavor, BTag, BTagPhys, NCharged, NNeutrals, Constituents, ConstituentsIdx

### `FullReco_Electron`

Charge, D0, DZ, Eta, Phi, PT, ErrorD0, ErrorDZ, EhadOverEem, IsolationVar, IsolationVarRhoCorr

### `FullReco_MuonTight`

Charge, D0, DZ, Eta, Phi, PT, ErrorD0, ErrorDZ, IsolationVar, IsolationVarRhoCorr

### `FullReco_PhotonTight`

Eta, Phi, PT, EhadOverEem, IsolationVar, IsolationVarRhoCorr

### `FullReco_MET` / `FullReco_PUPPIMET`

MET, Eta, Phi

### `FullReco_Rho`

Rho

### `FullReco_ScalarHT`

HT

### `Vertex_*`

Index, X, Y, Z, T, NDF, SumPT2, Constituents

### `Event_*`

Number, ProcessID, Weight, CrossSection, CrossSectionError, Scale, AlphaQCD, AlphaQED, ID1, ID2, X1, X2, PDF1, PDF2, ScalePDF

---

## 4. Stage 2: parquet → training tensors

**Not yet run for `collide-1m-ptrawv1.0`** — only Stage 1 (§1-3) has been
built for this dataset so far; the description below is the general Stage 2
design (still accurate and applicable whenever tensors get built from it),
not a description of files that currently exist on disk for this dataset.

Stage 1 (§1-3) writes per-sample regionized parquet. Stage 2 reads that
parquet and produces the `{'pf', 'label', 'obj'}` (or the 2-AE DisCo
baseline's `{'obj'}`) `.pt` tensor files every training/eval script actually
consumes — no further candidate selection, no further filtering, just
gather + pad + (for backgrounds) merge-and-split.

**Scripts**:
- [`convert_data.py`](../convert_data.py) — CLI entrypoint (`--format tensors` / `--format tensors_ae_split`; `--stats_only` prints per-sample candidate-count percentiles for sizing `n_objects` before a full conversion)
- [`src/aida_scout/data/converters.py`](../src/aida_scout/data/converters.py) — `convert_regionized_to_tensors()` / `convert_regionized_ae_split()` (orchestration), `gather_pfcands_regionized()` / `gather_objects_for_ae_regionized()` / `gather_objects_ae_split_regionized()` (per-tensor gathering)
- [`src/aida_scout/data/preprocessing.py`](../src/aida_scout/data/preprocessing.py) — `PFPreProcessor`, the model-level input layer that turns `pf`'s raw columns into the features actually fed to the encoder

### `pf` tensor: `[N_events, n_objects, 14]`

Padded/truncated to `data_processing.n_objects` candidates per event
(ragged in the source parquet — real per-event counts vary enormously by
class, from ~2 for MinBias to 40+ for TT/signals). Column order (saved
alongside the tensor as `pf_columns`, so it's read back by name, never by
position):

| Column | Source | Notes |
|---|---|---|
| `pt` | `pt_weighted` (default) or `pt` per `data_processing.pf_pt_mode` | **Always column 0 — the padding sentinel.** Padded candidate slots have `pt == 0.0` exactly; `PFCandsDataset.make_padding_mask`/`ContrastiveModel`'s mask both key off this. Re-sorted descending by whichever field this is (Stage 1's own on-disk order is raw-`pt` descending, which no longer matches column 0 if `pf_pt_mode: weighted`). |
| `eta`, `phi`, `dxy`, `dxysig`, `pdgId`, `charge`, `puppi_weight`, `e`, `mass`, `dz`, `error_dz`, `is_pu`, `is_reco_pu` | `L1T_PUPPIPart`, unchanged | See §2's `L1T_PUPPIPart` table for each field's definition. `funique_id` is not gathered (no physics meaning). |

Padding is applied as `ak.pad_none` → `ak.to_numpy` → `torch.tensor`
(`NaN` in padded slots at that point) → `assemble_and_save`'s
`nan_to_num_(nan=0.0, ...)`, which is what actually zeroes every column,
not just `pt`, in padded rows.

### `obj` tensor: `[N_events, 23, 4]` (main) or `[N_events, 23, 3]` (2-AE split)

Unchanged from the pre-migration schema: 10 jets + 4 muons + 4 electrons +
4 photons + 1 MET, each `[pt, eta, phi, type_id]` (main AE view,
`obj_columns = ["pt","eta","phi","type_id"]`) or `[pt, eta, phi]` with
slot position encoding type instead (2-AE split view — jets ΔR-cleaned
against muons/electrons first; see `AE_SPLIT_SLOTS`/`take_ae_split_slots`).
MET `eta` is forced to `0.0` in both views — the real `L1T_MET.Eta` field
exists but is non-conventional (not the object's actual pseudorapidity),
and this migration deliberately didn't start feeding it to the AE silently.

### `PFPreProcessor`'s default feature set

The `pf` tensor above is what's *saved*; `PFPreProcessor` (config key
`data.pf_feature_set`, omit for the default below) is what actually turns
those 14 raw columns into the features the contrastive encoder sees, all
batch-normalized over valid (non-padded) candidates only:

| Feature | Derivation | Why |
|---|---|---|
| `log_pt_frac` | `log(pt / sum(pt) per event)` | Event-relative, scale-invariant version of `pt` |
| `eta` | raw | — |
| `sin_phi`, `cos_phi` | `sin(phi)`, `cos(phi)` | Removes `phi`'s ±π discontinuity |
| `tanh_dxy` | `tanh(dxy)` | Bounded; raw `dxy` is long-tailed |
| `dxysig_clipped` | `clip(dxysig, -50, 50)` | Raw `dxysig`'s `EPS` denominator can spike to ~1e4 and dominate batch-norm unclipped |
| `log_e_frac` | `log(e / sum(e) per event)` | Event-relative energy, mirrors `log_pt_frac` |
| `puppi_weight` | raw | Online-available pileup discriminant |
| `tanh_dz` | `tanh(dz)` | Bounded, mirrors `tanh_dxy` |
| `charge` | raw | -1/0/+1 |
| `pdgId_onehot` | one-hot over `[0, 11, 13, 22, 211]` | 5-bucket realistic PID scheme (§2) |

Deliberately **excluded** from the default set: `is_pu`/`is_reco_pu` are
MC-truth pileup flags — a model trained on them would be exploiting
information no real L1 trigger has online, even though they're gathered
into `pf` (available for explicit ablation via `pf_feature_set`, or
diagnostics). `mass`/`error_dz` are dropped as near-fully redundant with
`pdgId`/`puppi_weight`+`dz` respectively. Padded rows are explicitly
re-zeroed feature-by-feature at aggregation time (not left to each
builder to be naturally zero-preserving — `cos(0) == 1`, not `0`, so
`cos_phi` needs this explicitly).

Every checkpoint persists a `preproc_config` (`{version, input_columns,
feature_set}`); `DiscoLoader`/`NurdLoader` raise if a checkpoint has none
or its version doesn't match the installed `PFPreProcessor.SCHEMA_VERSION`
— this dataset's 14-column `pf` happens to produce the same default
feature count (15) as the pre-migration schema did, so a stale checkpoint
could otherwise load with no shape error and produce silently-garbage
embeddings.

## 5. Summary: what changes at each stage

Two hops between the preprocessed parquet and what the network actually
sees, condensed from §4 above:

**Parquet → `.pt`** (`convert_regionized_to_tensors`, run once, standalone)
- Pick a pT definition (`pt_weighted` = pt×puppi_weight, default; or raw
  `pt`), rename it to `pt`, re-sort each event's candidates descending by it
- Keep 14 raw `L1T_PUPPIPart` columns per candidate, unchanged from source
- Pad/truncate every event's ragged candidate list to a fixed `n_objects`
  width (dense, no longer ragged)
- Cast to `float32`; padding-induced `NaN`s → exact `0.0` (`pt == 0` is the
  padding sentinel everything downstream keys off)
- Merge backgrounds by label into train/test splits; each signal saved
  standalone, never merged
- Build the separate fixed-23-slot `obj` tensor for the AE
- **No normalization** — everything stored in raw physical units

**`.pt` → network input** (`PFPreProcessor.forward`, run fresh every step)
- `pt == 0` marks padded slots; every output is re-zeroed there
- Derive features from the raw columns: `log_pt_frac`, `sin_phi`/`cos_phi`
  (replaces raw `phi`), `tanh_dxy`/`tanh_dz`, `dxysig` clipped to ±50,
  `log_e_frac` — nonlinear transforms, not present in the `.pt` file itself
- Continuous derived features run through `BatchNorm1d`, fit only over
  valid (non-padded) candidates — learned running stats, checkpointed with
  the model, not precomputed
- Discrete features: `charge` raw, `pdgId` one-hot over the 5-bucket scheme
- Feature set (`pf_feature_set`) is config-selectable per experiment — the
  same `.pt` file supports different derived-feature choices with no
  reconversion

The parquet → `.pt` step is a standalone offline conversion today, run
once ahead of training. It could instead become a per-batch step inside
the DataLoader (reading parquet directly, padding on the fly) to avoid
materializing the padded `.pt` file's disk footprint — not implemented
yet, noted here as a possible future direction.
