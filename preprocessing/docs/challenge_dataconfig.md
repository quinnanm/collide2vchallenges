# Per-challenge dataconfig.yml reference

This documents the config-driven system built on top of the original fixed
recipe described in `docs/central_dataset_preprocessing.md` (which is still
accurate as a description of that recipe -- it just predates all of the keys
below). Each hackathon challenge directory (`C1_HH4b/`, `C5_foundation_model/`,
`C9_robust_tagging/`) has its own `dataconfig.yml` using this schema, driving
the shared `preprocessing/` code with its own process list, event counts,
collections, selections, and train/eval split -- no code changes needed per
challenge.

Every key in this doc is **optional**. Omitting all of them reproduces the
original fixed recipe exactly (see "Defaults" under each section). Run with:

```bash
python convert_data.py --config ../C1_HH4b/dataconfig.yml --overwrite
```

## `samples:`

```yaml
samples:
  - name: HH_4b               # EOS subdirectory name
    label: 1                   # explicit integer label (see below)
    target_events: 200000      # see "Event-count targeting" below
  - name: QCD_HT50toInf
    label: 0
    target_events: 1000000
    event_selection:            # see "event_selection" below
      - {collection: FullReco_ScalarHT, field: HT, op: '>', value: 240}
  - name: minbias               # can still mix in the old max_files/files: style
    max_files: 20
```

- **`label`**: explicit integer class label for this sample, taking priority
  over the default `regionize.label_for_sample(name)` lookup (which only
  knows AIDA-Scout's own QCD/MinBias/TT/WJets/DY scheme). **Set this
  explicitly for every sample in a challenge config** -- a challenge's own
  process list (e.g. `HH_4b`, `ggHbb`, ...) won't match that shared scheme and
  would otherwise silently fall through to `label=5` ("Other") for everything.
- **`target_events`**: alternative to `max_files`/`files:`. Keeps reading
  files (in `dataset_version`-filtered discovery order) until this sample's
  running **kept** event count (after event_selection and the empty-axis
  filter) is `>= target_events` -- **whole-file granularity: it never splits
  a file to hit the target exactly, so the actual count can overshoot**, and
  it logs a warning if the sample's directory runs out of files first.
  Ignored on a `files:`-pinned entry (pinning means "use exactly this list",
  full stop).
- **`event_selection`**: a list of cuts deciding whether to keep this
  sample's events at all (as opposed to `collections.*.object_selection`
  below, which keeps/drops individual *objects* but never the whole event --
  see "collections:" for that). Each cut:

  ```yaml
  {collection: <registry name>, field: <field, default: first registry field>,
   op: '>' | '>=' | '<' | '<=' | '==' | '!=', value: <number>,
   reduce: scalar | count | any | all | leading | max | min}
  ```

  All cuts in the list must pass (AND). Evaluated on **raw** fields,
  independent of `collections:` -- a cut can reference a collection that
  isn't even being saved to output (e.g. cut on `FullReco_ScalarHT` without
  including it in `collections:`).

  `reduce` picks how a multi-object collection collapses to one true/false
  per event (required for any collection except a `fixed_scalar` one, which
  defaults to `scalar` and errors if `reduce` is set to anything else):
    - `scalar`: exactly one value/event (`fixed_scalar` collections only --
      `L1T_MET`/`ScalarHT`/etc.; errors if the field isn't actually
      single-valued, e.g. `L1T_Rho`'s 5 entries/event).
    - `count`: object multiplicity, e.g. `{collection: L1T_JetAK4,
      reduce: count, op: '>=', value: 4}` for "at least 4 jets" (`field` is
      irrelevant here, any field in that collection has the same count).
    - `any` / `all`: per-object comparison, then OR/AND across the event's
      objects. `all` is **vacuously true for a zero-object event** (no
      object violates the condition) -- watch for this if you need "at least
      one object AND all objects pass X", which needs a separate `count`
      cut too.
    - `max` / `min`: reduce `field` to its max/min across the event's
      objects, then compare that one number (false for a zero-object event --
      nothing to reduce).
    - `leading`: the field value of the collection's own highest-`rank_field`
      object (`PT` descending for everything except `Vertex`, ranked by
      `SumPT2`) -- "leading jet's Mass > X", for instance. False for a
      zero-object event.

  Not yet supported: a nested "count objects that themselves pass a
  per-object condition" cut (e.g. "at least 4 jets with PT>30") -- `count`
  today is a plain multiplicity check with no inner condition.

## `collections:`

```yaml
collections:
  L1T_PUPPIPart: 18              # shorthand: cap only (meaning set by candidate_selection.mode)
  L1T_JetAK4:
    cap: 10
    object_selection:
      - {field: PT, op: '>', value: 20}
    drop_fields: [Constituents]
  L1T_MuonTight: 4
  L1T_Electron: 4
  L1T_PhotonTight: 4
  L1T_MET: 1                      # fixed-size collections ignore cap/object_selection
```

Keys are collection names from `converters.COLLECTION_REGISTRY` (open that
file for the authoritative list -- it covers every `L1T_*`/`FullReco_*`/
`Gen_*`/`Vertex_*`/`Event_*` collection in `docs/eos_dataset_schema.md`). Each
value is either:
- a plain **cap** -- an int (per-event object limit) or `null` (no cap,
  keep every object in original order); or
- a **dict** `{cap, object_selection, drop_fields, total_cap}` for finer control:
  - `cap`: as above.
  - `object_selection`: a list of `{field, op, value}` cuts (ALL must pass),
    evaluated on this collection's own **raw** fields, applied to each
    *individual object* (not the whole event) before ranking/capping --
    e.g. drop individual jets below a PT threshold. `variable_object`
    collections only (not `fixed_scalar` -- there's only one "object" there,
    the event itself; use `event_selection` for that). Composes with `cap`:
    survivors of `object_selection` are then ranked/capped as usual.
  - `drop_fields`: output field names to omit entirely from what's written
    (e.g. `[is_pu, is_reco_pu]` on `L1T_PUPPIPart` to drop the MC-truth
    pileup flags from the output, or `[Constituents]` to save space on a jet
    collection you don't need constituent indices for). The field is still
    *read* regardless (in case `object_selection` needs it) -- this only
    affects what's written.
  - `total_cap`: **`L1T_PUPPIPart` only.** A secondary flat ceiling on the
    FLATTENED per-event candidate count, applied AFTER the primary
    region/`flat_topn`/none selection. Only meaningful under `mode: region`,
    where `cap` is a PER-REGION limit (so a busy event's total can still run
    up to `N_REGIONS * cap` -- e.g. up to 1620 at the default `cap: 18`) --
    `total_cap` lets you keep the region-geometry selection logic while
    still bounding the whole event's candidate count with one flat number,
    e.g. `{cap: 18, total_cap: 500}` = region-select as usual, per-region,
    then additionally trim the flattened result to the top 500 by raw pT if
    it exceeds that. Rejected (a config error) on any other collection --
    every other collection's own `cap` already is the final per-event limit.

**Omit the whole `collections:` key** to get the original default set: every
`L1T_*` collection except `L1T_PFPart` (redundant with unweighted
`L1T_PUPPIPart`), `L1T_PUPPIPart` at the original fixed 18/region, everything
else uncapped, no object_selection/drop_fields.

**Truncation rule**: any capped collection other than `L1T_PUPPIPart` keeps
the top-N (post-`object_selection`) objects/event by that collection's
registry `rank_field` -- `PT` descending for every object collection except
`Vertex` (no `PT` field; ranked by `SumPT2` descending instead, the natural
"hardness"/primary-vertex proxy). Fixed-size collections (`*_MET`,
`*_PUPPIMET`, `*_Rho`, `*_ScalarHT`, `Gen_MissingET`, `Event`) ignore
cap/object_selection -- they're always fully populated regardless of real
per-event activity (`drop_fields` still applies to them).

**Caveat**: `Vertex_*`/`Event_*` raw column names are assumed to follow the
same `<prefix>_<field>` convention confirmed for every `L1T_*`/`FullReco_*`
collection (e.g. `Vertex_Index`, `Event_Number`) -- this hasn't been
independently checked against a real EOS file's schema. Before trusting a
conversion that requests one of those two, confirm with a quick
`pyarrow.parquet.read_schema(...)` against a real file.

**Truth-level warning**: `Gen_*` collections are generator-level MC truth --
safe for offline validation/labeling, **not safe as a training feature** (no
real trigger has access to it online).

## `candidate_selection:`

```yaml
candidate_selection:
  mode: region          # "region" (default) or "flat_topn"
  pt: weighted           # "weighted" (default) / "raw" / "none"
  floor_gev: 1.0           # pT floor before ranking/truncation
  realistic_pid: true       # collapse PID to the sanitized 5-bucket scheme (default) or keep it raw
```

`L1T_PUPPIPart` also supports `collections.L1T_PUPPIPart`'s `object_selection`/
`drop_fields` (see above) -- e.g. `object_selection: [{field: PT, op: '>=',
value: 1.0}]` for a per-candidate pT floor beyond the `floor_gev`
selection-criterion floor below, or `{field: Charge, op: '!=', value: 0}` to
keep only charged candidates. `object_selection` fields are this collection's
**raw** names (`PT`, `Eta`, `Phi`, `PID`, `Charge`, `E`, `Mass`, `D0`, `DZ`,
`ErrorD0`, `ErrorDZ`, `IsPU`, `IsRecoPU`, `PuppiW`, `fUniqueID`), not the
derived output names (`pt`, `pdgId`, ...).

- **`mode`**: `region` (default) is the original design -- the real 90-region
  PFL1 geometry, with `collections.L1T_PUPPIPart.cap` meaning "candidates kept
  per region" (so total candidates/event can reach up to 90x that number).
  `flat_topn` drops the region geometry entirely: the cap becomes a flat
  "top-N candidates for the whole event" instead. Use `flat_topn` for a
  challenge that doesn't need the L1T hardware-geometry detail and just
  wants a simple highest-pT candidate list.
- **`pt`**: which pT drives the `floor_gev` floor and the ranking --
  `weighted` (raw pT x PUPPI weight, the design default), `raw` (ignores
  PUPPI weight for selection), or `none` (no floor, no cap at all -- every
  candidate with `pt_raw > 0` is kept, still subject to `object_selection`;
  a no-cuts baseline, not a production mode). Falls back to the legacy
  top-level `candidate_selection_pt` key if `pt` isn't set here (so old
  configs using that key are unaffected), then to `weighted`.
- **`floor_gev`**: the pT floor (in the units of whichever `pt` mode is
  active) applied before ranking/truncation. Default: the original fixed
  1.0 GeV constant.
- **`realistic_pid`**: `true` (default) collapses the raw `PID` field the way
  this pipeline always has -- the unsigned 5-bucket scheme (`0`=neutral
  hadron, `11`=electron, `13`=muon, `22`=photon, `211`=charged-hadron
  catch-all), stored as `pdgId` (int16). `false` stores the raw `PID` value
  unmodified (int32) instead -- an explicit opt-out for a dataset that
  deliberately wants generator-level particle ID; **this reintroduces the
  truth-leakage the realistic scheme exists to avoid**, so don't set this to
  `false` for anything meant to train a model that has to work with only
  what a real L1 trigger can see online.

`report_diagnostics: true` (a `data_processing:` key, unchanged from before)
only supports `mode: region` and `pt` != `none` -- the region-based
cut-accounting has no equivalent for a flat top-N or a no-cuts run, and
doesn't account for `object_selection` drops.

## `split:`

```yaml
split:
  train_frac: 0.9
  eval_frac: 0.1
  seed: 42
```

**Omit entirely** for the original single-output-per-sample behavior (no
split at all). When set, every kept event is independently assigned to
`train` or `eval` via `np.random.default_rng(seed)` (one RNG per sample, so
different samples don't share the exact same draw sequence) and written under
separate top-level `train/`/`eval/` directories -- see "Output layout" below.
`train_frac + eval_frac` must sum to 1.0.

For a hackathon: ship `train/` to participants, keep `eval/` private for
scoring.

## `flush_every_events:`

```yaml
flush_every_events: 2000000
```

Optional, default `2_000_000`. Once a split's (or the whole sample's, if no
`split:`) buffered row count reaches this threshold, it's written out as the
next parquet fragment and the buffer is cleared -- bounds memory regardless
of how large `target_events` gets (needed once a single process's target
reaches into the tens of millions of events; holding all of it in memory
before one final write, the original design's approach, risks the same
OOM failure mode already called out in `nrp/preprocess_template.yaml`'s
resource-sizing comments). A sample under this threshold still produces
exactly one fragment.

## `max_events_per_file:`

Optional (default `-1`, no cap) -- caps how many rows are read from *each*
source file, independent of `target_events`/`max_files` (which control how
many *files* are read). Mostly useful for a fast local smoke-test run
against real EOS files without waiting on full-size reads; not something a
production challenge config needs to set.

## Output layout

```
<out_path>/<sample>/<sample>_00000.parquet          # no split: fragment(s) directly here
<out_path>/<sample>/<sample>_source_files.txt

<out_path>/train/<sample>/<sample>_00000.parquet     # split configured: split-then-process nesting
<out_path>/eval/<sample>/<sample>_00000.parquet
<out_path>/<sample>_source_files.txt                  # shared, top-level (not nested under either split)
```

**This is the one user-visible behavior change even with every new key
omitted**: a sample now always produces `<sample>_NNNNN.parquet` fragment(s)
instead of always exactly one file named `<sample>.parquet`. With the
defaults (no `split`, `flush_every_events` at its default), an
ordinary-size sample (well under 2M events) still produces exactly one
fragment (`_00000`), just with that suffix added to the filename.
`pyarrow`/`pandas` both read a directory of fragments as a single dataset
(`pyarrow.dataset.dataset(dir)` or `pd.read_parquet(dir)`), so this shouldn't
require any change on the consuming side.

`--overwrite` now applies per fragment directory: without it, any existing
fragment for that sample/split is a hard error; with it, existing fragments
for that sample/split are deleted before new ones are written.
