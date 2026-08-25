# collide2vpreproc

Standalone Stage 1 of AIDA-Scout's data pipeline: EOS `collide2v`
foundational-model-dataset (raw CMS Phase-2 L1T trigger-emulation parquet)
-> per-sample **regionized parquet** (real 90-region PFL1 candidate-selection
geometry, one fixed integer label per sample). This produces preprocessed
parquet files only -- it does not build training tensors, and has no
dependency on PyTorch or any other ML framework.

Extracted from [AIDA-Scout/aidascoutrepo](https://github.com/AIDA-Scout/aidascoutrepo)
(`src/aida_scout/data/converters.py`, `regionize.py`, `config.py`; commit
`c145ce6` on branch `melissa`) so it can run standalone against a different
EOS sample selection, on a different PVC/volume and Kubernetes namespace,
for a different project. `converters.py`/`regionize.py` are extracted
verbatim from that repo's Stage 1 code (verified against its own test suite
-- see `tests/`); only Stage 2 (parquet -> `.pt` training tensors, and the
torch dependency that comes with it) was left out, since it's specific to
AIDA-Scout's own NURD/DisCo training and not needed just to produce
preprocessed parquet.

## What's here

```
config.py         # YAML config loader (Config/DataConfig/join_remote)
constants.py       # EPS (numerical epsilon)
regionize.py        # 90-region PFL1 geometry, candidate selection, sample labeling
converters.py         # convert_collide2v_regionized() -- the actual conversion
convert_data.py         # CLI entrypoint
configs/
  example_convert.yaml    # template data_processing: config -- copy and edit per project
docs/
  central_dataset_preprocessing.md   # design doc: region geometry, candidate
                                      # selection, dataset_version filtering,
                                      # the empty-axis event filter
  eos_dataset_schema.md               # raw EOS parquet schema reference
nrp/
  preprocess_template.yaml   # template Kubernetes Job manifest (NRP/Nautilus) --
                              # every namespace/PVC/secret-specific line marked EDIT
tests/
  test_converters.py   # real synthetic-data tests of the candidate-selection
                        # logic (13 tests, adapted from aidascoutrepo's own
                        # test suite, passing against this extraction)
```

## Setup

```bash
pip install -r requirements.txt          # numpy, awkward, pyarrow, fsspec-xrootd, PyYAML
pip install pytest && python -m pytest tests/ -v   # optional: verify the extraction
```

EOS access over xrootd needs a CMS grid certificate/proxy -- see
aidascoutrepo's `nrp/README.md` §4 for the one-time setup (VOMS role, x509
proxy secret) if running on a Kubernetes cluster; `nrp/preprocess_template.yaml`
here assumes that same setup exists in your own namespace, under your own
secret name.

## Usage

1. Copy `configs/example_convert.yaml`, point `sample_dir`/`redir` at your
   EOS path, `out_path` at your own output volume, and list the `samples:`
   you need (see the config's own comments for the two entry forms --
   auto-discovered via `max_files`, or an exact pinned `files:` list).
2. Edit `regionize.py`'s `SAMPLE_LABELS` dict if you need a different
   background-class scheme than AIDA-Scout's own (QCD/MinBias/TT/WJets/DY) --
   everything else in this package is independent of that choice. Any
   sample not listed there gets label `5` ("Other"), the same convention
   AIDA-Scout uses for signals/non-canonical backgrounds.
3. Run:

```bash
python convert_data.py --config configs/example_convert.yaml --overwrite
```

Output: one `<out_path>/<sample>/<sample>.parquet` per sample, plus a
`<sample>_source_files.txt` listing which EOS files went into it. See
`converters.py`'s `convert_collide2v_regionized()` docstring for the full
`data_processing:` config reference (candidate selection mode, diagnostics,
dataset-version filtering) and `docs/central_dataset_preprocessing.md` for
the design rationale behind each default.

On a Kubernetes cluster: `nrp/preprocess_template.yaml` is a starting point
-- every namespace/PVC/secret-specific line is marked `EDIT`.

## Keeping this in sync

This is a point-in-time extraction, not a live subtree/submodule -- if
aidascoutrepo's Stage 1 logic changes (bug fixes, schema changes), those
changes won't automatically appear here. If you need to pull in a later fix,
diff against `src/aida_scout/data/converters.py` (lines implementing
`convert_collide2v_regionized` and its helpers, from `PUPPI_CAND_RAW_FIELDS`
through the end of that function) and `src/aida_scout/data/regionize.py` in
the source repo.
