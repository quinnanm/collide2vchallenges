#!/usr/bin/env python
"""
Per-collection feature-histogram PDF for one challenge's converted output.

For every process/sample listed in a dataconfig.yml, reads ONE output
parquet fragment (the first train fragment if `split:` is configured, else
the first eval fragment, else the first no-split fragment) and histograms
every field of every collection found in that file. Output is a single PDF:
one big divider page per process, then one page per (process, collection)
pair -- histograms laid out in a `--cols`-wide grid (default 5) to keep the
page compact, paginated further if a collection has more fields than fit on
one page.

Meant as a quick visual QA pass over a just-converted dataset (feature
sanity, floor cuts landing where expected, no all-zero/all-NaN columns) --
not a physics-analysis tool. Only reads a `--max-events`-sized prefix of the
one file per process (default 20,000) via pyarrow's batched reader, since a
candidate collection (FullReco_PFPart/PUPPIPart at cap=500) can make even a
single fragment tens of GB if read in full -- see the loading note below.

Usage
-----
python plot_features.py --config ../C1_HH4b/dataconfig.yml
python plot_features.py --config ../C9_robust_tagging/dataconfig.yml --out ../C9_robust_tagging/features.pdf --cols 4
"""
import argparse
import datetime
import logging
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from config import DataConfig

logger = logging.getLogger(__name__)

# Bookkeeping fields written onto every record by converters.py -- not a
# physics "collection", skipped for plotting.
_META_FIELDS = {"label", "source_file", "source_row"}


def find_one_fragment(out_path: Path, sample_name: str) -> Path | None:
    """First output parquet fragment for `sample_name`, preferring the train
    split (eval is meant to be held back for scoring; no-split layout is the
    fallback for a config without `split:`). Layout is <out_path>/<split>/
    <sample>/<sample>_NNNNN.parquet when split: is configured, or
    <out_path>/<sample>/<sample>_NNNNN.parquet otherwise -- split is the
    outer directory, not the sample (see convert_collide2v_regionized)."""
    for split_dir in ("train", "eval"):
        frags = sorted((out_path / split_dir / sample_name).glob(f"{sample_name}_*.parquet"))
        if frags:
            return frags[0]
    frags = sorted((out_path / sample_name).glob(f"{sample_name}_*.parquet"))
    return frags[0] if frags else None


def load_prefix(path: Path, max_events: int) -> ak.Array:
    """Read only the first `max_events` rows of `path`, via pyarrow's batched
    reader -- ak.from_parquet/ak.to_parquet write each call as a single row
    group, so a plain row-group read wouldn't limit memory the way batched
    reading does (batch_size caps how many rows get decoded at once)."""
    pf = pq.ParquetFile(path)
    batch = next(pf.iter_batches(batch_size=max_events))
    return ak.from_arrow(batch)


def flatten_numeric(values: ak.Array) -> np.ndarray | None:
    """Flatten a field down to a flat numpy array for histogramming, however
    deeply nested it is (e.g. FullReco_JetAK4.ConstituentsIdx is a per-jet
    list of indices, one extra jagged level beyond the collection's own
    event->object nesting). Returns None if the field isn't numeric after
    flattening (nothing in the current schema hits this, but a future
    string/bytes field shouldn't crash the whole PDF)."""
    while values.ndim > 1:
        values = ak.flatten(values, axis=1)
    try:
        np_values = ak.to_numpy(values)
    except Exception:
        return None
    if not np.issubdtype(np_values.dtype, np.number):
        return None
    finite = np_values[np.isfinite(np_values)] if np.issubdtype(np_values.dtype, np.floating) else np_values
    return finite


def collection_panels(arr: ak.Array, collection: str) -> list[tuple[str, np.ndarray]]:
    """[(panel_label, values), ...] for one collection: one panel per field,
    plus a leading "n_objects/event" panel if the collection is per-event
    variable-length (jagged) rather than one-value-per-event."""
    coll = arr[collection]
    panels = []
    if coll.ndim > 1:
        counts = ak.to_numpy(ak.num(coll, axis=1)).astype(np.float64)
        panels.append(("n_objects / event", counts))
    for field in coll.fields:
        values = flatten_numeric(coll[field])
        if values is not None and len(values) > 0:
            panels.append((field, values))
        else:
            panels.append((field, None))  # plotted as a "skipped" placeholder
    return panels


def plot_grid(pdf: PdfPages, title: str, panels: list, cols: int, bins: int,
              clip_percentiles: tuple) -> None:
    """One or more PDF pages for `panels` (see collection_panels), `cols`-wide,
    paginating if there are more panels than fit on one page (5 rows/page)."""
    rows_per_page = 5
    per_page = cols * rows_per_page
    n_pages = max(1, -(-len(panels) // per_page))  # ceil div

    for page in range(n_pages):
        chunk = panels[page * per_page:(page + 1) * per_page]
        n_rows = -(-len(chunk) // cols)
        fig, axes = plt.subplots(n_rows, cols, figsize=(3.0 * cols, 2.4 * n_rows), squeeze=False)
        page_suffix = f"  (page {page + 1}/{n_pages})" if n_pages > 1 else ""
        fig.suptitle(f"{title}{page_suffix}", fontsize=12, fontweight="bold")

        for i, ax in enumerate(axes.flat):
            if i >= len(chunk):
                ax.axis("off")
                continue
            label, values = chunk[i]
            if values is None:
                ax.text(0.5, 0.5, "skipped\n(non-numeric)", ha="center", va="center", fontsize=8)
                ax.set_title(label, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            lo, hi = np.percentile(values, clip_percentiles) if len(values) > 1 else (values.min(), values.max())
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            ax.hist(values, bins=bins, range=(lo, hi), color="#3b6fa0")
            ax.set_title(f"{label}  (n={len(values)})", fontsize=9)
            ax.tick_params(labelsize=7)

        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig)
        plt.close(fig)


def divider_page(pdf: PdfPages, text: str, subtext: str = "") -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.55, text, ha="center", va="center", fontsize=22, fontweight="bold")
    if subtext:
        fig.text(0.5, 0.45, subtext, ha="center", va="center", fontsize=11)
    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def make_feature_pdf(cfg: DataConfig, out_pdf: Path, cols: int = 5, bins: int = 50,
                      max_events: int = 20000, clip_percentiles: tuple = (0.5, 99.5)) -> None:
    ds_name = cfg.get_ds_name()
    out_path = Path(cfg.dp("out_path"))
    samples = cfg.dp("samples", [])

    with PdfPages(out_pdf) as pdf:
        divider_page(pdf, ds_name,
                      f"Feature histograms -- {len(samples)} processes\n"
                      f"generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                      f"1 file/process, first {max_events} events")

        for sample in samples:
            name = sample["name"]
            label = sample.get("label")
            frag = find_one_fragment(out_path, name)
            if frag is None:
                logger.warning(f"{name}: no output fragment found under {out_path / name} -- skipping")
                divider_page(pdf, name, f"label={label}\n\nNO OUTPUT FILE FOUND\nexpected under {out_path / name}")
                continue

            arr = load_prefix(frag, max_events)
            n_read = len(arr)
            collections = [f for f in arr.fields if f not in _META_FIELDS]
            divider_page(pdf, name, f"label={label}\nfile: {frag.name}\nevents plotted: {n_read}\n"
                                     f"collections: {len(collections)}")

            for collection in collections:
                panels = collection_panels(arr, collection)
                plot_grid(pdf, f"{name}  |  {collection}", panels, cols, bins, clip_percentiles)

            logger.info(f"{name}: plotted {len(collections)} collections from {frag} ({n_read} events)")

    logger.info(f"Wrote {out_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a challenge's dataconfig.yml.")
    parser.add_argument("--out", default=None, help="Output PDF path (default: <ds_name>_features.pdf next to "
                                                      "--config).")
    parser.add_argument("--cols", type=int, default=5, help="Histograms per row (default: 5).")
    parser.add_argument("--bins", type=int, default=50, help="Bins per histogram (default: 50).")
    parser.add_argument("--max-events", type=int, default=20000,
                         help="Events read (from the front of the one file used per process) for histogramming "
                              "(default: 20000). Keeps memory bounded for high-multiplicity candidate collections.")
    parser.add_argument("--clip-percentiles", type=float, nargs=2, default=(0.5, 99.5),
                         help="Display-range percentile clip per histogram, low high (default: 0.5 99.5) -- "
                              "affects only the plotted x-range, not the data.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cfg = DataConfig(args.config)
    out_pdf = Path(args.out) if args.out else Path(args.config).resolve().parent / f"{cfg.get_ds_name()}_features.pdf"

    make_feature_pdf(cfg, out_pdf, cols=args.cols, bins=args.bins, max_events=args.max_events,
                      clip_percentiles=tuple(args.clip_percentiles))
