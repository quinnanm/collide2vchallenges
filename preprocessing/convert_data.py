#!/usr/bin/env python
"""
EOS collide2v parquet -> per-sample regionized parquet. Standalone,
single-stage CLI (Stage 1 only -- see converters.py's module docstring for
what that means and where this was extracted from).

Usage
-----
python convert_data.py --config configs/example_convert.yaml
python convert_data.py --config configs/example_convert.yaml --overwrite
python convert_data.py --config configs/example_convert.yaml --overwrite --resume
"""
import argparse
import datetime
import logging

from config import DataConfig
from converters import convert_collide2v_regionized

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a data_processing YAML config -- see "
                                                          "configs/example_convert.yaml.")
    parser.add_argument("--overwrite", action="store_true", help="Without this, ANY pre-existing output for a "
                                                                   "sample is a hard error. With it, a sample's "
                                                                   "existing output is wiped and that sample is "
                                                                   "reconverted from scratch (every sample, unless "
                                                                   "--resume is also given).")
    parser.add_argument("--resume", action="store_true", help="Skip a sample entirely (no re-discovery, no "
                                                                "re-reading, existing output left untouched) if it "
                                                                "already finished on a prior run -- for restarting "
                                                                "after a crash without re-doing every "
                                                                "already-completed sample from the first one again. "
                                                                "Only meaningful together with --overwrite (an "
                                                                "incomplete sample is still wiped and redone).")
    args = parser.parse_args()

    cfg = DataConfig(args.config)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    import os
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                         handlers=[logging.StreamHandler(),
                                   logging.FileHandler(f"logs/convert_{cfg.get_ds_name()}_{ts}.log")])

    convert_collide2v_regionized(cfg, overwrite=args.overwrite, resume=args.resume)
