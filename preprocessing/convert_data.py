#!/usr/bin/env python
"""
EOS collide2v parquet -> per-sample regionized parquet. Standalone,
single-stage CLI (Stage 1 only -- see converters.py's module docstring for
what that means and where this was extracted from).

Usage
-----
python convert_data.py --config configs/example_convert.yaml
python convert_data.py --config configs/example_convert.yaml --overwrite
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
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files "
                                                                   "instead of skipping already-converted samples.")
    args = parser.parse_args()

    cfg = DataConfig(args.config)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    import os
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                         handlers=[logging.StreamHandler(),
                                   logging.FileHandler(f"logs/convert_{cfg.get_ds_name()}_{ts}.log")])

    convert_collide2v_regionized(cfg, overwrite=args.overwrite)
