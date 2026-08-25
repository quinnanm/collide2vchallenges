"""
Unified YAML config loader used by converters.py.

Standalone extraction of AIDA-Scout's src/aida_scout/config.py
(github.com/AIDA-Scout/aidascoutrepo, commit c145ce6) -- Config/DataConfig/
join_remote are byte-for-byte identical to the source; wandb_init (a
training-only helper, no relevance to parquet preprocessing) is dropped.
"""
from pathlib import Path
from typing import Any, Union

import yaml


class Config:
    """Thin dict wrapper with dotted-path lookup, a `default` fallback, and an
    optional live W&B-sweep override (`.get(key, default, sweepable=True)`),
    matching the original `train_config.hp()` sweep-awareness.
    """

    def __init__(self, path: Union[str, Path]):
        path = Path(path)
        try:
            with open(path, "r") as f:
                self._cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {path}")
        self.path = path

    def get(self, dotted_key: str, default: Any = None, sweepable: bool = False) -> Any:
        """Look up `dotted_key` (e.g. "hyperparameters.lr"). If `sweepable`,
        an active W&B sweep's `wandb.config` takes priority over the file
        (mirrors the original `train_config.hp()` behavior).
        """
        if sweepable:
            try:
                import wandb

                if wandb.run is not None and dotted_key.split(".")[-1] in wandb.config:
                    return wandb.config[dotted_key.split(".")[-1]]
            except ImportError:
                pass

        node = self._cfg
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def hp(self, key: str, default: Any = None) -> Any:
        """Sweep-aware hyperparameter getter under the `hyperparameters:` section."""
        return self.get(f"hyperparameters.{key}", default, sweepable=True)

    def is_sweep(self) -> bool:
        try:
            import wandb

            return wandb.run is not None and wandb.run.sweep_id is not None
        except ImportError:
            return False

    @property
    def raw(self) -> dict:
        return self._cfg

    def __getitem__(self, key: str) -> Any:
        return self._cfg[key]

    def section(self, name: str) -> dict:
        return self._cfg.get(name, {})


class DataConfig(Config):
    """Config for the data-conversion (`converters.py`) and dataset-composition
    (`class_weights`, `nevents_per_class`, ...) options. Same schema as the
    original `data_*.yaml` files.
    """

    def get_file_label_map(self):
        """[(file_glob, class_label), ...] -- list order defines the class index."""
        samples = self.section("data_processing").get("samples", [])
        if not samples:
            raise ValueError("`data_processing.samples` not found in the config file.")
        out = []
        for cls_label, sample_dict in enumerate(samples):
            fnames = list(sample_dict.values())[0]
            if fnames is not None:
                out += [(fname, cls_label) for fname in fnames]
        return out

    def get_label_name_map(self) -> dict:
        samples = self.section("data_processing").get("samples", [])
        return {i: list(d.keys())[0] for i, d in enumerate(samples)}

    def get_ds_name(self) -> str:
        name = self._cfg.get("ds_name", "")
        if not name:
            raise ValueError("`ds_name` not found in the config file.")
        return name

    def get_sample_dir(self) -> str:
        """`data_processing.sample_dir`, validated -- without this, a missing
        key silently returns None and the converters crash later with an
        opaque `TypeError: ... not 'NoneType'` inside `Path(None)`."""
        sample_dir = self.dp("sample_dir")
        if not sample_dir:
            raise ValueError("`data_processing.sample_dir` not found in the config file.")
        return sample_dir

    def dp(self, key: str, default: Any = None) -> Any:
        """Getter under the `data_processing:` section."""
        return self.section("data_processing").get(key, default)


def join_remote(prefix: str, p) -> str:
    """Join an xrootd redirector like root://host with a local/posix path."""
    p_str = p.as_posix() if isinstance(p, Path) else str(p)
    sep = "//" if p_str.startswith("/") else "/"
    return f"{prefix.rstrip('/')}{sep}{p_str.lstrip('/')}"
