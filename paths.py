"""Central registry for filesystem paths.

Reads `dir.toml` at the repo root (gitignored, per-machine). Copy
`example_dir.toml` to `dir.toml` and edit for your VM. Every path used
by the repo is exposed here — no path should be hardcoded elsewhere.
"""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


_PATHS_FILE = Path(__file__).resolve().parent / "dir.toml"
if not _PATHS_FILE.exists():
    raise FileNotFoundError(
        f"{_PATHS_FILE} not found — copy example_dir.toml to dir.toml and edit."
    )
_PATHS = tomllib.loads(_PATHS_FILE.read_text())


MERGED_DATASET_DIR = _PATHS["merged_dataset_dir"]
POSE_NPZ_DIR = _PATHS["pose_npz_dir"]
POSE_DATA_DIR = _PATHS["pose_data_dir"]
AUDIO_DATA_DIR = _PATHS["audio_data_dir"]
XABI_TOKENIZER_PATH = _PATHS["xabi_tokenizer_path"]
JAMES_TOKENIZER_PATH = _PATHS["james_tokenizer_path"]
