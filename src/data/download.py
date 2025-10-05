"""Download the Elliptic Bitcoin transaction dataset."""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "ellipticco/elliptic-data-set"
EXPECTED_FILES = [
    "elliptic_txs_features.csv",
    "elliptic_txs_edgelist.csv",
    "elliptic_txs_classes.csv",
]


def check_data_exists(raw_dir: str = "data/raw") -> bool:
    return all((Path(raw_dir) / f).exists() for f in EXPECTED_FILES)


def download_elliptic(raw_dir: str = "data/raw") -> None:
    """Download Elliptic dataset via Kaggle API.

    Requires: pip install kaggle, plus ~/.kaggle/kaggle.json with your API key.
    Manual alternative: download from kaggle.com/datasets/ellipticco/elliptic-data-set
    and extract CSVs into data/raw/.
    """
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)

    if check_data_exists(raw_dir):
        logger.info("elliptic data already present, skipping download")
        return

    logger.info("downloading elliptic dataset from kaggle...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(raw), "--unzip"],
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "kaggle CLI not found. install with: pip install kaggle\n"
            "then set up API key: https://www.kaggle.com/docs/api"
        )

    # kaggle sometimes nests files in a subdirectory
    for subdir in [raw / "elliptic_bitcoin_dataset", raw / "elliptic-data-set"]:
        if subdir.exists():
            for f in subdir.iterdir():
                f.rename(raw / f.name)
            subdir.rmdir()

    if not check_data_exists(raw_dir):
        missing = [f for f in EXPECTED_FILES if not (raw / f).exists()]
        raise FileNotFoundError(f"missing after download: {missing}")

    logger.info(f"elliptic dataset ready in {raw}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_elliptic()
