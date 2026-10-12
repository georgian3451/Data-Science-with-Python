"""Download the public datasets used in the internship.

    python scripts/download_data.py            # Titanic + Fashion-MNIST
    python scripts/download_data.py --titanic  # only Titanic (tiny)

Wine (Week 3) ships with scikit-learn, so nothing to download.
"""
import argparse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
FASHION_BASE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FASHION_FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                 "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print("already there:", dest.relative_to(ROOT))
        return
    print("downloading   :", url)
    urllib.request.urlretrieve(url, dest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--titanic", action="store_true", help="download only the Titanic CSV")
    args = ap.parse_args()
    fetch(TITANIC_URL, ROOT / "data" / "raw" / "titanic.csv")
    if not args.titanic:
        for f in FASHION_FILES:
            fetch(FASHION_BASE + f, ROOT / "data" / "raw" / "fashion-mnist" / f)
    print("Done.")
