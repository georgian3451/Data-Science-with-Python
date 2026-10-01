"""Download the public Titanic dataset used in Weeks 1-2.

    python scripts/download_data.py
"""
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
dest = ROOT / "data" / "raw" / "titanic.csv"

if dest.exists():
    print("already there:", dest.relative_to(ROOT))
else:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("downloading:", TITANIC_URL)
    urllib.request.urlretrieve(TITANIC_URL, dest)
print("Done.")
