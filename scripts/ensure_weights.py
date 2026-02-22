import os
from pathlib import Path
import urllib.request

# Put them where the current code expects
BASE = Path("runs/bikes_ensemble")
BASE.mkdir(parents=True, exist_ok=True)

FILES = {
    BASE / "m1_convnext" / "best.pt": os.getenv("URL_M1_BEST"),
    BASE / "m1_convnext" / "last.pt": os.getenv("URL_M1_LAST"),
    BASE / "m2_effnetb0" / "best.pt": os.getenv("URL_M2_BEST"),
    BASE / "m2_effnetb0" / "last.pt": os.getenv("URL_M2_LAST"),
    BASE / "m3_vit_small" / "best.pt": os.getenv("URL_M3_BEST"),
    BASE / "m3_vit_small" / "last.pt": os.getenv("URL_M3_LAST"),
}

def download(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    print(f"Downloading {url} -> {out_path}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        f.write(r.read())
    tmp.replace(out_path)

def main():
    for out_path, url in FILES.items():
        if not url:
            raise RuntimeError(f"Missing env var for {out_path.name}")
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"OK cached: {out_path}")
            continue
        download(url, out_path)

    print("Weights ready.")

if __name__ == "__main__":
    main()