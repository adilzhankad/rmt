"""Download the four benchmark TSVs into $LMUData and check them against the md5 pinned
in VLMEvalKit (and config.yaml). A file with the wrong md5 is deleted, never used.

    python experiment/download_data.py

The URLs are the ones in VLMEvalKit at the pinned commit. opencompass.openxlab.space
(CCOCR, olmOCRBench) served an EXPIRED TLS certificate on 2026-10-20, so for that host
only the certificate check is skipped; integrity is guaranteed by the md5 check instead.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import lmu_root, load_config, md5_file  # noqa: E402

URLS = {
    "OCRBench_v2": "https://huggingface.co/datasets/QYWH/ocrbench_v2/resolve/main/OCRBench_v2.tsv?download=true",
    "CCOCR": "https://opencompass.openxlab.space/utils/VLMEval/CCOCR.tsv",
    "OmniDocBench": "https://huggingface.co/datasets/ouyanglinke/OmniDocBench_tsv/resolve/main/OmniDocBench.tsv",
    "olmOCRBench": "https://opencompass.openxlab.space/utils/VLMEval/olmOCRBench.tsv",
}
EXPIRED_CERT_HOSTS = ("opencompass.openxlab.space",)


def fetch(url, dest):
    cmd = ["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(dest), url]
    if any(h in url for h in EXPIRED_CERT_HOSTS):
        cmd.insert(1, "-k")
    subprocess.run(cmd, check=True)


def nltk_data():
    """Corpora the evaluators need (METEOR in OCRBench v2 / OmniDocBench uses wordnet)."""
    import nltk
    import zipfile
    for pkg in ("wordnet", "omw-1.4", "punkt", "punkt_tab"):
        nltk.download(pkg, quiet=True)
    # some nltk versions download wordnet as a zip but cannot read it zipped
    for name in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{name}")
        except LookupError:
            for d in nltk.data.path:
                z = Path(d) / "corpora" / f"{name}.zip"
                if z.exists():
                    zipfile.ZipFile(z).extractall(z.parent)
                    break


def main():
    nltk_data()
    cfg = load_config()
    root = lmu_root()
    root.mkdir(parents=True, exist_ok=True)
    for bench, b in cfg["benchmarks"].items():
        dest = root / f"{b['dataset']}.tsv"
        if not (dest.exists() and md5_file(dest) == b["md5"]):
            fetch(URLS[bench], dest)
        got = md5_file(dest)
        if got != b["md5"]:
            dest.unlink()
            raise SystemExit(f"{dest.name}: md5 {got} != pinned {b['md5']} (file deleted)")
        print(f"{dest.name:20s} {dest.stat().st_size / 1e9:6.2f} GB  md5 {got}  OK")


if __name__ == "__main__":
    main()
