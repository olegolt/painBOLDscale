"""Scan a folder before it is made public.

    python check_release.py data/ [code/] [README.md] [--allow-email=name@host]

Flags: contrast-image-like files and 4D images, NIfTI headers with free text, absolute paths, original-style
subject IDs (sub-<number>; the pseudonyms sub-p<number> are fine), e-mail addresses, and files above GitHub's 100 MB limit.
Exit code 1 if anything is found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TEXT = {"", ".tsv", ".csv", ".json", ".txt", ".md", ".py", ".ipynb", ".xml", ".yml", ".yaml", ".cff", ".toml", ".bidsignore"}
PATTERNS = {
    "absolute path": re.compile(r"(?<![\w.])(/home/|/Users/|/mnt/|/data/|/Volumes/|[A-Z]:\\\\)[\w.\-/\\]+"),
    "original-style subject ID": re.compile(r"sub[-_]\d{2,}"),
    "e-mail address": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
}
IMAGE_NAME = re.compile(r"(s\d*w?con_?\d{3,4}|spmT|beta_\d|mean_wskull|sub-)", re.IGNORECASE)
MAX_BYTES = 100 * 1024 * 1024


ALLOWED_EMAILS: set[str] = set()
SKIP_DIRS = {".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__", ".ipynb_checkpoints"}


def scan_text(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return [f"cannot read ({exc})"]
    found = []
    for label, pattern in PATTERNS.items():
        for match in list(pattern.finditer(text))[:3]:
            if label == "e-mail address" and match.group(0) in ALLOWED_EMAILS:
                continue
            found.append(f"{label}: {match.group(0)[:80]}")
    return found


def scan_image(path: Path) -> list[str]:
    try:
        import nibabel as nib
    except ImportError:
        return ["nibabel not installed: image headers not checked"]
    img = nib.load(str(path))
    found = []
    if len(img.shape) > 3 and img.shape[3] > 1:
        found.append(f"4D image {img.shape}: looks like subject-level data")
    for field in ("descrip", "aux_file", "intent_name", "db_name"):
        value = bytes(img.header[field]).strip(b"\x00 ").decode("ascii", "ignore") if field in img.header else ""
        if value:
            found.append(f"header field {field} is not empty: {value[:60]!r}")
    return found


def check(roots: list[Path]) -> int:
    problems = 0
    for root in roots:
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file() and not SKIP_DIRS & set(p.parts)]
        for path in sorted(files):
            findings = []
            if path.stat().st_size > MAX_BYTES:
                findings.append(f"{path.stat().st_size / 1e6:.0f} MB: too large for a normal GitHub file")
            if path.suffix == ".gz" and path.name.endswith(".nii.gz") or path.suffix == ".nii":
                findings += scan_image(path)
                if IMAGE_NAME.search(path.name):
                    findings.append("file name looks like a subject-level image")
            elif path.suffix in TEXT:
                findings += scan_text(path)
            elif path.suffix in {".npy", ".npz", ".joblib", ".pkl"}:
                findings.append(f"{path.suffix} file: check what it contains")
            for finding in findings:
                print(f"{path}: {finding}")
            problems += len(findings)
    print("Nothing suspicious found." if not problems else f"{problems} finding(s). Fix them before publishing.")
    return int(bool(problems))


if __name__ == "__main__":
    ALLOWED_EMAILS.update(a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--allow-email="))
    targets = [Path(a) for a in sys.argv[1:] if not a.startswith("--")] or [Path(".")]
    missing = [t for t in targets if not t.exists()]
    if missing:
        sys.exit(f"Not found: {missing}")
    sys.exit(check(targets))
