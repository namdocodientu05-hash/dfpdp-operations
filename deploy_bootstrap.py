from pathlib import Path
import hashlib
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
PARTS = sorted(ROOT.glob("DFPDP_BUNDLE_*.part"))
ZIP_PATH = ROOT / "DFPDP_RENDER_READY.zip"
APP_DIR = ROOT / "DFPDP_RENDER_READY"
EXPECTED_SHA256 = "454f89048ec6f888c51d14ccc6eaaedcb75684f85083d2c1be3ffea725aac1a7"

if not PARTS:
    raise SystemExit("Khong tim thay DFPDP_BUNDLE_*.part trong thu muc goc repository.")

print(f"[bootstrap] Found {len(PARTS)} bundle parts")
with ZIP_PATH.open("wb") as out:
    for part in PARTS:
        print(f"[bootstrap] Joining {part.name} ({part.stat().st_size} bytes)")
        with part.open("rb") as src:
            shutil.copyfileobj(src, out)

digest = hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest()
print(f"[bootstrap] SHA256={digest}")
if digest != EXPECTED_SHA256:
    raise SystemExit(
        "Bundle checksum mismatch. Expected "
        + EXPECTED_SHA256
        + " but received "
        + digest
    )

if APP_DIR.exists():
    shutil.rmtree(APP_DIR)

with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    zf.extractall(ROOT)

required = [
    APP_DIR / "app.py",
    APP_DIR / "requirements-render.txt",
    APP_DIR / "core" / "checkpoints" / "current_pilot_best.pt",
    APP_DIR / "web" / "control.html",
]
missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
if missing:
    raise SystemExit("Bundle extracted but required files are missing: " + ", ".join(missing))

print("[bootstrap] Bundle reconstructed and extracted successfully.")
