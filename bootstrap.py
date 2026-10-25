from pathlib import Path
import base64, hashlib, io, shutil, zipfile

ROOT = Path(__file__).resolve().parent
APP = ROOT / 'app'

def b64_file(path: Path) -> bytes:
    return base64.b64decode(path.read_text(encoding='ascii'))

def restore_source():
    if APP.exists():
        shutil.rmtree(APP)
    APP.mkdir(parents=True, exist_ok=True)
    raw = b64_file(ROOT / 'payload' / 'source_zip.b64')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        z.extractall(APP)

def restore_checkpoint(label: str, target_name: str):
    d = ROOT / 'payload' / f'ckpt_{label}'
    parts = sorted(d.glob('part_*.b64'))
    if not parts:
        raise RuntimeError(f'No checkpoint parts for {label}')
    data = b''.join(base64.b64decode(p.read_text(encoding='ascii')) for p in parts)
    expected = (d / 'sha256.txt').read_text().strip()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise RuntimeError(f'Checkpoint checksum mismatch for {label}: {actual} != {expected}')
    out = APP / 'core' / 'checkpoints' / target_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f'Restored {target_name}: {len(data)/1024/1024:.2f} MiB, sha256={actual}')

restore_source()
restore_checkpoint('current', 'current_pilot_best.pt')
restore_checkpoint('best', 'best_actor_critic.pt')
print('Bootstrap completed.')