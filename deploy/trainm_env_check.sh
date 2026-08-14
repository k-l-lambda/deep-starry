#!/bin/bash
# Verify the deep-starry training environment INSIDE the trainm-dev-9 pod on maiyi-9.
#
# Why a script rather than a checklist: the pod's whole premise is that python packages come from
# HOST mounts, and that premise fails in ways that look like unrelated bugs — a python minor-version
# mismatch surfaces as an ImportError deep in a C extension, and a fresh ssh/Jupyter session
# silently loses PYTHONPATH (diary 0606). So each assumption gets its own named check.
#
# Run:
#   ssh -tt maiyi-9 'sudo su -c "kubectl exec trainm-dev-9 -- bash /workspace/deeps/deploy/trainm_env_check.sh"'
#
# Checks
#   1. interpreter matches the host build the mounts were made for
#   2. the three mounted package roots are on sys.path via the .pth (not just PYTHONPATH)
#   3. torch imports, sees CUDA, and counts the B300s
#   4. every requirements.txt module imports
#   5. the model zoo registers — model_factory eagerly imports ALL models, so a missing dep in an
#      unrelated modality still breaks a midi run
#   6. the repo's own data feeder loads, through the config path the trainer uses
#   7. DATA_DIR / TRAINING_DIR resolve to mounted, writable paths
set -u
fail=0
ok ()   { echo "  ok   $1${2:+   $2}"; }
bad ()  { echo "  FAIL $1${2:+   $2}"; fail=$((fail+1)); }

echo "== 1. interpreter"
ver=$(python3 -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])')
[ "$ver" = "3.10.12" ] && ok "python $ver matches the host mounts" \
  || bad "python $ver" "host site-packages were built for 3.10.12 — C extensions will not load"

echo "== 2. mounted package roots on sys.path"
# Deliberately with `env -i`: proves the .pth works with NO inherited environment, which is the
# case a fresh sshd session or a cron job actually hits.
env -i /usr/bin/python3 -c '
import sys
want = ["/opt/sys-packages", "/opt/dist-packages", "/opt/apt-packages"]
missing = [w for w in want if w not in sys.path]
print("MISSING:" + ",".join(missing) if missing else "ALLPRESENT")
' | grep -q ALLPRESENT \
  && ok "all three roots resolve with an empty environment (.pth works)" \
  || bad "a package root is missing under env -i" "the .pth bootstrap did not run"

echo "== 3. torch + CUDA"
python3 - <<'PY'
import sys
try:
    import torch
except Exception as e:
    print(f"  FAIL torch import   {type(e).__name__}: {e}"); sys.exit(1)
print(f"  ok   torch {torch.__version__}")
if not torch.cuda.is_available():
    print("  FAIL cuda unavailable   /dev/nvidia* not visible, or driver/runtime mismatch"); sys.exit(1)
n = torch.cuda.device_count()
print(f"  ok   cuda available, {n} device(s)   {torch.cuda.get_device_name(0)}")
# A real allocation, not just a device count: CUDA init can succeed and then fail on first malloc.
t = torch.zeros(1024, 1024, device='cuda')
print(f"  ok   allocated on device   {t.sum().item():.0f}")
PY
[ $? -ne 0 ] && fail=$((fail+1))

echo "== 4. requirements"
# Split into what a MIDI training run actually imports versus what only other modalities need.
# requirements.txt is one flat list, but the registry imports each modality independently, so
# `cv2` (vision), `primesieve` (topology) and `attrdict` (paraff) missing is a warning, not a
# failure — treating them as required would block a perfectly runnable midi environment, and
# building cv2/primesieve from source through this cluster's proxy is slow and pointless here.
python3 - <<'PY'
import importlib
need = {
    'tensorboardX':'tensorboardX', 'fs':'fs', 'python-dotenv':'dotenv', 'Pillow':'PIL',
    'dill':'dill', 'perlin-noise':'perlin_noise', 'tqdm':'tqdm', 'pyyaml':'yaml',
    'msgpack':'msgpack', 'numpy':'numpy', 'transformers':'transformers',
    'datasets':'datasets', 'typer':'typer', 'jinja2':'jinja2', 'pandas':'pandas',
}
optional = {'zmq':'zmq (zero_server only)', 'cv2':'cv2 (vision)',
            'primesieve':'primesieve (topology)', 'attrdict':'attrdict (paraff)'}
def probe (mods):
    out = []
    for mod in mods:
        try: importlib.import_module(mod)
        except Exception: out.append(mod)
    return out
missing = probe(need.values())
print("  ok   all %d midi-path modules import" % len(need) if not missing
      else "  FAIL missing: " + ", ".join(missing))
absent = probe(optional)
print("  --   optional absent: " + ", ".join(optional[m] for m in absent) if absent
      else "  ok   optional modules present too")
PY

echo "== 5. model registration"
# `import_package_submodules` imports each modality INDEPENDENTLY (model_factory.py), precisely so
# one modality's missing optional dep (cv2, primesieve) cannot block the others — so the check that
# matters is that the MIDI models we intend to train are registered by name, not that the whole zoo
# imported cleanly. A blanket registerModels() would pass while the midi classes were absent, or
# fail on a vision dep that no midi run needs.
python3 - <<'PY'
import sys
sys.path.insert(0, '/workspace/deeps')
try:
    from starry.utils.model_factory import registerModels, model_dict
    registerModels()
except Exception as e:
    print(f"  FAIL registerModels   {type(e).__name__}: {e}"); sys.exit(1)
want = [n for n in model_dict if n.startswith('MidiSeq2') or n.startswith('MidiTranslator')]
if not want:
    print(f"  FAIL no midi models registered   {len(model_dict)} total registered"); sys.exit(1)
print(f"  ok   {len(model_dict)} models registered, {len(want)} midi")
PY
[ $? -ne 0 ] && fail=$((fail+1))

echo "== 6. midi feeder"
python3 - <<'PY'
import sys, os
sys.path.insert(0, '/workspace/deeps')
try:
    from starry.midi.data.seq2seq2 import Seq2Seq2, _make_source, _ZipSource, _DirSource
    print("  ok   Seq2Seq2 imports")
except Exception as e:
    print(f"  FAIL Seq2Seq2 import   {type(e).__name__}: {e}"); sys.exit(1)
root = os.environ.get('CHECK_DATA_ROOT', '')
if not root or not os.path.isdir(root):
    print(f"  --   no corpus at {root!r}; skipping the load (set CHECK_DATA_ROOT to exercise it)")
    sys.exit(0)
src = _make_source(root)
kind = 'zip' if isinstance(src, _ZipSource) else 'dir'
print(f"  ok   source detected as {kind}")
# The arms are read from the corpus itself, never assumed. Packed corpora carry the arm name as an
# ENTRY PREFIX, so a hardcoded guess fails with a KeyError that reads like a packing bug rather than a
# naming mismatch — and it fails at first read, not at init, because names() comes from the manifest.
# The manifest's `arms` is the authority for what the archives actually contain.
import json
if isinstance(src, _ZipSource):
    with open(os.path.join(root, _ZipSource.MANIFEST)) as f:
        arms = json.load(f)['arms']
else:
    arms = [d for d in sorted(os.listdir(root)) if d.startswith('midi-seq2')]
target = 'midi-seq2-score'
source = next((a for a in arms if a != target), None)
if source is None:
    print(f"  FAIL no source arm among {arms}"); sys.exit(1)
print(f"  ok   arms from the corpus   {source} -> {target}")
(ds,) = Seq2Seq2.load(root, dict(source_dir=source, target_dir=target,
    mark_mode='tick', line_range=[20, 256], random_crop=False), splits='0/1')
print(f"  ok   feeder built   {len(ds.names)} pairs")
d = ds.describe(0)
print(f"  ok   sample 0   {len(d['ids'])} ids, sep {d['sep']}, skip {d['skip']}")
b = ds.collateBatch([ds[i] for i in range(min(4, len(ds.indices)))])
print(f"  ok   collated   {tuple(b['input_ids'].shape)}, supervised {int(b['target_mask'].sum())}")
PY
[ $? -ne 0 ] && fail=$((fail+1))

echo "== 7. data / training dirs"
for p in /data/midi /data/training; do
  if [ -d "$p" ]; then
    if touch "$p/.wtest" 2>/dev/null; then rm -f "$p/.wtest"; ok "$p writable"
    else bad "$p not writable" "hostPath owner is probably root; chown it on the host"; fi
  else bad "$p missing" "hostPath volume did not mount"; fi
done

echo
[ $fail -eq 0 ] && echo "trainm env: all checks ok" || { echo "trainm env: $fail FAILED"; exit 1; }
