#!/usr/bin/env bash
set -euo pipefail
source /mnt/public/raojiaji/RPent-env/libero.sh
cd /mnt/public/raojiaji/RPent-context
FULL_GPU="${FULL_GPU:-2}"
ALL_GPU="${ALL_GPU:-3}"
export BATCH_DIR="${BATCH_DIR:-$PWD/logs/parallel_low_$(date +%Y%m%d_%H%M%S)}"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$BATCH_DIR"
python - "$FULL_GPU" "$ALL_GPU" <<'PY'
import socket,subprocess,sys
ids=subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).split()
assert all(x in ids for x in sys.argv[1:]), f'GPU indices must be in {ids}'
assert sys.argv[1] != sys.argv[2], 'Use distinct GPUs for the two groups'
with socket.create_connection(('127.0.0.1',19170),timeout=5):pass
sockets=[]
try:
    for port in range(19200,19208):
        s=socket.socket();s.bind(('127.0.0.1',port));sockets.append(s)
finally:
    for s in sockets:s.close()
PY
PYTHONPATH="/mnt/public/raojiaji/RPent-context-deps:$PWD" python - <<'PY'
from rpent.context.features import Features
f=Features('/mnt/public/raojiaji/context-encoder')
assert f.similarity('cup','cup') > .99
print('Encoder preflight OK')
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
for i in 1 2 3 4; do
  nohup python scripts/context_benchmark/worker.py \
    --batch "$BATCH_DIR" --mode full --gpu "$FULL_GPU" --repeat "$i" \
    --port "$((19199+i))" \
    > "$BATCH_DIR/full_${i}.worker.log" 2>&1 < /dev/null &
  echo "$!" > "$BATCH_DIR/full_${i}.worker.pid"
  nohup python scripts/context_benchmark/worker.py \
    --batch "$BATCH_DIR" --mode all --gpu "$ALL_GPU" --repeat "$i" \
    --port "$((19203+i))" \
    > "$BATCH_DIR/all_${i}.worker.log" 2>&1 < /dev/null &
  echo "$!" > "$BATCH_DIR/all_${i}.worker.pid"
done
printf 'Started 4 Full + 4 All. Results: %s\n' "$BATCH_DIR"
