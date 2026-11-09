#!/bin/sh
# Fill the benchmark feature cache with N parallel workers (the cache is per file, so this is safe to interrupt).
cd "$(dirname "$0")/.."
N=${1:-5}
i=0
while [ $i -lt $N ]; do
  BENCH_THREADS=3 python scripts/run_bench.py --render --part=$i/$N > data/bench-render-$i.log 2>&1 &
  i=$((i+1))
done
wait
echo ALL_DONE
