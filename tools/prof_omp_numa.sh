#!/usr/bin/env bash
# Minimal OMP/NUMA profiling helper for the C++ main binary.
# Runs a set of controlled experiments to quantify the impact of
# - OpenMP threads per process (OMP_NUM_THREADS)
# - CPU pinning (taskset)
# - NUMA binding (numactl --cpunodebind/--membind)
#
# Usage:
#   tools/prof_omp_numa.sh [--binary ./main] [--dataset dataset/Cora] \
#                          [--algo 9] [--r 0.5] [--socket 0] [--outdir tools/prof_results]
#
# Recommendations:
# - Run from repo root on the target server.
# - Ensure GNU time is available at /usr/bin/time for accurate metrics; otherwise we fall back to wall clock.
# - numactl is recommended. If missing, we skip NUMA binding.

set -euo pipefail

BINARY="./main"
DATASET="dataset/Cora"
ALGO=9
RVAL=0.5
SOCKET=0
OUTDIR="tools/prof_results"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary)   BINARY="$2"; shift 2;;
    --dataset)  DATASET="$2"; shift 2;;
    --algo)     ALGO="$2"; shift 2;;
    --r)        RVAL="$2"; shift 2;;
    --socket)   SOCKET="$2"; shift 2;;
    --outdir)   OUTDIR="$2"; shift 2;;
    -h|--help)
      sed -n '1,80p' "$0"; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

mkdir -p "$OUTDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUTDIR/run_$STAMP"
mkdir -p "$RUN_DIR"

# Tool detection
TIME_CMD=""
if [[ -x "/usr/bin/time" ]]; then
  TIME_CMD="/usr/bin/time -v"
elif command -v gtime >/dev/null 2>&1; then
  TIME_CMD="gtime -v"
fi

HAVE_NUMACTL=0
if command -v numactl >/dev/null 2>&1; then
  HAVE_NUMACTL=1
fi

echo "[INFO] Binary:     $BINARY"
echo "[INFO] Dataset:    $DATASET"
echo "[INFO] Algo code:  $ALGO"
echo "[INFO] r value:    $RVAL"
echo "[INFO] Socket:     $SOCKET"
echo "[INFO] Output dir: $RUN_DIR"
echo "[INFO] GNU time:   ${TIME_CMD:-unavailable}"
echo "[INFO] numactl:    $([[ $HAVE_NUMACTL -eq 1 ]] && echo yes || echo no)"

# Build CPU lists per socket (first logical CPU per core; avoid HT siblings)
build_cpu_lists() {
  local tmpfile
  tmpfile=$(mktemp)
  lscpu -e=CPU,CORE,SOCKET,ONLINE > "$tmpfile"
  mapfile -t SOCKET0 < <(awk 'NR>1 && $4=="yes" && $3==0 && !seen[$2]++ {print $1}' "$tmpfile")
  mapfile -t SOCKET1 < <(awk 'NR>1 && $4=="yes" && $3==1 && !seen[$2]++ {print $1}' "$tmpfile")
  rm -f "$tmpfile"
}

build_cpu_lists

choose_socket_cpus() {
  local s=$1
  if [[ "$s" == "0" ]]; then
    printf '%s\n' "${SOCKET0[@]}"
  else
    printf '%s\n' "${SOCKET1[@]}"
  fi
}

join_by_comma() {
  local IFS=","; echo "$*"
}

run_cmd_logged() {
  local label="$1"; shift
  local logfile="$RUN_DIR/${label}.log"
  echo "[RUN] $label" | tee "$logfile"
  echo "CMD: $*" | tee -a "$logfile"
  local start end
  start=$(date +%s.%N)
  if [[ -n "$TIME_CMD" ]]; then
    { $TIME_CMD "$@"; } >> "$logfile" 2>&1 || true
  else
    { "$@"; } >> "$logfile" 2>&1 || true
  fi
  end=$(date +%s.%N)
  awk -v s="$start" -v e="$end" 'BEGIN{printf "WALL_SECONDS: %.3f\n", (e-s)}' >> "$logfile"
}

single_process_omp_sweep() {
  echo "[SECTION] Single-process OMP sweep on socket $SOCKET"
  local cpus=($(choose_socket_cpus "$SOCKET"))
  local first_cpu="${cpus[0]}"
  local ts
  for ts in 1 2 3 4 6 8 12 16 24; do
    local label="A_single_omp${ts}"
    local cmd=( )
    if [[ $HAVE_NUMACTL -eq 1 ]]; then
      cmd+=(numactl --cpunodebind="$SOCKET" --membind="$SOCKET")
    fi
    cmd+=(taskset -c "$first_cpu" env OMP_NUM_THREADS="$ts" "$BINARY" "$ALGO" "$DATASET" "$RVAL")
    run_cmd_logged "$label" "${cmd[@]}"
  done
}

proc_thread_partitions() {
  # Build partitions with equal total threads ~= NUM_CORES (socket cores count)
  local cpus=($(choose_socket_cpus "$SOCKET"))
  local n=${#cpus[@]}
  local parts=( )
  # e.g. n x 1, n/2 x 2, n/4 x 4, n/8 x 8, 1 x n
  local g procs
  for g in 1 2 4 8; do
    procs=$(( n / g ))
    if (( procs >= 1 )); then
      parts+=("${procs}x${g}")
    fi
  done
  # also 1 x n (single proc OMP=n)
  parts+=("1x${n}")
  printf '%s\n' "${parts[@]}"
}

multi_process_equal_budget() {
  echo "[SECTION] Multi-process equal-budget on socket $SOCKET"
  local cpus=($(choose_socket_cpus "$SOCKET"))
  local parts=( $(proc_thread_partitions) )
  local part
  for part in "${parts[@]}"; do
    local procs=${part%x*}
    local omp=${part#*x}
    local label="B_${procs}proc_omp${omp}"
    local logfile="$RUN_DIR/${label}.log"
    echo "[RUN] $label" | tee "$logfile"
    local start=$(date +%s.%N)
    local i=0
    while (( i < procs )); do
      local begin=$(( i * omp ))
      local end=$(( begin + omp - 1 ))
      local slice=("${cpus[@]:$begin:$omp}")
      local cpu_range=$(join_by_comma "${slice[@]}")
      (
        if [[ $HAVE_NUMACTL -eq 1 ]]; then
          exec numactl --cpunodebind="$SOCKET" --membind="$SOCKET" \
            taskset -c "$cpu_range" env OMP_NUM_THREADS="$omp" \
            "$BINARY" "$ALGO" "$DATASET" "$RVAL"
        else
          exec taskset -c "$cpu_range" env OMP_NUM_THREADS="$omp" \
            "$BINARY" "$ALGO" "$DATASET" "$RVAL"
        fi
      ) >> "$logfile" 2>&1 &
      i=$(( i + 1 ))
    done
    wait
    local end=$(date +%s.%N)
    awk -v s="$start" -v e="$end" 'BEGIN{printf "WALL_SECONDS: %.3f\n", (e-s)}' >> "$logfile"
  done
}

echo "[START] Profiling OMP/NUMA impact (results under $RUN_DIR)"

# Section A: single-process OMP sweep (socket-local)
single_process_omp_sweep

# Section B: multi-process equal total threads (socket-local)
multi_process_equal_budget

echo "[DONE] See logs under: $RUN_DIR"

