#!/usr/bin/env python3
import argparse
import csv
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run_cmd(cmd, log_path: Path) -> int:
    with open(log_path, 'w') as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        proc.wait()
        return proc.returncode


def lscpu_sockets() -> dict:
    # Returns mapping socket_id -> list of logical CPUs (first logical per core)
    # Requires: lscpu
    out = subprocess.check_output(['lscpu', '-e=CPU,CORE,SOCKET,ONLINE'], text=True)
    lines = out.strip().splitlines()
    header = lines[0].split()
    idx_cpu = header.index('CPU')
    idx_core = header.index('CORE')
    idx_socket = header.index('SOCKET')
    idx_online = header.index('ONLINE')

    seen_per_socket_core = {}
    cpus_by_socket = {}
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        online = parts[idx_online]
        if online.lower() != 'yes':
            continue
        cpu = int(parts[idx_cpu])
        core = int(parts[idx_core])
        sock = int(parts[idx_socket])
        seen = seen_per_socket_core.setdefault(sock, set())
        if core in seen:
            continue
        seen.add(core)
        cpus_by_socket.setdefault(sock, []).append(cpu)
    # Keep stable order
    for s in cpus_by_socket:
        cpus_by_socket[s].sort()
    return cpus_by_socket


def comma_join(ints) -> str:
    return ','.join(str(x) for x in ints)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def section_single_omp_sweep(run_dir: Path, binary: str, dataset: str, algo: int, r_val: float, socket: int,
                             omp_list=(1, 2, 3, 4, 6, 8, 12, 16, 24)):
    cpus_by_socket = lscpu_sockets()
    if socket not in cpus_by_socket:
        raise RuntimeError(f"Socket {socket} not found by lscpu")
    first_cpu = cpus_by_socket[socket][0]
    have_numactl = which('numactl')
    results = []
    for omp in omp_list:
        label = f"A_socket{socket}_single_omp{omp}"
        log_path = run_dir / f"{label}.log"
        cmd = []
        if have_numactl:
            cmd += ['numactl', f'--cpunodebind={socket}', f'--membind={socket}']
        cmd += ['taskset', '-c', str(first_cpu), 'env', f'OMP_NUM_THREADS={omp}', binary, str(algo), dataset, str(r_val)]
        start = time.perf_counter()
        rc = run_cmd(cmd, log_path)
        end = time.perf_counter()
        results.append({
            'section': 'A',
            'label': label,
            'socket': socket,
            'procs': 1,
            'omp': omp,
            'wall_seconds': round(end - start, 3),
            'rc': rc,
        })
        with open(log_path, 'a') as f:
            f.write(f"\nWALL_SECONDS: {results[-1]['wall_seconds']:.3f}\n")
    return results


def section_multi_equal_budget(run_dir: Path, binary: str, dataset: str, algo: int, r_val: float, socket: int):
    cpus_by_socket = lscpu_sockets()
    if socket not in cpus_by_socket:
        raise RuntimeError(f"Socket {socket} not found by lscpu")
    cpus = cpus_by_socket[socket]
    n = len(cpus)
    have_numactl = which('numactl')
    parts = []
    for g in (1, 2, 4, 8):
        procs = n // g
        if procs >= 1:
            parts.append((procs, g))
    parts.append((1, n))  # 1 x n

    results = []
    for procs, omp in parts:
        label = f"B_socket{socket}_{procs}proc_omp{omp}"
        log_path = run_dir / f"{label}.log"
        procs_list = []
        start = time.perf_counter()
        with open(log_path, 'w') as f:
            f.write(f"LABEL: {label}\n")
        for i in range(procs):
            begin = i * omp
            slice_cpus = cpus[begin:begin + omp]
            cpu_arg = comma_join(slice_cpus)
            cmd = []
            if have_numactl:
                cmd += ['numactl', f'--cpunodebind={socket}', f'--membind={socket}']
            cmd += ['taskset', '-c', cpu_arg, 'env', f'OMP_NUM_THREADS={omp}', binary, str(algo), dataset, str(r_val)]
            p = subprocess.Popen(cmd, stdout=open(log_path, 'a'), stderr=subprocess.STDOUT)
            procs_list.append(p)
        rc = 0
        for p in procs_list:
            p.wait()
            rc = rc or p.returncode
        end = time.perf_counter()
        res = {
            'section': 'B',
            'label': label,
            'socket': socket,
            'procs': procs,
            'omp': omp,
            'wall_seconds': round(end - start, 3),
            'rc': rc,
        }
        results.append(res)
        with open(log_path, 'a') as f:
            f.write(f"\nWALL_SECONDS: {res['wall_seconds']:.3f}\n")
    return results


def save_csv(path: Path, rows):
    if not rows:
        return
    cols = ['section', 'label', 'socket', 'procs', 'omp', 'wall_seconds', 'rc']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def try_plot(run_dir: Path, csv_path: Path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as e:
        print(f"[WARN] Plotting skipped (matplotlib/pandas not available): {e}")
        return
    import pandas as pd
    df = pd.read_csv(csv_path)
    for section in ['A', 'B']:
        d = df[df['section'] == section].copy()
        if d.empty:
            continue
        d.sort_values('wall_seconds', inplace=True)
        plt.figure(figsize=(10, max(3, len(d) * 0.3)))
        plt.barh(d['label'], d['wall_seconds'])
        plt.xlabel('Wall seconds (lower is better)')
        plt.title(f'OMP/NUMA Profile Section {section}')
        plt.tight_layout()
        out = run_dir / f'plot_section_{section}.png'
        plt.savefig(out)
        plt.close()
        print(f"[INFO] Saved plot: {out}")


def main():
    ap = argparse.ArgumentParser(description='Profile OMP/NUMA impact for C++ main binary')
    ap.add_argument('--binary', default='./main')
    ap.add_argument('--dataset', default='dataset/Cora')
    ap.add_argument('--algo', type=int, default=9)
    ap.add_argument('--r', type=float, default=0.5)
    ap.add_argument('--sockets', default='0', help='Comma-separated socket ids, e.g., 0 or 0,1')
    ap.add_argument('--outdir', default='tools/prof_results')
    args = ap.parse_args()

    if not Path(args.binary).exists():
        print(f"[ERROR] Binary not found: {args.binary}")
        sys.exit(2)

    outdir = Path(args.outdir)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = outdir / f'run_{stamp}'
    ensure_dir(run_dir)

    print(f"[INFO] Writing to: {run_dir}")
    sockets = [int(s.strip()) for s in args.sockets.split(',') if s.strip()]

    all_rows = []
    for s in sockets:
        all_rows += section_single_omp_sweep(run_dir, args.binary, args.dataset, args.algo, args.r, s)
        all_rows += section_multi_equal_budget(run_dir, args.binary, args.dataset, args.algo, args.r, s)

    csv_path = run_dir / 'results.csv'
    save_csv(csv_path, all_rows)
    print(f"[INFO] Saved CSV: {csv_path}")

    try_plot(run_dir, csv_path)
    print("[DONE] Inspect CSV and plots for fastest configuration.")


if __name__ == '__main__':
    main()

