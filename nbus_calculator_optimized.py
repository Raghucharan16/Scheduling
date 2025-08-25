#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized N-Bus Calculator with parallel processing, caching, and smart heuristics
"""

from pathlib import Path
import sys
import json
import pickle
import hashlib
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Remove ortools import from here - will import inside solve() function
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache
import time
from dataclasses import dataclass, asdict
import platform

# ─────────────────────────────────────────────────────────────
# Global time grid & ops rules
# ─────────────────────────────────────────────────────────────
SLOTS_PER_DAY   = 48                      # 30-minute slots
FORBIDDEN       = {1,2,3,4,5,6}           # 00:30–03:00 forbidden for departures
ALLOWED         = [s for s in range(SLOTS_PER_DAY) if s not in FORBIDDEN]
TIME_LIMIT_SEC  = 18000                     # per-solve time (seconds)

# ─────────────────────────────────────────────────────────────
# Cache configuration
# ─────────────────────────────────────────────────────────────
CACHE_DIR = Path("cache/nbus_solutions")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class SolutionResult:
    """Store solution results for caching"""
    N: int
    A: int
    B: int
    trips: int
    avg_epk: float
    total_epk: float
    compute_time: float
    
    @property
    def is_valid(self) -> bool:
        return self.trips > 0

# ─────────────────────────────────────────────────────────────
# Helpers (same as original)
# ─────────────────────────────────────────────────────────────
def slot_to_idx(s: str) -> int:
    s = s.strip()
    parts = s.split(":")
    if len(parts) == 2:
        hh, mm = parts
    elif len(parts) == 3:
        hh, mm, _ = parts
    else:
        raise ValueError(f"Invalid time format: {s}")
    return int(hh) * 2 + (int(mm) // 30)

def idx_to_hhmm(idx: int) -> str:
    return f"{idx//2:02d}:{(idx%2)*30:02d}:00"

def read_month_days(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.isdigit()], key=lambda x: int(x))

# ─────────────────────────────────────────────────────────────
# Cache management
# ─────────────────────────────────────────────────────────────
def get_cache_key(route_fwd: str, route_rev: str, travel_h: float, 
                  charge_h: float, idle_h: float, A: int, B: int,
                  day_labels: List[str]) -> str:
    """Generate unique cache key for a specific problem configuration"""
    params = {
        'route_fwd': route_fwd,
        'route_rev': route_rev,
        'travel_h': travel_h,
        'charge_h': charge_h,
        'idle_h': idle_h,
        'A': A,
        'B': B,
        'days': len(day_labels)
    }
    key_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()

def load_from_cache(cache_key: str) -> Optional[SolutionResult]:
    """Load cached solution if exists"""
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    return None

def save_to_cache(cache_key: str, result: SolutionResult):
    """Save solution to cache"""
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)

# ─────────────────────────────────────────────────────────────
# Original solver functions (keeping your optimized version)
# ─────────────────────────────────────────────────────────────
def load_corridor(csv_path: Path,
                  route_fwd: str, route_rev: str
                 ) -> Tuple[Dict[Tuple[str,str,int], float], List[str]]:
    df = pd.read_csv(csv_path)
    for col in ("slot","route"):
        if col not in df.columns:
            raise ValueError(f"{csv_path}: missing required column '{col}'")
    day_cols = read_month_days(df)
    if not day_cols:
        raise ValueError(f"{csv_path}: no day columns found (expect '1'..'31').")

    df = df[df["route"].isin([route_fwd, route_rev])].copy()
    if df.empty:
        raise ValueError(f"{csv_path}: routes {route_fwd},{route_rev} not found.")

    df["slot_idx"] = (
        pd.to_timedelta(df["slot"]).dt.components.hours * 2 +
        pd.to_timedelta(df["slot"]).dt.components.minutes // 30
    ).astype(int)

    epk_map: Dict[Tuple[str,str,int], float] = {}
    for _, row in df.iterrows():
        r = row["route"]
        s = int(row["slot_idx"])
        for d in day_cols:
            epk_map[(r, d, s)] = float(row[d])
    return epk_map, day_cols

# [Include your original solve() function here - keeping it as is]
def solve(epk, days, routeAB, routeBA,
          busesA, busesB,
          travel_s, charge_s, idle_s, limit=300,
          deterministic=True, seed=123, enable_logs=False,
          keep_per_day=None,
          buffer_per_day=6,
          try_chain_prune=True,
          escalate_steps=(12, 24, 48)
          ):
    """Your original optimized solver - keeping as is"""
    # [Copy your entire solve function here]
    # I'm abbreviating for space, but include the full function
    from ortools.sat.python import cp_model

    TOTAL = travel_s + charge_s
    SLOTS = SLOTS_PER_DAY
    D     = len(days)
    H     = D * SLOTS
    B     = busesA + busesB

    home = ["A"] * busesA + ["B"] * busesB
    def route_for(b, k):
        if home[b] == "A":
            return routeAB if (k % 2 == 0) else routeBA
        else:
            return routeBA if (k % 2 == 0) else routeAB

    def dep_station(rt): return "A" if rt == routeAB else "B"

    def build_route_day_candidates(perday_keep):
        route_day_abs = {routeAB: {d: [] for d in range(D)},
                         routeBA: {d: [] for d in range(D)}}
        for rt in (routeAB, routeBA):
            for d in range(D):
                cand = []
                for s in ALLOWED:
                    v = float(epk.get((rt, d, s), 0.0))
                    cand.append((v, d * SLOTS + s))
                cand.sort(key=lambda x: (x[0], x[1]), reverse=True)
                keep = cand[:max(perday_keep, B + buffer_per_day)]
                if len(keep) < B:
                    keep = [(float(epk.get((rt, d, s), 0.0)), d * SLOTS + s) for s in ALLOWED]
                    keep.sort(key=lambda x: (x[0], x[1]), reverse=True)
                route_day_abs[rt][d] = [abs_t for _, abs_t in keep]
        return route_day_abs

    def chain_prune(route_day_abs, first_route):
        C = []
        for k in range(2 * D):
            rt = first_route if (k % 2 == 0) else (routeBA if first_route == routeAB else routeAB)
            C.append(sorted(route_day_abs[rt][d] for d in range(D)))
            C[-1] = sorted([t for sub in C[-1] for t in sub])

        changed = True
        while changed:
            changed = False
            for k in range(2 * D - 1):
                nxt = C[k + 1]
                if not nxt: return None
                keep = []
                j = 0
                for t in C[k]:
                    lo, hi = t + TOTAL, t + TOTAL + idle_s
                    while j < len(nxt) and nxt[j] < lo: j += 1
                    if j < len(nxt) and nxt[j] <= hi:
                        keep.append(t)
                if len(keep) != len(C[k]):
                    C[k] = keep; changed = True
                if not C[k]: return None
            for k in range(1, 2 * D):
                prv = C[k - 1]
                if not prv: return None
                keep = []
                i = 0
                for t in C[k]:
                    lo, hi = t - (TOTAL + idle_s), t - TOTAL
                    while i < len(prv) and prv[i] < lo: i += 1
                    if i < len(prv) and prv[i] <= hi:
                        keep.append(t)
                if len(keep) != len(C[k]):
                    C[k] = keep; changed = True
                if not C[k]: return None
        return C

    perday_base = (B + buffer_per_day) if keep_per_day is None else keep_per_day
    tried_keeps = [perday_base] + [max(perday_base, k) for k in escalate_steps]
    tried_keeps.append(len(ALLOWED))

    last_error = None
    for perday_keep in tried_keeps:
        try:
            route_day_abs = build_route_day_candidates(perday_keep)

            if try_chain_prune:
                C_A = chain_prune(route_day_abs, routeAB)
                C_B = chain_prune(route_day_abs, routeBA)
                if (busesA > 0 and C_A is None) or (busesB > 0 and C_B is None):
                    raise RuntimeError("chain_prune_failed")
            else:
                C_A = [sorted([t for d in range(D) for t in route_day_abs[routeAB if k % 2 == 0 else routeBA][d]])
                       for k in range(2 * D)]
                C_B = [sorted([t for d in range(D) for t in route_day_abs[routeBA if k % 2 == 0 else routeAB][d]])
                       for k in range(2 * D)]

            m = cp_model.CpModel()
            K = 2 * D
            y = {}
            st = {}
            occ = {}
            day_route_vars = {(routeAB, d): [] for d in range(D)}
            day_route_vars.update({(routeBA, d): [] for d in range(D)})
            first_starts_A, first_starts_B = [], []

            for b in range(B):
                seq = C_A if home[b] == "A" else C_B
                for k in range(K):
                    cands = seq[k]
                    lits = [m.NewBoolVar(f"y_{b}_{k}_{abs_t}") for abs_t in cands]
                    for abs_t, v in zip(cands, lits):
                        y[(b, k, abs_t)] = v
                        stn = dep_station(route_for(b, k))
                        occ.setdefault((stn, abs_t), []).append(v)
                        d = abs_t // SLOTS
                        day_route_vars[(route_for(b, k), d)].append(v)
                    m.AddExactlyOne(lits)

                    st[(b, k)] = m.NewIntVar(0, H - 1, f"st_{b}_{k}")
                    m.Add(st[(b, k)] == sum(abs_t * y[(b, k, abs_t)] for abs_t in cands))

                if home[b] == "A":
                    first_starts_A.append(st[(b, 0)])
                else:
                    first_starts_B.append(st[(b, 0)])

            for b in range(B):
                for k in range(K - 1):
                    m.Add(st[(b, k + 1)] >= st[(b, k)] + TOTAL)
                    m.Add(st[(b, k + 1)] - st[(b, k)] <= TOTAL + idle_s)
                for k in range(K - 2):
                    m.Add(st[(b, k + 2)] >= st[(b, k)] + SLOTS)

            for (stn, abs_t), vars_list in occ.items():
                m.AddAtMostOne(vars_list)

            for d in range(D):
                m.Add(sum(day_route_vars[(routeAB, d)]) == B)
                m.Add(sum(day_route_vars[(routeBA, d)]) == B)

            for arr in (first_starts_A, first_starts_B):
                for i in range(len(arr) - 1):
                    m.Add(arr[i] <= arr[i + 1])

            obj = []
            for (b, k, abs_t), var in y.items():
                d = abs_t // SLOTS
                s = abs_t % SLOTS
                rt = route_for(b, k)
                val = float(epk.get((rt, d, s), 0.0))
                obj.append(int(val * 100000) * var + abs_t * var)
            m.Maximize(sum(obj))

            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = limit
            solver.parameters.log_search_progress = enable_logs
            solver.parameters.num_search_workers = 8
            if deterministic:
                solver.parameters.random_seed = seed

            status = solver.Solve(m)
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                raise RuntimeError("solve_failed")

            def s2t_local(s): return f"{s//2:02d}:{(s%2)*30:02d}:00"
            sched = {}
            for b in range(B):
                bus = f"Bus-{b+1:02d}"
                sched[bus] = {dy: [] for dy in days}
                for k in range(K):
                    abs_val = int(solver.Value(st[(b, k)]))
                    d = abs_val // SLOTS
                    s = abs_val % SLOTS
                    rt = route_for(b, k)
                    if 0 <= d < D:
                        sched[bus][days[d]].append({
                            "route": rt,
                            "startTime": s2t_local(s),
                            "midPointTime": s2t_local((s + travel_s // 2) % SLOTS),
                            "endTime": s2t_local((s + travel_s) % SLOTS),
                            "epk": round(float(epk.get((rt, d, s), 0.0)), 2)
                        })
                for dy in days:
                    sched[bus][dy].sort(key=lambda tr: tr["startTime"])
            return sched

        except RuntimeError as e:
            last_error = str(e)
            continue

    raise RuntimeError(f"No feasible schedule – {last_error or 'model infeasible'}.")

def solve_corridor_cached(
    epk_map: Dict[Tuple[str,str,int], float],
    day_labels: List[str],
    route_fwd: str, route_rev: str,
    travel_h: float, charge_h: float,
    n_origin: int, n_dest: int,
    idle_h: float = 4.0,
    time_limit: int = TIME_LIMIT_SEC,
    use_cache: bool = True,
    deterministic: bool = True,
    seed: int = 123,
    enable_logs: bool = False,
) -> SolutionResult:
    """Cached version of solve_corridor that returns SolutionResult"""
    
    start_time = time.time()
    
    # Check cache first
    if use_cache:
        cache_key = get_cache_key(route_fwd, route_rev, travel_h, charge_h, 
                                 idle_h, n_origin, n_dest, day_labels)
        cached = load_from_cache(cache_key)
        if cached:
            print(f"  Cache hit for A={n_origin}, B={n_dest}")
            return cached
    
    # Prepare indexed EPK data
    D = len(day_labels)
    day_to_idx = {d:i for i,d in enumerate(day_labels)}
    epk_indexed: Dict[Tuple[str,int,int], float] = {}
    for d_lbl, d_idx in day_to_idx.items():
        for s in ALLOWED:
            epk_indexed[(route_fwd, d_idx, s)] = float(epk_map.get((route_fwd, d_lbl, s), 0.0))
            epk_indexed[(route_rev, d_idx, s)] = float(epk_map.get((route_rev, d_lbl, s), 0.0))

    travel_s = int(round(travel_h * 2))
    charge_s = int(round(charge_h * 2))
    idle_s   = int(round(idle_h   * 2))

    try:
        sched = solve(
            epk=epk_indexed,
            days=day_labels,
            routeAB=route_fwd, routeBA=route_rev,
            busesA=n_origin, busesB=n_dest,
            travel_s=travel_s, charge_s=charge_s, idle_s=idle_s,
            limit=time_limit,
            deterministic=deterministic, seed=seed, enable_logs=enable_logs
        )
        
        # Calculate metrics
        SLOTS = SLOTS_PER_DAY
        trips = 0
        total_epk = 0.0
        
        for _, by_day in sched.items():
            for d_lbl in day_labels:
                for tr in by_day.get(d_lbl, []):
                    trips += 1
                    total_epk += float(tr["epk"])
        
        avg_epk = (total_epk / trips) if trips > 0 else 0.0
        
    except Exception as e:
        # If solve fails, return zero result
        trips = 0
        avg_epk = 0.0
        total_epk = 0.0
    
    compute_time = time.time() - start_time
    result = SolutionResult(
        N=n_origin + n_dest,
        A=n_origin,
        B=n_dest,
        trips=trips,
        avg_epk=avg_epk,
        total_epk=total_epk,
        compute_time=compute_time
    )
    
    # Save to cache
    if use_cache:
        cache_key = get_cache_key(route_fwd, route_rev, travel_h, charge_h, 
                                 idle_h, n_origin, n_dest, day_labels)
        save_to_cache(cache_key, result)
    
    return result

# ─────────────────────────────────────────────────────────────
# Parallel processing wrapper
# ─────────────────────────────────────────────────────────────
def solve_split_task(args):
    """Worker function for parallel processing"""
    (epk_map, day_labels, route_fwd, route_rev, 
     travel_h, charge_h, A, B, idle_h, time_limit) = args
    
    return solve_corridor_cached(
        epk_map, day_labels,
        route_fwd, route_rev,
        travel_h, charge_h,
        n_origin=A, n_dest=B,
        idle_h=idle_h,
        time_limit=time_limit,
        use_cache=True
    )

# ─────────────────────────────────────────────────────────────
# Smart exploration strategies
# ─────────────────────────────────────────────────────────────
def get_smart_split_order(N: int) -> List[Tuple[int, int]]:
    """
    Return splits in smart order:
    1. Balanced splits first (often optimal)
    2. Then gradually explore extremes
    """
    splits = []
    center = N // 2
    
    # Start with balanced
    splits.append((center, N - center))
    if N % 2 == 1:
        splits.append((center + 1, N - center - 1))
    
    # Expand outward
    for offset in range(1, center + 1):
        if center - offset >= 0:
            splits.append((center - offset, N - (center - offset)))
        if center + offset <= N and offset != 0:
            splits.append((center + offset, N - (center + offset)))
    
    # Ensure all splits are included (avoid duplicates)
    all_splits = set((a, N - a) for a in range(N + 1))
    for s in all_splits:
        if s not in splits:
            splits.append(s)
    
    # Remove any duplicates that might have been created
    seen = set()
    unique_splits = []
    for s in splits:
        if s not in seen:
            seen.add(s)
            unique_splits.append(s)
    
    return unique_splits

def estimate_solution_quality(A: int, B: int, N: int) -> float:
    """
    Heuristic to estimate solution quality without solving.
    Balanced splits tend to be better.
    """
    balance_score = 1.0 - abs(A - B) / N
    return balance_score

# ─────────────────────────────────────────────────────────────
# Balanced-window split generator (prunes search space)
# ─────────────────────────────────────────────────────────────
def get_balanced_window_splits(N: int, window: int = 2) -> List[Tuple[int, int]]:
    """
    Return splits centered at the balanced configuration with a limited window.
    Example: N=10, window=2 → consider A in {3,4,5,6,7} with pairs (A, N-A),
    ordered balanced-first.
    """
    if N < 0:
        return []
    center = N // 2
    candidates: List[Tuple[int,int]] = []
    # Balanced first
    candidates.append((center, N - center))
    # If odd, include the other central split next
    if N % 2 == 1:
        candidates.append((center + 1, N - (center + 1)))

    # Expand within window
    for offset in range(1, window + 1):
        left = center - offset
        right = center + offset
        if 0 <= left <= N:
            candidates.append((left, N - left))
        if 0 <= right <= N and right != left:
            candidates.append((right, N - right))

    # Deduplicate preserving order
    seen = set()
    pruned: List[Tuple[int,int]] = []
    for s in candidates:
        if s not in seen:
            seen.add(s)
            pruned.append(s)
    return pruned

def get_required_splits(N: int, window: int = 2) -> List[Tuple[int, int]]:
    """
    Exact split set per user's rule:
    - N=1: (1,0), (0,1)
    - N=2: (1,1)
    - N>=3: let base=ceil(N/2). Consider deltas d=0..window and pairs
      (base±d, N-(base±d)), but skip pairs where min(A,B) < base-window or A==0 or B==0.
      Order: balanced first (d=0), then increasing d with + then -.
    """
    if N <= 0:
        return []
    if N == 1:
        return [(1,0), (0,1)]
    if N == 2:
        return [(1,1)]

    import math
    base = math.ceil(N / 2)
    min_allowed = max(1, base - window)
    splits: List[Tuple[int,int]] = []

    # Helper to add if valid
    def maybe_add(a: int):
        b = N - a
        if a < 0 or a > N or b < 0 or b > N:
            return
        if a == 0 or b == 0:
            return
        if min(a, b) < min_allowed:
            return
        pair = (a, b)
        if pair not in splits:
            splits.append(pair)

    # d = 0
    maybe_add(base)
    # d = 1..window (plus then minus)
    for d in range(1, window + 1):
        maybe_add(base + d)
        maybe_add(base - d)

    return splits

# ─────────────────────────────────────────────────────────────
# Optimized main solver with parallel processing
# ─────────────────────────────────────────────────────────────
def run_optimized_total_N(
    epk_map, day_labels,
    route_fwd, route_rev,
    travel_h, charge_h,
    N: int,
    idle_h: float = 4.0,
    time_limit: int = TIME_LIMIT_SEC,
    select_metric: str = "avg_epk",
    max_workers: int = 4,
    use_cache: bool = True,
    progressive_time: bool = True,
    early_stop_threshold: float = 0.95,  # Stop if we find a solution this good
    use_balanced_window: bool = True,
    balanced_window_size: int = 2
):
    """
    Optimized solver with multiple strategies:
    - Parallel processing of splits
    - Smart exploration order
    - Caching of results
    - Progressive time limits
    - Early stopping
    """
    
    print(f"\n🚀 Optimized solver for N={N} buses")
    print(f"   Strategies: parallel={max_workers} workers, cache={use_cache}, progressive={progressive_time}")
    
    start_time = time.time()
    
    # Metric sanitization
    if select_metric not in ("avg_epk", "total_epk", "trips"):
        select_metric = "avg_epk"

    # Build split list per user's rule for minimal coverage
    splits = get_required_splits(N, window=balanced_window_size) if use_balanced_window else get_smart_split_order(N)
    print(f"   Evaluating {len(splits)} splits in optimized order...")
    
    # Prepare tasks for parallel processing (balanced splits get more time)
    tasks = []
    center = N // 2
    for i, (A, B) in enumerate(splits):
        # Progressive time limit: give more time to promising splits
        if progressive_time:
            if use_balanced_window:
                # prioritize within balanced window; degrade with distance
                dist = abs(A - center)
                # Within window → full time; outside → reduced
                in_window = dist <= balanced_window_size
                base = 0.75 if in_window else 0.5
                quality_est = estimate_solution_quality(A, B, N)
                split_time = int(time_limit * (base + 0.5 * quality_est))
            else:
                quality_est = estimate_solution_quality(A, B, N)
                split_time = int(time_limit * (0.5 + 0.5 * quality_est))
        else:
            split_time = time_limit
        
        tasks.append((
            epk_map, day_labels, route_fwd, route_rev,
            travel_h, charge_h, A, B, idle_h, split_time
        ))
    
    # Process in parallel with early stopping
    results = []
    best_so_far = None
    
    # Choose executor based on platform
    if platform.system() == "Windows":
        # Use ThreadPoolExecutor on Windows to avoid multiprocessing issues
        executor_class = ThreadPoolExecutor
        print(f"   Using ThreadPoolExecutor (Windows compatibility)")
    else:
        # Use ProcessPoolExecutor on Unix-like systems
        executor_class = ProcessPoolExecutor
        print(f"   Using ProcessPoolExecutor (Unix/Linux)")
    
    with executor_class(max_workers=max_workers) as executor:
        # Submit only up to number of tasks; avoid oversubmitting
        capacity = min(max_workers, len(tasks))
        futures = {executor.submit(solve_split_task, tasks[i]): i for i in range(capacity)}
        next_idx = capacity
        
        while futures:
            # Process completed futures (no per-result timeout; as_completed yields only finished ones)
            done_futures = []
            try:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        done_futures.append(future)

                        # Update best
                        if result.is_valid:
                            if best_so_far is None or is_better(result, best_so_far, select_metric):
                                best_so_far = result
                                print(f"   ✓ New best: A={result.A}, B={result.B}, "
                                      f"EPK={result.avg_epk:.2f}, trips={result.trips}")

                            # Early stopping: do not submit more tasks; just drain current
                            if select_metric == "avg_epk" and result.avg_epk >= early_stop_threshold * 100:
                                print(f"   🎯 Early stop: Found excellent solution (EPK={result.avg_epk:.2f})")
                                next_idx = len(tasks)
                                break
                    except Exception:
                        done_futures.append(future)
                        # keep going; treat as a failed task
                        continue
            except Exception:
                # Defensive: if as_completed raises unexpectedly, break to avoid stall
                pass
            
            # Remove completed futures
            for f in done_futures:
                del futures[f]
            
            # Submit next task(s) if available
            while next_idx < len(tasks) and len(futures) < max_workers:
                future = executor.submit(solve_split_task, tasks[next_idx])
                futures[future] = next_idx
                next_idx += 1
    
    # Create DataFrame with results
    df = pd.DataFrame([asdict(r) for r in results])
    df = df.sort_values(['avg_epk', 'trips', 'A'], ascending=[False, False, True])
    
    # Save results
    output_file = f"optimized_N{N}_results.csv"
    df.to_csv(output_file, index=False)
    
    compute_time = time.time() - start_time
    
    if best_so_far:
        print(f"\n✅ Best solution for N={N}:")
        print(f"   A={best_so_far.A}, B={best_so_far.B}")
        print(f"   Trips={best_so_far.trips}, Avg EPK={best_so_far.avg_epk:.2f}")
        print(f"   Total time: {compute_time:.1f}s (vs ~{len(splits)*time_limit}s sequential)")
        print(f"   Speedup: {len(splits)*time_limit/compute_time:.1f}x")
        print(f"   Results saved to {output_file}")
    else:
        print(f"\n❌ No feasible solution found for N={N}")
    
    return best_so_far, df

def is_better(cand: SolutionResult, best: SolutionResult, metric: str) -> bool:
    """Compare two solutions based on metric"""
    if metric == "avg_epk":
        if cand.avg_epk != best.avg_epk:
            return cand.avg_epk > best.avg_epk
        if cand.trips != best.trips:
            return cand.trips > best.trips
        return cand.A < best.A
    elif metric == "total_epk":
        if cand.total_epk != best.total_epk:
            return cand.total_epk > best.total_epk
        if cand.trips != best.trips:
            return cand.trips > best.trips
        return cand.A < best.A
    else:  # trips
        if cand.trips != best.trips:
            return cand.trips > best.trips
        if cand.avg_epk != best.avg_epk:
            return cand.avg_epk > best.avg_epk
        return cand.A < best.A

# ─────────────────────────────────────────────────────────────
# Scan multiple N values efficiently
# ─────────────────────────────────────────────────────────────
def scan_n_range_optimized(
    epk_map, day_labels,
    route_fwd, route_rev,
    travel_h, charge_h,
    N_min: int, N_max: int,
    idle_h: float = 4.0,
    time_limit: int = TIME_LIMIT_SEC,
    select_metric: str = "avg_epk",
    max_workers: int = 4
):
    """Scan a range of N values and find the best overall"""
    
    print(f"\n📊 Scanning N from {N_min} to {N_max}")
    
    all_results = []
    best_overall = None
    
    for N in range(N_min, N_max + 1):
        best_n, _ = run_optimized_total_N(
            epk_map, day_labels,
            route_fwd, route_rev,
            travel_h, charge_h,
            N=N,
            idle_h=idle_h,
            time_limit=time_limit,
            select_metric=select_metric,
            max_workers=max_workers
        )
        
        if best_n:
            all_results.append(best_n)
            if best_overall is None or is_better(best_n, best_overall, select_metric):
                best_overall = best_n
    
    # Save summary
    summary_df = pd.DataFrame([asdict(r) for r in all_results])
    summary_file = f"n_scan_{N_min}to{N_max}_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    # Plot results
    if all_results:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot([r.N for r in all_results], [r.avg_epk for r in all_results], 'o-')
        plt.xlabel("Total Buses (N)")
        plt.ylabel("Best Average EPK")
        plt.title("Average EPK vs Total Buses")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot([r.N for r in all_results], [r.trips for r in all_results], 's-')
        plt.xlabel("Total Buses (N)")
        plt.ylabel("Total Trips")
        plt.title("Total Trips vs Total Buses")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = f"n_scan_{N_min}to{N_max}_plot.png"
        plt.savefig(plot_file, dpi=160)
        plt.close()
        
        print(f"\n📈 Results saved:")
        print(f"   Summary: {summary_file}")
        print(f"   Plot: {plot_file}")
    
    if best_overall:
        print(f"\n🏆 Best overall (by {select_metric}):")
        print(f"   N={best_overall.N} (A={best_overall.A}, B={best_overall.B})")
        print(f"   Trips={best_overall.trips}, Avg EPK={best_overall.avg_epk:.2f}")
    
    return best_overall, summary_df

def generate_epk_curve(
    epk_map, day_labels,
    route_fwd, route_rev,
    travel_h, charge_h,
    N_max: int,
    idle_h: float = 4.0,
    time_limit: int = TIME_LIMIT_SEC,
    select_metric: str = "avg_epk",
    max_workers: int = 4,
    save_plots: bool = True,
    plot_title: str = None,
    balanced_window_size: int = 1
):
    """
    Generate EPK curve from 1 to N_max buses with detailed visualization.
    This is the main function you want to use for generating the EPK vs N graph.
    """
    
    if plot_title is None:
        plot_title = f"EPK Curve: {route_fwd} ↔ {route_rev} (Travel: {travel_h}h, Charge: {charge_h}h)"
    
    print(f"\n📈 Generating EPK Curve from 1 to {N_max} buses")
    print(f"   Route: {route_fwd} ↔ {route_rev}")
    print(f"   Travel time: {travel_h}h, Charge time: {charge_h}h")
    print(f"   Time limit per solve: {time_limit}s")
    print(f"   Workers: {max_workers}")
    
    start_time = time.time()
    all_results = []
    
    # Scan from 1 to N_max buses
    for N in range(1, N_max + 1):
        print(f"\n🚌 Evaluating N={N} buses...")
        
        try:
            best_n, _ = run_optimized_total_N(
                epk_map, day_labels,
                route_fwd, route_rev,
                travel_h, charge_h,
                N=N,
                idle_h=idle_h,
                time_limit=time_limit,
                select_metric=select_metric,
                max_workers=max_workers,
                use_balanced_window=True,
                balanced_window_size=balanced_window_size
            )
            
            if best_n and best_n.is_valid:
                all_results.append(best_n)
                print(f"   ✅ N={N}: Best EPK={best_n.avg_epk:.2f}, Trips={best_n.trips}, Split=({best_n.A},{best_n.B})")
            else:
                print(f"   ❌ N={N}: No feasible solution found")
                # Add a placeholder for plotting
                all_results.append(SolutionResult(
                    N=N, A=0, B=0, trips=0, avg_epk=0.0, total_epk=0.0, compute_time=0.0
                ))
                
        except Exception as e:
            print(f"   ❌ N={N}: Error - {e}")
            # Add a placeholder for plotting
            all_results.append(SolutionResult(
                N=N, A=0, B=0, trips=0, avg_epk=0.0, total_epk=0.0, compute_time=0.0
            ))
    
    total_time = time.time() - start_time
    
    # Create comprehensive visualization
    if save_plots:
        create_epk_curve_plots(all_results, plot_title, route_fwd, route_rev)
    
    # Save detailed results
    results_df = pd.DataFrame([asdict(r) for r in all_results])
    results_file = f"epk_curve_1to{N_max}_detailed.csv"
    results_df.to_csv(results_file, index=False)

    # Save compact JSON summary per N as requested
    try:
        import json
        json_records = []
        for r in all_results:
            json_records.append({
                "n_buses": int(r.N),
                "buses_at_a": int(r.A),
                "buses_at_b": int(r.B),
                "epk": float(round(r.avg_epk, 2))
            })
        json_file = f"epk_curve_1to{N_max}_summary.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_records, f, ensure_ascii=False, indent=2)
    except Exception:
        json_file = None
    
    # Print summary
    print(f"\n" + "="*60)
    print(f"📊 EPK CURVE GENERATION COMPLETE")
    print(f"="*60)
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Buses evaluated: 1 to {N_max}")
    print(f"   Successful solves: {len([r for r in all_results if r.is_valid])}/{len(all_results)}")
    print(f"   Results saved to: {results_file}")
    if json_file:
        print(f"   JSON summary saved to: {json_file}")
    
    if save_plots:
        print(f"   Plots saved to: epk_curve_1to{N_max}_*.png")
    
    # Find best overall
    valid_results = [r for r in all_results if r.is_valid]
    if valid_results:
        best_overall = max(valid_results, key=lambda x: x.avg_epk)
        print(f"\n🏆 Best overall configuration:")
        print(f"   N={best_overall.N} buses (A={best_overall.A}, B={best_overall.B})")
        print(f"   EPK={best_overall.avg_epk:.2f}, Trips={best_overall.trips}")
    
    return all_results, results_df

def create_epk_curve_plots(all_results, plot_title, route_fwd, route_rev):
    """Create comprehensive EPK curve visualization plots"""
    
    # Filter out invalid results for plotting
    valid_results = [r for r in all_results if r.is_valid]
    all_n_values = [r.N for r in all_results]
    
    if not valid_results:
        print("   ⚠️ No valid results to plot")
        return
    
    # Main EPK curve plot
    plt.figure(figsize=(14, 10))
    
    # Plot 1: EPK vs N (main curve)
    plt.subplot(2, 2, 1)
    epk_values = [r.avg_epk for r in valid_results]
    n_values = [r.N for r in valid_results]
    
    plt.plot(n_values, epk_values, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    plt.fill_between(n_values, epk_values, alpha=0.3, color='#2E86AB')
    
    # Highlight best point
    best_idx = epk_values.index(max(epk_values))
    plt.plot(n_values[best_idx], epk_values[best_idx], 'ro', markersize=12, label=f'Best: N={n_values[best_idx]}')
    
    plt.xlabel("Total Buses (N)", fontsize=12)
    plt.ylabel("Best Average EPK", fontsize=12)
    plt.title(f"EPK Curve: {route_fwd} ↔ {route_rev}", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Trips vs N
    plt.subplot(2, 2, 2)
    trip_values = [r.trips for r in valid_results]
    plt.plot(n_values, trip_values, 's-', linewidth=2, markersize=8, color='#A23B72')
    plt.fill_between(n_values, trip_values, alpha=0.3, color='#A23B72')
    plt.xlabel("Total Buses (N)", fontsize=12)
    plt.ylabel("Total Trips", fontsize=12)
    plt.title("Trips vs Total Buses", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Split distribution (A vs B)
    plt.subplot(2, 2, 3)
    a_values = [r.A for r in valid_results]
    b_values = [r.B for r in valid_results]
    
    # Create stacked bar chart
    x_pos = range(len(n_values))
    plt.bar(x_pos, a_values, label='Buses at A', color='#F18F01', alpha=0.8)
    plt.bar(x_pos, b_values, bottom=a_values, label='Buses at B', color='#C73E1D', alpha=0.8)
    
    plt.xlabel("Total Buses (N)", fontsize=12)
    plt.ylabel("Number of Buses", fontsize=12)
    plt.title("Optimal Split Distribution (A vs B)", fontsize=14, fontweight='bold')
    plt.xticks(x_pos, n_values)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency analysis
    plt.subplot(2, 2, 4)
    # Calculate efficiency (EPK per bus)
    efficiency = [epk/n for epk, n in zip(epk_values, n_values)]
    plt.plot(n_values, efficiency, '^-', linewidth=2, markersize=8, color='#2E8B57')
    plt.fill_between(n_values, efficiency, alpha=0.3, color='#2E8B57')
    plt.xlabel("Total Buses (N)", fontsize=12)
    plt.ylabel("EPK per Bus", fontsize=12)
    plt.title("Efficiency: EPK per Bus", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(plot_title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save high-quality plot
    plot_file = f"epk_curve_1to{max(all_n_values)}_comprehensive.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a simple, clean EPK curve for quick reference
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, epk_values, 'o-', linewidth=3, markersize=10, color='#2E86AB')
    plt.fill_between(n_values, epk_values, alpha=0.2, color='#2E86AB')
    
    # Highlight best point
    plt.plot(n_values[best_idx], epk_values[best_idx], 'ro', markersize=15, 
             label=f'Optimal: N={n_values[best_idx]} buses\nEPK={epk_values[best_idx]:.1f}')
    
    plt.xlabel("Total Buses (N)", fontsize=14)
    plt.ylabel("Best Average EPK", fontsize=14)
    plt.title(f"EPK Curve: {route_fwd} ↔ {route_rev}", fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add trend line if enough points
    if len(n_values) > 2:
        z = np.polyfit(n_values, epk_values, 2)
        p = np.poly1d(z)
        x_trend = np.linspace(min(n_values), max(n_values), 100)
        plt.plot(x_trend, p(x_trend), '--', alpha=0.7, color='red', linewidth=2, label='Trend')
    
    plt.tight_layout()
    simple_plot_file = f"epk_curve_1to{max(all_n_values)}_simple.png"
    plt.savefig(simple_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Created comprehensive plots:")
    print(f"      • {plot_file} (4-panel detailed view)")
    print(f"      • {simple_plot_file} (clean EPK curve)")

# ─────────────────────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────────────────────
def ask_path(msg, default): v=input(f"{msg} [{default}]: ").strip(); return Path(v) if v else Path(default)
def ask_int (msg, default): v=input(f"{msg} [{default}]: ").strip(); return int(v) if v else default
def ask_float(msg, default): v=input(f"{msg} [{default}]: ").strip(); return float(v) if v else default
def ask_text(msg, default): v=input(f"{msg} [{default}]: ").strip(); return v if v else default
def ask_bool(msg, default): 
    v=input(f"{msg} [{'Y' if default else 'N'}]: ").strip().lower()
    return v in ['y', 'yes', 'true', '1'] if v else default

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    print("⚡ Optimized Electric Bus Corridor Calculator\n")
    
    # Basic inputs
    csv_path    = ask_path("Month CSV path", "cache/epk_h_v_month.csv")
    route_fwd   = ask_text("Forward route code", "H-V")
    route_rev   = ask_text("Reverse route code", "V-H")
    travel_h    = ask_float("Travel hours (one way)", 9.0)
    charge_h    = ask_float("Charge hours at destination", 2.0)
    idle_h      = ask_float("Max extra idle hours", 4.0)
    
    # Mode selection
    print("\nSelect mode:")
    print("1. Single N value (optimized)")
    print("2. Range of N values")
    print("3. Generate EPK curve (1 to N_max)")
    print("4. Compare optimized vs original (benchmark)")
    mode = ask_int("Mode", 1)
    
    # Load data
    print("\nLoading data...")
    epk_map, day_labels = load_corridor(csv_path, route_fwd, route_rev)
    print(f"Loaded {len(day_labels)} days of data")
    
    if mode == 1:
        # Single N
        N = ask_int("Total buses (N)", 5)
        time_limit = ask_int("Time limit per split (seconds)", 60)
        max_workers = ask_int("Parallel workers", 4)
        metric = ask_text("Selection metric [avg_epk|total_epk|trips]", "avg_epk")
        
        best, df = run_optimized_total_N(
            epk_map, day_labels,
            route_fwd, route_rev,
            travel_h, charge_h,
            N=N,
            idle_h=idle_h,
            time_limit=time_limit,
            select_metric=metric,
            max_workers=max_workers
        )
        
    elif mode == 2:
        # Range of N
        N_min = ask_int("Minimum N", 1)
        N_max = ask_int("Maximum N", 10)
        time_limit = ask_int("Time limit per split (seconds)", 30)
        max_workers = ask_int("Parallel workers", 4)
        metric = ask_text("Selection metric [avg_epk|total_epk|trips]", "avg_epk")
        
        best, df = scan_n_range_optimized(
            epk_map, day_labels,
            route_fwd, route_rev,
            travel_h, charge_h,
            N_min, N_max,
            idle_h=idle_h,
            time_limit=time_limit,
            select_metric=metric,
            max_workers=max_workers
        )
        
    elif mode == 3:
        # EPK Curve generation mode
        N_max = ask_int("Maximum number of buses to evaluate (1 to N_max)", 10)
        time_limit = ask_int("Time limit per split (seconds)", 300)
        max_workers = ask_int("Parallel workers", 4)
        metric = ask_text("Selection metric [avg_epk|total_epk|trips]", "avg_epk")
        bw = ask_int("Balanced window size (±k around N/2)", 2)
        
        print(f"\n📈 Generating EPK curve from 1 to {N_max} buses...")
        print(f"   This will evaluate all N values and create visualization plots")
        
        all_results, results_df = generate_epk_curve(
            epk_map, day_labels,
            route_fwd, route_rev,
            travel_h, charge_h,
            N_max=N_max,
            idle_h=idle_h,
            time_limit=time_limit,
            select_metric=metric,
            max_workers=max_workers,
            save_plots=True
        )
        
        print(f"\n🎉 EPK curve generation complete!")
        print(f"   Check the generated PNG files for visualizations")
        
    else:
        # Benchmark mode
        N = ask_int("Total buses (N) for benchmark", 4)
        time_limit = ask_int("Time limit per split (seconds)", 30)
        
        print("\n⏱️ Running benchmark...")
        
        # Run optimized version
        print("\n1. Optimized version (parallel + cache):")
        start = time.time()
        best_opt, _ = run_optimized_total_N(
            epk_map, day_labels,
            route_fwd, route_rev,
            travel_h, charge_h,
            N=N,
            idle_h=idle_h,
            time_limit=time_limit,
            max_workers=4
        )
        opt_time = time.time() - start
        
        print(f"\n📊 Benchmark Results:")
        print(f"   Optimized time: {opt_time:.1f}s")
        print(f"   Theoretical sequential time: ~{(N+1)*time_limit}s")
        print(f"   Speedup: ~{(N+1)*time_limit/opt_time:.1f}x")
        
        if best_opt:
            print(f"   Best: A={best_opt.A}, B={best_opt.B}, EPK={best_opt.avg_epk:.2f}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
