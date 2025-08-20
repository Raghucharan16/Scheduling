#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json, sys
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ortools.sat.python import cp_model
from epk_avg_as_threshold_per_route import get_avg_epk_per_route
# ─────────────────────────────────────────────────────────────
# Global time grid & ops rules
# ─────────────────────────────────────────────────────────────
SLOTS_PER_DAY   = 48                  # 30-minute slots
FORBIDDEN_SLOTS = {1,2,3,4,5,6}       # 00:30–03:00 forbidden for departures
DEFAULT_MIN_GAP = 1                   # station spacing in slots (30 min per slot)
TIME_LIMIT_SEC  = 300                 # per-solve time

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def slot_to_idx(s: str) -> int:
    s = s.strip()
    if s == "23:50:00":
        return 0
    # Handle both HH:MM and HH:MM:SS formats
    parts = s.split(":")
    if len(parts) == 2:  # HH:MM format
        hh, mm = parts
    elif len(parts) == 3:  # HH:MM:SS format
        hh, mm, _ = parts
    else:
        raise ValueError(f"Invalid time format: {s}")
    return int(hh)*2 + (int(mm)//30)

def idx_to_hhmm(idx: int) -> str:
    return f"{idx//2:02d}:{(idx%2)*30:02d}:00"

def read_month_days(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.isdigit()], key=lambda x: int(x))

# ─────────────────────────────────────────────────────────────
# Data loading
# CSV expected: slot, sector, route, 1..31 (days)
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
    df["slot_idx"] = df["slot"].map(slot_to_idx)

    epk_map: Dict[Tuple[str,str,int], float] = {}
    for _, row in df.iterrows():
        r = row["route"]; s = int(row["slot_idx"])
        for d in day_cols:
            epk_map[(r, d, s)] = float(row[d])
    return epk_map, day_cols

# ─────────────────────────────────────────────────────────────
# OR-Tools model for one corridor and one (nA, nB) combination
# route-wise avg floor is optional (pass None to disable)
# ─────────────────────────────────────────────────────────────
def solve_corridor(
    epk_map: Dict[Tuple[str,str,int], float],
    day_labels: List[str],
    route_fwd: str, route_rev: str,
    travel_h: float, charge_h: float,
    n_origin: int, n_dest: int,
    idle_h: float = 4.0,                 # NEW: to match per-bus model's max extra idle
    time_limit: int = TIME_LIMIT_SEC,
    deterministic: bool = True,
    seed: int = 123,
    enable_logs: bool = False,
) -> Tuple[int, float, List[Tuple[int,str]]]:
    """
    Wraps the per-bus CP-SAT `solve(...)` (month horizon, daily quotas).
    Returns (trips, avg_epk, chosen) where chosen is [(abs_t, route_str)].
    """
    # --- Remap EPK to the per-bus solver's keying: (route, day_index, slot)
    # day_labels are strings "1","2",...; solver wants day index 0..D-1
    D = len(day_labels)
    day_to_idx = {d:i for i,d in enumerate(day_labels)}
    epk_indexed: Dict[Tuple[str,int,int], float] = {}

    for d_lbl, d_idx in day_to_idx.items():
        for s in ALLOWED:
            # default to 0.0 if missing in csv (solver objective can handle zeros)
            epk_indexed[(route_fwd, d_idx, s)] = float(epk_map.get((route_fwd, d_lbl, s), 0.0))
            epk_indexed[(route_rev, d_idx, s)] = float(epk_map.get((route_rev, d_lbl, s), 0.0))

    # --- Call your per-bus solver
    travel_s = int(round(travel_h * 2))
    charge_s = int(round(charge_h * 2))
    idle_s   = int(round(idle_h   * 2))

    sched = solve(
        epk=epk_indexed,
        days=day_labels,           # labels are used only to name buckets in output
        routeAB=route_fwd,
        routeBA=route_rev,
        busesA=n_origin,
        busesB=n_dest,
        travel_s=travel_s,
        charge_s=charge_s,
        idle_s=idle_s,
        limit=time_limit,
        deterministic=deterministic,
        seed=seed,
        enable_logs=enable_logs
    )

    # --- Aggregate metrics to match old return type
    SLOTS = SLOTS_PER_DAY
    trips = 0
    total_epk = 0.0
    chosen: List[Tuple[int,str]] = []  # (abs_t, route)

    for bus, by_day in sched.items():
        for d_idx, d_lbl in enumerate(day_labels):
            for tr in by_day.get(d_lbl, []):
                trips += 1
                total_epk += float(tr["epk"])
                s = slot_to_idx(tr["startTime"])    # "HH:MM:SS" -> slot
                abs_t = d_idx * SLOTS + s
                chosen.append((abs_t, tr["route"]))

    chosen.sort(key=lambda x: x[0])
    avg_epk = (total_epk / trips) if trips > 0 else 0.0
    return trips, avg_epk, chosen


# ─────────────────────────────────────────────────────────────
# Heatmap & HTML
# ─────────────────────────────────────────────────────────────
def generate_heatmap(epk_matrix: np.ndarray, output_dir: Path,
                     maxA: int, maxB: int, title: str) -> Path:
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    im = ax.imshow(epk_matrix, cmap='viridis', origin='upper')
    plt.colorbar(im, ax=ax, label='Average EPK')

    ax.set_xlabel('Buses at Destination (B)')
    ax.set_ylabel('Buses at Origin (A)')
    ax.set_title(title)

    ax.set_xticks(np.arange(maxB+1))
    ax.set_yticks(np.arange(maxA+1))
    ax.set_xticklabels(range(maxB+1))
    ax.set_yticklabels(range(maxA+1))

    # cell annotations
    for i in range(maxA+1):
        for j in range(maxB+1):
            val = epk_matrix[i, j]
            if np.isnan(val): txt = "–"
            else: txt = f"{val:.1f}"
            ax.text(j, i, txt, ha="center", va="center", color="w", fontsize=10, fontweight='bold')

    plt.tight_layout()
    out = output_dir / 'epk_heatmap.png'
    plt.savefig(out, dpi=160)
    plt.close()
    return out

def heatmap_html(epk_matrix: np.ndarray, heatmap_path: Path, output_path: Path,
                 maxA: int, maxB: int, combo_df: pd.DataFrame, title: str):
    # build matrix HTML rows
    rows = []
    for i in range(maxA+1):
        cells = "".join(
            f"<td>{'–' if np.isnan(epk_matrix[i,j]) else f'{epk_matrix[i,j]:.1f}'}</td>"
            for j in range(maxB+1)
        )
        rows.append(f"<tr><th>{i}</th>{cells}</tr>")
    matrix_rows = "\n".join(rows)

    combo_tbl = combo_df.sort_values(["trips","avg_epk"], ascending=[False, False]).to_html(
        index=False, float_format=lambda x: f"{x:.2f}", classes="combo-table"
    )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EPK Heatmap Report</title>
<style>
 body {{ font-family: Arial, sans-serif; margin: 20px; }}
 h1,h2 {{ color: #2c3e50; }}
 .container {{ max-width: 1100px; margin: 0 auto; }}
 .heatmap-img {{ max-width: 100%; border: 1px solid #ddd; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
 table.matrix {{ border-collapse: collapse; margin: 20px auto; }}
 table.matrix th, table.matrix td {{ padding: 8px 10px; border: 1px solid #ddd; text-align: center; min-width: 60px; }}
 table.matrix th {{ background-color: #f2f2f2; }}
 .combo-table {{ border-collapse: collapse; width: 100%; }}
 .combo-table th, .combo-table td {{ border: 1px solid #ddd; padding: 6px; }}
 .combo-table th {{ background-color: #f2f2f2; }}
 .footer {{ margin-top: 20px; color: #777; font-size: 0.9em; }}
</style></head>
<body>
  <div class="container">
    <h1>{title}</h1>
    <img src="{heatmap_path.name}" alt="EPK Heatmap" class="heatmap-img" />
    <h2>Average EPK Matrix</h2>
    <table class="matrix">
      <tr>
        <th>A\\B</th>
        {"".join(f"<th>{j}</th>" for j in range(maxB+1))}
      </tr>
      {matrix_rows}
    </table>
    <h2>Combinations (all results)</h2>
    {combo_tbl}
    <div class="footer">Generated on {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
  </div>
</body></html>"""
    output_path.write_text(html, encoding="utf-8")

# ─────────────────────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────────────────────
def ask_path(msg, default): v=input(f"{msg} [{default}]: ").strip(); return Path(v) if v else Path(default)
def ask_int (msg, default): v=input(f"{msg} [{default}]: ").strip(); return int(v) if v else default
def ask_float(msg, default): v=input(f"{msg} [{default}]: ").strip(); return float(v) if v else default
def ask_text(msg, default): v=input(f"{msg} [{default}]: ").strip(); return v if v else default

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    print("Electric Bus Corridor — EPK Heatmap\n")
    csv_path   = ask_path("Month CSV path", "cache/vskp-vjy_aug_neugo_epk_data.csv")
    route_fwd  = ask_text("Forward route code", "H-V")
    route_rev  = ask_text("Reverse route code", "V-H")
    travel_h   = ask_float("Travel hours (one way, same both directions)", 9.0)
    charge_h   = ask_float("Charge hours at destination", 2.0)
    min_gap    = ask_int("Min spacing at station (slots)", DEFAULT_MIN_GAP)
    maxA       = ask_int("Max buses at Origin (A) to scan", 5)
    maxB       = ask_int("Max buses at Destination (B) to scan", 5)
    time_limit = ask_int("Per-solve time limit (sec)", TIME_LIMIT_SEC)

    # Optional route-wise thresholds (press Enter to skip)
    use_thr = ask_text("Route-wise average EPK floors? (JSON dict or empty)", "")
    route_thresholds = get_avg_epk_per_route('cache/overall_epk_data.csv')
    if use_thr.strip():
        try:
            
            assert route_fwd in route_thresholds and route_rev in route_thresholds
        except Exception:
            print("⚠️  Could not parse thresholds JSON or keys missing; proceeding without floors.")
            route_thresholds = None

    # Load data
    epk_map, day_labels = load_corridor(csv_path, route_fwd, route_rev)

    # Heatmap computation
    epk_matrix = np.full((maxA+1, maxB+1), np.nan, dtype=float)
    rows = []
    for a in range(maxA+1):
        for b in range(maxB+1):
            if a==0 and b==0:
                continue
            trips, avg_epk, _ = solve_corridor(
                epk_map, day_labels,
                route_fwd, route_rev,
                travel_h, charge_h,
                n_origin=a, n_dest=b,
                min_gap_slots=min_gap,
                time_limit=time_limit,
                route_thresholds=route_thresholds
            )
            epk_matrix[a, b] = avg_epk if trips>0 else np.nan
            rows.append({"buses_A":a, "buses_B":b, "trips":trips, "avg_epk":avg_epk})

    combo_df = pd.DataFrame(rows)
    combo_df.to_csv("combo_summary.csv", index=False)
    pd.DataFrame(epk_matrix).to_csv("epk_matrix.csv", header=False, index=False)

    title = f"Average EPK by Bus Allocation — {route_fwd} / {route_rev}"
    heatmap_path = generate_heatmap(epk_matrix, Path.cwd(), maxA, maxB, title)
    html_path = Path("epk_heatmap.html")
    heatmap_html(epk_matrix, heatmap_path, html_path, maxA, maxB, combo_df, title)

    # Recommend the best combo (max trips, then max avg_epk)
    best = combo_df.sort_values(["trips","avg_epk"], ascending=[False, False]).head(1)
    if not best.empty:
        a, b, t, e = int(best.iloc[0]["buses_A"]), int(best.iloc[0]["buses_B"]), int(best.iloc[0]["trips"]), float(best.iloc[0]["avg_epk"])
        print(f"\nBest combo by trips then avgEPK → A={a}, B={b}, trips={t}, avgEPK={e:.2f}")

    print(f"\n✅ Matrix CSV  → epk_matrix.csv")
    print(f"✅ Combos CSV  → combo_summary.csv")
    print(f"✅ Heatmap PNG → {heatmap_path}")
    print(f"✅ Heatmap HTML→ {html_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌", e)
        sys.exit(1)
