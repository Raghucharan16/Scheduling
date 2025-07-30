#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json, sys
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import numpy as np
from ortools.sat.python import cp_model

# ─────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────
FORBIDDEN_SLOTS = {1, 2, 3, 4, 5, 6}  # 00:30–03:00 blocked for departures
SLOTS_PER_DAY   = 48                  # 30-min grid
MIN_GAP_SLOTS   = 1                   # ≥1h spacing at a station
TIME_LIMIT_SEC  = 300

ROUTES   = ["A-B", "B-A", "B-C", "C-B"]
ORIG     = {"A-B":"A","B-A":"B","B-C":"B","C-B":"C"}
DEST     = {"A-B":"B","B-A":"A","B-C":"C","C-B":"B"}

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def slot_str_to_index(s: str) -> int:
    s = s.strip()
    if s == "23:50:00":  # special midnight label sometimes used
        return 0
    hh, mm, _ = s.split(":")
    return int(hh)*2 + (int(mm)//30)

def index_to_time(idx: int) -> str:
    return f"{idx//2:02d}:{(idx%2)*30:02d}:00"

# ─────────────────────────────────────────────────────────────
# EPK loading + IMPUTATION
# ─────────────────────────────────────────────────────────────
def load_and_impute(csv_path: Path,
                    treat_zero_as_missing: bool = True,
                    imputed_weight: float = 0.95
                   ) -> Tuple[
                        Dict[Tuple[str,str,int], float],   # epk_map
                        Dict[Tuple[str,str,int], float],   # weight_map
                        List[str],                         # day_labels
                        Set[Tuple[str,str,int]]            # imputed_keys
                   ]:
    """
    Returns:
      epk_map[(route, day_label, slot)] = epk_value (dense except forbidden)
      weight_map[...] = 1.0 for observed, 'imputed_weight' for imputed
      day_labels (string list)
      imputed_keys = set of imputed keys (for debug)
    CSV format: slot, route, 1..31 (day-of-month columns).
    """
    df = pd.read_csv(csv_path)
    if "route" not in df.columns or "slot" not in df.columns:
        raise ValueError("CSV must have columns: slot, route, and day-of-month 1..31.")
    day_cols = sorted([c for c in df.columns if c.isdigit()], key=int)
    if not day_cols:
        raise ValueError("CSV must include numeric day columns '1'..'31'.")

    df["slot_idx"] = df["slot"].map(slot_str_to_index)
    df = df[df["route"].isin(ROUTES)].copy()

    epk_map: Dict[Tuple[str,str,int], float] = {}
    weight_map: Dict[Tuple[str,str,int], float] = {}
    imputed_keys: Set[Tuple[str,str,int]] = set()
    allowed_slots = [s for s in range(SLOTS_PER_DAY) if s not in FORBIDDEN_SLOTS]

    # Month medians per route (observed > 0 unless treat_zero_as_missing=False)
    med_route = {}
    for r in ROUTES:
        sub = df[df["route"]==r]
        vals=[]
        for _,row in sub.iterrows():
            s = int(row["slot_idx"])
            if s in FORBIDDEN_SLOTS: continue
            for d in day_cols:
                v = float(row[d])
                if treat_zero_as_missing and v==0: continue
                vals.append(v)
        med_route[r] = float(np.median(vals)) if vals else 0.0

    paired = {"A-B":"B-A", "B-A":"A-B", "B-C":"C-B", "C-B":"B-C"}

    for r in ROUTES:
        sub = df[df["route"]==r].set_index("slot_idx")[day_cols].reindex(range(SLOTS_PER_DAY))
        # never create values in forbidden window
        sub.loc[list(FORBIDDEN_SLOTS)] = np.nan

        for d in day_cols:
            series = sub[d].copy()
            if treat_zero_as_missing:
                series.replace(0, np.nan, inplace=True)

            if series.dropna().empty:
                # whole day missing → route median or paired route median*0.9
                fill_val = med_route[r]
                if fill_val == 0.0:
                    p = paired[r]
                    if med_route[p] > 0:
                        fill_val = med_route[p] * 0.9
                for s in allowed_slots:
                    epk_map[(r, d, s)] = float(fill_val)
                    weight_map[(r, d, s)] = imputed_weight
                    imputed_keys.add((r, d, s))
                continue

            # Interpolate + short extrapolate (new API to avoid warnings)
            s_work = series.copy()
            s_work = s_work.interpolate(method="linear", limit_direction="both")
            s_work = s_work.ffill().bfill()

            # Keep forbidden as NaN
            s_work.loc[list(FORBIDDEN_SLOTS)] = np.nan

            for s in allowed_slots:
                val = s_work.loc[s]
                if pd.isna(val):
                    v = med_route[r] if med_route[r] > 0 else 0.0
                    epk_map[(r,d,s)] = float(v)
                    weight_map[(r,d,s)] = imputed_weight * 0.9
                    imputed_keys.add((r,d,s))
                else:
                    orig_val = df.loc[(df["route"]==r) & (df["slot_idx"]==s), d]
                    was_observed = (not orig_val.empty) and \
                                   (not pd.isna(orig_val.values[0])) and \
                                   (not (treat_zero_as_missing and float(orig_val.values[0])==0.0))
                    epk_map[(r,d,s)] = float(val)
                    weight_map[(r,d,s)] = 1.0 if was_observed else imputed_weight
                    if not was_observed:
                        imputed_keys.add((r,d,s))

    return epk_map, weight_map, day_cols, imputed_keys

# ─────────────────────────────────────────────────────────────
# DURATIONS
# ─────────────────────────────────────────────────────────────
def build_durations(travel_hours: Dict[str, float], charge_h: float):
    travel_slots = {}
    total_slots  = {}
    ch = int(round(charge_h * 2))
    for r, th in travel_hours.items():
        tr = int(round(th * 2))
        travel_slots[r] = tr
        total_slots[r]  = tr + ch
    return travel_slots, total_slots

# ─────────────────────────────────────────────────────────────
# Prefilter: first-leg must have a compatible second within 24h (across days)
# ─────────────────────────────────────────────────────────────
def has_compatible_second_abs(route: str,
                              t_abs: int,                 # absolute time slot
                              day_labels: List[str],
                              epk_map: Dict[Tuple[str,str,int], float],
                              travel_slots: Dict[str,int],
                              total_slots: Dict[str,int]) -> bool:
    """
    True iff there exists a second leg (r2, t2_abs) with:
        t2_abs >= t_abs + total_slots[route]            (ready after first incl. charge)
        t2_abs <= t_abs + 48 - travel_slots[r2]         (second TRAVEL finishes within 24h)
        EPK exists at (r2, day_of(t2_abs), slot_of(t2_abs))
    Checks same day and next day if needed.
    """
    if route == "A-B":
        second_routes = ("B-A", "B-C")
    elif route == "C-B":
        second_routes = ("B-A", "B-C")
    elif route == "B-A":
        second_routes = ("A-B",)
    elif route == "B-C":
        second_routes = ("C-B",)
    else:
        return False

    first_ready_abs = t_abs + total_slots[route]
    horizon_last_abs = t_abs + SLOTS_PER_DAY
    T = len(day_labels) * SLOTS_PER_DAY

    for r2 in second_routes:
        latest_start_abs = horizon_last_abs - travel_slots[r2]
        start_abs = max(first_ready_abs, 0)
        end_abs   = min(latest_start_abs, T - 1)
        if end_abs < start_abs:
            continue
        for t2_abs in range(start_abs, end_abs + 1):
            d2_idx = t2_abs // SLOTS_PER_DAY
            if not (0 <= d2_idx < len(day_labels)):
                continue
            s2 = t2_abs % SLOTS_PER_DAY
            if s2 in FORBIDDEN_SLOTS:
                continue
            dlab2 = day_labels[d2_idx]
            if (r2, dlab2, s2) in epk_map:
                return True
    return False

# ─────────────────────────────────────────────────────────────
# Stage 1: CP-SAT flow (aggregate), with absolute-time prefilter
# ─────────────────────────────────────────────────────────────
def solve_flow(epk_map, weight_map, day_labels, nA, nB, nC,
               travel_slots, total_slots,
               min_gap=MIN_GAP_SLOTS, timelimit=TIME_LIMIT_SEC):
    D = len(day_labels)
    T = D * SLOTS_PER_DAY
    def day_of(t): return day_labels[t // SLOTS_PER_DAY]
    def slot_of(t): return t % SLOTS_PER_DAY

    m = cp_model.CpModel()
    Y = {}

    # departure var only where EPK exists AND has a compatible second across 24h
    for t in range(T):
        s = slot_of(t)
        if s in FORBIDDEN_SLOTS: 
            continue
        dlab = day_of(t)
        for r in ROUTES:
            if (r, dlab, s) not in epk_map:
                continue
            if not has_compatible_second_abs(
                    route=r, t_abs=t, day_labels=day_labels,
                    epk_map=epk_map, travel_slots=travel_slots, total_slots=total_slots):
                continue
            Y[(r,t)] = m.NewBoolVar(f"D_{r.replace('-','')}_{t}")

    # inventories — PRE-ALLOCATE ALL keys to avoid KeyError
    fleet = nA+nB+nC
    I = {}
    for st in ("A","B","C"):
        for tt in range(0, T+1):  # 0..T inclusive
            I[(st,tt)] = m.NewIntVar(0, fleet, f"I_{st}_{tt}")
    # initial inventories
    m.Add(I[("A",0)]==nA)
    m.Add(I[("B",0)]==nB)
    m.Add(I[("C",0)]==nC)

    routes_out = {"A":["A-B"], "B":["B-A","B-C"], "C":["C-B"]}
    routes_in  = {"A":["B-A"], "B":["A-B","C-B"], "C":["B-C"]}

    for t in range(T):
        for st in ("A","B","C"):
            dep = [Y[(r,t)] for r in routes_out[st] if (r,t) in Y]
            arr = []
            for r in routes_in[st]:
                tt = t - total_slots[r]
                if tt >= 0 and (r,tt) in Y:
                    arr.append(Y[(r,tt)])
            m.Add(I[(st,t+1)] == I[(st,t)] - sum(dep) + sum(arr))

    # spacing per station
    for t in range(T):
        for st in ("A","B","C"):
            in_slot = [Y[(r,t)] for r in routes_out[st] if (r,t) in Y]
            if in_slot:
                m.Add(sum(in_slot) <= 1)
            for g in range(1, min_gap):
                t2 = t+g
                if t2 >= T: break
                both=[]
                for r in routes_out[st]:
                    if (r,t)  in Y: both.append(Y[(r,t)])
                    if (r,t2) in Y: both.append(Y[(r,t2)])
                if both:
                    m.Add(sum(both) <= 1)

    # objective: epk * weight (slight penalty for imputed)
    obj=[]
    for (r,t),var in Y.items():
        dlab = day_of(t); s = slot_of(t)
        epk  = epk_map[(r,dlab,s)]
        w    = weight_map.get((r,dlab,s), 1.0)
        obj.append(int(epk*w*100)*var)
    m.Maximize(sum(obj))

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = timelimit
    s.parameters.num_search_workers = 8
    if s.Solve(m) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("Flow model infeasible")

    deps = [(t,r) for (r,t),v in Y.items() if s.Value(v)==1]
    deps.sort(key=lambda x:x[0])
    return deps, T

# ─────────────────────────────────────────────────────────────
# Stage 2: per-bus assignment (immediate second reservation; 2-in-24)
# ─────────────────────────────────────────────────────────────
class Bus:
    __slots__=("name","station","ready","anchor","count","first_route")
    def __init__(self, name, station):
        self.name=name; self.station=station
        self.ready=0; self.anchor=0; self.count=0; self.first_route=None

def allowed_first(station:str)->List[str]:
    if station=="A": return ["A-B"]
    if station=="C": return ["C-B"]
    if station=="B": return ["B-A","B-C"]
    return []

def allowed_second(first_route:str)->List[str]:
    if first_route=="A-B": return ["B-A","B-C"]
    if first_route=="C-B": return ["B-A","B-C"]
    if first_route=="B-A": return ["A-B"]
    if first_route=="B-C": return ["C-B"]
    return []

def assign_buses_with_pairing(departures: List[Tuple[int,str]],
                              travel_slots: Dict[str,int],
                              total_slots: Dict[str,int],
                              day_labels: List[str],
                              nA:int, nB:int, nC:int) -> Dict[str, List[dict]]:
    unassigned = departures[:]  # sorted
    from collections import defaultdict, deque
    route_idx = defaultdict(deque)
    for i,(t,r) in enumerate(unassigned):
        route_idx[r].append(i)
    taken = [False]*len(unassigned)
    def take(i): taken[i]=True
    def next_cand(route, earliest, latest)->Optional[int]:
        if latest < earliest: return None
        for i in route_idx[route]:
            if taken[i]: continue
            t,_ = unassigned[i]
            if t < earliest: continue
            if t <= latest: return i
            break
        return None

    # Fleet
    buses=[]
    def add(k,st,base):
        for i in range(k): buses.append(Bus(f"Bus-{base+i:02d}", st))
    add(nA,"A",1); add(nB,"B",nA+1); add(nC,"C",nA+nB+1)

    pools={"A":list(range(0,nA)),
           "B":list(range(nA, nA+nB)),
           "C":list(range(nA+nB, nA+nB+nC))}
    out = {d: [] for d in day_labels}

    for i,(t,r) in enumerate(unassigned):
        if taken[i]: continue
        o,d = ORIG[r], DEST[r]
        ready_ids = [idx for idx in pools[o] if buses[idx].ready <= t]
        ready_ids.sort(key=lambda idx:(buses[idx].count!=1, buses[idx].ready))

        chosen=None; as_second=False
        # (A) Try to satisfy as SECOND if some bus is waiting
        for idx in ready_ids:
            b=buses[idx]
            if b.count==1 and ORIG[r]==b.station:
                latest = b.anchor + SLOTS_PER_DAY - travel_slots[r]  # travel must finish within 24h
                if t <= latest:
                    chosen=idx; as_second=True; break

        # (B) Otherwise as FIRST, but only if we can reserve SECOND immediately
        if chosen is None:
            for idx in ready_ids:
                b=buses[idx]
                if b.count!=0 or r not in allowed_first(b.station): continue
                anchor = t
                first_ready = t + total_slots[r]
                pick_j=None; pick_r2=None
                for r2 in allowed_second(r):
                    if ORIG[r2] != d:  continue
                    latest2 = anchor + SLOTS_PER_DAY - travel_slots[r2]
                    j = next_cand(r2, earliest=first_ready, latest=latest2)
                    if j is not None: pick_j=j; pick_r2=r2; break
                if pick_j is None:
                    continue  # cannot pair → skip this first
                # Commit FIRST + SECOND
                chosen=idx
                # FIRST
                start=t; end_rdy=t+total_slots[r]; end_trv=t+travel_slots[r]
                day1 = day_labels[start // SLOTS_PER_DAY]
                out[day1].append({
                    "busNumber": buses[chosen].name,
                    "trip": {"route": r, "startStation": o, "endStation": d,
                             "startTime": index_to_time(start % SLOTS_PER_DAY),
                             "endTime":   index_to_time(end_trv % SLOTS_PER_DAY)}
                })
                take(i)
                pools[o].remove(chosen)
                b.station=d; b.ready=end_rdy; b.anchor=anchor; b.count=1; b.first_route=r
                # SECOND
                t2,_=unassigned[pick_j]
                o2,d2=ORIG[pick_r2], DEST[pick_r2]
                day2 = day_labels[t2 // SLOTS_PER_DAY]
                out[day2].append({
                    "busNumber": b.name,
                    "trip": {"route": pick_r2, "startStation": o2, "endStation": d2,
                             "startTime": index_to_time(t2 % SLOTS_PER_DAY),
                             "endTime":   index_to_time((t2+travel_slots[pick_r2]) % SLOTS_PER_DAY)}
                })
                take(pick_j)
                b.station=d2; b.ready=t2+total_slots[pick_r2]; b.count=2; b.first_route=None
                pools[d2].append(chosen)
                break

        if chosen is not None and as_second:
            b=buses[chosen]
            pools[o].remove(chosen)
            start=t; end_rdy=t+total_slots[r]; end_trv=t+travel_slots[r]
            dayx = day_labels[start // SLOTS_PER_DAY]
            out[dayx].append({
                "busNumber": b.name,
                "trip": {"route": r, "startStation": o, "endStation": d,
                         "startTime": index_to_time(start % SLOTS_PER_DAY),
                         "endTime":   index_to_time(end_trv % SLOTS_PER_DAY)}
            })
            take(i)
            b.station=d; b.ready=end_rdy; b.count=2; b.first_route=None
            pools[d].append(chosen)

    # Sort each day
    for d in day_labels:
        out[d].sort(key=lambda a:(a["trip"]["route"], a["trip"]["startTime"], a["busNumber"]))
    return out

# ─────────────────────────────────────────────────────────────
# HTML rendering (columns = routes)
# ─────────────────────────────────────────────────────────────
def write_routes_html(schedule_by_day, html_path: Path):
    routes=set()
    for arr in schedule_by_day.values():
        for a in arr: routes.add(a["trip"]["route"])
    ordered=[r for r in ["A-B","B-A","B-C","C-B"] if r in routes] + \
            sorted([r for r in routes if r not in {"A-B","B-A","B-C","C-B"}])
    days=sorted(schedule_by_day.keys(), key=lambda x:int(x))
    css = """<!doctype html><html><head><meta charset="utf-8">
<title>Route-wise Schedule</title>
<style>
body{font-family:Arial,Helvetica,sans-serif;margin:16px}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #ddd;padding:6px;vertical-align:top}
th{background:#333;color:#fff}
.day{white-space:nowrap;text-align:center;font-weight:bold}
.tag{display:inline-block;padding:3px 8px;margin:2px;border-radius:10px;color:#fff;font-size:12px}
.A-B{background:#e76f51}.B-A{background:#2a9d8f}.B-C{background:#8ab17d}.C-B{background:#6d597a}
.wrap{display:flex;flex-wrap:wrap;gap:4px}
</style></head><body>
"""
    header = f"<h2>Route-wise Schedule ({len(days)} days)</h2>\n"
    header += "<table><thead><tr><th>Day</th>"
    header += "".join(f"<th>{r}</th>" for r in ordered)
    header += "</tr></thead><tbody>\n"
    rows=[]
    for d in days:
        buckets={r:[] for r in ordered}
        for a in schedule_by_day.get(d, []):
            r=a["trip"]["route"]
            if r in buckets: buckets[r].append(a)
        for r in ordered:
            buckets[r].sort(key=lambda a:a["trip"]["startTime"])
        cells=[]
        for r in ordered:
            chips=" ".join(
                f"<span class='tag {r}' title='{a['busNumber']} {a['trip']['startTime']}→{a['trip']['endTime']}'>"
                f"{a['busNumber']} {a['trip']['startTime']}→{a['trip']['endTime']}</span>"
                for a in buckets[r]
            )
            cells.append(f"<td><div class='wrap'>{chips}</div></td>")
        rows.append(f"<tr><td class='day'><b>{d}</b></td>{''.join(cells)}</tr>")
    html = css + header + "\n".join(rows) + "</tbody></table></body></html>"
    html_path.write_text(html, encoding="utf-8")

# ─────────────────────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────────────────────
def ask_path(msg, default): v=input(f"{msg} [{default}]: ").strip(); return Path(v) if v else Path(default)
def ask_int (msg, default): v=input(f"{msg} [{default}]: ").strip(); return int(v) if v else default
def ask_float(msg, default): v=input(f"{msg} [{default}]: ").strip(); return float(v) if v else default
def ask_bool(msg, default=True):
    dv = "Y/n" if default else "y/N"
    v = input(f"{msg} [{dv}]: ").strip().lower()
    if v==""  : return default
    if v in ("y","yes","true","1"): return True
    if v in ("n","no","false","0"): return False
    return default

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    csv_path = ask_path("EPK CSV path", "cache/abc.csv")
    day_start = ask_int("First day-of-month", 1)
    day_end   = ask_int("Last day-of-month", 30)
    nA = ask_int("Buses initially at A", 1)
    nB = ask_int("Buses initially at B", 3)
    nC = ask_int("Buses initially at C", 2)

    print("\nEnter TRAVEL hours per leg (exclude charge):")
    th_AB = ask_float("A-B travel hours", 9.0)
    th_BA = ask_float("B-A travel hours", 9.0)
    th_BC = ask_float("B-C travel hours", 10.0)
    th_CB = ask_float("C-B travel hours", 10.0)
    ch_h  = ask_float("Destination-depot CHARGE hours", 2.0)

    min_gap = ask_int("Min spacing at a station (slots)", MIN_GAP_SLOTS)
    treat_zero_as_missing = ask_bool("Treat 0 EPK as missing (interpolate)?", True)
    imputed_weight = ask_float("Weight for imputed EPK (<=1 for slight penalty)", 0.95)

    json_out = ask_path("JSON output", "schedule_hub.json")
    html_out = ask_path("HTML output", "schedule_hub.html")

    # Load + Impute
    epk_map, weight_map, all_days, _ = load_and_impute(
        csv_path, treat_zero_as_missing=treat_zero_as_missing, imputed_weight=imputed_weight
    )
    day_labels = [str(d) for d in range(day_start, day_end+1) if str(d) in all_days]
    if not day_labels:
        raise ValueError("Selected day range not present in CSV.")

    # Durations
    travel_slots, total_slots = build_durations(
        {"A-B": th_AB, "B-A": th_BA, "B-C": th_BC, "C-B": th_CB},
        ch_h
    )

    # Stage 1: flow with absolute-time prefilter for 2nd legs
    departures, _ = solve_flow(
        epk_map=epk_map, weight_map=weight_map,
        day_labels=day_labels, nA=nA, nB=nB, nC=nC,
        travel_slots=travel_slots, total_slots=total_slots,
        min_gap=min_gap, timelimit=TIME_LIMIT_SEC
    )

    # Stage 2: per-bus assignment with immediate second reservation
    schedule_by_day = assign_buses_with_pairing(
        departures=departures,
        travel_slots=travel_slots,
        total_slots=total_slots,
        day_labels=day_labels,
        nA=nA, nB=nB, nC=nC
    )

    # Write outputs
    json_out.write_text(json.dumps({"schedule": schedule_by_day}, indent=2), encoding="utf-8")
    write_routes_html(schedule_by_day, html_out)
    print(f"✅ JSON → {json_out}")
    print(f"✅ HTML → {html_out}")

    trips = sum(len(v) for v in schedule_by_day.values())
    print(f"Trips scheduled in days [{day_labels[0]}..{day_labels[-1]}]: {trips}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌", e)
        sys.exit(1)
