#!/usr/bin/env python3
"""
hub_two_trip_scheduler.py

Optimise a hubbed corridor (A, B, C) with directed legs:
  A-B, B-A, B-C, C-B

Stage 1: CP-SAT time-expanded fleet flow picks departures (maximises EPK)
         with station inventories, forbidden window, spacing.
Stage 2: Greedy per-bus assignment enforces:
         • identity & continuity,
         • exactly two trips within each bus's 24h service window,
         • B hub cross-trip rules (A→B then B→A/B→C; C→B then B→A/B→C; etc).

CSV format (day-of-month columns):
slot,sector,route,1,2,3,...,30
23:50:00,H-V,A-B, ...
00:30:00,H-V,A-B, ...
...
00:00:00,H-V,B-A, ...
00:00:00,H-V,B-C, ...
00:00:00,H-V,C-B, ...

Run:
  pip install ortools pandas
  python hub_two_trip_scheduler.py
"""

from pathlib import Path
import json, sys
from typing import Dict, List, Tuple
import pandas as pd
from ortools.sat.python import cp_model

# ─────────────────────────────────────────────────────────────
# Parameters (you can change defaults at runtime via prompts)
# ─────────────────────────────────────────────────────────────
FORBIDDEN_SLOTS = {1, 2, 3, 4, 5, 6}  # 00:30–03:00 cannot START
SLOTS_PER_DAY   = 48                  # 30-min grid
ALLOWED_SLOTS   = [s for s in range(SLOTS_PER_DAY) if s not in FORBIDDEN_SLOTS]
MIN_GAP_SLOTS   = 1                   # ≥ 1h spacing per station (slot = 30m)
TIME_LIMIT_SEC  = 300

ROUTES  = ["A-B", "B-A", "B-C", "C-B"]
STATION_OF = {
    "A-B": ("A", "B"),
    "B-A": ("B", "A"),
    "B-C": ("B", "C"),
    "C-B": ("C", "B"),
}
STATIONS = ["A", "B", "C"]

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def slot_str_to_index(s: str) -> int:
    """ Map HH:MM:SS to 0..47. Special-case '23:50:00' as midnight slot 0. """
    s = s.strip()
    if s == "23:50:00":
        return 0
    h, m, _ = s.split(":")
    return int(h) * 2 + (int(m) // 30)

def index_to_time(idx: int) -> str:
    return f"{idx//2:02d}:{(idx%2)*30:02d}:00"

def read_epk_csv(csv_path: Path) -> Tuple[Dict[Tuple[str, str, int], float], List[str]]:
    """ epk[(route, day_label, slot_idx)] = epk_value; returns also ordered day labels """
    df = pd.read_csv(csv_path)
    day_cols = sorted([c for c in df.columns if c.isdigit()], key=int)
    if not day_cols:
        raise ValueError("CSV must include numeric day columns '1'..'31'.")
    df["slot_idx"] = df["slot"].map(slot_str_to_index)
    epk = {}
    for _, r in df.iterrows():
        route = str(r["route"]).strip()
        sidx = int(r["slot_idx"])
        if sidx in FORBIDDEN_SLOTS:
            continue
        for dlabel in day_cols:
            epk[(route, dlabel, sidx)] = float(r[dlabel])
    return epk, day_cols

def build_durations(travel_hours: Dict[str, float], depot_charge_h: float) -> Dict[str, int]:
    """ total duration in slots for each directed route (travel + destination charge) """
    d = {}
    for r in ROUTES:
        tot_h = travel_hours[r] + depot_charge_h
        d[r] = int(round(tot_h * 2))  # 2 slots per hour
    return d

# ─────────────────────────────────────────────────────────────
# CP-SAT fleet-flow (aggregate departures, no per-bus vars)
# ─────────────────────────────────────────────────────────────
def solve_flow(epk_map: Dict[Tuple[str,str,int], float],
               day_labels: List[str],
               nA:int, nB:int, nC:int,
               durations: Dict[str,int],
               min_gap:int = MIN_GAP_SLOTS,
               timelimit:int = TIME_LIMIT_SEC):
    """Choose departures D[(route,t)] ∈ {0,1} maximising EPK with inventories & spacing."""

    D_days = len(day_labels)
    T = D_days * SLOTS_PER_DAY

    def day_label_of(t:int) -> str: return day_labels[t // SLOTS_PER_DAY]
    def slot_of(t:int) -> int: return t % SLOTS_PER_DAY

    model = cp_model.CpModel()

    # Decision: departure on (route, absolute slot t)
    D = {}
    for t in range(T):
        s = slot_of(t)
        if s in FORBIDDEN_SLOTS:
            continue
        dlab = day_label_of(t)
        for r in ROUTES:
            if (r, dlab, s) in epk_map:
                D[(r,t)] = model.NewBoolVar(f"D_{r.replace('-','')}_{t}")

    # Inventories I[(station, t)] with conservation across time
    I = {}
    fleet = nA + nB + nC
    for s in STATIONS:
        for t in range(T+1):
            I[(s,t)] = model.NewIntVar(0, fleet, f"I_{s}_{t}")
    model.Add(I[("A",0)] == nA)
    model.Add(I[("B",0)] == nB)
    model.Add(I[("C",0)] == nC)

    routes_out = {"A": ["A-B"], "B": ["B-A","B-C"], "C": ["C-B"]}
    routes_in  = {"A": ["B-A"], "B": ["A-B","C-B"], "C": ["B-C"]}

    for t in range(T):
        for s in STATIONS:
            dep = [D[(r,t)] for r in routes_out[s] if (r,t) in D]
            arr = []
            for r in routes_in[s]:
                tt = t - durations[r]
                if tt >= 0 and (r,tt) in D:
                    arr.append(D[(r,tt)])
            model.Add(I[(s,t+1)] == I[(s,t)] - sum(dep) + sum(arr))

    # Station spacing: at-most-one per slot; plus ≥min_gap between any two departs
    for t in range(T):
        for s in STATIONS:
            # at-most-one in slot t
            in_slot = [D[(r,t)] for r in routes_out[s] if (r,t) in D]
            if in_slot:
                model.Add(sum(in_slot) <= 1)
            # min-gap with future slots within gap window
            for g in range(1, min_gap):
                t2 = t + g
                if t2 >= T: break
                both = []
                for r in routes_out[s]:
                    if (r,t) in D:   both.append(D[(r,t)])
                    if (r,t2) in D:  both.append(D[(r,t2)])
                if both:
                    model.Add(sum(both) <= 1)

    # Objective: Maximise total EPK
    obj = []
    for (r,t), var in D.items():
        dlab = day_label_of(t)
        s    = slot_of(t)
        val  = epk_map.get((r, dlab, s), 0.0)
        obj.append(int(val * 100) * var)
    model.Maximize(sum(obj))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timelimit
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"Flow model infeasible: {solver.StatusName(status)}")

    # Extract chosen departures
    dep = []
    for (r,t), var in D.items():
        if solver.Value(var) == 1:
            dep.append((t,r))
    dep.sort(key=lambda x: x[0])
    return dep, T

# ─────────────────────────────────────────────────────────────
# Per-bus assignment with 2-trips-per-24h windows & hub rules
# ─────────────────────────────────────────────────────────────
class BusState:
    __slots__ = ("name","station","ready","win_anchor","win_count","first_route")
    def __init__(self, name:str, station:str):
        self.name    = name
        self.station = station      # current station of bus
        self.ready   = 0            # earliest absolute slot ready to depart
        self.win_anchor = 0         # start slot of current 24h window
        self.win_count  = 0         # 0 or 1 (we’ll never let it reach >2)
        self.first_route = None     # route used as first leg in current window

def allowed_first_routes(station:str) -> List[str]:
    if station == "A": return ["A-B"]
    if station == "C": return ["C-B"]
    if station == "B": return ["B-A","B-C"]
    return []

def allowed_second_routes(first_route:str) -> List[str]:
    if first_route == "A-B": return ["B-A","B-C"]
    if first_route == "C-B": return ["B-A","B-C"]
    if first_route == "B-A": return ["A-B"]
    if first_route == "B-C": return ["C-B"]
    return []

def assign_buses(departures: List[Tuple[int,str]],
                 durations: Dict[str,int],
                 day_labels: List[str],
                 nA:int, nB:int, nC:int) -> Dict[str, List[dict]]:
    """
    Greedy assignment:
      - pick among ready buses at the origin station at time t
      - enforce 2 trips per bus inside [win_anchor, win_anchor+48)
      - enforce route pairing rules at the hub
      - update bus ready time and station after each trip
    Output format: schedule[day_label] = [ {busNumber, trip{...}}, ... ]
    """
    T = len(day_labels) * SLOTS_PER_DAY

    # Create buses with identities and initial stations
    buses: List[BusState] = []
    def add_buses(count:int, station:str, start_index:int):
        for i in range(count):
            buses.append(BusState(f"Bus-{start_index+i:02d}", station))

    add_buses(nA, "A", 1)
    add_buses(nB, "B", nA+1)
    add_buses(nC, "C", nA+nB+1)

    # Pools by station (store indices of buses)
    pools = {"A": [], "B": [], "C": []}
    for i,b in enumerate(buses):
        pools[b.station].append(i)

    # Output container
    out = {d: [] for d in day_labels}

    # helper: ensure window advanced if we’re past 24h since anchor
    def advance_window_if_needed(b:BusState, t:int):
        while t >= b.win_anchor + SLOTS_PER_DAY:
            b.win_anchor += SLOTS_PER_DAY
            b.win_count   = 0
            b.first_route = None

    # iterate departures in chronological order
    for t, route in departures:
        orig, dest = STATION_OF[route]
        # choose a suitable bus at 'orig'
        candidates = pools[orig][:]  # indices
        # filter feasible by ready time
        candidates = [idx for idx in candidates if buses[idx].ready <= t]
        # among feasible, enforce window + route rules and pick best
        chosen = None
        for idx in sorted(candidates, key=lambda i:(buses[i].ready, buses[i].win_count)):
            b = buses[idx]
            advance_window_if_needed(b, t)
            if b.win_count == 0:
                # first leg must match allowed_first_routes(station)
                if route not in allowed_first_routes(b.station):
                    continue
            else:  # second leg
                # must be within same 24h window
                if t >= b.win_anchor + SLOTS_PER_DAY:
                    # window rolled; interpret as new first leg instead
                    b.win_anchor += SLOTS_PER_DAY
                    b.win_count = 0
                    b.first_route = None
                    if route not in allowed_first_routes(b.station):
                        continue
                else:
                    # within window: must be allowed second route
                    if route not in allowed_second_routes(b.first_route):
                        continue
            chosen = idx
            break

        if chosen is None:
            # if no bus is ready under window rules, try any bus ready at origin and treat as new window first leg
            fallback = [idx for idx in pools[orig] if buses[idx].ready <= t]
            fallback.sort(key=lambda i:(buses[i].ready, buses[i].win_count))
            for idx in fallback:
                b = buses[idx]
                advance_window_if_needed(b, t)
                # start a new window (first leg) if needed
                if b.win_count != 0 and t < b.win_anchor + SLOTS_PER_DAY:
                    # cannot take more within window
                    continue
                # Now t >= anchor+24h or win_count == 0 → treat as first
                if route in allowed_first_routes(b.station):
                    chosen = idx
                    break

        if chosen is None:
            # As a last resort, skip this departure (should be rare if flow is consistent)
            # print(f"[warn] Unable to assign {route} at t={t}")
            continue

        # Assign to chosen bus
        b = buses[chosen]
        pools[orig].remove(chosen)  # leaving origin
        start_abs = t
        end_abs   = t + durations[route]
        start_day = day_labels[start_abs // SLOTS_PER_DAY]
        start_tim = index_to_time(start_abs % SLOTS_PER_DAY)
        end_tim   = index_to_time(end_abs % SLOTS_PER_DAY)

        out[start_day].append({
            "busNumber": b.name,
            "trip": {
                "route": route,
                "startStation": orig,
                "endStation": dest,
                "startTime": start_tim,
                "endTime": end_tim
            }
        })

        # update bus state
        b.station = dest
        b.ready   = end_abs
        if b.win_count == 0:
            b.first_route = route
            b.win_count   = 1
        else:
            b.win_count   = 2  # window will roll once we hit next anchor time

        # add back to destination pool
        pools[dest].append(chosen)

    # final tidy: sort day lists by (route, start time, bus)
    for d in day_labels:
        out[d].sort(key=lambda a: (a["trip"]["route"], a["trip"]["startTime"], a["busNumber"]))
    return out

# ─────────────────────────────────────────────────────────────
# HTML rendering (one column per actual route)
# ─────────────────────────────────────────────────────────────
def write_routes_html(schedule_by_day: Dict[str, List[dict]],
                      html_path: Path):
    # detect routes present
    routes = set()
    for d, arr in schedule_by_day.items():
        for a in arr:
            routes.add(a["trip"]["route"])
    routes = [r for r in ["A-B","B-A","B-C","C-B"] if r in routes] + \
             sorted([r for r in routes if r not in {"A-B","B-A","B-C","C-B"}])

    days = sorted(schedule_by_day.keys(), key=lambda x:int(x))
    css = """
<!doctype html><html><head><meta charset="utf-8">
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
<h2>Route-wise Schedule ({} days)</h2>
<table><thead><tr><th>Day</th>{}</tr></thead><tbody>
""".format(len(days), "".join(f"<th>{r}</th>" for r in routes))
    rows=[]
    for d in days:
        # bucket trips by route
        buckets = {r:[] for r in routes}
        for a in schedule_by_day[d]:
            r = a["trip"]["route"]
            if r in buckets:
                buckets[r].append(a)
        for r in routes:
            buckets[r].sort(key=lambda a:a["trip"]["startTime"])
        cells = []
        for r in routes:
            chips = " ".join(
                f"<span class='tag {r}' title='{a['busNumber']} {a['trip']['startTime']}→{a['trip']['endTime']}'>"
                f"{a['busNumber']} {a['trip']['startTime']}→{a['trip']['endTime']}</span>"
                for a in buckets[r]
            )
            cells.append(f"<td><div class='wrap'>{chips}</div></td>")
        rows.append(f"<tr><td class='day'><b>{d}</b></td>{''.join(cells)}</tr>")
    html = css + "\n".join(rows) + "</tbody></table></body></html>"
    html_path.write_text(html, encoding="utf-8")

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def ask_path(msg, default): 
    v = input(f"{msg} [{default}]: ").strip(); 
    return Path(v) if v else Path(default)

def ask_int(msg, default): 
    v = input(f"{msg} [{default}]: ").strip(); 
    return int(v) if v else default

def ask_float(msg, default): 
    v = input(f"{msg} [{default}]: ").strip(); 
    return float(v) if v else default

def main():
    # Inputs
    csv_path = ask_path("EPK CSV path", "cache/abc.csv")
    day_start = ask_int("First day-of-month to schedule", 1)
    day_end   = ask_int("Last day-of-month to schedule", 30)
    nA = ask_int("Buses initially at A", 1)
    nB = ask_int("Buses initially at B", 3)
    nC = ask_int("Buses initially at C", 2)

    print("\nEnter TRAVEL hours (each leg; excludes depot charge):")
    th_AB = ask_float("A-B travel hours", 9.0)
    th_BA = ask_float("B-A travel hours", 9.0)
    th_BC = ask_float("B-C travel hours", 10.0)
    th_CB = ask_float("C-B travel hours", 10.0)
    depot_h = ask_float("Depot charge hours at destination", 2.0)

    min_gap = ask_int("Min spacing between departures at a station (slots)", MIN_GAP_SLOTS)
    json_out = ask_path("JSON output", "schedule_hub.json")
    html_out = ask_path("HTML output", "schedule_hub.html")

    # Load EPK
    epk_map, all_day_cols = read_epk_csv(csv_path)
    day_labels = [str(d) for d in range(day_start, day_end+1) if str(d) in all_day_cols]
    if not day_labels:
        raise ValueError("Selected day range not present in CSV columns.")

    # Durations (slots)
    durations = build_durations(
        {"A-B": th_AB, "B-A": th_BA, "B-C": th_BC, "C-B": th_CB},
        depot_charge_h=depot_h
    )

    # Stage 1: CP-SAT aggregate flow optimisation
    departures, T = solve_flow(
        epk_map=epk_map,
        day_labels=day_labels,
        nA=nA, nB=nB, nC=nC,
        durations=durations,
        min_gap=min_gap,
        timelimit=TIME_LIMIT_SEC
    )

    # Stage 2: per-bus assignment with 2-trips-per-24h windows & hub rules
    schedule_by_day = assign_buses(
        departures=departures,
        durations=durations,
        day_labels=day_labels,
        nA=nA, nB=nB, nC=nC
    )

    # Write JSON
    out = {"schedule": schedule_by_day}
    json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"✅ JSON written → {json_out}")

    # Write HTML (route-wise grid)
    write_routes_html(schedule_by_day, html_out)
    print(f"✅ HTML written → {html_out}")

    # Quick summary
    trips = sum(len(v) for v in schedule_by_day.values())
    print(f"Trips scheduled in [{day_labels[0]}..{day_labels[-1]}]: {trips}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌", e)
        sys.exit(1)
