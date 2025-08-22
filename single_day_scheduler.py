#!/usr/bin/env python3
"""
ev_month_scheduler.py  – schedule an electric‑bus fleet for a month.

CSV format:
slot,sector,route,1,2,3, … 30
00:00:00,H‑V,H‑V,97.9,76.3, …
00:00:00,V‑H,V‑H,95.1,77.2, …
(Exactly two symmetric routes.)

Requires:  pip install ortools pandas jinja2
"""

from pathlib import Path
import json, sys
import pandas as pd
from ortools.sat.python import cp_model
from jinja2 import Template

# ─────────────────────────  CONSTANTS  ─────────────────────────
FORBIDDEN = {1,2,3,4,5,6}      # 00:30‑03:00 departure slots
SLOTS_PER_DAY = 48             # 30‑min grid
ALLOWED = [s for s in range(SLOTS_PER_DAY) if s not in FORBIDDEN]
TRIPS_PER_DAY = 2              # per bus
SPACING = 1                    # ≥ 1 h between same‑station departs
DEFAULT_JSON = Path("schedule.json")
DEFAULT_HTML = Path("schedule.html")

def s2t(s): return f"{s//2:02d}:{(s%2)*30:02d}:00"

HTML = """<!doctype html><html><head>
<meta charset="utf-8"><style>
body{font-family:Arial,Helvetica,sans-serif;margin:0;padding:1rem;font-size:14px}
table{border-collapse:collapse;width:100%;margin-top:20px}
th,td{border:1px solid #ccc;padding:8px;text-align:center;vertical-align:top}
th{background:#333;color:#fff;font-weight:bold}
tr:nth-child(odd){background:#f9f9f9}
.bus{background:#4CAF50;color:#fff;border-radius:4px;padding:6px;margin:2px;display:inline-block;width:80px;font-size:12px;font-weight:bold}
.bus-time{font-size:11px;margin-top:2px}
.bus-epk{font-size:10px;opacity:0.9;margin-top:2px}
.route-header{background:#00bcd4;color:#fff;padding:8px;text-align:center;font-weight:bold}

/* Route A (Hyderabad-Vijayawada) - Blue theme */
.bus-route-a{background:#2196F3;color:#fff}
.bus-route-a:hover{background:#1976D2}

/* Route B (Vijayawada-Hyderabad) - Orange theme */
.bus-route-b{background:#FF9800;color:#fff}
.bus-route-b:hover{background:#F57C00}

/* Individual bus colors for better identification */
.bus-01{background:#E91E63}
.bus-02{background:#9C27B0}
.bus-03{background:#673AB7}
.bus-04{background:#3F51B5}
.bus-05{background:#2196F3}
.bus-06{background:#00BCD4}
.bus-07{background:#009688}
.bus-08{background:#4CAF50}
.bus-09{background:#8BC34A}
.bus-10{background:#CDDC39}
.bus-11{background:#FFEB3B}
.bus-12{background:#FFC107}
.bus-13{background:#FF9800}
.bus-14{background:#FF5722}
.bus-15{background:#795548}
.bus-16{background:#607D8B}
.bus-17{background:#9E9E9E}
.bus-18{background:#F44336}
.bus-19{background:#E91E63}
.bus-20{background:#9C27B0}

/* Hover effects */
.bus:hover{opacity:0.8;transform:scale(1.05);transition:all 0.2s ease}
</style></head><body>
<div style="background:#00bcd4;color:#fff;padding:1rem;text-align:center;">
  <h2>EV Bus Schedule</h2>
  <div>
    <strong>Route 1:</strong> {{ routeAB_name }} &nbsp; | &nbsp;
    <strong>Route 2:</strong> {{ routeBA_name }}
  </div>
  <div>
    <strong>No. of Buses:</strong> {{ total_buses }} &nbsp; | &nbsp;
    <strong>Avg EPK:</strong> {{ avg_epk }}
  </div>
</div>

<table>
<thead>
<tr>
<th style="width:80px">Day</th>
<th style="width:45%">
  <div class="route-header" style="background:#2196F3">{{ routeAB_name }}</div>
</th>
<th style="width:45%">
  <div class="route-header" style="background:#FF9800">{{ routeBA_name }}</div>
</th>
</tr>
</thead>
<tbody>
{% for d in range(days|length) %}
<tr>
<td style="background:#f0f0f0;font-weight:bold">{{days[d]}}</td>
<td>
  {% for bus,tr in ab[d] %}
  {% set bus_num = bus.split('-')[1] %}
  <div class="bus bus-{{bus_num}} bus-route-a" title="{{bus}} - {{tr.startTime}} - EPK: {{tr.epk}}">
    {{bus}}
    <div class="bus-time">{{tr.startTime}}</div>
    <div class="bus-epk">EPK: {{tr.epk}}</div>
  </div>
  {% endfor %}
</td>
<td>
  {% for bus,tr in ba[d] %}
  {% set bus_num = bus.split('-')[1] %}
  <div class="bus bus-{{bus_num}} bus-route-b" title="{{bus}} - {{tr.startTime}} - EPK: {{tr.epk}}">
    {{bus}}
    <div class="bus-time">{{tr.startTime}}</div>
    <div class="bus-epk">EPK: {{tr.epk}}</div>
  </div>
  {% endfor %}
</td>
</tr>
{% endfor %}
</tbody>
</table>

<div style="margin-top:20px;padding:15px;background:#f5f5f5;border-radius:5px;">
  <h4>Color Legend:</h4>
  <div style="display:flex;justify-content:space-around;flex-wrap:wrap;">
    <div style="margin:5px;">
      <span style="background:#2196F3;color:#fff;padding:5px;border-radius:3px;">{{ routeAB_name }}</span>
    </div>
    <div style="margin:5px;">
      <span style="background:#FF9800;color:#fff;padding:5px;border-radius:3px;">{{ routeBA_name }}</span>
    </div>
  </div>
  <p style="font-size:12px;color:#666;margin-top:10px;">
    <strong>Note:</strong> Each bus maintains the same color across all days for easy identification.
  </p>
</div>
</body></html>"""

# ─────────────────────────  LOAD CSV  ─────────────────────────
def load_epk(csv_path: Path):
    df = pd.read_csv(csv_path)
    day_cols = sorted([c for c in df.columns if c.isdigit()], key=int)
    if not day_cols: raise ValueError("No numeric day columns found (1,2,3,…).")

    df["slot_idx"] = (
        pd.to_timedelta(df["slot"]).dt.components.hours*2 +
        pd.to_timedelta(df["slot"]).dt.components.minutes//30
    ).astype(int)

    routes = sorted(df["route"].unique())
    if len(routes)!=2: raise ValueError("CSV must contain exactly two routes.")
    routeAB, routeBA = routes

    epk = {}
    for _,r in df.iterrows():
        s = int(r["slot_idx"])
        if s in FORBIDDEN: continue
        for d,col in enumerate(day_cols):
            epk[(r["route"], d, s)] = float(r[col])
    return epk, day_cols, routeAB, routeBA

# ─────────────────────────  SAFE ADD  ─────────────────────────
def safe(model, vars_, sense, rhs):
    if vars_: model.Add((sum(vars_)).__getattribute__(sense)(rhs))

# ─────────────────────────  SOLVER  ─────────────────────────
from ortools.sat.python import cp_model

def solve(epk, days, routeAB, routeBA,
          busesA, busesB,
          travel_s, charge_s, idle_s,
          min_gap_slots=2,
          enable_logs=False,
          deterministic=True, seed=123,
          time_limit=None):
    """
    Fast & optimal 24h cyclic template solved via pair selection.
    Decision = pick B pairs (sAB, sBA) with:
      - Δ = (sBA - sAB) mod 48 ∈ [TOTAL, TOTAL+idle] AND (sAB - sBA) mod 48 ∈ [TOTAL, TOTAL+idle]
      - station 'cool-down' min_gap_slots on A and B
      - exactly B pairs chosen
    Objective (monthly): sum_d [EPK(AB, d, sAB) + EPK(BA, d, sBA)].
    Deterministic, 3-pass lexicographic optimum (FIXED_SEARCH).
    """

    # -------- constants & helpers --------
    SLOTS = SLOTS_PER_DAY
    D     = len(days)
    B     = busesA + busesB
    TOTAL = travel_s + charge_s

    def dep_station(rt): return "A" if rt == routeAB else "B"

    # valid slots per route (same ALLOWED for both)
    S = sorted(ALLOWED)

    # monthly weights on 24h grid
    Wab = {s: 0.0 for s in S}
    Wba = {s: 0.0 for s in S}
    for d_idx in range(D):
        for s in S:
            Wab[s] += float(epk.get((routeAB, d_idx, s), 0.0))
            Wba[s] += float(epk.get((routeBA, d_idx, s), 0.0))

    # Δ must satisfy both forward and wrap-around windows
    lo1, hi1 = TOTAL, TOTAL + idle_s
    lo2, hi2 = (SLOTS - (TOTAL + idle_s)), (SLOTS - TOTAL)
    d_lo, d_hi = max(lo1, lo2), min(hi1, hi2)
    if d_lo > d_hi:
        raise RuntimeError("No delta satisfies both turnaround and wrap-around. Adjust travel/charge/idle.")

    valid_delta = set(range(d_lo, d_hi + 1))

    # -------- build feasible pair universe P = {(sAB, sBA)} --------
    P = []                          # list of pairs
    idx = {}                        # map pair -> index
    for sAB in S:
        for dlt in valid_delta:
            sBA = (sAB + dlt) % SLOTS
            if sBA in Wba:          # must be in allowed set
                p = (sAB, sBA)
                idx[p] = len(P)
                P.append(p)

    if len(P) < B:
        raise RuntimeError("Not enough feasible (AB,BA) pairs to schedule all buses.")

    # -------- model --------
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x_{i}") for i in range(len(P))]

    # exactly B pairs
    m.Add(sum(x) == B)

    # station cool-down (circular) on A (AB times) and B (BA times)
    # bucket pairs by station time
    A_at = {s: [] for s in S}  # x indices that use AB time s
    B_at = {s: [] for s in S}  # x indices that use BA time s
    for i, (sAB, sBA) in enumerate(P):
        A_at[sAB].append(x[i])
        B_at[sBA].append(x[i])

    # per-slot capacity and min-gap (circular) for station A
    if min_gap_slots <= 0:
        min_gap_slots = 1
    for s0 in S:
        # capacity at the slot itself
        if A_at[s0]:
            m.AddAtMostOne(A_at[s0])
        # cool-down with next slots
        for g in range(1, min_gap_slots):
            s1 = (s0 + g) % SLOTS
            if s1 in A_at and (A_at[s0] or A_at[s1]):
                m.AddAtMostOne(A_at[s0] + A_at.get(s1, []))

    # per-slot capacity and min-gap for station B
    for s0 in S:
        if B_at[s0]:
            m.AddAtMostOne(B_at[s0])
        for g in range(1, min_gap_slots):
            s1 = (s0 + g) % SLOTS
            if s1 in B_at and (B_at[s0] or B_at[s1]):
                m.AddAtMostOne(B_at[s0] + B_at.get(s1, []))

    # -------- deterministic solver factory --------
    def make_solver():
        s = cp_model.CpSolver()
        s.parameters.log_search_progress = enable_logs
        s.parameters.num_search_workers  = 1
        s.parameters.search_branching    = cp_model.FIXED_SEARCH
        s.parameters.randomize_search    = False
        if deterministic:
            s.parameters.random_seed     = seed
        if isinstance(time_limit, (int, float)) and time_limit and time_limit > 0:
            s.parameters.max_time_in_seconds = float(time_limit)
        return s

    # Decision order: pairs in sorted (sAB, sBA) — stable & reproducible.
    order = sorted(range(len(P)), key=lambda i: P[i])
    m.AddDecisionStrategy([x[i] for i in order],
                          cp_model.CHOOSE_FIRST,
                          cp_model.SELECT_MAX_VALUE)

    # -------- 3-pass lexicographic objective --------
    scale = 100
    w1 = [int(round((Wab[sAB] + Wba[sBA]) * scale)) for (sAB, sBA) in P]
    w2 = [ (sAB + sBA) for (sAB, sBA) in P ]  # prefer later times overall
    w3 = [ (sAB + SLOTS * sBA) for (sAB, sBA) in P ]  # canonical tiny tie

    # pass 1
    m1 = m.Clone(); m1.Maximize(sum(w1[i] * x[i] for i in range(len(P))))
    s1 = make_solver(); st = s1.Solve(m1)
    if st != cp_model.OPTIMAL:
        name = ("INFEASIBLE" if st == cp_model.INFEASIBLE else
                "MODEL_INVALID" if st == cp_model.MODEL_INVALID else
                "FEASIBLE" if st == cp_model.FEASIBLE else "UNKNOWN")
        if st == cp_model.FEASIBLE:
            raise RuntimeError("Stopped before proving optimal monthly EPK (remove/raise time_limit).")
        raise RuntimeError(f"No feasible schedule. Solver status: {name}.")
    opt1 = int(round(s1.ObjectiveValue()))

    # pass 2
    m2 = m.Clone()
    m2.Add(sum(w1[i] * x[i] for i in range(len(P))) == opt1)
    m2.Maximize(sum(w2[i] * x[i] for i in range(len(P))))
    s2 = make_solver(); st = s2.Solve(m2)
    if st != cp_model.OPTIMAL:
        raise RuntimeError("Failed to prove optimal tie-break (later times).")
    opt2 = int(round(s2.ObjectiveValue()))

    # pass 3
    m3 = m.Clone()
    m3.Add(sum(w1[i] * x[i] for i in range(len(P))) == opt1)
    m3.Add(sum(w2[i] * x[i] for i in range(len(P))) == opt2)
    m3.Maximize(sum(w3[i] * x[i] for i in range(len(P))))
    s3 = make_solver(); st = s3.Solve(m3)
    if st != cp_model.OPTIMAL:
        raise RuntimeError("Failed to prove canonical optimum (pass 3).")

    # -------- build month by repeating the chosen pairs --------
    def s2t_local(v): return f"{v//2:02d}:{(v%2)*30:02d}:00"
    chosen = [P[i] for i in range(len(P)) if s3.Value(x[i]) == 1]
    # stable order for assignment
    chosen.sort()

    sched = {}
    # assign first busesA as home A, remaining as home B (display only)
    # but every pair yields one AB and one BA departure per day.
    for b in range(B):
        bus = f"Bus-{b+1:02d}"
        sched[bus] = {dy: [] for dy in days}
        sAB, sBA = chosen[b]
        for d_idx, d_lbl in enumerate(days):
            # AB
            sched[bus][d_lbl].append({
                "route": routeAB,
                "startTime": s2t_local(sAB),
                "midPointTime": s2t_local((sAB + travel_s // 2) % SLOTS),
                "endTime": s2t_local((sAB + travel_s) % SLOTS),
                "epk": round(float(epk.get((routeAB, d_idx, sAB), 0.0)), 2)
            })
            # BA
            sched[bus][d_lbl].append({
                "route": routeBA,
                "startTime": s2t_local(sBA),
                "midPointTime": s2t_local((sBA + travel_s // 2) % SLOTS),
                "endTime": s2t_local((sBA + travel_s) % SLOTS),
                "epk": round(float(epk.get((routeBA, d_idx, sBA), 0.0)), 2)
            })
        for d_lbl in days:
            sched[bus][d_lbl].sort(key=lambda tr: tr["startTime"])
    return sched


def compute_metrics(sched):
    total_buses = len(sched)
    total_epk = 0
    total_trips = 0
    for bus in sched.values():
        for day in bus.values():
            for trip in day:
                total_epk += trip["epk"]
                total_trips += 1
    avg_epk = round(total_epk / total_trips, 2) if total_trips else 0
    return total_buses, avg_epk

# ─────────────────────────  HTML WRITER  ─────────────────────────
def html_out(sched, html, days, routeAB, routeBA):
    ab=[[] for _ in days]; ba=[[] for _ in days]
    for bus,mp in sched.items():
        for i,d in enumerate(days):
            for tr in mp[d]:
                (ab if tr["route"]==routeAB else ba)[i].append((bus,tr))
    for i in range(len(days)):
        ab[i].sort(key=lambda bt:bt[1]["startTime"])
        ba[i].sort(key=lambda bt:bt[1]["startTime"])
    total_buses, avg_epk = compute_metrics(sched)
    html.write_text(
        Template(HTML).render(
            days=days,
            ab=ab,
            ba=ba,
            routeAB_name=routeAB,
            routeBA_name=routeBA,
            trips=TRIPS_PER_DAY,
            total_buses=total_buses,
            avg_epk=avg_epk
        ),
        encoding="utf-8"
    )

# ─────────────────────────  CLI PROMPTS  ─────────────────────────
def ask_int(msg, default):
    v=input(f"{msg} [{default}]: ").strip(); return int(v) if v else default
def ask_path(msg, default):
    v=input(f"{msg} [{default}]: ").strip(); return Path(v) if v else Path(default)

def main():
    print("Electric‑Bus Month Scheduler")
    csv = ask_path("Path to EPK CSV", "cache/epk_h_v_month.csv")
    busesA = ask_int("Buses at Station A",5)
    busesB = ask_int("Buses at Station B",5)
    travel_h = ask_int("Travel hours (one leg)",9)
    charge_h = ask_int("Depot charge hours",2)
    idle_h   = ask_int("Max *extra* idle hours",4)
    jsonf = ask_path("JSON output", DEFAULT_JSON)
    htmlf = ask_path("HTML output", DEFAULT_HTML)

    print("⏳ Loading CSV …")
    epk, days, rAB, rBA = load_epk(csv)
    print("⚙️  Solving …")
    sched = solve(epk, days, rAB, rBA,
                  busesA, busesB,
                  travel_h*2, charge_h*2, idle_h*2)

    print("💾 Writing files …")
    jsonf.write_text(json.dumps({"schedule":sched},indent=2))
    html_out(sched, htmlf, days, rAB, rBA)
    print(f"✅ JSON → {jsonf}\n✅ HTML → {htmlf}")

if __name__=="__main__":
    try: main()
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)
