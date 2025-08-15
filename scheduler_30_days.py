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
def solve(epk, days, routeAB, routeBA,
          busesA, busesB,
          travel_s, charge_s, idle_s, limit=300):

    TOTAL = travel_s + charge_s
    FIRST_OK = [s for s in ALLOWED if s + TOTAL + idle_s <= SLOTS_PER_DAY - 1]

    B = busesA + busesB
    D = len(days)
    H = D * SLOTS_PER_DAY  # horizon in 30-min slots

    home = ["A"] * busesA + ["B"] * busesB
    seqA = [routeAB, routeBA]  # for A-home buses: A→B then B→A
    seqB = [routeBA, routeAB]  # for B-home buses: B→A then A→B

    m = cp_model.CpModel()

    y, start, abs_start = {}, {}, {}

    # Decision vars: choose a start slot for each (bus, day, trip index)
    for b in range(B):
        for d in range(D):
            for t in range(TRIPS_PER_DAY):
                rt = seqA[t] if home[b] == "A" else seqB[t]
                pool = FIRST_OK if t == 0 else ALLOWED

                # create y[b,d,t,s] only for feasible (route,day,slot) that have EPK
                for s in pool:
                    if (rt, d, s) in epk:
                        y[(b, d, t, s)] = m.NewBoolVar(f"y_{b}_{d}_{t}_{s}")

                # exactly one start slot must be chosen
                chosen = [y[(b, d, t, s)] for s in pool if (b, d, t, s) in y]
                if not chosen:
                    # If no slot has EPK for this (b,d,t), the instance is infeasible.
                    # Create a dummy constraint to fail fast and explain.
                    m.AddBoolOr([])  # always false -> infeasible if hit
                else:
                    m.Add(sum(chosen) == 1)

                # start time in local day
                start[(b, d, t)] = m.NewIntVar(0, SLOTS_PER_DAY - 1, f"st_{b}_{d}_{t}")
                m.Add(start[(b, d, t)] ==
                      sum(s * y[(b, d, t, s)] for s in pool if (b, d, t, s) in y))

                # absolute start time across the whole month
                abs_start[(b, d, t)] = m.NewIntVar(0, H - 1, f"abs_st_{b}_{d}_{t}")
                m.Add(abs_start[(b, d, t)] == start[(b, d, t)] + d * SLOTS_PER_DAY)

            # Same-day precedence + (bounded) extra idle between the 2 daily trips
            m.Add(start[(b, d, 1)] >= start[(b, d, 0)] + TOTAL)
            m.Add(start[(b, d, 1)] - start[(b, d, 0)] <= TOTAL + idle_s)

    # === NEW: Cross-day chaining (hard) ===
    # Next day’s first trip cannot depart before previous day’s second trip
    # completes travel + charge.
    for b in range(B):
        for d in range(D - 1):
            m.Add(abs_start[(b, d + 1, 0)] >= abs_start[(b, d, 1)] + TOTAL)

    # Spacing at each station (within day)
    def safe(model, vars_, sense, rhs):
        if vars_:
            model.Add((sum(vars_)).__getattribute__(sense)(rhs))

    for d in range(D):
        for station, rt in (("A", routeAB), ("B", routeBA)):
            group = [b for b in range(B) if home[b] == station]
            idxA, idxB = (0, 1) if station == "A" else (1, 0)
            for s in ALLOWED:
                now = [y[(b, d, idxA, s)] for b in group if (b, d, idxA, s) in y] + \
                      [y[(b, d, idxB, s)] for b in group if (b, d, idxB, s) in y]
                safe(m, now, "__le__", 1)
                for δ in range(1, SPACING):
                    s2 = s + δ
                    if s2 not in ALLOWED:
                        continue
                    gap = now + \
                          [y[(b, d, idxA, s2)] for b in group if (b, d, idxA, s2) in y] + \
                          [y[(b, d, idxB, s2)] for b in group if (b, d, idxB, s2) in y]
                    safe(m, gap, "__le__", 1)

    # Objective: maximize total EPK
    obj = []
    for (b, d, t, s), var in y.items():
        rt = seqA[t] if home[b] == "A" else seqB[t]
        obj.append(int(epk.get((rt, d, s), 0) * 100) * var)
    m.Maximize(sum(obj))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = limit
    solver.parameters.num_search_workers = 8

    status = solver.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible schedule – try larger max idle or check EPK coverage.")

    # Build schedule (unchanged)
    sched = {}
    for b in range(B):
        bus = f"Bus-{b + 1:02d}"
        sched[bus] = {}
        for d, dy in enumerate(days):
            trips = []
            for t in range(TRIPS_PER_DAY):
                rt = seqA[t] if home[b] == "A" else seqB[t]
                s = int(solver.Value(start[(b, d, t)]))
                trips.append({
                    "route": rt,
                    "startTime": s2t(s),
                    "midPointTime": s2t((s + travel_s // 2) % SLOTS_PER_DAY),
                    "endTime": s2t((s + travel_s) % SLOTS_PER_DAY),
                    "epk": round(epk.get((rt, d, s), 0), 2)
                })
            trips.sort(key=lambda x: x["startTime"])
            sched[bus][dy] = trips

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
