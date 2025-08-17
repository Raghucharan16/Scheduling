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
          travel_s, charge_s, idle_s, limit=300,
          deterministic=True, seed=123, enable_logs=False):

    from ortools.sat.python import cp_model

    # ---------- constants ----------
    TOTAL = travel_s + charge_s        # min gap between consecutive trips of same bus (in 30-min slots)
    SLOTS = SLOTS_PER_DAY              # 48
    D = len(days)
    H = D * SLOTS
    B = busesA + busesB                # total buses

    # route selection along a bus's chain (alternating)
    home = ["A"] * busesA + ["B"] * busesB
    def route_for(b, k):
        if home[b] == "A":
            return routeAB if (k % 2 == 0) else routeBA
        else:
            return routeBA if (k % 2 == 0) else routeAB

    def dep_station(rt):  # which station the route departs from
        return "A" if rt == routeAB else "B"

    # ---------- candidates: every ALLOWED absolute slot per route ----------
    route_abs = {routeAB: [], routeBA: []}  # sorted lists of abs times
    for rt in (routeAB, routeBA):
        for d in range(D):
            for s in ALLOWED:
                route_abs[rt].append(d * SLOTS + s)
        route_abs[rt].sort()

    # ---------- model ----------
    m = cp_model.CpModel()
    K = 2 * D                                # trips per bus across month
    y = {}                                   # (b,k,abs) -> Bool, start at absolute time 'abs'
    start_abs = {}                           # (b,k)     -> Int absolute start in [0, H-1]

    # station occupancy tracker: at most one depart per station per absolute slot
    occ = {("A", t): [] for t in range(H)}
    occ.update({("B", t): [] for t in range(H)})

    # Build variables (exactly-one start per (b,k))
    for b in range(B):
        for k in range(K):
            rt = route_for(b, k)
            vars_k = []
            for abs_t in route_abs[rt]:
                v = m.NewBoolVar(f"y_{b}_{k}_{abs_t}")
                y[(b, k, abs_t)] = v
                vars_k.append((abs_t, v))
                occ[(dep_station(rt), abs_t)].append(v)
            # exactly one time for this trip
            m.Add(sum(v for _, v in vars_k) == 1)
            # link to integer start time
            st = m.NewIntVar(0, H - 1, f"st_{b}_{k}")
            start_abs[(b, k)] = st
            m.Add(st == sum(abs_t * y[(b, k, abs_t)] for abs_t, _ in vars_k))

    # ---------- same-bus sequencing ----------
    # (a) turnaround min/max between consecutive trips
    for b in range(B):
        for k in range(K - 1):
            m.Add(start_abs[(b, k + 1)] >= start_abs[(b, k)] + TOTAL)
            m.Add(start_abs[(b, k + 1)] - start_abs[(b, k)] <= TOTAL + idle_s)

    # (b) NO "3 trips in a 24h window": at most 2 in any 48-slot span
    for b in range(B):
        for k in range(K - 2):
            m.Add(start_abs[(b, k + 2)] >= start_abs[(b, k)] + SLOTS)

    # ---------- station spacing (per absolute slot) ----------
    for t in range(H):
        if occ[("A", t)]: m.Add(sum(occ[("A", t)]) <= 1)
        if occ[("B", t)]: m.Add(sum(occ[("B", t)]) <= 1)
    # If you require a ≥1h gap, loop δ=1 and also bound t/t+δ together (not shown).

    # ---------- per-day per-route quotas (fixes Day-1/Day-30 imbalance) ----------
    # exactly B departures of AB and B of BA every day
    for d in range(D):
        lo, hi = d * SLOTS, (d + 1) * SLOTS - 1
        # AB on day d
        ab_vars = []
        # BA on day d
        ba_vars = []
        for b in range(B):
            for k in range(K):
                rt = route_for(b, k)
                # add all y[b,k,abs] whose abs is in day d
                # (checking key existence keeps it efficient)
                if rt == routeAB:
                    for s in range(lo, hi + 1):
                        key = (b, k, s)
                        if key in y: ab_vars.append(y[key])
                else:
                    for s in range(lo, hi + 1):
                        key = (b, k, s)
                        if key in y: ba_vars.append(y[key])
        # enforce quotas
        m.Add(sum(ab_vars) == B)
        m.Add(sum(ba_vars) == B)

    # ---------- objective: maximize total EPK (tiny time tie-breaker) ----------
    obj = []
    for (b, k, abs_t), var in y.items():
        d = abs_t // SLOTS
        s = abs_t % SLOTS
        rt = route_for(b, k)
        val = epk.get((rt, d, s), 0.0)
        # strong EPK term + tiny preference for later times to break ties toward nights
        obj.append(int(val * 100000) * var + abs_t * var)
    m.Maximize(sum(obj))

    # ---------- solve ----------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = limit
    solver.parameters.log_search_progress = enable_logs
    if deterministic:
        solver.parameters.num_search_workers = 8
        solver.parameters.random_seed = seed
    else:
        solver.parameters.num_search_workers = 8

    status = solver.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        name = ("INFEASIBLE" if status == cp_model.INFEASIBLE else
                "MODEL_INVALID" if status == cp_model.MODEL_INVALID else "UNKNOWN")
        raise RuntimeError(f"No feasible schedule. Solver status: {name}.")

    # ---------- build schedule back into your day buckets ----------
    def s2t_local(s): return f"{s//2:02d}:{(s%2)*30:02d}:00"
    sched = {}
    for b in range(B):
        bus = f"Bus-{b+1:02d}"
        sched[bus] = {dy: [] for dy in days}
        for k in range(K):
            abs_val = int(solver.Value(start_abs[(b, k)]))
            d = abs_val // SLOTS
            s = abs_val % SLOTS
            rt = route_for(b, k)
            if 0 <= d < D:
                sched[bus][days[d]].append({
                    "route": rt,
                    "startTime": s2t_local(s),
                    "midPointTime": s2t_local((s + travel_s // 2) % SLOTS),
                    "endTime": s2t_local((s + travel_s) % SLOTS),
                    "epk": round(epk.get((rt, d, s), 0.0), 2)
                })
        for dy in days:
            sched[bus][dy].sort(key=lambda tr: tr["startTime"])
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
