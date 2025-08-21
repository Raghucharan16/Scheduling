#!/usr/bin/env python3
"""
ev_month_scheduler.py  – schedule an electric-bus fleet for a month.

CSV format:
slot,sector,route,1,2,3, … 30
00:00:00,H-V,H-V,97.9,76.3, …
00:00:00,V-H,V-H,95.1,77.2, …
(Exactly two symmetric routes.)

Requires:  pip install ortools pandas jinja2
"""

from pathlib import Path
import json, sys
import pandas as pd
from ortools.sat.python import cp_model
from jinja2 import Template

# ─────────────────────────  CONSTANTS  ─────────────────────────
FORBIDDEN = {0,1,2,3,4,5,6}        # disallow 00:00..03:00 (inclusive)
SLOTS_PER_DAY = 48                 # 30-min grid
ALLOWED = [s for s in range(SLOTS_PER_DAY) if s not in FORBIDDEN]
TRIPS_PER_DAY = 2                  # per bus
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
.bus-route-a{background:#2196F3;color:#fff}
.bus-route-b{background:#FF9800;color:#fff}
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
<th style="width:45%"><div class="route-header" style="background:#2196F3">{{ routeAB_name }}</div></th>
<th style="width:45%"><div class="route-header" style="background:#FF9800">{{ routeBA_name }}</div></th>
</tr>
</thead>
<tbody>
{% for d in range(days|length) %}
<tr>
<td style="background:#f0f0f0;font-weight:bold">{{days[d]}}</td>
<td>
  {% for bus,tr in ab[d] %}
  <div class="bus bus-route-a" title="{{bus}} - {{tr.startTime}} - EPK: {{tr.epk}}">
    {{bus}}<div class="bus-time">{{tr.startTime}}</div><div class="bus-epk">EPK: {{tr.epk}}</div>
  </div>
  {% endfor %}
</td>
<td>
  {% for bus,tr in ba[d] %}
  <div class="bus bus-route-b" title="{{bus}} - {{tr.startTime}} - EPK: {{tr.epk}}">
    {{bus}}<div class="bus-time">{{tr.startTime}}</div><div class="bus-epk">EPK: {{tr.epk}}</div>
  </div>
  {% endfor %}
</td>
</tr>
{% endfor %}
</tbody>
</table>
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

# ─────────────────────────  SOLVER  ─────────────────────────
def _build_model(epk, days, routeAB, routeBA,
                 busesA, busesB,
                 travel_s, charge_s, idle_s,
                 min_gap_slots,
                 lock_epk=None, lock_sec=None,
                 objective="primary",
                 enable_logs=False,
                 deterministic=True, seed=123,
                 limit=None):  # <-- added
    """Build a fresh model for each lexicographic pass."""
    SLOTS = SLOTS_PER_DAY
    D     = len(days)
    H     = D * SLOTS
    B     = busesA + busesB
    K     = 2 * D
    TOTAL = travel_s + charge_s

    home = ["A"] * busesA + ["B"] * busesB
    def route_for(b, k):
        if home[b] == "A":
            return routeAB if (k % 2 == 0) else routeBA
        else:
            return routeBA if (k % 2 == 0) else routeAB

    def dep_station(rt): return "A" if rt == routeAB else "B"

    # candidates
    route_abs = {routeAB: [], routeBA: []}
    for rt in (routeAB, routeBA):
        for d in range(D):
            for s in ALLOWED:
                route_abs[rt].append(d * SLOTS + s)
        route_abs[rt].sort()

    if any(len(route_abs[route_for(b, k)]) == 0 for b in range(B) for k in range(K)):
        raise RuntimeError("Some (b,k) has no candidate starts — check ALLOWED/FORBIDDEN and data.")

    m = cp_model.CpModel()
    y, st = {}, {}
    occ = {}
    day_route_vars = {(routeAB, d): [] for d in range(D)}
    day_route_vars.update({(routeBA, d): [] for d in range(D)})
    all_y = []

    # vars & linking
    for b in range(B):
        for k in range(K):
            rt = route_for(b, k)
            cands = route_abs[rt]
            lits  = []
            for abs_t in cands:
                v = m.NewBoolVar(f"y_{b}_{k}_{abs_t}")
                y[(b, k, abs_t)] = v
                lits.append(v)
                all_y.append(v)
                occ.setdefault((dep_station(rt), abs_t), []).append(v)
                d = abs_t // SLOTS
                day_route_vars[(rt, d)].append(v)
            m.AddExactlyOne(lits)
            st[(b, k)] = m.NewIntVar(0, H - 1, f"st_{b}_{k}")
            m.Add(st[(b, k)] == sum(abs_t * y[(b, k, abs_t)] for abs_t in cands))

    # same-bus sequencing
    for b in range(B):
        for k in range(K - 1):
            m.Add(st[(b, k + 1)] >= st[(b, k)] + TOTAL)
            m.Add(st[(b, k + 1)] - st[(b, k)] <= TOTAL + idle_s)
        for k in range(K - 2):
            m.Add(st[(b, k + 2)] >= st[(b, k)] + SLOTS)

    # station capacity + cool-down
    for (stn, abs_t), vars_list in occ.items():
        m.AddAtMostOne(vars_list)
    if min_gap_slots > 1:
        by_station = {"A": sorted(t for (s, t) in occ if s == "A"),
                      "B": sorted(t for (s, t) in occ if s == "B")}
        for stn, times in by_station.items():
            for t in times:
                for g in range(1, min_gap_slots):
                    t2 = t + g
                    if (stn, t2) in occ:
                        m.AddAtMostOne(occ[(stn, t)] + occ[(stn, t2)])

    # per-day per-route quotas
    for d in range(D):
        m.Add(sum(day_route_vars[(routeAB, d)]) == B)
        m.Add(sum(day_route_vars[(routeBA, d)]) == B)

    # decision strategy
    m.AddDecisionStrategy(all_y,
                          cp_model.CHOOSE_FIRST,
                          cp_model.SELECT_MAX_VALUE)

    # objectives
    epk_scale = 100
    prim_terms = []
    for (b, k, abs_t), var in y.items():
        d = abs_t // SLOTS
        s  = abs_t %  SLOTS
        rt = route_for(b, k)
        epk_i = int(round(float(epk.get((rt, d, s), 0.0)) * epk_scale))
        prim_terms.append(epk_i * var)
    sec_terms = [abs_t * var for (b, k, abs_t), var in y.items()]

    if lock_epk is not None:
        m.Add(sum(prim_terms) == int(lock_epk))
    if lock_sec is not None:
        m.Add(sum(sec_terms)  == int(lock_sec))

    if objective == "primary":
        m.Maximize(sum(prim_terms))
    elif objective == "secondary":
        m.Maximize(sum(sec_terms))
    elif objective == "tie":
        S = H
        tie_terms = []
        for (b, k, abs_t), var in y.items():
            tie_id = abs_t + S * (k + (2 * D) * b)
            tie_terms.append(tie_id * var)
        m.Maximize(sum(tie_terms))
    else:
        raise ValueError("unknown objective")

    # solver
    s = cp_model.CpSolver()
    s.parameters.log_search_progress = enable_logs
    s.parameters.num_search_workers  = 1
    s.parameters.search_branching    = cp_model.FIXED_SEARCH
    s.parameters.randomize_search    = False
    if deterministic:
        s.parameters.random_seed     = seed
    if isinstance(limit, (int, float)) and limit is not None and limit > 0:
        s.parameters.max_time_in_seconds = float(limit)

    return m, s, y, st, route_abs, (SLOTS, D, H, B, K)

def solve(epk, days, routeAB, routeBA,
          busesA, busesB,
          travel_s, charge_s, idle_s,
          limit=None,                 # <-- added, default unlimited
          deterministic=True, seed=123,
          enable_logs=False,
          min_gap_slots=2):
    """3-pass lexicographic solve → deterministic, canonical optimum."""
    # Pass 1: maximize EPK
    m1, s1, y1, st1, route_abs, dims = _build_model(
        epk, days, routeAB, routeBA,
        busesA, busesB, travel_s, charge_s, idle_s,
        min_gap_slots, objective="primary",
        enable_logs=enable_logs, deterministic=deterministic, seed=seed,
        limit=limit
    )
    st = s1.Solve(m1)
    if st != cp_model.OPTIMAL:
        name = ("INFEASIBLE" if st == cp_model.INFEASIBLE else
                "MODEL_INVALID" if st == cp_model.MODEL_INVALID else
                "FEASIBLE" if st == cp_model.FEASIBLE else "UNKNOWN")
        if st == cp_model.FEASIBLE:
            raise RuntimeError("Stopped before proving optimal EPK in pass 1. Increase time or set limit=None.")
        raise RuntimeError(f"No feasible schedule. Solver status: {name}.")
    opt_epk = int(round(s1.ObjectiveValue()))

    # Pass 2: lock EPK, maximize late times
    m2, s2, y2, st2, _, _ = _build_model(
        epk, days, routeAB, routeBA,
        busesA, busesB, travel_s, charge_s, idle_s,
        min_gap_slots, lock_epk=opt_epk, objective="secondary",
        enable_logs=enable_logs, deterministic=deterministic, seed=seed,
        limit=limit
    )
    st = s2.Solve(m2)
    if st != cp_model.OPTIMAL:
        name = ("INFEASIBLE" if st == cp_model.INFEASIBLE else
                "MODEL_INVALID" if st == cp_model.MODEL_INVALID else
                "FEASIBLE" if st == cp_model.FEASIBLE else "UNKNOWN")
        raise RuntimeError(f"Failed to prove optimal tie-break (pass 2). Status: {name}.")
    best_sec = int(round(s2.ObjectiveValue()))

    # Pass 3: lock both, canonical tiny tie
    m3, s3, y3, st3, _, (SLOTS, D, H, B, K) = _build_model(
        epk, days, routeAB, routeBA,
        busesA, busesB, travel_s, charge_s, idle_s,
        min_gap_slots, lock_epk=opt_epk, lock_sec=best_sec, objective="tie",
        enable_logs=enable_logs, deterministic=deterministic, seed=seed,
        limit=limit
    )
    st = s3.Solve(m3)
    if st != cp_model.OPTIMAL:
        name = ("INFEASIBLE" if st == cp_model.INFEASIBLE else
                "MODEL_INVALID" if st == cp_model.MODEL_INVALID else
                "FEASIBLE" if st == cp_model.FEASIBLE else "UNKNOWN")
        raise RuntimeError(f"Failed to prove canonical optimum (pass 3). Status: {name}.")

    # Build schedule from chosen y3
    def s2t_local(s): return f"{s//2:02d}:{(s%2)*30:02d}:00"
    sched = {}
    home = ["A"] * busesA + ["B"] * busesB
    def route_for(b, k):
        return routeAB if (home[b] == "A" and k % 2 == 0) or (home[b] == "B" and k % 2 == 1) else routeBA

    for b in range(B):
        bus = f"Bus-{b+1:02d}"
        sched[bus] = {dy: [] for dy in days}
        for k in range(2 * D):
            abs_val = None
            rt = route_for(b, k)
            for abs_t in route_abs[rt]:
                var = y3[(b, k, abs_t)]
                if s3.Value(var) == 1:
                    abs_val = abs_t
                    break
            if abs_val is None:
                continue
            d = abs_val // SLOTS
            s  = abs_val %  SLOTS
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

# ─────────────────────────  METRICS / HTML  ─────────────────────────
def compute_metrics(sched):
    total_buses = len(sched)
    total_epk, total_trips = 0, 0
    for bus in sched.values():
        for day in bus.values():
            for trip in day:
                total_epk += trip["epk"]
                total_trips += 1
    avg_epk = round(total_epk / total_trips, 2) if total_trips else 0
    return total_buses, avg_epk

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
            ab=ab, ba=ba,
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
    print("Electric-Bus Month Scheduler")
    csv = ask_path("Path to EPK CSV", "cache/epk_h_v_month.csv")
    busesA = ask_int("Buses at Station A",5)
    busesB = ask_int("Buses at Station B",5)
    travel_h = ask_int("Travel hours (one leg)",9)
    charge_h = ask_int("Depot charge hours",2)
    idle_h   = ask_int("Max *extra* idle hours",4)
    jsonf = ask_path("JSON output", DEFAULT_JSON)
    htmlf = ask_path("HTML output", DEFAULT_HTML)

    print("⏳ Loading CSV …")
    epk, days, rAB, rBA = load_epk(csv)   # <- rAB/rBA names

    print("⚙️  Solving … (deterministic, canonical optimum)")
    sched = solve(
        epk=epk, days=days,
        routeAB=rAB, routeBA=rBA,          # <- use rAB/rBA here
        busesA=busesA, busesB=busesB,
        travel_s=int(round(travel_h*2)),
        charge_s=int(round(charge_h*2)),
        idle_s=int(round(idle_h*2)),                      # no time limit → prove OPTIMAL in each pass
        deterministic=True,
        seed=123,
        enable_logs=True,
        min_gap_slots=2
    )

    print("💾 Writing files …")
    jsonf.write_text(json.dumps({"schedule":sched},indent=2))
    html_out(sched, htmlf, days, rAB, rBA)
    print(f"✅ JSON → {jsonf}\n✅ HTML → {htmlf}")

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)