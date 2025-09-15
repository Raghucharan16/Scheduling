#!/usr/bin/env python3
"""
EV month scheduler — constant template with priority for existing services.

If a Base Services CSV is provided, we:
  - Build a 24h constant template (two trips per bus per day) that repeats for all days.
  - Snap any selection within ±30 mins to the exact base time.
  - Prefer (or require) using base times first, then maximize monthly EPK.
  - Annotate trips with serviceKey when within ±30 min of a base time;
    otherwise mark as NEW (JSON + HTML).

If no Base Services CSV is provided, the script falls back to a dynamic month solver.

Requires:  pip install ortools pandas jinja2
"""

from pathlib import Path
import json, sys, re
import pandas as pd
from ortools.sat.python import cp_model
from jinja2 import Template

# ─────────────────────────  CONSTANTS / HTML  ─────────────────────────
FORBIDDEN = {0,1,2,3,4,5,6}        # forbid 00:00..03:00 departures
SLOTS_PER_DAY = 48                 # 30-min grid
ALLOWED = [s for s in range(SLOTS_PER_DAY) if s not in FORBIDDEN]
TRIPS_PER_DAY = 2
DEFAULT_JSON = Path("schedule.json")
DEFAULT_HTML = Path("schedule.html")

def s2t(s): return f"{s//2:02d}:{(s%2)*30:02d}:00"
def t2s(t):  # "HH:MM[:SS]" -> slot idx
    parts = t.strip().split(":")
    hh, mm = int(parts[0]), int(parts[1])
    return hh*2 + (mm//30)

HTML = """<!doctype html><html><head>
<meta charset="utf-8"><style>
body{font-family:Arial,Helvetica,sans-serif;margin:0;padding:1rem;font-size:14px}
table{border-collapse:collapse;width:100%;margin-top:20px}
th,td{border:1px solid #ccc;padding:8px;text-align:center;vertical-align:top}
th{background:#333;color:#fff;font-weight:bold}
tr:nth-child(odd){background:#f9f9f9}
.bus{background:#4CAF50;color:#fff;border-radius:4px;padding:6px;margin:2px;display:inline-block;width:118px;font-size:12px;font-weight:bold}
.bus-time{font-size:11px;margin-top:2px}
.bus-epk{font-size:10px;opacity:0.9;margin-top:2px}
.bus-key{font-size:10px;opacity:0.95;margin-top:4px}
.badge{display:inline-block;padding:2px 6px;border-radius:10px;font-size:10px;margin-top:4px}
.badge-base{background:#263238;color:#fff}
.badge-new{background:#F44336;color:#fff}
.route-header{background:#00bcd4;color:#fff;padding:8px;text-align:center;font-weight:bold}
.bus-route-a{background:#2196F3;color:#fff}
.bus-route-b{background:#FF9800;color:#fff}
.bus:hover{opacity:0.9;transform:scale(1.03);transition:all .15s ease}
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
<table><thead><tr>
<th style="width:80px">Day</th>
<th style="width:45%"><div class="route-header" style="background:#2196F3">{{ routeAB_name }}</div></th>
<th style="width:45%"><div class="route-header" style="background:#FF9800">{{ routeBA_name }}</div></th>
</tr></thead><tbody>
{% for d in range(days|length) %}
<tr>
<td style="background:#f0f0f0;font-weight:bold">{{days[d]}}</td>
<td>
{% for bus,tr in ab[d] %}
<div class="bus bus-route-a" title="{{bus}} - {{tr.startTime}} - EPK: {{tr.epk}}">
  {{bus}}<div class="bus-time">{{tr.startTime}}</div>
  <div class="bus-epk">EPK: {{tr.epk}}</div>
  {% if tr.isNew %}
    <span class="badge badge-new">NEW</span>
  {% elif tr.serviceKey %}
    <div class="bus-key">Key: {{tr.serviceKey}}</div>
  {% endif %}
</div>
{% endfor %}
</td>
<td>
{% for bus,tr in ba[d] %}
<div class="bus bus-route-b" title="{{bus}} - {{tr.startTime}} - EPK: {{tr.epk}}">
  {{bus}}<div class="bus-time">{{tr.startTime}}</div>
  <div class="bus-epk">EPK: {{tr.epk}}</div>
  {% if tr.isNew %}
    <span class="badge badge-new">NEW</span>
  {% elif tr.serviceKey %}
    <div class="bus-key">Key: {{tr.serviceKey}}</div>
  {% endif %}
</div>
{% endfor %}
</td>
</tr>
{% endfor %}
</tbody></table>
</body></html>"""

# ─────────────────────────  IO  ─────────────────────────
def load_epk(csv_path: Path):
    df = pd.read_csv(csv_path)
    day_cols = sorted([c for c in df.columns if str(c).isdigit()], key=lambda x:int(x))
    if not day_cols: raise ValueError("No numeric day columns (1,2,3,…).")
    td = pd.to_timedelta(df["slot"])
    df["slot_idx"] = (td.dt.components.hours*2 + td.dt.components.minutes//30).astype(int)
    routes = sorted(df["route"].unique())
    if len(routes)!=2: raise ValueError("EPK CSV must contain exactly two routes.")
    routeAB, routeBA = routes

    epk = {}
    for _,r in df.iterrows():
        s = int(r["slot_idx"])
        if s in FORBIDDEN: continue
        for d,col in enumerate(day_cols):
            epk[(r["route"], d, s)] = float(r[col])
    return epk, day_cols, routeAB, routeBA

def _norm_route_name(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())

def load_base_services(base_path: Path, routeAB: str, routeBA: str):
    """
    Returns:
      base_slots: {route: set(slots)}
      base_keys:  {route: {slot: serviceKey}}
    Accepts:
      - Columns: ['origin','destination','serviceKey','time']  OR
      - Columns: ['route','serviceKey','time']
    """
    df = pd.read_csv(base_path, sep=None, engine="python")  # auto-detect delimiters
    cols = {c.lower().strip(): c for c in df.columns}

    def map_to_epk(rt_txt: str) -> str | None:
        candidates = [
            routeAB, routeBA,
            routeAB.replace("-", "|"), routeBA.replace("-", "|"),
            routeAB.replace("–", "|").replace("—","|"), routeBA.replace("–","|").replace("—","|"),
        ]
        rt_norm = _norm_route_name(rt_txt)
        for cand in candidates:
            if _norm_route_name(cand) == rt_norm:
                return cand
        return None

    routes = [routeAB, routeBA]
    base_slots = {routeAB: set(), routeBA: set()}
    base_keys  = {routeAB: {}, routeBA: {}}

    if "route" in cols and "time" in cols:
        for _,r in df.iterrows():
            rt_raw = str(r[cols["route"]]).strip()
            rt = map_to_epk(rt_raw) or (map_to_epk(rt_raw.replace("|","-")) if "|" in rt_raw else None)
            if rt not in routes:  # ignore out-of-corridor rows
                continue
            tstr = str(r[cols["time"]]).strip()
            try:
                s = t2s(tstr if ":" in tstr else f"{tstr}:00")
            except:
                continue
            if s in FORBIDDEN: 
                continue
            base_slots[rt].add(s)
            if "servicekey" in cols or "serviceKey" in cols:
                key = str(r[cols.get("servicekey","serviceKey")]).strip()
                if key and s not in base_keys[rt]:
                    base_keys[rt][s] = key
    elif all(k in cols for k in ["origin","destination","time"]):
        for _,r in df.iterrows():
            o = str(r[cols["origin"]]).strip().upper()
            d = str(r[cols["destination"]]).strip().upper()
            rt_guess = f"{o}|{d}"
            rt = map_to_epk(rt_guess) or map_to_epk(f"{o}-{d}") or map_to_epk(f"{o}{d}")
            if rt not in routes: 
                continue
            tstr = str(r[cols["time"]]).strip()
            try:
                s = t2s(tstr if ":" in tstr else f"{tstr}:00")
            except:
                continue
            if s in FORBIDDEN: 
                continue
            base_slots[rt].add(s)
            if "servicekey" in cols or "serviceKey" in cols:
                key = str(r[cols.get("servicekey","serviceKey")]).strip()
                if key and s not in base_keys[rt]:
                    base_keys[rt][s] = key
    else:
        raise ValueError("Base CSV needs either (route, time[, serviceKey]) OR (origin, destination, time[, serviceKey]).")

    return base_slots, base_keys

# ─────────────────────────  CONSTANT TEMPLATE SOLVER (with base priority)  ─────────────────────────
def solve_constant_with_base(epk, days, routeAB, routeBA,
                             busesA, busesB,
                             travel_s, charge_s, idle_s,
                             base_slots,                # {route:set(slots)}
                             snap_radius_slots=1,       # snap within ±30min
                             require_all_base=False,    # if True, cover every base slot on each route
                             min_gap_slots=1,
                             limit=300, workers=8,
                             deterministic=True, seed=123,
                             enable_logs=False,
                             base_weight=1_000_000_000, # primary
                             epk_weight=1_000_000):     # secondary
    """
    Pick 24h template y[b,k,s], repeat for all days.
    Primary objective: cover as many base slots as possible (hard if require_all_base=True).
    Secondary: max monthly EPK (sum over days).
    Tertiary: prefer later slots (tiny).
    Snap rule: near a base slot p, remove p±1 from the candidate set so chosen time equals p.
    """

    SLOTS = SLOTS_PER_DAY
    D     = len(days)
    B     = busesA + busesB
    TOTAL = travel_s + charge_s
    home  = ["A"] * busesA + ["B"] * busesB

    def route_pos(b, k):  # k in {0,1}
        return routeAB if (home[b]=="A" and k==0) or (home[b]=="B" and k==1) else routeBA
    def dep_station(rt): return "A" if rt==routeAB else "B"

    # Monthly EPK weights on 24h grid
    W = {routeAB:{s:0.0 for s in ALLOWED}, routeBA:{s:0.0 for s in ALLOWED}}
    for d in range(D):
        for s in ALLOWED:
            W[routeAB][s] += float(epk.get((routeAB,d,s),0.0))
            W[routeBA][s] += float(epk.get((routeBA,d,s),0.0))

    # Build "snapped" allowed pools
    allow_rt = {routeAB:set(ALLOWED), routeBA:set(ALLOWED)}
    for rt in (routeAB, routeBA):
        near = set()
        for p in base_slots.get(rt, set()):
            for g in range(-snap_radius_slots, snap_radius_slots+1):
                if g == 0: continue
                s2 = p + g
                if 0 <= s2 < SLOTS and s2 in allow_rt[rt]:
                    near.add(s2)
        allow_rt[rt] -= near
        allow_rt[rt] |= (base_slots.get(rt, set()) & set(ALLOWED))

    if require_all_base:
        for rt in (routeAB, routeBA):
            need = len(base_slots.get(rt, set()) & set(ALLOWED))
            if need > B:
                raise RuntimeError(
                    f"require_all_base=True but route '{rt}' has {need} base times and only {B} buses/day."
                )

    m = cp_model.CpModel()
    y = {}             # y[b,k,s]
    t = {}             # start slot integer per (b,k)
    occ = {("A", s): [] for s in ALLOWED}
    occ.update({("B", s): [] for s in ALLOWED})

    # Decision variables
    for b in range(B):
        for k in (0,1):
            rt = route_pos(b,k)
            lits = []
            for s in sorted(allow_rt[rt]):
                v = m.NewBoolVar(f"y_{b}_{k}_{s}")
                y[(b,k,s)] = v
                lits.append(v)
                occ[(dep_station(rt), s)].append(v)
            if not lits:
                m.AddBoolOr([])  # infeasible
            else:
                m.Add(sum(lits) == 1)
            t[(b,k)] = m.NewIntVar(0, SLOTS-1, f"t_{b}_{k}")
            m.Add(t[(b,k)] == sum(s * y[(b,k,s)] for s in allow_rt[rt]))

    # Same-bus cyclic sequencing
    for b in range(B):
        m.Add(t[(b,1)] - t[(b,0)] >= TOTAL)
        m.Add(t[(b,1)] - t[(b,0)] <= TOTAL + idle_s)
        m.Add(t[(b,0)] - t[(b,1)] >= TOTAL - SLOTS)
        m.Add(t[(b,0)] - t[(b,1)] <= TOTAL + idle_s - SLOTS)

    # Station capacity and min-gap (circular)
    for (stn, s), lits in occ.items():
        m.Add(sum(lits) <= 1)
    if min_gap_slots and min_gap_slots > 1:
        for stn in ("A","B"):
            base = sorted(s for (x,s) in occ if x==stn)
            base_set = set(base)
            for s in base:
                for g in range(1, min_gap_slots):
                    s2 = (s+g) % SLOTS
                    if s2 in base_set:
                        m.AddAtMostOne(occ[(stn,s)] + occ[(stn,s2)])

    # Track coverage of base slots
    use_base = {}
    for rt in (routeAB, routeBA):
        for p in sorted(base_slots.get(rt, set()) & set(ALLOWED)):
            u = m.NewBoolVar(f"use_{_norm_route_name(rt)}_{p}")
            use_base[(rt,p)] = u
            sums = []
            for b in range(B):
                for k in (0,1):
                    if route_pos(b,k)==rt and (b,k,p) in y:
                        sums.append(y[(b,k,p)])
            if sums:
                m.Add(u <= sum(sums))
            else:
                m.Add(u == 0)

    if require_all_base:
        for rt in (routeAB, routeBA):
            for p in base_slots.get(rt,set()) & set(ALLOWED):
                m.Add(use_base[(rt,p)] == 1)

    # Objective (big weights emulate lexicographic)
    obj_terms = []
    obj_terms += [base_weight * use_base[(rt,p)] for (rt,p) in use_base]
    for (b,k,s), var in y.items():
        rt = route_pos(b,k)
        obj_terms.append(int(W[rt][s] * epk_weight) * var)
        obj_terms.append(s * var)  # tiny tie
    m.Maximize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(limit) if limit and limit>0 else 0
    solver.parameters.log_search_progress = enable_logs
    solver.parameters.num_search_workers = max(1,int(workers))
    if deterministic:
        solver.parameters.random_seed = seed

    status = solver.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        name = ("INFEASIBLE" if status == cp_model.INFEASIBLE else
                "MODEL_INVALID" if status == cp_model.MODEL_INVALID else "UNKNOWN")
        raise RuntimeError(f"No feasible constant template. Solver status: {name}.")

    # Build repeating schedule for all days; serviceKey tagging happens in main().
    sched = {}
    for b in range(B):
        bus = f"Bus-{b+1:02d}"
        sched[bus] = {dy: [] for dy in days}
        chosen = []
        for k in (0,1):
            chosen_s = None
            for s in allow_rt[route_pos(b,k)]:
                if solver.Value(y[(b,k,s)]) == 1:
                    chosen_s = s
                    break
            if chosen_s is None:
                continue
            chosen.append((k, chosen_s))
        for d_idx, d_lbl in enumerate(days):
            for k, s in chosen:
                rt = route_pos(b,k)
                sched[bus][d_lbl].append({
                    "route": rt,
                    "startTime": s2t(s),
                    "midPointTime": s2t((s + travel_s//2) % SLOTS),
                    "endTime": s2t((s + travel_s) % SLOTS),
                    "epk": round(float(epk.get((rt,d_idx,s),0.0)),2),
                    "serviceKey": None,
                    "isNew": True  # default; will be overwritten if we find a base match
                })
        for d_lbl in days:
            sched[bus][d_lbl].sort(key=lambda tr: tr["startTime"])
    return sched

# ─────────────────────────  DYNAMIC (fallback if no base CSV)  ─────────────────────────
def solve_abs_month(epk, days, routeAB, routeBA,
                    busesA, busesB, travel_s, charge_s, idle_s,
                    limit=300, workers=8, min_gap_slots=1,
                    deterministic=True, seed=123, enable_logs=False):

    SLOTS, D = SLOTS_PER_DAY, len(days)
    H = D * SLOTS
    B = busesA + busesB
    TOTAL = travel_s + charge_s

    home = ["A"] * busesA + ["B"] * busesB
    def route_for(b, k):
        if home[b] == "A":
            return routeAB if (k % 2 == 0) else routeBA
        else:
            return routeBA if (k % 2 == 0) else routeAB
    def dep_station(rt): return "A" if rt == routeAB else "B"

    route_abs = {routeAB: [], routeBA: []}
    for rt in (routeAB, routeBA):
        for d in range(D):
            for s in ALLOWED:
                route_abs[rt].append(d * SLOTS + s)
        route_abs[rt].sort()

    m = cp_model.CpModel()
    K = 2 * D
    y, st = {}, {}
    occ = {("A", t): [] for t in range(H)}
    occ.update({("B", t): [] for t in range(H)})

    for b in range(B):
        for k in range(K):
            rt = route_for(b, k)
            lits = []
            for a in route_abs[rt]:
                v = m.NewBoolVar(f"y_{b}_{k}_{a}")
                y[(b, k, a)] = v
                lits.append(v)
                occ[(dep_station(rt), a)].append(v)
            m.Add(sum(lits) == 1)
            st[(b, k)] = m.NewIntVar(0, H-1, f"st_{b}_{k}")
            m.Add(st[(b, k)] == sum(a * y[(b, k, a)] for a in route_abs[rt]))

    for b in range(B):
        for k in range(K-1):
            m.Add(st[(b, k+1)] >= st[(b, k)] + TOTAL)
            m.Add(st[(b, k+1)] - st[(b, k)] <= TOTAL + idle_s)
        for k in range(K-2):
            m.Add(st[(b, k+2)] >= st[(b, k)] + SLOTS)

    for (stn, t), lits in occ.items():
        if lits: m.Add(sum(lits) <= 1)

    if min_gap_slots and min_gap_slots > 1:
        for stn in ("A","B"):
            times = sorted(t for (s,t) in occ if s==stn)
            for t in times:
                for g in range(1, min_gap_slots):
                    t2 = t + g
                    if (stn, t2) in occ:
                        m.AddAtMostOne(occ[(stn,t)] + occ[(stn,t2)])

    # per-day per-route quotas (balance)
    for d in range(D):
        lo, hi = d*SLOTS, (d+1)*SLOTS - 1
        ab_vars, ba_vars = [], []
        for b in range(B):
            for k in range(K):
                rt = route_for(b,k)
                for a in range(lo, hi+1):
                    key = (b,k,a)
                    if key in y:
                        (ab_vars if rt==routeAB else ba_vars).append(y[key])
        m.Add(sum(ab_vars) == B)
        m.Add(sum(ba_vars) == B)

    BIG = 100_000
    obj = []
    for (b,k,a), var in y.items():
        d = a // SLOTS
        s = a % SLOTS
        rt = route_for(b,k)
        val = float(epk.get((rt, d, s), 0.0))
        obj.append(int(val*BIG) * var + a * var)
    m.Maximize(sum(obj))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(limit) if limit and limit>0 else 0
    solver.parameters.log_search_progress = enable_logs
    solver.parameters.num_search_workers = max(1,int(workers))
    if deterministic: solver.parameters.random_seed = seed

    status = solver.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        name = ("INFEASIBLE" if status == cp_model.INFEASIBLE else
                "MODEL_INVALID" if status == cp_model.MODEL_INVALID else "UNKNOWN")
        raise RuntimeError(f"No feasible schedule. Solver status: {name}.")

    sched = {}
    for b in range(B):
        bus = f"Bus-{b+1:02d}"
        sched[bus] = {dy: [] for dy in days}
        for k in range(K):
            a = int(solver.Value(st[(b,k)]))
            d = a // SLOTS
            s = a % SLOTS
            rt = route_for(b,k)
            if 0 <= d < D:
                sched[bus][days[d]].append({
                    "route": rt,
                    "startTime": s2t(s),
                    "midPointTime": s2t((s + travel_s//2) % SLOTS),
                    "endTime": s2t((s + travel_s) % SLOTS),
                    "epk": round(float(epk.get((rt,d,s),0.0)), 2),
                    "serviceKey": None,
                    "isNew": True
                })
        for dy in days:
            sched[bus][dy].sort(key=lambda tr: tr["startTime"])
    return sched

# ─────────────────────────  SERVICE-KEY ANNOTATION (± buffer)  ─────────────────────────
def annotate_service_keys(sched, days, base_slots, base_keys, snap_radius_slots):
    """
    For each trip, if its startTime slot is within ±snap_radius_slots of any
    base slot for that route, attach that base serviceKey and mark as not NEW.
    Otherwise mark as NEW.
    """
    for bus, mp in sched.items():
        for d in days:
            for tr in mp[d]:
                rt = tr["route"]
                s  = t2s(tr["startTime"])
                best_p, best_delta = None, None
                for p in base_slots.get(rt, set()):
                    delta = abs(p - s)
                    if delta <= snap_radius_slots:
                        if best_delta is None or delta < best_delta:
                            best_p, best_delta = p, delta
                if best_p is not None:
                    key = base_keys.get(rt, {}).get(best_p)
                    if key:
                        tr["serviceKey"] = key
                        tr["isNew"] = False
                    else:
                        # matched a base time but no key provided; still treat as not NEW
                        tr["serviceKey"] = None
                        tr["isNew"] = False
                else:
                    tr["serviceKey"] = "NEW"
                    tr["isNew"] = True
    return sched

# ─────────────────────────  METRICS / HTML  ─────────────────────────
def compute_metrics(sched):
    total_buses = len(sched)
    total_epk, total_trips = 0.0, 0
    for bus in sched.values():
        for day in bus.values():
            for trip in day:
                total_epk += trip["epk"]
                total_trips += 1
    avg_epk = round(total_epk / total_trips, 2) if total_trips else 0.0
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
            days=days, ab=ab, ba=ba,
            routeAB_name=routeAB, routeBA_name=routeBA,
            trips=TRIPS_PER_DAY, total_buses=total_buses, avg_epk=avg_epk),
        encoding="utf-8"
    )

# ─────────────────────────  CLI  ─────────────────────────
def ask_int(msg, default):
    v=input(f"{msg} [{default}]: ").strip(); return int(v) if v else default
def ask_path(msg, default):
    v=input(f"{msg} [{default}]: ").strip(); return Path(v) if v else Path(default)
def ask_str(msg, default):
    v=input(f"{msg} [{default}]: ").strip(); return v if v else default
def ask_bool(msg, default=False):
    v=input(f"{msg} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    if not v: return default
    return v in ("y","yes","true","1")


# ─────────────────────────  METRICS: SCHEDULE vs BASE  ─────────────────────────
from collections import Counter, defaultdict
import math

def _slot_ceil_from_timestr(t: str) -> int:
    """Ceil a 'HH:MM' (or 'HH:MM:SS') to the next 30-min slot."""
    t = t.strip()
    parts = t.split(":")
    hh = int(parts[0]); mm = int(parts[1])
    slot = hh * 2 + math.ceil(mm / 30.0)
    # keep in [0, 47]; values like 24:00 (slot 48) are pushed out of range
    return slot if 0 <= slot < SLOTS_PER_DAY else -1

def _load_base_times_ceil_for_two_routes(base_csv: Path, routeAB: str, routeBA: str):
    """
    Returns Counter of *rounded-up* slots per route for the two EPK routes only.
    Accepts 'route,time[,serviceKey]' OR 'origin,destination,time[,serviceKey]'.
    """
    df = pd.read_csv(base_csv, sep=None, engine="python")
    cols = {c.lower().strip(): c for c in df.columns}

    def map_to_epk(rt_txt: str) -> str | None:
        rt_txt = str(rt_txt).strip()
        cands = [routeAB, routeBA,
                 routeAB.replace("-", "|"), routeBA.replace("-", "|"),
                 routeAB.replace("–","|").replace("—","|"),
                 routeBA.replace("–","|").replace("—","|")]
        key = _norm_route_name(rt_txt)
        for cand in cands:
            if _norm_route_name(cand) == key:
                return cand
        return None

    ctr = {routeAB: Counter(), routeBA: Counter()}

    if "route" in cols and "time" in cols:
        for _, r in df.iterrows():
            rt_raw = r[cols["route"]]
            rt = map_to_epk(rt_raw) or (map_to_epk(str(rt_raw).replace("|","-")) if "|" in str(rt_raw) else None)
            if rt not in ctr: 
                continue
            s = _slot_ceil_from_timestr(str(r[cols["time"]]))
            if s == -1 or s in FORBIDDEN: 
                continue
            ctr[rt][s] += 1
    elif all(k in cols for k in ["origin", "destination", "time"]):
        for _, r in df.iterrows():
            o = str(r[cols["origin"]]).strip().upper()
            d = str(r[cols["destination"]]).strip().upper()
            rt_guess = f"{o}|{d}"
            rt = map_to_epk(rt_guess) or map_to_epk(f"{o}-{d}") or map_to_epk(f"{o}{d}")
            if rt not in ctr:
                continue
            s = _slot_ceil_from_timestr(str(r[cols["time"]]))
            if s == -1 or s in FORBIDDEN:
                continue
            ctr[rt][s] += 1
    else:
        raise ValueError("Base CSV must contain (route,time) or (origin,destination,time).")

    return ctr

def _load_schedule_json(schedule_json_path: Path):
    data = json.loads(Path(schedule_json_path).read_text(encoding="utf-8"))
    return data.get("schedule", data)  # support either {"schedule": …} or raw dict

def _collect_sched_slots(schedule_dict):
    """
    Returns:
      counts_by_route: {route: Counter(slot)}
      total_trips: int
    """
    counts = defaultdict(Counter)
    total = 0
    for bus, daymap in schedule_dict.items():
        for trips in daymap.values():
            for tr in trips:
                rt = tr.get("route")
                s  = t2s(tr.get("startTime"))
                counts[rt][s] += 1
                total += 1
    return counts, total

def compare_schedule_to_base(base_csv: Path, schedule_json_path: Path,
                             routeAB: str, routeBA: str, report=True):
    """
    Metrics (denominator = total trips in schedule JSON, across both routes):
      1) exact_up: schedule equals ceil(base_time)       → exact adherence
      2) buffer_30: schedule equals ceil(base_time)±1    → exactly 30-min offset (excludes #1)
      3) new: neither of the above
    Returns {counts:…, pct:…} and prints a short report if report=True.
    """
    sched = _load_schedule_json(schedule_json_path)
    sched_counts, total = _collect_sched_slots(sched)
    base_ceil = _load_base_times_ceil_for_two_routes(base_csv, routeAB, routeBA)

    # work on copies (we'll decrement matches)
    sc = {rt: Counter(sched_counts.get(rt, Counter())) for rt in (routeAB, routeBA)}
    bc = {rt: Counter(base_ceil.get(rt, Counter()))  for rt in (routeAB, routeBA)}

    exact = 0
    # pass 1: exact (after rounding up)
    for rt in (routeAB, routeBA):
        for s, cnt in list(sc[rt].items()):
            m = min(cnt, bc[rt][s])
            if m:
                sc[rt][s] -= m
                bc[rt][s] -= m
                exact += m

    # pass 2: exactly one slot away (±30 min), excluding exact we already matched
    buffer_30 = 0
    for rt in (routeAB, routeBA):
        for s, cnt in list(sc[rt].items()):
            while cnt > 0:
                matched = False
                # prefer whichever neighbor has availability; try -1 then +1
                if bc[rt].get(s-1, 0) > 0:
                    bc[rt][s-1] -= 1
                    matched = True
                elif bc[rt].get(s+1, 0) > 0:
                    bc[rt][s+1] -= 1
                    matched = True

                if matched:
                    buffer_30 += 1
                    sc[rt][s] -= 1
                    cnt -= 1
                else:
                    break  # no more neighbors to match at this s

    new_trips = total - exact - buffer_30
    pct = lambda x: round(100.0 * x / total, 2) if total else 0.0

    result = {
        "counts": {
            "total_trips": total,
            "exact_match_after_roundup": exact,
            "buffer_30min_match": buffer_30,
            "new_trips": new_trips,
        },
        "percentages": {
            "exact_match_after_roundup_pct": pct(exact),
            "buffer_30min_match_pct": pct(buffer_30),
            "new_trips_pct": pct(new_trips),
            "adherence_overall_pct": pct(exact + buffer_30),
        },
        "notes": (
            "Exact matching compares schedule start slots against base times "
            "rounded UP to the next 30-minute slot (e.g., 22:10→22:30, 22:45→23:00). "
            "Buffer matching counts schedule times that are exactly ±30 minutes from "
            "that rounded base time and excludes exact matches."
        )
    }

    if report:
        print("\n📊 Adherence report (schedule JSON vs base services)")
        print(f"   Total trips in schedule: {total}")
        print(f"   1) Exact match after rounding up base times: {exact}  ({pct(exact)}%)")
        print(f"   2) Exactly ±30-min from rounded base time:  {buffer_30}  ({pct(buffer_30)}%)")
        print(f"   3) NEW (no exact/±30 match):               {new_trips}  ({pct(new_trips)}%)")
        print(f"   Overall adherence (1+2):                   {pct(exact + buffer_30)}%")
        print(f"   Note: Base times are rounded UP to the 30-min grid for metric #1.")
    return result


def main():
    print("EV Month Scheduler — base-priority constant template (or dynamic fallback)")

    csv = ask_path("Path to EPK CSV", "cache/epk_h_v_month.csv")
    base_csv_in = ask_str("Base services CSV (blank for none)", "")
    busesA = ask_int("Buses at Station A", 5)
    busesB = ask_int("Buses at Station B", 5)
    travel_h = ask_int("Travel hours (one leg)", 9)
    charge_h = ask_int("Depot charge hours", 2)
    idle_h   = ask_int("Max *extra* idle hours", 4)
    min_gap  = ask_int("Min station gap (slots, 30-min)", 1)
    limit    = ask_int("Solve time limit (s)", 420)
    workers  = ask_int("Solver threads", 8)
    snap_rad = ask_int("Snap radius in slots (±N*30min around base)", 1)
    require_all = ask_bool("Require covering ALL base times?", False)
    jsonf = ask_path("JSON output", DEFAULT_JSON)
    htmlf = ask_path("HTML output", DEFAULT_HTML)

    print("⏳ Loading EPK …")
    epk, days, rAB, rBA = load_epk(csv)

    if base_csv_in:
        print("📚 Loading base services …")
        base_slots, base_keys = load_base_services(Path(base_csv_in), rAB, rBA)
        print("⚙️  Solving constant template with base priority …")
        sched = solve_constant_with_base(
            epk=epk, days=days, routeAB=rAB, routeBA=rBA,
            busesA=busesA, busesB=busesB,
            travel_s=int(round(travel_h*2)), charge_s=int(round(charge_h*2)),
            idle_s=int(round(idle_h*2)),
            base_slots=base_slots,
            snap_radius_slots=int(snap_rad),
            require_all_base=require_all,
            min_gap_slots=int(min_gap),
            limit=limit, workers=workers,
            deterministic=True, enable_logs=False
        )
        # Annotate keys or NEW using ±snap radius
        sched = annotate_service_keys(
            sched, days, base_slots, base_keys, snap_radius_slots=int(snap_rad)
        )
    else:
        print("⚙️  Base CSV not provided → dynamic month solve (fallback).")
        sched = solve_abs_month(
            epk=epk, days=days, routeAB=rAB, routeBA=rBA,
            busesA=busesA, busesB=busesB,
            travel_s=int(round(travel_h*2)), charge_s=int(round(charge_h*2)),
            idle_s=int(round(idle_h*2)),
            limit=limit, workers=workers, min_gap_slots=int(min_gap),
            deterministic=True, enable_logs=False
        )

    print("💾 Writing files …")
    jsonf.write_text(json.dumps({"schedule": sched}, indent=2))
    html_out(sched, htmlf, days, rAB, rBA)
    print(f"✅ JSON → {jsonf}\n✅ HTML → {htmlf}")
    if base_csv_in:
        print("🔎 Computing adherence metrics against base CSV …")
        compare_schedule_to_base(Path(base_csv_in), jsonf, rAB, rBA, report=True)

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)
