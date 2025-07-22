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
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #ccc;padding:4px;text-align:center}
th{background:#333;color:#fff}tr:nth-child(odd){background:#f9f9f9}
.bus{border-radius:6px;padding:4px;margin:2px;display:inline-block;width:90px;color:#fff}
.ab{background:#e76f51}.ba{background:#2a9d8f}
</style></head><body>
<h2>EV Schedule ({{days|length}} days, {{trips}} trips/bus/day)</h2>
<table><thead><tr><th>Day</th><th>{{ab}}</th><th>{{ba}}</th></tr></thead><tbody>
{% for d in range(days|length) %}
<tr><td><strong>{{days[d]}}</strong></td>
<td>{% for bus,tr in ab[d] %}<div class="bus ab">{{bus}}<br>{{tr.startTime}}<br>{{tr.epk}}</div>{% endfor %}</td>
<td>{% for bus,tr in ba[d] %}<div class="bus ba">{{bus}}<br>{{tr.startTime}}<br>{{tr.epk}}</div>{% endfor %}</td>
</tr>{% endfor %}
</tbody></table></body></html>"""

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
    FIRST_OK = [s for s in ALLOWED if s+TOTAL+idle_s <= SLOTS_PER_DAY-1]

    B = busesA + busesB
    D = len(days)
    home = ["A"]*busesA + ["B"]*busesB
    seqA = [routeAB, routeBA]; seqB=[routeBA, routeAB]

    m = cp_model.CpModel()
    y,start={},{}
    for b in range(B):
        for d in range(D):
            for t in range(TRIPS_PER_DAY):
                rt = seqA[t] if home[b]=="A" else seqB[t]
                pool = FIRST_OK if t==0 else ALLOWED
                for s in pool:
                    if (rt,d,s) in epk:
                        y[(b,d,t,s)] = m.NewBoolVar(f"y_{b}_{d}_{t}_{s}")
                m.Add(sum(y[(b,d,t,s)] for s in pool if (b,d,t,s) in y)==1)
                start[(b,d,t)] = m.NewIntVar(0,SLOTS_PER_DAY-1,f"st_{b}_{d}_{t}")
                m.Add(start[(b,d,t)] ==
                      sum(s*y[(b,d,t,s)] for s in pool if (b,d,t,s) in y))

            # precedence + idle
            m.Add(start[(b,d,1)] >= start[(b,d,0)] + TOTAL)
            m.Add(start[(b,d,1)] - start[(b,d,0)] <= TOTAL + idle_s)

    # spacing
    for d in range(D):
        for station,rt in (("A",routeAB),("B",routeBA)):
            group=[b for b in range(B) if home[b]==station]
            idxA,idxB=(0,1) if station=="A" else (1,0)
            for s in ALLOWED:
                now=[y[(b,d,idxA,s)] for b in group if (b,d,idxA,s) in y]+\
                    [y[(b,d,idxB,s)] for b in group if (b,d,idxB,s) in y]
                safe(m,now,"__le__",1)
                for δ in range(1,SPACING):
                    s2=s+δ
                    if s2 not in ALLOWED:continue
                    gap=now+[y[(b,d,idxA,s2)] for b in group if (b,d,idxA,s2) in y]+\
                              [y[(b,d,idxB,s2)] for b in group if (b,d,idxB,s2) in y]
                    safe(m,gap,"__le__",1)

    # objective: sum(epk * y)
    obj=[]
    for (b,d,t,s),var in y.items():
        rt = seqA[t] if home[b]=="A" else seqB[t]
        obj.append(int(epk.get((rt,d,s),0)*100)*var)
    m.Maximize(sum(obj))

    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=limit
    solver.parameters.num_search_workers=8
    if solver.Solve(m) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible schedule – try larger max idle.")

    # build schedule
    sched={}
    for b in range(B):
        bus=f"Bus-{b+1:02d}"; sched[bus]={}
        for d,dy in enumerate(days):
            trips=[]
            for t in range(TRIPS_PER_DAY):
                rt=seqA[t] if home[b]=="A" else seqB[t]
                s=int(solver.Value(start[(b,d,t)]))
                trips.append({"route":rt,
                              "startTime":s2t(s),
                              "midPointTime":s2t((s+travel_s//2)%SLOTS_PER_DAY),
                              "endTime":s2t((s+travel_s)%SLOTS_PER_DAY),
                              "epk":round(epk.get((rt,d,s),0),2)})
            trips.sort(key=lambda x:x["startTime"])
            sched[bus][dy]=trips
    return sched

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
    html.write_text(
    Template(HTML).render(
        days=days,
        ab=ab,            # list of trips A→B
        ba=ba,            # list of trips B→A
        routeAB_name=routeAB,
        routeBA_name=routeBA,
        trips=TRIPS_PER_DAY
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
    csv = ask_path("Path to EPK CSV", "epk_data.csv")
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
