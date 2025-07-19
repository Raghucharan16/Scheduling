#!/usr/bin/env python3
"""
peak_solver.py  –  Optimise Fri–Sun schedule given Thu & Mon anchoring.

Assumptions
-----------
* Your steady JSON (from the previous solver) has numeric day labels:
  "1" … "7"   (1=Mon, 4=Thu, 5=Fri, 7=Sun).
* Exactly two symmetric routes appear in the EPK CSV, e.g. 'H-V' and 'V-H'.
* Travel‑time + charge‑time ≤ 12 h so at most two departures per bus per day.
"""

from pathlib import Path
import json, sys
import pandas as pd
from ortools.sat.python import cp_model
from jinja2 import Template

# -----------------  CONFIGURABLE CONSTANTS  -----------------
FORBIDDEN  = {1,2,3,4,5,6}           # 00:30‑03:00 start slots
SLOTS_PER  = 48                      # 30‑min granularity
ALLOWED    = [s for s in range(SLOTS_PER) if s not in FORBIDDEN]

PEAK_DAYS  = ["5","6","7"]           # Fri, Sat, Sun
NIGHT_SLOTS= {s for s in ALLOWED if 38 <= s <= 47}      # 19:00‑23:30
EPK_THRESH = 90.0                    # high‑EPK threshold
UTIL_FLOOR = 0.95                    # ≥95 % of max trips

# HTML stub (kept minimal)
HTML = """<!doctype html><html><head><meta charset="utf-8">
<style>body{font-family:Arial;margin:0;padding:1rem;font-size:14px}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #ccc;padding:4px;text-align:center}
th{background:#333;color:#fff}tr:nth-child(odd){background:#f9f9f9}
.bus{border-radius:6px;padding:4px;margin:2px;display:inline-block;width:90px;color:#fff}
.ab{background:#e76f51}.ba{background:#2a9d8f}</style></head><body>
<h2>Peak‑weekend schedule (Fri–Sun)</h2>
<table><thead><tr><th>Day</th><th>{{ab}}</th><th>{{ba}}</th></tr></thead><tbody>
{% for i,d in enumerate(days) %}<tr><td><b>{{d}}</b></td>
<td>{% for bus,tr in ab[i] %}<div class="bus ab">{{bus}}<br>{{tr.start}}</div>{% endfor %}</td>
<td>{% for bus,tr in ba[i] %}<div class="bus ba">{{bus}}<br>{{tr.start}}</div>{% endfor %}</td>
</tr>{% endfor %}</tbody></table></body></html>"""

# -----------------  HELPERS  -----------------
def slot2str(s): return f"{s//2:02d}:{(s%2)*30:02d}:00"

def load_epk(csv:Path):
    df=pd.read_csv(csv)
    day_cols=[c for c in df.columns if c.isdigit()]
    day_cols.sort(key=int)
    df["slot_idx"]=(pd.to_timedelta(df["slot"]).dt.components.hours*2 +
                    pd.to_timedelta(df["slot"]).dt.components.minutes//30).astype(int)
    routes=sorted(df["route"].unique())
    if len(routes)!=2: raise ValueError("Need exactly two routes in CSV.")
    rAB,rBA=routes
    epk={}
    for _,r in df.iterrows():
        s=int(r["slot_idx"])
        if s in FORBIDDEN: continue
        for d,c in enumerate(day_cols):
            epk[(r["route"],d,c,s)] = float(r[c])   # key by (route, d‑idx, label, slot)
    return epk,day_cols,rAB,rBA

def initial_and_target(json_path:Path, rAB, rBA):
    data=json.load(open(json_path))
    init=[]; target=[]
    for b in range(len(data["schedule"])):
        bus=f"Bus-{b+1:02d}"
        # Thursday last trip
        last_tr=data["schedule"][bus]["4"][-1]["route"]
        init.append("A" if last_tr==rBA else "B")
        # Monday first trip
        first_tr=data["schedule"][bus]["1"][0]["route"]
        target.append("A" if first_tr==rAB else "B")
    return init,target

# -----------------  SOLVER  -----------------
def solve_peak(epk, epk_days, rAB, rBA,
               travel_s, charge_s, idle_s,
               init_st, target_st,
               limit=300):
    TOTAL=travel_s+charge_s
    FIRST_ALLOWED=[s for s in ALLOWED if s+TOTAL+idle_s<=SLOTS_PER-1]

    B=len(init_st); D=len(PEAK_DAYS)
    m=cp_model.CpModel()
    y,start={},{}                      # decision vars
    trips_per_bus=[[] for _ in range(B)]
    high_vars=[]; epk_terms=[]

    for b in range(B):
        for di,day in enumerate(PEAK_DAYS):
            for t in range(2):         # at most 2 trips
                rt0 = rAB if init_st[b]=="A" else rBA
                rt  = (rt0 if t==0 else (rBA if rt0==rAB else rAB))
                pool=FIRST_ALLOWED if t==0 else ALLOWED
                for s in pool:
                    if (rt, epk_days.index(day), day, s) in epk:
                        y[(b,di,t,s)]=m.NewBoolVar(f"y_{b}_{di}_{t}_{s}")
                # mandatory first trip, optional second
                required=1 if t==0 else 0
                m.Add(sum(y[(b,di,t,s)] for s in pool if (b,di,t,s) in y)>=required)
                m.Add(sum(y[(b,di,t,s)] for s in pool if (b,di,t,s) in y)<=1)
                # start time var
                st=m.NewIntVar(0,SLOTS_PER-1,f"st_{b}_{di}_{t}")
                m.Add(st==sum(s*y[(b,di,t,s)] for s in pool if (b,di,t,s) in y))
                start[(b,di,t)]=st
                if t==1:
                    m.Add(st >= start[(b,di,0)] + TOTAL)
                    # idle upper bound only for Saturday (di==1) – else soft
                    if day not in ("5","7"):
                        m.Add(st - start[(b,di,0)] <= TOTAL+idle_s)

                # bookkeeping
                trips_per_bus[b].append((rt,di,t,st))
                epk_val = epk.get((rt, epk_days.index(day), day, 0),0)  # any slot key to fetch route existence
            # end t loop
        # end day loop
    # ----- 50 % night‑high rule -----
    for di,day in enumerate(PEAK_DAYS):
        if day=="5":
            demand=rAB
        elif day=="7":
            demand=rBA
        else:
            continue
        total=[]; high=[]
        for (b,ddi,t,s),var in y.items():
            if ddi!=di: continue
            rt = rAB if (t==0) ^ (init_st[b]=="B") else rBA
            total.append(var)
            if rt==demand and s in NIGHT_SLOTS and \
               epk.get((rt, epk_days.index(day), day, s),0) >= EPK_THRESH:
                high.append(var)
        if total:
            m.Add( 2*sum(high) >= sum(total) )

    # ----- utilisation floor (95 %) -----
    max_trips = B * 2 * D
    m.Add(sum(y.values()) >= int(UTIL_FLOOR * max_trips))

    # ----- parity constraint to hit Monday start station -----
    for b in range(B):
        tot = m.NewIntVar(0, 2*D, f"tot_{b}")
        m.Add(tot == sum(y[(b,di,t,s)] for (bb,di,t,s) in y if bb==b))
        par = m.NewIntVar(0,1,f"par_{b}")
        m.AddModuloEquality(par, tot, 2)
        need = 0 if init_st[b]==target_st[b] else 1
        m.Add(par == need)

    # ----- objective: maximise #high slots first, then EPK -----
    for (b,di,t,s),var in y.items():
        rt = rAB if (t==0) ^ (init_st[b]=="B") else rBA
        epk_val = epk.get((rt, epk_days.index(PEAK_DAYS[di]), PEAK_DAYS[di], s),0)
        epk_terms.append(int(epk_val*100)*var)
        if ((PEAK_DAYS[di]=="5" and rt==rAB) or (PEAK_DAYS[di]=="7" and rt==rBA)) and \
            s in NIGHT_SLOTS and epk_val>=EPK_THRESH:
            high_vars.append(var)

    m.Maximize( 1_000_000*sum(high_vars) + sum(epk_terms) )

    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=limit
    solver.parameters.num_search_workers=8
    if solver.Solve(m) not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        raise RuntimeError("No feasible peak schedule (relax thresholds or idle).")

    # ----- build result -----
    out={}
    for b in range(B):
        bus=f"Bus-{b+1:02d}"; out[bus]={}
        for di,day in enumerate(PEAK_DAYS):
            trips=[]
            for t in range(2):
                st=int(solver.Value(start[(b,di,t)]))
                if st<0 or st>=SLOTS_PER: continue   # trip not selected
                rt=rAB if (t==0) ^ (init_st[b]=="B") else rBA
                trips.append({"route":rt,"start":slot2str(st)})
            trips.sort(key=lambda x:x["start"])
            out[bus][day]=trips
    return out

# -----------------  HTML writer -----------------
def html_report(sched, html_path, days, rAB, rBA):
    ab=[[] for _ in days]; ba=[[] for _ in days]
    for bus,mp in sched.items():
        for i,d in enumerate(days):
            for tr in mp[d]:
                (ab if tr["route"]==rAB else ba)[i].append((bus,tr))
    for i in range(len(days)):
        ab[i].sort(key=lambda bt:bt[1]["start"])
        ba[i].sort(key=lambda bt:bt[1]["start"])
    html_path.write_text(
        Template(HTML).render(days=days,ab=ab,ba=ba,
                              ab_label=rAB,ba_label=rBA), encoding="utf-8")

# -----------------  CLI  -----------------
def ask_path(msg,defv): v=input(f"{msg} [{defv}]: ").strip(); return Path(v) if v else Path(defv)
def ask_int (msg,defv): v=input(f"{msg} [{defv}]: ").strip(); return int(v) if v else defv

def main():
    csv = ask_path("EPK CSV", "epk_data.csv")
    j_prev = ask_path("Previous JSON schedule", "schedule.json")
    out_json = ask_path("Peak JSON output", "peak_schedule.json")
    out_html = ask_path("Peak HTML output", "peak_schedule.html")
    travel_h = ask_int("Travel hours (one leg)", 9)
    charge_h = ask_int("Depot charge hours", 2)
    idle_h   = ask_int("Max *extra* idle hours", 6)

    print("⏳ Loading data …")
    epk, epk_days, rAB, rBA = load_epk(csv)
    init,target = initial_and_target(j_prev, rAB, rBA)

    print("⚙️  Solving peak‑weekend …")
    sched = solve_peak(epk, epk_days, rAB, rBA,
                       travel_h*2, charge_h*2, idle_h*2,
                       init, target)

    print("💾 Writing outputs …")
    out_json.write_text(json.dumps({"schedule":sched},indent=2))
    html_report(sched, out_html, PEAK_DAYS, rAB, rBA)
    print(f"✅ Peak JSON → {out_json}\n✅ Peak HTML → {out_html}")

if __name__=="__main__":
    try: main()
    except Exception as e:
        print("❌",e); sys.exit(1)
