#!/usr/bin/env python3
"""
ev_month_scheduler.py - Schedule an electric‑bus fleet with EPK heatmap
"""

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
<table><thead><tr><th>Day</th><th>{{ab}}</th><th>{{ba}}</th></tr></thead><tbody>
{% for d in range(days|length) %}
<tr><td><strong>{{days[d]}}</strong></td>
<td>{% for bus,tr in ab[d] %}<div class="bus ab">{{bus}}<br>{{tr.startTime}}<br>{{tr.epk}}</div>{% endfor %}</td>
<td>{% for bus,tr in ba[d] %}<div class="bus ba">{{bus}}<br>{{tr.startTime}}<br>{{tr.epk}}</div>{% endfor %}</td>
</tr>{% endfor %}
</tbody></table></body></html>"""

from pathlib import Path
import json, sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ortools.sat.python import cp_model
from jinja2 import Template

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

# ─────────────────────────  HEATMAP GENERATOR  ─────────────────────────
def generate_heatmap(epk_matrix, output_dir):
    """Generate and save heatmap visualization of EPK results"""
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # Create heatmap with colorbar
    im = ax.imshow(epk_matrix, cmap='viridis', origin='upper')
    plt.colorbar(im, ax=ax, label='Average EPK')
    
    # Set labels and title
    ax.set_xlabel('Buses at Station B')
    ax.set_ylabel('Buses at Station A')
    ax.set_title('Average EPK by Bus Allocation')
    
    # Set ticks
    ax.set_xticks(np.arange(0, 6, 1))
    ax.set_yticks(np.arange(0, 6, 1))
    ax.set_xticklabels(range(6))
    ax.set_yticklabels(range(6))
    
    # Add text annotations
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{epk_matrix[i][j]:.1f}",
                    ha="center", va="center", color="w",
                    fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save as image
    heatmap_path = output_dir / 'epk_heatmap.png'
    plt.savefig(heatmap_path)
    plt.close()
    
    return heatmap_path

def heatmap_html(epk_matrix, heatmap_path, output_path):
    """Generate HTML report for heatmap results"""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EPK Heatmap Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        .heatmap-img {{ max-width: 100%; border: 1px solid #ddd; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        .matrix-table {{ margin: 20px auto; border-collapse: collapse; }}
        .matrix-table th, .matrix-table td {{ 
            padding: 10px; 
            text-align: center; 
            border: 1px solid #ddd;
            min-width: 60px;
        }}
        .matrix-table th {{ background-color: #f2f2f2; }}
        .footer {{ margin-top: 20px; font-size: 0.9em; color: #777; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Average EPK by Bus Allocation</h1>
        
        <div class="heatmap">
            <img src="{heatmap_path.name}" alt="EPK Heatmap" class="heatmap-img">
        </div>
        
        <h2>Data Matrix</h2>
        <table class="matrix-table">
            <tr>
                <th>A\B</th>
                <th>0</th>
                <th>1</th>
                <th>2</th>
                <th>3</th>
                <th>4</th>
                <th>5</th>
            </tr>
            {"".join(
                f'<tr><th>{i}</th>' + 
                "".join(f'<td>{epk_matrix[i][j]:.1f}</td>' for j in range(6)) + 
                '</tr>' 
                for i in range(6)
            )}
        </table>
        
        <div class="footer">
            <p>Generated on {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
    </div>
</body>
</html>
    """
    output_path.write_text(html_content, encoding="utf-8")

# ─────────────────────────  MODIFIED MAIN FUNCTION  ─────────────────────────
def main():
    print("Electric‑Bus Month Scheduler - EPK Heatmap Generator")
    
    # Get user inputs
    csv = ask_path("Path to EPK CSV", "cache/epk_h_v_month.csv")
    travel_h = ask_int("Travel hours (one leg)", 9)
    charge_h = ask_int("Depot charge hours", 2)
    idle_h   = ask_int("Max *extra* idle hours", 4)
    
    print("⏳ Loading CSV …")
    epk, days, rAB, rBA = load_epk(csv)
    
    # Create 6x6 matrix to store results
    epk_matrix = [[0.0 for _ in range(6)] for _ in range(6)]
    
    print("⚙️  Solving for all bus combinations...")
    for busesA in range(6):
        for busesB in range(6):
            print(f"  A={busesA}, B={busesB}... ", end="", flush=True)
            try:
                # Get schedule for this configuration
                sched = solve(epk, days, rAB, rBA,
                              busesA, busesB,
                              travel_h*2, charge_h*2, idle_h*2)
                
                # Calculate average EPK
                _, avg_epk = compute_metrics(sched)
                epk_matrix[busesA][busesB] = avg_epk
                print(f"✅ EPK={avg_epk:.1f}")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                epk_matrix[busesA][busesB] = 0.0  # Mark as 0 on error
    
    # Generate and save heatmap
    output_dir = Path.cwd()
    print("\n📊 Generating heatmap visualization...")
    heatmap_path = generate_heatmap(epk_matrix, output_dir)
    
    # Generate HTML report
    html_path = output_dir / "epk_heatmap.html"
    heatmap_html(epk_matrix, heatmap_path, html_path)
    
    print(f"\n✅ Heatmap image → {heatmap_path}")
    print(f"✅ Heatmap report → {html_path}")

if __name__=="__main__":
    try: 
        main()
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)