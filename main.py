#!/usr/bin/env python3
"""
ev_weekly_scheduler.py  –  interactive version
------------------------------------------------------------
• Two trips per bus per day (Mon‑Thu).
• Forbidden departure slots: 00:30–03:00.
• Spacing ≥ 2 slots (1 h) between departures from the same station.
• Gap between a bus’s two trips:  travel+charge  ≤  gap  ≤  travel+charge+max_idle.
• Objective: maximise total EPK (earning per km).
------------------------------------------------------------
Install prerequisites:
    pip install ortools pandas jinja2
"""

from pathlib import Path
from datetime import timedelta
import json, sys
import pandas as pd
from ortools.sat.python import cp_model
from jinja2 import Template

# ─────────────────────────────────────────────────────────────
#  Helper & template
# ─────────────────────────────────────────────────────────────
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday"]
FORBIDDEN = {1, 2, 3, 4, 5, 6}                 # 00:30‑03:00
SLOTS_PER_DAY = 48                             # 30‑min granularity
ALLOWED = [s for s in range(SLOTS_PER_DAY) if s not in FORBIDDEN]

def slot_to_str(slot: int) -> str:
    return f"{slot//2:02d}:{(slot%2)*30:02d}:00"

HTML_TMPL = """<!doctype html><html><head>
<meta charset="utf-8">
<style>
body{font-family:Arial,Helvetica,sans-serif;margin:0;padding:1rem;font-size:14px}
table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:6px;text-align:center}
th{background:#333;color:#fff}tbody tr:nth-child(odd){background:#f9f9f9}
.bus{border-radius:6px;padding:4px;margin:2px;display:inline-block;width:90px;color:#fff}
.routeAB{background:#e76f51}.routeBA{background:#2a9d8f}
</style></head><body>
<h2>EV Bus Schedule (Mon–Thu, 2 trips / bus / day)</h2>
<table>
<thead><tr><th>Day</th><th>A → B</th><th>B → A</th></tr></thead><tbody>
{% for idx in range(days|length) %}
{% set day = days[idx] %}
<tr><td><strong>{{day}}</strong></td>
<td>{% for bus,trip in ab[idx] %}
<div class="bus routeAB">{{bus}}<br>{{trip.startTime}}<br>EPK {{trip.epk}}</div>{% endfor %}</td>
<td>{% for bus,trip in ba[idx] %}
<div class="bus routeBA">{{bus}}<br>{{trip.startTime}}<br>EPK {{trip.epk}}</div>{% endfor %}</td>
</tr>{% endfor %}
</tbody></table></body></html>"""

# ─────────────────────────────────────────────────────────────
#  Data‑loading
# ─────────────────────────────────────────────────────────────
def load_epk(csv_path: Path):
    """Return dict key=(route, day_idx, slot) → epk (float)."""
    df = pd.read_csv(csv_path)
    # slot ‑> integer index 0‑47
    df["slot_idx"] = (
        pd.to_timedelta(df["slot"]).dt.components.hours * 2 +
        pd.to_timedelta(df["slot"]).dt.components.minutes // 30
    ).astype(int)

    epk = {}
    for _, r in df.iterrows():
        s = int(r["slot_idx"])
        if s in FORBIDDEN:
            continue
        for d, col in enumerate(["monday", "tuesday", "wednesday", "thursday"]):
            epk[(r["route"], d, s)] = float(r[col])
    return epk

# ─────────────────────────────────────────────────────────────
#  Model helper
# ─────────────────────────────────────────────────────────────
def safe_add(model, vars_, sense, rhs):
    """Add constraint only if the vars_ list is non‑empty."""
    if vars_:
        expr = sum(vars_)
        if sense == "==":
            model.Add(expr == rhs)
        elif sense == "<=":
            model.Add(expr <= rhs)
        elif sense == ">=":
            model.Add(expr >= rhs)

# ─────────────────────────────────────────────────────────────
#  Core solver
# ─────────────────────────────────────────────────────────────
def build_and_solve(epk_map,
                    buses_a: int,
                    buses_b: int,
                    travel_slots: int,
                    charge_slots: int,
                    max_extra_idle: int,
                    min_station_gap: int,
                    solver_time_limit: int = 300):
    """
    Build and solve CP‑SAT model; return schedule dict + objective value.
    """
    TOTAL_SLOTS = travel_slots + charge_slots          # min gap between trips
    FIRST_OK = [s for s in ALLOWED
                if s + TOTAL_SLOTS + max_extra_idle <= SLOTS_PER_DAY - 1]

    B = buses_a + buses_b
    D = len(DAYS)
    home_station = ["A"] * buses_a + ["B"] * buses_b
    ROUTE_SEQ_A = ["A-B", "B-A"]     # bus sleeping at A: trip0 A‑B, trip1 B‑A
    ROUTE_SEQ_B = ["B-A", "A-B"]

    model = cp_model.CpModel()

    # Decision vars y[b,d,t,s] (t=0 first trip, t=1 second trip)
    y = {}
    start = {}

    for b in range(B):
        for d in range(D):
            for t in range(2):
                route = ROUTE_SEQ_A[t] if home_station[b] == "A" else ROUTE_SEQ_B[t]
                slots_pool = FIRST_OK if t == 0 else ALLOWED
                for s in slots_pool:
                    if (route, d, s) in epk_map:
                        y[b, d, t, s] = model.NewBoolVar(f"y_b{b}_d{d}_t{t}_s{s}")

                # exactly one slot per (b,d,t)
                cand = [y[b, d, t, s] for s in slots_pool if (b, d, t, s) in y]
                if not cand:
                    raise ValueError(f"No feasible slots for Bus {b} {DAYS[d]} trip {t}")
                model.Add(sum(cand) == 1)

                # helper int var = chosen start slot value
                start[b, d, t] = model.NewIntVar(0, SLOTS_PER_DAY - 1,
                                                 f"start_b{b}_d{d}_t{t}")
                coeffs, vars_ = zip(*[(s, y[b, d, t, s]) for s in slots_pool if (b, d, t, s) in y])
                model.Add(start[b, d, t] == sum(c * v for c, v in zip(coeffs, vars_)))

            # precedence + idle gap constraints
            model.Add(start[b, d, 1] >= start[b, d, 0] + TOTAL_SLOTS)
            model.Add(start[b, d, 1] - start[b, d, 0] <= TOTAL_SLOTS + max_extra_idle)

    # station‑level spacing ≥ min_station_gap
    for d in range(D):
        for station, route_tag in (("A", "A-B"), ("B", "B-A")):
            buses_here = [b for b in range(B) if home_station[b] == station]
            trip_idx = 0 if station == "A" else 1
            opp_idx  = 1 - trip_idx

            for s in ALLOWED:
                vars_now = [y[b, d, trip_idx, s] for b in buses_here
                            if (b, d, trip_idx, s) in y] + \
                           [y[b, d, opp_idx, s]   for b in buses_here
                            if (b, d, opp_idx, s) in y]
                safe_add(model, vars_now, "<=", 1)

                # enforce gap to neighbouring slots
                for delta in range(1, min_station_gap):
                    s2 = s + delta
                    if s2 not in ALLOWED:
                        continue
                    vars_gap = vars_now + \
                               [y[b, d, trip_idx, s2] for b in buses_here
                                if (b, d, trip_idx, s2) in y] + \
                               [y[b, d, opp_idx, s2]  for b in buses_here
                                if (b, d, opp_idx, s2) in y]
                    safe_add(model, vars_gap, "<=", 1)

    # objective
    obj_terms = []
    for (route, d, s), val in epk_map.items():
        if route == "A-B":
            trip_idx = 0
            elig = [b for b in range(B)
                    if (home_station[b] == "A" and trip_idx == 0) or
                       (home_station[b] == "B" and trip_idx == 1)]
        else:  # B‑A
            trip_idx = 0
            elig = [b for b in range(B)
                    if (home_station[b] == "B" and trip_idx == 0) or
                       (home_station[b] == "A" and trip_idx == 1)]
        for b in elig:
            if (b, d, trip_idx, s) in y:
                obj_terms.append(y[b, d, trip_idx, s] * int(val * 100))
        # second‑trip counterpart (opposite route)
        other_route = "B-A" if route == "A-B" else "A-B"
        other_idx = 1
        for b in range(B):
            if (home_station[b] == ("A" if other_route == "A-B" else "B")):
                if (b, d, other_idx, s) in y and (other_route, d, s) in epk_map:
                    obj_terms.append(y[b, d, other_idx, s] *
                                     int(epk_map[(other_route, d, s)] * 100))

    model.Maximize(sum(obj_terms))

    # solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = solver_time_limit
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible schedule found. Check inputs.")

    # build schedule dict
    sched = {}
    for b in range(B):
        bus_id = f"Bus-{b+1:02d}"
        sched[bus_id] = {}
        for d, day in enumerate(DAYS):
            trips = []
            for t in range(2):
                route = ROUTE_SEQ_A[t] if home_station[b] == "A" else ROUTE_SEQ_B[t]
                s = solver.Value(start[b, d, t])
                trips.append({
                    "route": route,
                    "startTime": slot_to_str(s),
                    "midPointTime": slot_to_str((s + travel_slots // 2) % SLOTS_PER_DAY),
                    "endTime": slot_to_str((s + travel_slots) % SLOTS_PER_DAY),
                    "epk": round(epk_map[(route, d, s)], 2)
                })
            trips.sort(key=lambda x: x["startTime"])
            sched[bus_id][day] = trips

    return sched, solver.ObjectiveValue() / 100

# ─────────────────────────────────────────────────────────────
#  HTML writer
# ─────────────────────────────────────────────────────────────
def write_html(schedule: dict, html_path: Path):
    ab = [[] for _ in DAYS]
    ba = [[] for _ in DAYS]
    for bus, day_map in schedule.items():
        for idx, day in enumerate(DAYS):
            for trip in day_map.get(day, []):
                target = ab if trip["route"] == "A-B" else ba
                target[idx].append((bus, trip))
    for idx in range(len(DAYS)):
        ab[idx].sort(key=lambda bt: bt[1]["startTime"])
        ba[idx].sort(key=lambda bt: bt[1]["startTime"])

    html = Template(HTML_TMPL).render(days=DAYS, ab=ab, ba=ba)
    html_path.write_text(html, encoding="utf-8")

# ─────────────────────────────────────────────────────────────
#  Main (interactive)
# ─────────────────────────────────────────────────────────────
def main():
    print("EV Bus Weekly Scheduler (Mon–Thu)")
    print("----------------------------------")

    # Prompt helpers
    def ask_int(prompt, default):
        val = input(f"{prompt} [{default}]: ").strip()
        return int(val) if val else default

    def ask_str(prompt, default):
        val = input(f"{prompt} [{default}]: ").strip()
        return Path(val) if val else Path(default)

    csv_path = ask_str("Path to epk_data.csv", "cache/epk.csv")
    buses_a  = ask_int("Number of buses at Station A", 5)
    buses_b  = ask_int("Number of buses at Station B", 5)
    travel_h = ask_int("Travel time (hours) X→Y or Y→X", 9)
    charge_h = ask_int("Depot charge time (hours)", 2)
    max_idle_h = ask_int("Max *extra* idle allowed (hours)", 2)
    json_out = ask_str("JSON output file", "schedule.json")
    html_out = ask_str("HTML output file", "schedule.html")

    travel_slots = travel_h * 2           # each slot = 30 min
    charge_slots = charge_h * 2
    max_extra_idle = max_idle_h * 2

    print("\n⏳ Loading EPK data…")
    epk = load_epk(csv_path)

    print("⚙️  Building & solving model…")
    sched, obj = build_and_solve(
        epk_map=epk,
        buses_a=buses_a,
        buses_b=buses_b,
        travel_slots=travel_slots,
        charge_slots=charge_slots,
        max_extra_idle=max_extra_idle,
        min_station_gap=1,
        solver_time_limit=300
    )

    print("💾 Writing outputs…")
    json_out.write_text(json.dumps({"schedule": sched,
                                    "total_epk": obj}, indent=2))
    write_html(sched, html_out)

    print(f"\n✅ Done!\n   JSON → {json_out}\n   HTML → {html_out}")
    print(f"📈 Objective (total EPK) = {obj:.2f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ {e}")
        sys.exit(1)
