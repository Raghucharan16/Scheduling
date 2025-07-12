#!/usr/bin/env python3
"""
debug_constraints.py

Loads your EPK CSV and then adds constraints in numbered groups.
For each group 1..8, it builds a model with *only* groups 1..g
and reports whether it remains feasible.
"""

import csv
from ortools.sat.python import cp_model

# ——— USER PARAMETERS ———
CSV_PATH    = "cache/epk_v-h-week-daywise.csv"
BUSES_A     = 5
BUSES_B     = 5
TRAVEL_H    = 9.0
CHARGE_H    = 2.0
MAX_IDLE_H  = 2.0
THRESHOLD   = 95.0
MIN_TOTAL   = 130
FORBIDDEN   = {30,60,90,120,150,180}
HUMAN_DAYS  = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]

# load once
def load_trips():
    trips = []
    idx = 0
    with open(CSV_PATH) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            h,m,_ = row["slot"].split(":")
            start = int(h)*60 + int(m)
            if start in FORBIDDEN:
                continue
            origin = row["route"][0]
            for d, day in enumerate(HUMAN_DAYS):
                trips.append({
                    "id":     idx,
                    "day":    d,
                    "start":  start,
                    "gstart": d*1440 + start,
                    "origin": origin,
                    "route":  row["route"],
                    "epk":    float(row[day])
                })
                idx += 1
    return trips

TRIPS = load_trips()


def test_feasible(groups_to_include):
    """ Build a CP-SAT model with only constraint groups 1..g included. """
    B = BUSES_A + BUSES_B
    cycle = int((TRAVEL_H + CHARGE_H)*60)
    max_idle = int(MAX_IDLE_H*60)
    model = cp_model.CpModel()

    # Decision vars
    x = {}
    for b in range(B):
        for tr in TRIPS:
            x[b,tr["id"]] = model.NewBoolVar(f"x[{b},{tr['id']}]")

    # sum_trips[b,d] and eq1[b,d]
    sum_trips = {}
    eq1       = {}
    for b in range(B):
        for d in range(7):
            sum_trips[b,d] = model.NewIntVar(0,2,f"sum[{b},{d}]")
            eq1[b,d]       = model.NewBoolVar(f"eq1[{b},{d}]")
            if 5 in groups_to_include:  # group 5 enforces eq1 logic
                model.Add(sum_trips[b,d] == 1).OnlyEnforceIf(eq1[b,d])
                model.Add(sum_trips[b,d] != 1).OnlyEnforceIf(eq1[b,d].Not())

    # 1) each trip ≤1 bus
    if 1 in groups_to_include:
        for tr in TRIPS:
            model.Add(sum(x[b,tr["id"]] for b in range(B)) <= 1)

    # 2) no two depart same station/time
    if 2 in groups_to_include:
        for d in range(7):
            starts = set(t["start"] for t in TRIPS if t["day"]==d)
            for s in starts:
                for orig in ("A","B"):
                    ids = [t["id"] for t in TRIPS
                           if t["day"]==d and t["start"]==s and t["origin"]==orig]
                    if ids:
                        model.Add(
                          sum(x[b,i] for b in range(B) for i in ids)
                          <= 1
                        )

    # 3) link sum_trips
    if 3 in groups_to_include:
        for b in range(B):
            for d in range(7):
                ids = [t["id"] for t in TRIPS if t["day"]==d]
                model.Add(sum(x[b,i] for i in ids) == sum_trips[b,d])

    # 4) ≤2 trips/day & ≤1 per direction
    if 4 in groups_to_include:
        for b in range(B):
            for d in range(7):
                day_ids = [t["id"] for t in TRIPS if t["day"]==d]
                model.Add(sum(x[b,i] for i in day_ids) <= 2)
                ab = [t["id"] for t in TRIPS if t["day"]==d and t["route"]=="A-B"]
                ba = [t["id"] for t in TRIPS if t["day"]==d and t["route"]=="B-A"]
                if ab: model.Add(sum(x[b,i] for i in ab) <= 1)
                if ba: model.Add(sum(x[b,i] for i in ba) <= 1)

    # 5) single-trip threshold
    if 5 in groups_to_include:
        high = {t["id"]:(1 if t["epk"]>=THRESHOLD else 0) for t in TRIPS}
        for b in range(B):
            for d in range(7):
                ids = [t["id"] for t in TRIPS if t["day"]==d]
                if ids:
                    model.Add(
                      sum(high[i]*x[b,i] for i in ids)
                      >= eq1[b,d]
                    )

    # 6) downtime & no-overlap
    if 6 in groups_to_include:
        ivs = []
        for b in range(B):
            row = []
            for t in TRIPS:
                iv = model.NewOptionalIntervalVar(
                    t["gstart"], cycle, t["gstart"]+cycle,
                    x[b,t["id"]],
                    f"iv[{b},{t['id']}]"
                )
                row.append(iv)
            model.AddNoOverlap(row)

    # 7) idle-gap within same day
    if 7 in groups_to_include:
        for b in range(B):
            for d in range(7):
                day = sorted([t for t in TRIPS if t["day"]==d], key=lambda x:x["gstart"])
                for i in range(len(day)):
                    for j in range(i+1, len(day)):
                        t1,t2 = day[i], day[j]
                        finish = t1["gstart"] + cycle
                        if t2["gstart"] - finish > max_idle:
                            model.Add(x[b,t1["id"]] + x[b,t2["id"]] <= 1)


    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5
    stat = solver.Solve(model)
    return stat in (cp_model.OPTIMAL, cp_model.FEASIBLE)


if __name__=="__main__":
    print("Testing constraint groups incrementally:")
    for g in range(1,9):
        ok = test_feasible(list(range(1, g+1)))
        print(f"  Groups 1..{g:>2} → {'feasible' if ok else 'INFEASIBLE'}")
