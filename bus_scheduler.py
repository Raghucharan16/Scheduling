#!/usr/bin/env python3
import csv, json
from datetime import timedelta
from ortools.sat.python import cp_model

DAYS      = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
FORBIDDEN = {30,60,90,120,150,180}

def load_trips(csv_path, a1,b1,c1,a2,b2,c2):
    trips=[]
    idx=0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            h,m,_ = row["slot"].split(":")
            start = int(h)*60 + int(m)
            if start in FORBIDDEN: continue
            route  = row["route"]
            origin = route[0]
            if route=="X-Y":
                cycle, mid_off = a1+b1+c1, a1
            else:
                cycle, mid_off = a2+b2+c2, a2
            for d,day in enumerate(DAYS):
                trips.append({
                  "id":     idx,
                  "day":    d,
                  "start":  start,
                  "origin": origin,
                  "route":  route,
                  "epk":    float(row[day]),
                  "cycle":  cycle,
                  "mid_off":mid_off
                })
                idx+=1
    return trips

def hhmmss(mins):
    td=timedelta(minutes=mins)
    h=td.seconds//3600; m=(td.seconds%3600)//60; s=td.seconds%60
    return f"{h:02d}:{m:02d}:{s:02d}"

def build_and_solve(trips,nX,nY,charge_depo):
    B=nX+nY
    model=cp_model.CpModel()
    x, intervals = {}, {b:[] for b in range(B)}

    # Decision + interval vars
    for b in range(B):
      for t in trips:
        lit = model.NewBoolVar(f"x[{b},{t['id']}]")
        x[b,t["id"]] = lit
        s = t["day"]*1440 + t["start"]
        iv = model.NewOptionalIntervalVar(s, t["cycle"], s+t["cycle"], lit, f"iv[{b},{t['id']}]")
        intervals[b].append(iv)

    # 1) each trip ≤1 bus
    for t in trips:
      model.Add(sum(x[b,t["id"]] for b in range(B)) <= 1)

    # 2) no overlap per bus
    for b in range(B):
      model.AddNoOverlap(intervals[b])

    # 3) no two departures same origin/time
    for d in range(7):
      for orig in ("X","Y"):
        for s in {t["start"] for t in trips if t["day"]==d and t["origin"]==orig}:
          ids=[t["id"] for t in trips if t["day"]==d and t["start"]==s and t["origin"]==orig]
          model.Add(sum(x[b,i] for b in range(B) for i in ids) <=1)

    # 4) ≤2 trips/day
    for b in range(B):
      for d in range(7):
        day_ids=[t["id"] for t in trips if t["day"]==d]
        model.Add(sum(x[b,i] for i in day_ids) <= 2)

    # 5) depot chaining
    for b in range(B):
      home = "X" if b<nX else "Y"
      for t in trips:
        if t["origin"]!=home:
          earlier=[u for u in trips if (u["day"]<t["day"]) or (u["day"]==t["day"] and u["start"]<t["start"])]
          if not any((b,u["id"]) in x for u in earlier):
            model.Add(x[b,t["id"]]==0)

    # 6) continuity + time-gating
    for b in range(B):
      for t1 in trips:
        if (b,t1["id"]) not in x: continue
        g1   = t1["day"]*1440 + t1["start"]
        end1 = g1 + t1["cycle"] + int(charge_depo*60)
        dest = "Y" if t1["origin"]=="X" else "X"
        for t2 in trips:
          if (b,t2["id"]) not in x: continue
          g2 = t2["day"]*1440 + t2["start"]
          if g2 <= g1: continue
          if t2["origin"]!=dest:
            model.Add(x[b,t1["id"]] + x[b,t2["id"]] <= 1)
          if g2 < end1:
            model.Add(x[b,t1["id"]] + x[b,t2["id"]] <= 1)

    # 7) objective: maximize trips then epk
    trip_count = sum(x[b,t["id"]] for b in range(B) for t in trips)
    max_epk100  = max(int(t["epk"]*100) for t in trips)
    W           = max_epk100*B*len(trips) + 1
    epk_terms   = sum(int(t["epk"]*100)*x[b,t["id"]] for b in range(B) for t in trips)
    model.Maximize(W*trip_count + epk_terms)

    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=60
    solver.parameters.num_search_workers=8
    status=solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
      print("❌ No feasible schedule!")
      return None

    out={day:{"assignments":[],"feasible":True} for day in DAYS}
    for b in range(B):
      chosen=[t for t in trips if solver.Value(x[b,t["id"]])]
      chosen.sort(key=lambda u:(u["day"],u["start"]))
      for t in chosen:
        out[DAYS[t["day"]]]["assignments"].append({
          "busNumber": b+1,
          "dayIndex":  t["day"],
          "dayName":   DAYS[t["day"]],
          "trip":{
            "route":      t["route"],
            "startTime":  hhmmss(t["start"]),
            "midPointTime": hhmmss(t["start"]+t["mid_off"]),
            "endTime":    hhmmss(t["start"]+t["cycle"]),
            "epk":        round(t["epk"],2)
          }
        })
    return out

if __name__=="__main__":
    csv_path    = "epk_data.csv"
    nX          = int(input("Buses at X: "))
    nY          = int(input("Buses at Y: "))
    a1,b1,c1    = map(float,input("X→M travel, charge; M→Y travel (h): ").split())
    a2,b2,c2    = map(float,input("Y→M travel, charge; M→X travel (h): ").split())
    charge_depo = float(input("Depot charge (h): "))
    trips = load_trips(csv_path,int(a1*60),int(b1*60),int(c1*60),
                               int(a2*60),int(b2*60),int(c2*60))
    sched = build_and_solve(trips,nX,nY,charge_depo)
    if sched:
      with open("schedule.json","w") as f:
        json.dump({"schedule":sched},f,indent=2)
      print("✅ schedule.json written.")
