# problem_solver.py
"""
Greedy EV Bus Weekly Scheduler
A fallback heuristic-based scheduler (no OR-Tools) that:
  - Reads EPK data from CSV
  - Assigns trips greedily by descending EPK
  - Respects all hard constraints:
      • Forbidden start slots (00:30–03:00)
      • Each bus ≤2 trips/day
      • After a trip, bus unavailable until travel+charge complete
      • No overlapping trips on a bus
      • Buses alternate A↔B
      • No two departures from same station in same or adjacent slots
  - Outputs schedule.json and schedule.html
"""
import pandas as pd
import json
from collections import defaultdict

def load_epk(csv_path):
    df = pd.read_csv(csv_path)
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    trips = []
    for _, row in df.iterrows():
        # parse time slot
        h, m, _ = row['slot'].split(':')
        start = int(h)*60 + int(m)
        # skip forbidden slots
        if start in {30,60,90,120,150,180}:
            continue
        for di,day in enumerate(days):
            trips.append({
                'day': di,
                'start': start,
                'route': row['route'],
                'origin': row['route'][0],  # 'A' or 'B'
                'epk': float(row[day])
            })
    return trips


def greedy_schedule(trips, buses_A, buses_B,
                    travel_time_h, charge_time_h,
                    slot_gap_min=30):
    # Initialize buses
    total = buses_A + buses_B
    # Each bus has: available_time, location ('A'/'B'), trips_today count per day
    bus_state = [
        {'avail':0, 'loc':('A' if i<buses_A else 'B'), 'last_depart':-999, 'daily_count':defaultdict(int)}
        for i in range(total)
    ]
    duration = int((travel_time_h + charge_time_h)*60)

    # Prepare output: day->route->list of (bus, start, epk)
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    schedule = {d:{'A-B':[], 'B-A':[]} for d in days}

    # Sort all trips by descending EPK globally
    sorted_trips = sorted(trips, key=lambda x: -x['epk'])

    # Track station/time occupancy (day, start) -> occupied
    occupied = set()

    for trip in sorted_trips:
        day = trip['day']
        start = trip['start']
        key1 = (day, start, trip['origin'])
        # also adjacent slots
        adj_keys = {(day, start-slot_gap_min, trip['origin']), (day, start+slot_gap_min, trip['origin'])}
        # try assign to some bus
        for b in range(total):
            st = bus_state[b]
            # can only depart if at correct origin
            if st['loc'] != trip['origin']:
                continue
            # bus must be available
            global_start = day*24*60 + start
            if st['avail'] > global_start:
                continue
            # daily count < 2
            if st['daily_count'][day] >= 2:
                continue
            # no departure same or adjacent
            if (day, start, trip['origin']) in occupied:
                continue
            if any((day, start+delta, trip['origin']) in occupied for delta in (-slot_gap_min, slot_gap_min)):
                continue
            # assign
            schedule[days[day]][trip['route']].append((b, start, trip['epk']))
            occupied.add(key1)
            for k in adj_keys:
                # prevent adjacent slot departure by marking occupied
                occupied.add(k)
            # update bus state
            st['avail'] = global_start + duration
            st['loc'] = 'B' if trip['origin']=='A' else 'A'
            st['last_depart'] = start
            st['daily_count'][day] += 1
            break

    return schedule


def write_outputs(schedule):
    # JSON
    out = []
    for day,routes in schedule.items():
        for route, trips in routes.items():
            for b,start,epk in trips:
                out.append({'day':day,'route':route,'bus':b,
                            'start':f"{start//60:02d}:{start%60:02d}",
                            'epk':epk})
    with open('schedule.json','w') as f:
        json.dump(out, f, indent=2)
    # HTML
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    html = ['<html><head><style>table{border-collapse:collapse;}td,th{border:1px solid#999;padding:5px;} .slot{margin:2px 0;}</style></head><body>']
    html.append('<h1>Weekly Schedule</h1><table><tr><th>Day</th><th>A→B</th><th>B→A</th></tr>')
    for day in days:
        html.append(f'<tr><td><b>{day}</b></td>')
        for route in ['A-B','B-A']:
            cell = ''
            for b,start,epk in schedule[day][route]:
                tstr = f"{start//60:02d}:{start%60:02d}"
                cell += f"<div class='slot'>Bus {b}@{tstr} (EPK={epk:.2f})</div>"
            html.append(f'<td>{cell}</td>')
        html.append('</tr>')
    html.append('</table></body></html>')
    with open('schedule.html','w') as f:
        f.write('\n'.join(html))


def main():
    csv_path = input("EPK CSV path: ")
    buses_A = int(input("Number of buses at A: "))
    buses_B = int(input("Number of buses at B: "))
    tt = float(input("Travel time (h): "))
    ct = float(input("Charge time (h): "))

    trips = load_epk(csv_path)
    schedule = greedy_schedule(trips, buses_A, buses_B, tt, ct)
    write_outputs(schedule)
    print("Done: schedule.json and schedule.html generated.")

if __name__=='__main__':
    main()
