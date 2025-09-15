import os
import json
import psycopg2
import pandas as pd
from datetime import datetime, date, timedelta
from math import inf
from dotenv import load_dotenv
# CONFIG
TIME_WINDOW_MIN = 60         # ± minutes threshold (set very large for experimentation)
BUFFER_MINUTES = 30          # Buffer for "close match" comparison
UTC_OFFSET_MIN = 330          # TripBoardingPoints stored UTC -> convert to IST for comparison
METRICS_OUT = "service_mappings_may_metrics.csv"
load_dotenv()
# Route code to (sourceId, destinationId)
ROUTE_CODE_MAP = {
    "H-V":  (3, 5),
    "V-H":  (5, 3),
    "Vij-Vsk": (5, 58),   # adjust IDs if needed
    "Vsk-Vij": (58, 5),
    "B-T": (7, 12),
    "T-B": (12, 7),
}

# Route name mapping for existing schedule data
ROUTE_NAME_MAP = {
    "BANGALORE": "B",
    "TIRUPATI": "T", 
    "HYDERABAD": "H",
    "VIJAYAWADA": "V",
    "VISAKHAPATNAM": "Vsk",
    "MYSORE": "M"
}

def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", 5432)
    )

def load_schedule(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("schedule", {})

def parse_hms(tstr: str):
    if len(tstr.split(":")) == 2:  # HH:MM
        tstr = tstr + ":00"
    return datetime.strptime(tstr, "%H:%M:%S").time()

def parse_existing_schedule(existing_data_str):
    """
    Parse the existing schedule data string into structured format.
    Returns list of dicts with route_code and schedule_time.
    """
    existing_trips = []
    # Split by | and process in groups of 4 (source, destination, bus_id, time)
    parts = [p.strip() for p in existing_data_str.split("|")]
    
    for i in range(0, len(parts), 4):
        if i + 3 < len(parts):
            source = parts[i].strip()
            destination = parts[i + 1].strip()
            bus_id = parts[i + 2].strip()
            time_str = parts[i + 3].strip()
            
            # Convert route names to route codes
            source_code = ROUTE_NAME_MAP.get(source)
            dest_code = ROUTE_NAME_MAP.get(destination)
            
            if source_code and dest_code:
                route_code = f"{source_code}-{dest_code}"
                try:
                    schedule_time = parse_hms(time_str)
                    existing_trips.append({
                        "route_code": route_code,
                        "schedule_time": schedule_time,
                        "bus_id": bus_id,
                        "source": source,
                        "destination": destination
                    })
                except ValueError:
                    continue  # Skip invalid time formats
    
    return existing_trips

def time_difference_minutes(time1, time2):
    """Calculate difference between two time objects in minutes."""
    dt1 = datetime.combine(date.today(), time1)
    dt2 = datetime.combine(date.today(), time2)
    
    # Handle times that cross midnight
    if time1 < time2 and (time2.hour - time1.hour) > 12:
        dt1 += timedelta(days=1)
    elif time2 < time1 and (time1.hour - time2.hour) > 12:
        dt2 += timedelta(days=1)
    
    return abs((dt1 - dt2).total_seconds() / 60.0)

def fetch_trips(start_date: date, end_date: date, needed_route_codes):
    """
    Returns list of dict:
      journeyDate (date)
      route_code
      tripId
      serviceId
      local_start_time (IST time object)
    """
    # Build filter for (sourceId,destinationId) pairs
    pairs = [ROUTE_CODE_MAP[rc] for rc in needed_route_codes if rc in ROUTE_CODE_MAP]
    if not pairs:
        return []
    # Build VALUES list for join
    # Use ANY on composite not simple; easier to OR them
    pair_filters = " OR ".join([f'(s."sourceId"={src} AND s."destinationId"={dst})' for src,dst in pairs])
    sql = f"""
        SELECT
          t.id AS trip_id,
          t."serviceId" AS service_id,
          t."journeyDate" AS journey_date,
          s."sourceId" AS src,
          s."destinationId" AS dst,
          MIN(b."scheduledTime") AS first_bp_utc
        FROM "Trips" t
        JOIN "Services" s ON s.id = t."serviceId"
        JOIN "TripBoardingPoints" b ON b."tripId" = t.id
        WHERE t.active = true
          AND t."journeyDate" BETWEEN %s AND %s
          AND ({pair_filters})
        GROUP BY t.id, t."serviceId", t."journeyDate", s."sourceId", s."destinationId";
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(sql, (start_date, end_date))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # Reverse route lookup map
    rev_map = {v: k for k,v in ROUTE_CODE_MAP.items()}

    trips = []
    for trip_id, service_id, jdate, src, dst, first_bp_utc in rows:
        if not first_bp_utc:
            continue
        # Convert UTC -> IST
        local_dt = first_bp_utc + timedelta(minutes=UTC_OFFSET_MIN)
        route_code = rev_map.get((src,dst))
        if not route_code:
            continue
        trips.append({
            "tripId": trip_id,
            "serviceId": service_id,
            "journeyDate": jdate,
            "route_code": route_code,
            "local_start_time": local_dt.time()
        })
    return trips

def expand_schedule(schedule_dict, base_start_date: date):
    """
    Converts schedule JSON to list of dict:
      date, dayIndex, bus, route_code, schedule_time (time)
    """
    legs = []
    for bus, days in schedule_dict.items():
        for day_str, entries in days.items():
            if not day_str.isdigit():
                continue
            day_idx = int(day_str)
            leg_date = base_start_date + timedelta(days=day_idx - 1)
            for seq, item in enumerate(entries):
                route_code = item["route"]
                if route_code not in ROUTE_CODE_MAP:
                    continue
                st = parse_hms(item["startTime"])
                legs.append({
                    "bus": bus,
                    "dayIndex": day_idx,
                    "date": leg_date,
                    "route_code": route_code,
                    "schedule_time": st,
                    "seq": seq
                })
    return legs

def minutes_diff(t1: datetime, t2: datetime):
    return abs((t1 - t2).total_seconds()) / 60.0

def hungarian(cost):
    """
    Simple Hungarian algorithm implementation.
    cost: 2D list (m x n). Returns assignment dict row->col minimizing cost.
    Assumes m <= n (pad if needed).
    """
    # Convert to square matrix
    m = len(cost)
    n = len(cost[0]) if cost else 0
    size = max(m, n)
    # Pad rows
    padded = [row + [1e9]*(size - n) for row in cost]
    # Pad extra rows
    for _ in range(size - m):
        padded.append([1e9]*size)

    # Hungarian
    u = [0]*(size+1)
    v = [0]*(size+1)
    p = [0]*(size+1)
    way = [0]*(size+1)

    for i in range(1, size+1):
        p[0] = i
        j0 = 0
        minv = [inf]*(size+1)
        used = [False]*(size+1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, size+1):
                if not used[j]:
                    cur = padded[i0-1][j-1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, size+1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # Augmenting
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    # p[j] = i
    assignment = {}
    for j in range(1, size+1):
        if p[j] != 0 and p[j] <= m and j <= n:
            assignment[p[j]-1] = j-1
    return assignment

def optimal_match_day_route(schedule_legs, trips):
    """
    schedule_legs: list with schedule_time (time)
    trips: list with local_start_time (time)
    Returns list of (schedule_index, trip_index, diff_minutes) only where diff <= TIME_WINDOW_MIN.
    Uses Hungarian to minimize total diff.
    """
    if not schedule_legs or not trips:
        return []

    # Build cost matrix (schedule rows x trip cols)
    cost = []
    diff_matrix = []
    for s in schedule_legs:
        row_cost = []
        row_diff = []
        sched_dt = datetime.combine(s["date"], s["schedule_time"])
        for t in trips:
            trip_dt = datetime.combine(t["journeyDate"], t["local_start_time"])
            diff = minutes_diff(sched_dt, trip_dt)
            row_diff.append(diff)
            row_cost.append(diff if diff <= TIME_WINDOW_MIN else 1e6)  # large penalty outside window
        cost.append(row_cost)
        diff_matrix.append(row_diff)

    assignment = hungarian(cost)
    matches = []
    for s_idx, t_idx in assignment.items():
        d = diff_matrix[s_idx][t_idx]
        if d <= TIME_WINDOW_MIN:
            matches.append((s_idx, t_idx, d))
    return matches

def run_matching(schedule_legs, trips):
    """Match per (date, route_code) bucket using optimal assignment.

    Adds diagnostic columns so if widening TIME_WINDOW_MIN doesn't drive ratio to 100%, you can
    see whether it's simply because there are fewer DB trips than scheduled legs, or because time
    differences still exceed the window (or route code absent).
    """
    # Group schedule legs and trips
    sched_map = {}
    for i, leg in enumerate(schedule_legs):
        key = (leg["date"], leg["route_code"])
        sched_map.setdefault(key, []).append((i, leg))
    trip_map = {}
    for j, tr in enumerate(trips):
        key = (tr["journeyDate"], tr["route_code"])
        trip_map.setdefault(key, []).append((j, tr))

    metrics_rows = []

    for key in sorted(sched_map.keys()):
        date_key, route_code = key
        sched_entries = sched_map[key]
        trip_entries = trip_map.get(key, [])
        sched_only = [se[1] for se in sched_entries]
        trips_only = [te[1] for te in trip_entries]

        # Optimal assignment within window
        matches = optimal_match_day_route(sched_only, trips_only)

        used_trip_local_indices = set()
        matched_legs = 0
        # For diagnostics: collect all raw diffs (even outside window) to know if window is the limiter
        all_diffs = []  # min diff per schedule leg
        for s_idx, s_leg in enumerate(sched_only):
            sched_dt = datetime.combine(s_leg["date"], s_leg["schedule_time"])
            diffs_this = []
            for t in trips_only:
                trip_dt = datetime.combine(t["journeyDate"], t["local_start_time"])
                diffs_this.append(minutes_diff(sched_dt, trip_dt))
            if diffs_this:
                all_diffs.append(min(diffs_this))
            else:
                all_diffs.append(None)

        for s_local_idx, t_local_idx, diff in matches:
            if t_local_idx in used_trip_local_indices:
                continue
            used_trip_local_indices.add(t_local_idx)
            matched_legs += 1

        scheduled_legs = len(sched_entries)
        trips_db = len(trip_entries)
        ratio_sched = (matched_legs / scheduled_legs * 100) if scheduled_legs else 0.0
        ratio_trips = (matched_legs / trips_db * 100) if trips_db else 0.0

        # Diagnostics
        shortage = max(0, scheduled_legs - trips_db)
        # Count schedule legs whose closest trip (if any) still outside window
        outside_window = sum(1 for d in all_diffs if d is not None and d > TIME_WINDOW_MIN)
        no_trips_for_leg = sum(1 for d in all_diffs if d is None)

        metrics_rows.append({
            "date": date_key,
            "route_code": route_code,
            "scheduled_legs": scheduled_legs,
            "trips_db": trips_db,
            "matched_legs": matched_legs,
            "match_ratio_sched_pct": round(ratio_sched, 2),
            "match_ratio_trips_pct": round(ratio_trips, 2),
            "scheduled_minus_trips": shortage,
            "legs_no_trips": no_trips_for_leg,
            "legs_closest_outside_window": outside_window,
            "window_min": TIME_WINDOW_MIN
        })

    return metrics_rows

def compare_schedules(json_schedule_legs, existing_schedule_trips):
    """
    Compare JSON schedule with existing schedule and return the three metrics:
    1. Exact matches (same route, same time)
    2. Buffer matches (same route, within 30 minutes)  
    3. New trips (not matching existing at all)
    """
    
    # Group existing trips by route for faster lookup
    existing_by_route = {}
    for trip in existing_schedule_trips:
        route = trip["route_code"]
        if route not in existing_by_route:
            existing_by_route[route] = []
        existing_by_route[route].append(trip)
    
    # Flatten JSON schedule legs to just route and time (ignore dates for comparison)
    json_trips = []
    for leg in json_schedule_legs:
        json_trips.append({
            "route_code": leg["route_code"],
            "schedule_time": leg["schedule_time"]
        })
    
    exact_matches = 0
    buffer_matches = 0
    new_trips = 0
    
    for json_trip in json_trips:
        route_code = json_trip["route_code"]
        json_time = json_trip["schedule_time"]
        
        # Check if route exists in existing schedule
        if route_code not in existing_by_route:
            new_trips += 1
            continue
            
        existing_trips_for_route = existing_by_route[route_code]
        
        # Check for exact match first
        exact_match_found = False
        for existing_trip in existing_trips_for_route:
            if json_time == existing_trip["schedule_time"]:
                exact_matches += 1
                exact_match_found = True
                break
        
        if exact_match_found:
            continue
            
        # Check for buffer match (within 30 minutes)
        buffer_match_found = False
        for existing_trip in existing_trips_for_route:
            time_diff = time_difference_minutes(json_time, existing_trip["schedule_time"])
            if time_diff <= BUFFER_MINUTES:
                buffer_matches += 1
                buffer_match_found = True
                break
                
        if not buffer_match_found:
            new_trips += 1
    
    total_json_trips = len(json_trips)
    
    results = {
        "total_json_trips": total_json_trips,
        "exact_matches": exact_matches,
        "buffer_matches": buffer_matches, 
        "new_trips": new_trips,
        "exact_match_percentage": round((exact_matches / total_json_trips * 100), 2) if total_json_trips > 0 else 0,
        "buffer_match_percentage": round((buffer_matches / total_json_trips * 100), 2) if total_json_trips > 0 else 0,
        "new_trips_percentage": round((new_trips / total_json_trips * 100), 2) if total_json_trips > 0 else 0
    }
    
    return results

def main():
    # Inputs
    schedule_json = "cache/hv-oct-refer.json"
    base_day1_date = date(2025, 9, 1)
    
    # Existing schedule data (provided by user)
    existing_schedule_data = """BANGALORE |TIRUPATI | 67|10:00 | BANGALORE |TIRUPATI | 8|12:00 | BANGALORE |TIRUPATI | 14|14:00 | BANGALORE |TIRUPATI | 35|17:00 | BANGALORE |TIRUPATI | 19|19:30 | BANGALORE |TIRUPATI | 22|20:30 | BANGALORE |TIRUPATI | 103|21:30 | BANGALORE |TIRUPATI | 9|22:20 | BANGALORE |TIRUPATI | 31|23:30 | BANGALORE |TIRUPATI | 68|5:00 | BANGALORE |TIRUPATI | 75|5:30 | BANGALORE |TIRUPATI | 36|6:00 | BANGALORE |TIRUPATI | 29|7:00 | HYDERABAD |VIJAYAWADA | 40|10:30 | HYDERABAD |VIJAYAWADA | 99|15:00 | HYDERABAD |VIJAYAWADA | 48|16:00 | HYDERABAD |VIJAYAWADA | 123|20:15 | HYDERABAD |VIJAYAWADA | 58|21:00 | HYDERABAD |VIJAYAWADA | 61|21:45 | HYDERABAD |VIJAYAWADA | 37|22:45 | HYDERABAD |VIJAYAWADA | 38|23:10 | HYDERABAD |VIJAYAWADA | 55|3:35 | HYDERABAD |VIJAYAWADA | 47|4:50 | HYDERABAD |VIJAYAWADA | 119|6:00 | HYDERABAD |VIJAYAWADA | 39|7:30 | HYDERABAD |VIJAYAWADA | 76|8:30 | HYDERABAD |VIJAYAWADA | 49|9:30 | MYSORE |BANGALORE | 127|17:00 | MYSORE |BANGALORE | 126|18:30 | MYSORE |BANGALORE | 131|19:30 | TIRUPATI |BANGALORE | 97|10:30 | TIRUPATI |BANGALORE | 10|11:30 | TIRUPATI |BANGALORE | 6|14:00 | TIRUPATI |BANGALORE | 34|16:00 | TIRUPATI |BANGALORE | 54|17:00 | TIRUPATI |BANGALORE | 102|18:00 | TIRUPATI |BANGALORE | 89|20:30 | TIRUPATI |BANGALORE | 21|22:00 | TIRUPATI |BANGALORE | 7|22:50 | TIRUPATI |BANGALORE | 13|23:50 | TIRUPATI |BANGALORE | 33|5:30 | TIRUPATI |BANGALORE | 90|6:30 | TIRUPATI |BANGALORE | 3|8:00 | VIJAYAWADA |HYDERABAD | 60|10:00 | VIJAYAWADA |HYDERABAD | 53|10:40 | VIJAYAWADA |HYDERABAD | 100|11:40 | VIJAYAWADA |HYDERABAD | 50|15:00 | VIJAYAWADA |HYDERABAD | 51|16:00 | VIJAYAWADA |HYDERABAD | 112|17:00 | VIJAYAWADA |HYDERABAD | 57|20:00 | VIJAYAWADA |HYDERABAD | 43|20:50 | VIJAYAWADA |HYDERABAD | 44|22:10 | VIJAYAWADA |HYDERABAD | 52|23:00 | VIJAYAWADA |HYDERABAD | 41|4:10 | VIJAYAWADA |HYDERABAD | 104|5:00 | VIJAYAWADA |HYDERABAD | 42|8:00 | VIJAYAWADA |HYDERABAD | 59|9:00 | VIJAYAWADA |VISAKHAPATNAM| 403|10:00 | VIJAYAWADA |VISAKHAPATNAM| 412|17:30 | VIJAYAWADA |VISAKHAPATNAM| 404|19:30 | VIJAYAWADA |VISAKHAPATNAM| 401|20:30 | VIJAYAWADA |VISAKHAPATNAM| 402|21:30 | VIJAYAWADA |VISAKHAPATNAM| 411|9:00 | VISAKHAPATNAM|VIJAYAWADA | 408|10:00 | VISAKHAPATNAM|VIJAYAWADA | 414|19:30 | VISAKHAPATNAM|VIJAYAWADA | 409|20:30 | VISAKHAPATNAM|VIJAYAWADA | 410|21:30 | VISAKHAPATNAM|VIJAYAWADA | 407|22:30"""

    # Load JSON schedule
    schedule = load_schedule(schedule_json)
    schedule_legs = expand_schedule(schedule, base_day1_date)
    
    # Parse existing schedule
    existing_trips = parse_existing_schedule(existing_schedule_data)
    
    print(f"JSON Schedule legs loaded: {len(schedule_legs)}")
    print(f"Existing schedule trips loaded: {len(existing_trips)}")
    
    # Get unique route codes from both schedules
    json_routes = sorted({leg["route_code"] for leg in schedule_legs})
    existing_routes = sorted({trip["route_code"] for trip in existing_trips})
    
    print(f"JSON schedule routes: {json_routes}")
    print(f"Existing schedule routes: {existing_routes}")
    
    # Compare schedules
    comparison_results = compare_schedules(schedule_legs, existing_trips)
    
    print("\n" + "="*60)
    print("SCHEDULE COMPARISON RESULTS")
    print("="*60)
    print(f"Total trips in JSON schedule: {comparison_results['total_json_trips']}")
    print(f"Buffer window for matching: {BUFFER_MINUTES} minutes")
    print()
    print(f"1. EXACT MATCHES: {comparison_results['exact_matches']} ({comparison_results['exact_match_percentage']}%)")
    print(f"   - Same route, same time")
    print()
    print(f"2. BUFFER MATCHES: {comparison_results['buffer_matches']} ({comparison_results['buffer_match_percentage']}%)")
    print(f"   - Same route, within {BUFFER_MINUTES} minutes difference")
    print()
    print(f"3. NEW TRIPS: {comparison_results['new_trips']} ({comparison_results['new_trips_percentage']}%)")
    print(f"   - Not matching existing schedule (new routes or >30min difference)")
    print()
    print(f"TOTAL VERIFIED: {comparison_results['exact_matches'] + comparison_results['buffer_matches'] + comparison_results['new_trips']} trips")
    print("="*60)

if __name__ == "__main__":
    main()