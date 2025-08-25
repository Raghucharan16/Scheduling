import os
import json
import psycopg2
import pandas as pd
from datetime import datetime, date, timedelta
from math import inf
from dotenv import load_dotenv
# CONFIG
TIME_WINDOW_MIN = 30         # ± minutes threshold (set very large for experimentation)
UTC_OFFSET_MIN = 330          # TripBoardingPoints stored UTC -> convert to IST for comparison
METRICS_OUT = "service_mappings_may_metrics.csv"
load_dotenv()
# Route code to (sourceId, destinationId)
ROUTE_CODE_MAP = {
    "H-V":  (3, 5),
    "V-H":  (5, 3),
    "Vij-Vsk": (4, 6),   # adjust IDs if needed
    "Vsk-Vij": (6, 4),
    "B-T": (7, 8),
    "T-B": (8, 7),
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

def main():
    # Inputs
    schedule_json = "C:/Users/Raghu/Downloads/hv-schedule.json"
    base_day1_date = date(2025, 5, 1)
    start_date = base_day1_date
    end_date = date(2025, 5, 31)

    schedule = load_schedule(schedule_json)
    schedule_legs = expand_schedule(schedule, base_day1_date)
    needed_route_codes = sorted({leg["route_code"] for leg in schedule_legs})

    print(f"Schedule legs loaded: {len(schedule_legs)} over {len(needed_route_codes)} route codes")

    trips = fetch_trips(start_date, end_date, needed_route_codes)
    print(f"Trips fetched: {len(trips)}")

    metrics_rows = run_matching(schedule_legs, trips)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["date","route_code"])
    metrics_df.to_csv(METRICS_OUT, index=False)

    # Overall summary (vs schedule only)
    total_sched = metrics_df["scheduled_legs"].sum()
    total_trips = metrics_df["trips_db"].sum()
    total_matched = metrics_df["matched_legs"].sum()
    overall_ratio_sched = round((total_matched / total_sched * 100), 2) if total_sched else 0.0
    overall_ratio_trips = round((total_matched / total_trips * 100), 2) if total_trips else 0.0

    print("Per-day metrics saved to", METRICS_OUT)
    print(f"Window (minutes): {TIME_WINDOW_MIN}")
    print(f"Overall scheduled legs: {total_sched}")
    print(f"Overall trips in DB: {total_trips}")
    print(f"Overall matched legs: {total_matched}")
    print(f"Overall match ratio vs schedule: {overall_ratio_sched}%")
    print(f"Overall match ratio vs trips: {overall_ratio_trips}%")
    # Helpful diagnostics to explain non-100% with huge window
    remaining_due_to_shortage = metrics_df['scheduled_minus_trips'].sum()
    remaining_outside_window = metrics_df['legs_closest_outside_window'].sum()
    no_trip_days = metrics_df['legs_no_trips'].sum()
    if remaining_due_to_shortage:
        print(f"Unmatched (schedule exceeds trips): {remaining_due_to_shortage}")
    if remaining_outside_window:
        print(f"Unmatched (even closest diff > window): {remaining_outside_window}")
    if no_trip_days:
        print(f"Schedule legs on days with zero trips: {no_trip_days}")

if __name__ == "__main__":
    main()