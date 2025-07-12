import csv
from typing import List, Dict

# forbidden half-hour slots (minutes past midnight)
FORBIDDEN = {30, 60, 90, 120, 150, 180}
DAYS = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]

def load_trips(csv_path: str) -> List[Dict]:
    """
    Load EPK CSV, skip forbidden slots, return list of:
      {id, day (0–6), start (minutes), origin ('A'/'B'), route, epk}
    """
    trips: List[Dict] = []
    idx = 0
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            h,m,_ = row["slot"].split(":")
            start = int(h)*60 + int(m)
            if start in FORBIDDEN:
                continue
            origin = row["route"][0]
            for d, day in enumerate(DAYS):
                trips.append({
                    "id":     idx,
                    "day":    d,
                    "start":  start,
                    "origin": origin,
                    "route":  row["route"],
                    "epk":    float(row[day])
                })
                idx += 1
    return trips
