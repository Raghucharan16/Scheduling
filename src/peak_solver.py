#!/usr/bin/env python3
"""
Pure OR-Tools CP-SAT Peak Phase Solver (Friday-Sunday)
Focus: EPK OPTIMIZATION + PREMIUM NIGHT COVERAGE with day continuity
"""

from typing import Dict, List, Tuple, Optional
from ortools.sat.python import cp_model
from .bus_manager import BusManager
from .loader import load_trips
import logging
from .constraints import global_bus_trip_chaining

class PeakCPSATSolver:
    """Pure OR-Tools CP-SAT solver for peak phase scheduling"""
    
    def __init__(self, bus_manager: BusManager):
        self.bus_manager = bus_manager
        self.logger = logging.getLogger(__name__)
        
    def solve_peak_phase(self, csv_path: str, end_states: List[Dict]) -> Dict:
        """
        Solve peak phase (Fri-Sun) using pure OR-Tools CP-SAT
        
        Returns:
            schedule_dict
        """
        print(f"\n🚀 PURE CP-SAT PEAK SOLVER (Fri-Sun)")
        print(f"   Strategy: OR-Tools CP-SAT with EPK optimization + night coverage")
        print(f"   Target: 18+ trips per day with premium EPK focus")
        
        # Apply end states from regular phase
        self.bus_manager.reset_for_phase(end_states)
        
        # Load and filter trips
        all_trips = load_trips(csv_path)
        peak_trips = [t for t in all_trips if 4 <= t["day"] <= 6]
        
        # Categorize trips by quality
        premium_trips = [t for t in peak_trips if t["epk"] >= 95.0]  # Premium
        night_premium = [t for t in peak_trips if t["epk"] >= 85.0 and 
                        (t["start"] >= 1320 or t["start"] <= 240)]  # Night premium
        good_trips = [t for t in peak_trips if t["epk"] >= 70.0]   # Good quality
        
        # Combine with priority: premium > night premium > good
        # Use dict to avoid duplicates since trips are dictionaries
        trip_dict = {}
        for t in premium_trips + night_premium + good_trips:
            trip_dict[t["id"]] = t
        viable_trips = list(trip_dict.values())
        
        print(f"   Available trips: {len(viable_trips)} total")
        print(f"     Premium (≥₹95): {len(premium_trips)} trips")
        print(f"     Night Premium (≥₹85, night): {len(night_premium)} trips")
        print(f"     Good (≥₹40): {len(good_trips)} trips")
        
        # Create CP-SAT model
        model = cp_model.CpModel()
        
        # Parameters
        B = self.bus_manager.total_buses
        # buses_a = self.bus_manager.buses_at_a
        travel_h = self.bus_manager.travel_h
        charge_h = self.bus_manager.charge_h
        cycle_minutes = int((travel_h + charge_h) * 60)
        
        # Decision variables: x[bus_idx, trip_id] = 1 if bus does trip
        x = {}
        for b in range(B):
            for t in viable_trips:
                x[b, t["id"]] = model.NewBoolVar(f"x[{b},{t['id']}]")
        
        # CONSTRAINT 1: No overlapping trips for same bus
        for b in range(B):
            intervals = []
            for t in viable_trips:
                start_time = t["day"] * 1440 + t["start"]
                end_time = start_time + cycle_minutes
                interval = model.NewOptionalIntervalVar(
                    start_time, cycle_minutes, end_time,
                    x[b, t["id"]], f"interval[{b},{t['id']}]"
                )
                intervals.append(interval)
            model.AddNoOverlap(intervals)
        
        # CONSTRAINT 2: Max 2 trips per day per bus
        for b in range(B):
            for day in range(4, 7):  # Days 4-6
                day_trips = [t["id"] for t in viable_trips if t["day"] == day]
                model.Add(sum(x[b, tid] for tid in day_trips) <= 2)
        
        # CONSTRAINT 3: Max 1 trip per direction per day per bus
        for b in range(B):
            for day in range(4, 7):
                ab_trips = [t["id"] for t in viable_trips if t["day"] == day and t["route"] == "A-B"]
                ba_trips = [t["id"] for t in viable_trips if t["day"] == day and t["route"] == "B-A"]
                
                if ab_trips:
                    model.Add(sum(x[b, tid] for tid in ab_trips) <= 1)
                if ba_trips:
                    model.Add(sum(x[b, tid] for tid in ba_trips) <= 1)
        
        # CONSTRAINT 4: No two buses same origin/time
        for day in range(4, 7):
            times = set(t["start"] for t in viable_trips if t["day"] == day)
            for time in times:
                for origin in ["A", "B"]:
                    trip_ids = [t["id"] for t in viable_trips 
                              if t["day"] == day and t["start"] == time and t["origin"] == origin]
                    if trip_ids:
                        model.Add(sum(x[b, tid] for b in range(B) for tid in trip_ids) <= 1)
        
        # CONSTRAINT 5: Cross-day continuity 
        for b in range(B):
            for t1 in viable_trips:
                end1 = t1["day"] * 1440 + t1["start"] + cycle_minutes
                next_day = t1["day"] + 1
                
                for t2 in viable_trips:
                    if t2["day"] == next_day:
                        start2 = t2["day"] * 1440 + t2["start"]
                        if start2 < end1:
                            # t2 starts before t1 finishes - forbid both
                            model.Add(x[b, t1["id"]] + x[b, t2["id"]] <= 1)

        # ENFORCE GLOBAL BUS TRIP CHAINING
        global_bus_trip_chaining(model, x, viable_trips, B, travel_h, charge_h)
        
        # CONSTRAINT 6: Route continuity (relaxed for feasibility)
        # Similar to regular phase - only enforce for same-day consecutive trips
        for b in range(B):
            for day in range(4, 7):
                day_trips = [t for t in viable_trips if t["day"] == day]
                day_trips.sort(key=lambda t: t["start"])
                
                for i in range(len(day_trips) - 1):
                    t1 = day_trips[i]
                    t2 = day_trips[i + 1]
                    
                    # Check if these could be consecutive for same bus
                    if (t2["start"] - t1["start"]) >= cycle_minutes // 60:  # Enough time gap
                        t1_dest = t1["route"].split('-')[1]
                        t2_origin = t2["origin"]
                        if t1_dest != t2_origin:
                            # These specific trips cannot be consecutive
                            model.Add(x[b, t1["id"]] + x[b, t2["id"]] <= 1)
        
        # CONSTRAINT 7: NIGHT COVERAGE - ensure some night trips
        for day in range(4, 7):
            night_trips = [t["id"] for t in viable_trips 
                          if t["day"] == day and (t["start"] >= 1320 or t["start"] <= 240)]
            if night_trips:
                # Ensure at least 2 night trips per day (relaxed)
                model.Add(sum(x[b, tid] for b in range(B) for tid in night_trips) >= 2)
        
        # CONSTRAINT 8: UTILIZATION - encourage bus usage (relaxed)
        for b in range(B):
            # Each bus should do at least 2 trips total across all 3 days
            all_trips = [t["id"] for t in viable_trips]
            model.Add(sum(x[b, tid] for tid in all_trips) >= 2)
        
        # CONSTRAINT 9: TARGET TRIP COUNT - aim for good utilization  
        for day in range(4, 7):
            day_trips = [t["id"] for t in viable_trips if t["day"] == day]
            # Try to achieve at least 6 trips per day (further relaxed)
            model.Add(sum(x[b, tid] for b in range(B) for tid in day_trips) >= 6)
        
        # OBJECTIVE: Maximize EPK + premium bonuses + night coverage
        total_epk = sum(t["epk"] * x[b, t["id"]] for b in range(B) for t in viable_trips)
        
        # Premium bonuses
        premium_bonus = sum(
            500 * x[b, t["id"]] for b in range(B) for t in viable_trips
            if t["epk"] >= 95.0
        )
        
        # Night premium bonus
        night_premium_bonus = sum(
            1000 * x[b, t["id"]] for b in range(B) for t in viable_trips
            if (t["start"] >= 1320 or t["start"] <= 240) and t["epk"] >= 85.0
        )
        
        # Night coverage bonus
        night_coverage = sum(
            200 * x[b, t["id"]] for b in range(B) for t in viable_trips
            if t["start"] >= 1320 or t["start"] <= 240
        )
        
        # Trip count bonus (secondary priority)
        total_trips = sum(x[b, t["id"]] for b in range(B) for t in viable_trips)
        
        # Weighted objective (prioritize EPK quality + night coverage)
        model.Maximize(
            100 * total_epk +           # Base EPK value
            premium_bonus +            # Premium trip bonus
            night_premium_bonus +      # Night premium bonus
            night_coverage +           # Night coverage bonus
            50 * total_trips          # Trip count bonus (lower priority)
        )
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0  # 1 minute timeout
        
        self.logger.info(f"Solving Peak CP-SAT model with {len(viable_trips)} trips, {B} buses")
        status = solver.Solve(model)
        
        schedule = {d: [] for d in range(4, 7)}
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            self.logger.info(f"CP-SAT Peak solution: {solver.StatusName(status)}")
            
            assignments = []
            for b in range(B):
                for t in viable_trips:
                    if solver.Value(x[b, t["id"]]) == 1:
                        bus_id = b + 1  # Convert to 1-based
                        day = t["day"]
                        
                        # Update bus manager
                        trip_start = day * 1440 + t["start"]
                        self.bus_manager.assign_trip(bus_id, trip_start, day, t["route"], t["epk"])
                        
                        # Add to schedule
                        trip_data = {
                            "bus": bus_id,
                            "id": t["id"],
                            "start": t["start"],
                            "route": t["route"],
                            "origin": t["origin"],
                            "epk": t["epk"],
                            "day": day,
                            "cpsat_solution": True
                        }
                        schedule[day].append(trip_data)
                        assignments.append(trip_data)
            
            # Print results
            total_trips = len(assignments)
            total_revenue = sum(a["epk"] for a in assignments)
            night_trips = len([a for a in assignments 
                             if a["start"] >= 1320 or a["start"] <= 240])
            premium_trips = len([a for a in assignments if a["epk"] >= 95.0])
            night_premium_trips = len([a for a in assignments 
                                     if (a["start"] >= 1320 or a["start"] <= 240) and a["epk"] >= 85.0])
            
            print(f"\n📊 PEAK CP-SAT RESULTS:")
            print(f"   Status: {solver.StatusName(status)}")
            print(f"   Total trips: {total_trips}")
            print(f"   Total revenue: ₹{total_revenue:.2f}")
            print(f"   Average EPK: ₹{total_revenue/total_trips:.2f}" if total_trips > 0 else "   No trips")
            print(f"   Premium trips (≥₹95): {premium_trips}")
            print(f"   Night coverage: {night_trips} trips")
            print(f"   Night premium: {night_premium_trips} trips")
            print(f"   Utilization: {total_trips/60*100:.1f}% (target: 60 trips)")
            
            for day in range(4, 7):
                day_name = ["Friday", "Saturday", "Sunday"][day - 4]
                day_trips = len(schedule[day])
                day_revenue = sum(t["epk"] for t in schedule[day])
                day_night = len([t for t in schedule[day] if t["start"] >= 1320 or t["start"] <= 240])
                day_premium = len([t for t in schedule[day] if t["epk"] >= 95.0])
                print(f"   {day_name}: {day_trips} trips, ₹{day_revenue:.2f}, {day_night} night, {day_premium} premium")
                
        else:
            self.logger.error(f"CP-SAT Peak failed: {solver.StatusName(status)}")
        
        return schedule 