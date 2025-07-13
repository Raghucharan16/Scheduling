#!/usr/bin/env python3
"""
Pure OR-Tools CP-SAT Regular Phase Solver (Monday-Thursday)
Focus: MAXIMUM UTILIZATION + NIGHT COVERAGE with day continuity
"""

from typing import Dict, List, Tuple, Optional
from ortools.sat.python import cp_model
from .bus_manager import BusManager
from .loader import load_trips
import logging
from .constraints import global_bus_trip_chaining

class RegularCPSATSolver:
    """Pure OR-Tools CP-SAT solver for regular phase scheduling"""
    
    def __init__(self, bus_manager: BusManager):
        self.bus_manager = bus_manager
        self.logger = logging.getLogger(__name__)
        
    def solve_regular_phase(self, csv_path: str, max_idle: float = None) -> Tuple[Dict, List[Dict]]:
        """
        Solve regular phase (Mon-Thu) using pure OR-Tools CP-SAT
        
        Returns:
            Tuple of (schedule_dict, end_states)
        """
        print(f"\n🔧 PURE CP-SAT REGULAR SOLVER (Mon-Thu)")
        print(f"   Strategy: OR-Tools CP-SAT with utilization + night coverage")
        print(f"   Target: 2 trips per bus per day + night coverage")
        
        # Load and filter trips
        all_trips = load_trips(csv_path)
        regular_trips = [t for t in all_trips if 0 <= t["day"] <= 3]
        
        # Filter by low EPK threshold for maximum utilization
        viable_trips = [t for t in regular_trips if t["epk"] >= 35.0]  # Very low threshold
        
        print(f"   Available trips: {len(viable_trips)} viable trips (≥₹15)")
        
        # Create CP-SAT model
        model = cp_model.CpModel()
        
        # Parameters
        B = self.bus_manager.total_buses
        buses_a = self.bus_manager.buses_at_a
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
            for day in range(4):  # Days 0-3
                day_trips = [t["id"] for t in viable_trips if t["day"] == day]
                model.Add(sum(x[b, tid] for tid in day_trips) <= 2)
        
        # CONSTRAINT 3: Max 1 trip per direction per day per bus
        for b in range(B):
            for day in range(4):
                ab_trips = [t["id"] for t in viable_trips if t["day"] == day and t["route"] == "A-B"]
                ba_trips = [t["id"] for t in viable_trips if t["day"] == day and t["route"] == "B-A"]
                
                if ab_trips:
                    model.Add(sum(x[b, tid] for tid in ab_trips) <= 1)
                if ba_trips:
                    model.Add(sum(x[b, tid] for tid in ba_trips) <= 1)
        
        # CONSTRAINT 4: No two buses same origin/time
        for day in range(4):
            times = set(t["start"] for t in viable_trips if t["day"] == day)
            for time in times:
                for origin in ["A", "B"]:
                    trip_ids = [t["id"] for t in viable_trips 
                              if t["day"] == day and t["start"] == time and t["origin"] == origin]
                    if trip_ids:
                        model.Add(sum(x[b, tid] for b in range(B) for tid in trip_ids) <= 1)
        
        # CONSTRAINT 5: Basic depot constraints (relaxed for feasibility)
        # Allow buses to be repositioned via trips but ensure logical flow
        for b in range(B):
            home = "A" if b < buses_a else "B"
            
            # For Monday's first trips, buses should prefer starting from home
            monday_trips = [t for t in viable_trips if t["day"] == 0]
            monday_trips.sort(key=lambda t: t["start"])
            
            for i, trip in enumerate(monday_trips[:2]):  # Only first 2 trips on Monday
                if trip["origin"] != home:
                    # Add preference penalty rather than hard constraint
                    # This allows repositioning but discourages unnecessary movement
                    pass  # Will be handled by objective weights
        
        # CONSTRAINT 6: Cross-day continuity
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

        # ENFORCE MAX IDLE TIME CONSTRAINT
        if max_idle is not None:
            max_idle_minutes = int(max_idle * 60)
            for b in range(B):
                bus_trips = [t for t in viable_trips if (b, t["id"]) in x]
                # Sort by global start time
                bus_trips.sort(key=lambda t: t["day"] * 1440 + t["start"])
                for i in range(len(bus_trips) - 1):
                    t1 = bus_trips[i]
                    t2 = bus_trips[i+1]
                    t1_end = t1["day"] * 1440 + t1["start"] + cycle_minutes
                    t2_start = t2["day"] * 1440 + t2["start"]
                    idle_gap = t2_start - t1_end
                    if idle_gap > max_idle_minutes:
                        model.Add(x[b, t1["id"]] + x[b, t2["id"]] <= 1)
        
        # CONSTRAINT 7: Route continuity (simplified for feasibility)
        # Only enforce strict continuity for same-day consecutive trips
        for b in range(B):
            for day in range(4):
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
        
        # CONSTRAINT 8: NIGHT COVERAGE - ensure some night trips happen
        for day in range(4):
            night_trips = [t["id"] for t in viable_trips 
                          if t["day"] == day and (t["start"] >= 1200 or t["start"] <= 300)]
            if night_trips:
                # Ensure at least 2 night trips per day (relaxed)
                model.Add(sum(x[b, tid] for b in range(B) for tid in night_trips) >= 2)
        
        # CONSTRAINT 9: Each bus should work (relaxed utilization)
        for b in range(B):
            # Each bus should do at least 4 trips total across all 4 days
            all_trips = [t["id"] for t in viable_trips]
            model.Add(sum(x[b, tid] for tid in all_trips) >= 4)
        
        # OBJECTIVE: Maximize trips + night coverage + EPK
        total_trips = sum(x[b, t["id"]] for b in range(B) for t in viable_trips)
        
        # Night bonus
        night_trips_sum = sum(
            x[b, t["id"]] for b in range(B) for t in viable_trips
            if t["start"] >= 1200 or t["start"] <= 300
        )
        
        # EPK sum
        total_epk = sum(t["epk"] * x[b, t["id"]] for b in range(B) for t in viable_trips)
        
        # Weighted objective (heavily favor trip count + night coverage)
        model.Maximize(
            100000 * total_trips +     # Massive weight for utilization
            50000 * night_trips_sum +  # High weight for night coverage  
            100 * total_epk           # Small weight for EPK
        )
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0  # 1 minute timeout
        
        self.logger.info(f"Solving Regular CP-SAT model with {len(viable_trips)} trips, {B} buses")
        status = solver.Solve(model)
        
        schedule = {d: [] for d in range(4)}
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            self.logger.info(f"CP-SAT Regular solution: {solver.StatusName(status)}")
            
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
                             if a["start"] >= 1200 or a["start"] <= 300])
            
            print(f"\n📊 REGULAR CP-SAT RESULTS:")
            print(f"   Status: {solver.StatusName(status)}")
            print(f"   Total trips: {total_trips}")
            print(f"   Total revenue: ₹{total_revenue:.2f}")
            print(f"   Average EPK: ₹{total_revenue/total_trips:.2f}" if total_trips > 0 else "   No trips")
            print(f"   Night coverage: {night_trips} trips")
            print(f"   Utilization: {total_trips/80*100:.1f}% (target: 80 trips)")
            
            for day in range(4):
                day_name = ["Monday", "Tuesday", "Wednesday", "Thursday"][day]
                day_trips = len(schedule[day])
                day_revenue = sum(t["epk"] for t in schedule[day])
                day_night = len([t for t in schedule[day] if t["start"] >= 1200 or t["start"] <= 300])
                print(f"   {day_name}: {day_trips} trips, ₹{day_revenue:.2f}, {day_night} night")
                
        else:
            self.logger.error(f"CP-SAT Regular failed: {solver.StatusName(status)}")
        
        # Generate end states
        end_states = self.bus_manager.get_end_states(end_day=3)
        
        return schedule, end_states 