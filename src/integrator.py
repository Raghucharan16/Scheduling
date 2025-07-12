#!/usr/bin/env python3
"""
Pure OR-Tools CP-SAT Integration System
Combines regular and peak CP-SAT solvers with no heuristic methods
"""

from typing import Dict, List, Tuple
from .bus_manager import BusManager
from .regular_solver import RegularCPSATSolver
from .peak_solver import PeakCPSATSolver
import logging
import os

class CPSATIntegrator:
    """Pure CP-SAT integration system"""
    
    def __init__(self, buses_at_a: int, buses_at_b: int, travel_h: float, charge_h: float):
        self.bus_manager = BusManager(buses_at_a, buses_at_b)
        self.bus_manager.travel_h = travel_h
        self.bus_manager.charge_h = charge_h
        
        self.regular_solver = RegularCPSATSolver(self.bus_manager)
        self.peak_solver = PeakCPSATSolver(self.bus_manager)
        
        self.logger = logging.getLogger(__name__)
        
    def run_complete_scheduling(self, csv_path: str) -> Dict:
        """
        Run complete two-phase CP-SAT scheduling
        
        Returns:
            Complete scheduling results
        """
        print(f"\n🚀 PURE CP-SAT TWO-PHASE SCHEDULING SYSTEM")
        print(f"======================================================================")
        print(f"📋 Configuration:")
        print(f"   Fleet: {self.bus_manager.total_buses} buses ({self.bus_manager.buses_at_a} at A, {self.bus_manager.buses_at_b} at B)")
        print(f"   Cycle: {self.bus_manager.travel_h}h travel + {self.bus_manager.charge_h}h charge")
        print(f"   Strategy: Pure OR-Tools CP-SAT optimization (no heuristics)")
        
        # Phase 1: Regular scheduling (Mon-Thu) with utilization + night coverage
        regular_schedule, end_states = self.regular_solver.solve_regular_phase(csv_path)
        
        # Phase 2: Peak scheduling (Fri-Sun) with EPK + night optimization
        peak_schedule = self.peak_solver.solve_peak_phase(csv_path, end_states)
        
        # Combine results
        complete_schedule = {}
        complete_schedule.update(regular_schedule)
        complete_schedule.update(peak_schedule)
        
        # Calculate final statistics
        all_assignments = []
        for day_schedule in complete_schedule.values():
            all_assignments.extend(day_schedule)
        
        regular_assignments = []
        for day in range(4):
            if day in regular_schedule:
                regular_assignments.extend(regular_schedule[day])
        
        peak_assignments = []
        for day in range(4, 7):
            if day in peak_schedule:
                peak_assignments.extend(peak_schedule[day])
        
        # Statistics
        total_trips = len(all_assignments)
        total_revenue = sum(a["epk"] for a in all_assignments)
        avg_epk = total_revenue / total_trips if total_trips > 0 else 0
        
        regular_trips = len(regular_assignments)
        regular_revenue = sum(a["epk"] for a in regular_assignments)
        regular_avg_epk = regular_revenue / regular_trips if regular_trips > 0 else 0
        
        peak_trips = len(peak_assignments)
        peak_revenue = sum(a["epk"] for a in peak_assignments)
        peak_avg_epk = peak_revenue / peak_trips if peak_trips > 0 else 0
        
        # Night coverage analysis
        total_night_trips = len([a for a in all_assignments 
                               if a["start"] >= 1320 or a["start"] <= 240])
        regular_night_trips = len([a for a in regular_assignments 
                                 if a["start"] >= 1200 or a["start"] <= 300])
        peak_night_trips = len([a for a in peak_assignments 
                              if a["start"] >= 1320 or a["start"] <= 240])
        
        # Premium analysis
        premium_trips = len([a for a in all_assignments if a["epk"] >= 95.0])
        peak_premium_trips = len([a for a in peak_assignments if a["epk"] >= 95.0])
        
        print(f"\n🏆 PURE CP-SAT SYSTEM FINAL RESULTS")
        print(f"==================================================")
        print(f"📊 Overall Performance:")
        print(f"   • Total trips: {total_trips} per week")
        print(f"   • Total revenue: ₹{total_revenue:.2f}")
        print(f"   • Average EPK: ₹{avg_epk:.2f}")
        print(f"   • Overall utilization: {total_trips/140*100:.1f}%")
        print(f"   • Night coverage: {total_night_trips} trips")
        print(f"   • Premium trips (≥₹95): {premium_trips}")
        
        print(f"\n🔧 Regular Phase (Mon-Thu):")
        print(f"   • Trips: {regular_trips} (₹{regular_revenue:.2f})")
        print(f"   • Utilization: {regular_trips/80*100:.1f}%")
        print(f"   • Average EPK: ₹{regular_avg_epk:.2f}")
        print(f"   • Night coverage: {regular_night_trips} trips")
        
        print(f"\n🎯 Peak Phase (Fri-Sun):")
        print(f"   • Trips: {peak_trips} (₹{peak_revenue:.2f})")
        print(f"   • Utilization: {peak_trips/60*100:.1f}%")
        print(f"   • Average EPK: ₹{peak_avg_epk:.2f}")
        print(f"   • Night coverage: {peak_night_trips} trips")
        print(f"   • Premium trips (≥₹95): {peak_premium_trips}")
        
        # Fleet utilization
        active_buses = len(set(a["bus"] for a in all_assignments))
        trips_per_bus = total_trips / active_buses if active_buses > 0 else 0
        
        print(f"\n🚌 Fleet Utilization:")
        print(f"   • Active buses: {active_buses}/{self.bus_manager.total_buses}")
        print(f"   • Trips per bus: {trips_per_bus:.1f} (avg)")
        
        # Success assessment
        regular_success = regular_trips >= 60  # At least 75% of 80 trips
        peak_success = peak_trips >= 45  # At least 75% of 60 trips
        night_success = total_night_trips >= 20  # Good night coverage
        premium_success = premium_trips >= 5  # Some premium capture
        
        print(f"\n✅ Success Criteria:")
        print(f"   • Regular utilization: {'✅ PASS' if regular_success else '❌ FAIL'}")
        print(f"   • Peak utilization: {'✅ PASS' if peak_success else '❌ FAIL'}")
        print(f"   • Night coverage: {'✅ PASS' if night_success else '❌ FAIL'}")
        print(f"   • Premium capture: {'✅ PASS' if premium_success else '❌ FAIL'}")
        
        overall_success = all([regular_success, peak_success, night_success, premium_success])
        print(f"\n🎯 OVERALL: {'🎉 SUCCESS' if overall_success else '⚠️ REVIEW NEEDED'}")
        
        if overall_success:
            print("   Pure CP-SAT system achieved all targets!")
        else:
            print("   Some criteria need attention - check individual phase results.")
        
        # Daily breakdown
        print(f"\n📅 Daily Breakdown:")
        for day in range(7):
            day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", 
                       "Friday", "Saturday", "Sunday"][day]
            phase = "Regular" if day < 4 else "Peak   "
            
            if day in complete_schedule:
                day_trips = len(complete_schedule[day])
                day_revenue = sum(t["epk"] for t in complete_schedule[day])
                day_night = len([t for t in complete_schedule[day] 
                               if (t["start"] >= 1320 or t["start"] <= 240) or 
                                  (day < 4 and (t["start"] >= 1200 or t["start"] <= 300))])
                day_premium = len([t for t in complete_schedule[day] if t["epk"] >= 95.0])
                
                print(f"   {day_name:9} ({phase}): {day_trips:2d} trips, ₹{day_revenue:7.1f}, "
                      f"{day_night} night, {day_premium} premium")
            else:
                print(f"   {day_name:9} ({phase}):  0 trips, ₹    0.0, 0 night, 0 premium")
        
        # Save enhanced schedule
        import json
        os.makedirs("outputs", exist_ok=True)
        
        # Convert schedule to serializable format
        serializable_schedule = {}
        for day, trips in complete_schedule.items():
            serializable_schedule[str(day)] = trips
        
        with open("outputs/enhanced_schedule.json", "w") as f:
            json.dump({
                "schedule": serializable_schedule,
                "statistics": {
                    "total_trips": total_trips,
                    "total_revenue": total_revenue,
                    "avg_epk": avg_epk,
                    "night_trips": total_night_trips,
                    "premium_trips": premium_trips,
                    "regular": {
                        "trips": regular_trips,
                        "revenue": regular_revenue,
                        "avg_epk": regular_avg_epk,
                        "night_trips": regular_night_trips
                    },
                    "peak": {
                        "trips": peak_trips,
                        "revenue": peak_revenue,
                        "avg_epk": peak_avg_epk,
                        "night_trips": peak_night_trips,
                        "premium_trips": peak_premium_trips
                    }
                }
            }, f, indent=2)
        
        print(f"\n💾 Enhanced schedule saved to outputs/enhanced_schedule.json")
        
        # Return structured results
        return {
            "complete_schedule": complete_schedule,
            "regular_schedule": regular_schedule,
            "peak_schedule": peak_schedule,
            "statistics": {
                "total_trips": total_trips,
                "total_revenue": total_revenue,
                "avg_epk": avg_epk,
                "night_trips": total_night_trips,
                "premium_trips": premium_trips,
                "regular": {
                    "trips": regular_trips,
                    "revenue": regular_revenue,
                    "avg_epk": regular_avg_epk,
                    "night_trips": regular_night_trips
                },
                "peak": {
                    "trips": peak_trips,
                    "revenue": peak_revenue,
                    "avg_epk": peak_avg_epk,
                    "night_trips": peak_night_trips,
                    "premium_trips": peak_premium_trips
                },
                "fleet": {
                    "active_buses": active_buses,
                    "trips_per_bus": trips_per_bus
                },
                "success": {
                    "regular": regular_success,
                    "peak": peak_success,
                    "night": night_success,
                    "premium": premium_success,
                    "overall": overall_success
                }
            }
        } 