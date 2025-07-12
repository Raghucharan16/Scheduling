#!/usr/bin/env python3
"""
main.py

Enhanced Electric Bus Scheduling System with proper bus tracking and day continuity.
Uses the new enhanced solvers with BusManager for better fleet management.
"""
import json
import os
from integrator import CPSATIntegrator
from src.schedule_visualizer import BusScheduleVisualizer

def get_user_inputs():
    """Get configuration - using defaults for testing"""
    print("🚌 Enhanced Electric Bus Scheduling System")
    print("=" * 55)
    
    # Use default parameters for testing the enhanced system
    buses_a = int(input("Enter the number of buses at depot A: "))  # Buses 1-5 at depot A
    buses_b = int(input("Enter the number of buses at depot B: "))  # Buses 6-10 at depot B  
    travel_h = float(input("Enter the travel time: "))  # 9 hours travel
    charge_h = float(input("Enter the charge time: "))  # 2 hours charge
    max_idle = float(input("Enter the max idle time: "))  # 2 hours max idle
    
    print(f"Using default test configuration:")
    print(f"   • Buses at depot A: {buses_a}")
    print(f"   • Buses at depot B: {buses_b}") 
    print(f"   • Travel time: {travel_h}h")
    print(f"   • Charge time: {charge_h}h")
    print(f"   • Max idle: {max_idle}h")
    
    return buses_a, buses_b, travel_h, charge_h, max_idle

def run_enhanced_scheduling():
    """Run the enhanced scheduling system"""
    csv_path = "cache/epk_v-h-week-daywise.csv"
    
    # Get configuration
    buses_a, buses_b, travel_h, charge_h, max_idle = get_user_inputs()
    
    print(f"\n🔧 Enhanced Configuration:")
    print(f"   Fleet Mapping:")
    print(f"     • Buses 1-{buses_a}: Home depot A")
    print(f"     • Buses {buses_a+1}-{buses_a+buses_b}: Home depot B")
    print(f"   Cycle: {travel_h}h travel + {charge_h}h charge = {travel_h + charge_h}h")
    print(f"   Max idle: {max_idle}h")
    print(f"   Strategy: Two-phase optimization (Utilization + EPK)")

    try:
        print(f"\n🚀 Running Enhanced Two-Phase Scheduling...")
        
        # Initialize pure CP-SAT integrator
        integrator = CPSATIntegrator(
            buses_at_a=buses_a,
            buses_at_b=buses_b,
            travel_h=travel_h,
            charge_h=charge_h
        )
        
        # Run complete scheduling
        results = integrator.run_complete_scheduling(csv_path)
        
        schedule = results["complete_schedule"]
        statistics = results["statistics"]
        
        print(f"\n🔗 Phase 3: Converting to legacy format...")
        # Convert to legacy format for visualization
        legacy_schedule = convert_to_legacy_format(schedule, travel_h)
        
        # Save legacy format
        with open("outputs/final_schedule.json", "w") as f:
            json.dump(legacy_schedule, f, indent=2)
        print("   ✅ Legacy format saved to outputs/final_schedule.json")
        
        print(f"\n🎨 Phase 4: Generating visual dashboard...")
        # Generate HTML visualization
        try:
            visualizer = BusScheduleVisualizer(legacy_schedule)
            visualizer.generate_html("outputs/bus_schedule.html")
            print("   ✅ Interactive dashboard saved to outputs/bus_schedule.html")
        except Exception as viz_error:
            print(f"   ⚠️  Visualization generation failed: {viz_error}")
            print("   📊 Schedule data is still available in JSON format")

        # Enhanced summary
        print(f"\n📊 ENHANCED SCHEDULE SUMMARY")
        print("=" * 40)
        print(f"🚌 Fleet Performance:")
        print(f"   • Total buses: {integrator.bus_manager.total_buses}")
        print(f"   • Active buses: {statistics['fleet']['active_buses']}")
        print(f"   • Trips per bus: {statistics['fleet']['trips_per_bus']:.1f} (avg)")
        
        print(f"\n📈 Overall Results:")
        print(f"   • Total trips: {statistics['total_trips']}")
        print(f"   • Total revenue: ₹{statistics['total_revenue']:.2f}")
        print(f"   • Average EPK: ₹{statistics['avg_epk']:.2f}")
        print(f"   • Overall utilization: {statistics['total_trips']/140*100:.1f}%")
        
        print(f"\n🔧 Regular Phase (Mon-Thu):")
        print(f"   • Focus: MAXIMUM UTILIZATION + NIGHT COVERAGE")
        print(f"   • Trips: {statistics['regular']['trips']} (₹{statistics['regular']['revenue']:.2f})")
        print(f"   • Utilization: {statistics['regular']['trips']/80*100:.1f}%")
        print(f"   • Average EPK: ₹{statistics['regular']['avg_epk']:.2f}")
        print(f"   • Night coverage: {statistics['regular']['night_trips']} trips")
        
        print(f"\n🎯 Peak Phase (Fri-Sun):")
        print(f"   • Focus: EPK OPTIMIZATION + PREMIUM NIGHT")
        print(f"   • Trips: {statistics['peak']['trips']} (₹{statistics['peak']['revenue']:.2f})")
        print(f"   • Utilization: {statistics['peak']['trips']/60*100:.1f}%")
        print(f"   • Average EPK: ₹{statistics['peak']['avg_epk']:.2f}")
        print(f"   • Premium trips (≥₹95): {statistics['peak']['premium_trips']}")
        print(f"   • Night coverage: {statistics['peak']['night_trips']} trips")

        # Daily breakdown
        print(f"\n📅 Daily Breakdown:")
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for d in range(7):
            day_trips = len(schedule[d])
            day_revenue = sum(t["epk"] for t in schedule[d])
            phase = "Backbone" if d < 4 else "Peak"
            print(f"   {days[d]:<9} ({phase:<8}): {day_trips:2d} trips, ₹{day_revenue:6.1f}")

        print(f"\n✨ ENHANCED SCHEDULING COMPLETE!")
        print("📁 Generated files:")
        print("   • outputs/enhanced_schedule.json - Complete enhanced schedule")
        print("   • outputs/final_schedule.json - Legacy format schedule")
        if os.path.exists("outputs/bus_schedule.html"):
            print("   • outputs/bus_schedule.html - Interactive dashboard")
        
        # Success assessment
        success_criteria = {
            "High regular utilization": statistics['success']['regular'],
            "Good peak utilization": statistics['success']['peak'],
            "Night coverage": statistics['success']['night'],
            "Premium capture": statistics['success']['premium']
        }
        
        print(f"\n🏆 Success Assessment:")
        for criterion, met in success_criteria.items():
            status = "✅ PASS" if met else "⚠️ REVIEW"
            print(f"   • {criterion}: {status}")
        
        overall_success = statistics['success']['overall']
        if overall_success:
            print(f"\n🎉 Pure CP-SAT system achieved all optimization targets!")
        else:
            print(f"\n📋 Pure CP-SAT system working - check individual criteria above.")
            
        return True

    except Exception as e:
        print(f"\n❌ Error during enhanced scheduling: {str(e)}")
        print("Please check your input parameters and try again.")
        import traceback
        traceback.print_exc()
        return False

def convert_to_legacy_format(enhanced_schedule, travel_h):
    """Convert enhanced schedule format to legacy format for visualization"""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    legacy_schedule = {}
    
    for day_idx in range(7):
        day_name = days[day_idx]
        day_trips = enhanced_schedule[day_idx]
        
        assignments = []
        for trip in day_trips:
            # Calculate timing
            start_minutes = trip["start"]
            start_h = start_minutes // 60
            start_m = start_minutes % 60
            
            # Calculate mid and end times
            mid_minutes = start_minutes + int(travel_h * 60 * 0.5)  # 50% through trip
            mid_h = (mid_minutes % 1440) // 60
            mid_m = (mid_minutes % 1440) % 60
            
            end_minutes = start_minutes + int(travel_h * 60)
            end_h = (end_minutes % 1440) // 60
            end_m = (end_minutes % 1440) % 60
            
            assignments.append({
                "busNumber": trip["bus"],
                "trip": {
                    "route": trip["route"],
                    "startTime": f"{start_h:02d}:{start_m:02d}:00",
                    "midPointTime": f"{mid_h:02d}:{mid_m:02d}:00",
                    "endTime": f"{end_h:02d}:{end_m:02d}:00",
                    "epk": round(trip["epk"], 2),
                    "origin": trip["origin"],
                    "day": day_idx
                }
            })
        
        # Sort by start time
        assignments.sort(key=lambda x: x["trip"]["startTime"])
        
        legacy_schedule[day_name] = {
            "assignments": assignments,
            "feasible": True,
            "totalTrips": len(assignments),
            "totalRevenue": round(sum(a["trip"]["epk"] for a in assignments), 2)
        }
    
    return legacy_schedule

if __name__ == "__main__":
    success = run_enhanced_scheduling()
    exit(0 if success else 1)
