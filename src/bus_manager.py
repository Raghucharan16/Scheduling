#!/usr/bin/env python3
"""
Bus Management System
Handles bus registry, location tracking, and day continuity management.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

@dataclass
class BusState:
    """Represents the current state of a bus"""
    bus_id: int
    home_depot: str  # 'A' or 'B'
    current_location: str  # 'A' or 'B'
    available_time: int  # Global minutes since Monday 00:00
    last_trip_day: int
    last_trip_route: Optional[str]
    total_trips: int
    total_revenue: float
    
    # NEW: Day continuity tracking
    needs_positioning: bool = False  # Whether bus needs specific positioning for next day
    positioning_target: Optional[str] = None  # Target depot for positioning
    day_assignments: Dict[int, int] = None  # Track assignments per day
    
    def __post_init__(self):
        if self.day_assignments is None:
            self.day_assignments = {}
    
    def is_available_at(self, global_time: int) -> bool:
        """Check if bus is available at given global time"""
        return global_time >= self.available_time
    
    def update_after_trip(self, trip_start: int, trip_day: int, route: str, 
                         epk: float, travel_h: float, charge_h: float):
        """Update bus state after completing a trip"""
        cycle_minutes = int((travel_h + charge_h) * 60)
        
        # Update location (bus ends at opposite depot)
        if route == "A-B":
            self.current_location = "B"
        else:  # "B-A"
            self.current_location = "A"
        
        # Update availability (trip start + full cycle)
        self.available_time = trip_start + cycle_minutes
        self.last_trip_day = trip_day
        self.last_trip_route = route
        self.total_trips += 1
        self.total_revenue += epk
        
        # Update day assignments tracking
        self.day_assignments[trip_day] = self.day_assignments.get(trip_day, 0) + 1
        
        # Clear positioning flags after assignment
        self.needs_positioning = False
        self.positioning_target = None

    def get_daily_trips(self, day: int) -> int:
        """Get number of trips assigned to this bus on a specific day"""
        return self.day_assignments.get(day, 0)
        
    def has_capacity_for_day(self, day: int, max_trips_per_day: int = 2) -> bool:
        """Check if bus has capacity for more trips on given day"""
        return self.get_daily_trips(day) < max_trips_per_day
        
    def mark_for_positioning(self, target_depot: str, reason: str = ""):
        """Mark bus for specific positioning needs"""
        self.needs_positioning = True
        self.positioning_target = target_depot
        logging.getLogger(__name__).info(f"Bus {self.bus_id}: Marked for positioning at {target_depot} - {reason}")

class BusManager:
    """Manages fleet of buses with location tracking and scheduling"""
    
    def __init__(self, buses_at_a: int, buses_at_b: int):
        self.buses_at_a = buses_at_a
        self.buses_at_b = buses_at_b
        self.total_buses = buses_at_a + buses_at_b
        
        # Initialize bus registry
        self.buses: Dict[int, BusState] = {}
        self._initialize_fleet()
        
        # Scheduling parameters
        self.travel_h = 9.0
        self.charge_h = 2.0
        self.max_slack_h = 2.0  # Maximum slack time for EPK optimization
        
        # Day continuity tracking
        self.next_day_requirements = {}  # Track next day's early trip needs
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _initialize_fleet(self):
        """Initialize all buses at their home depots"""
        bus_id = 1
        
        # Buses 1 to buses_at_a start at depot A
        for i in range(self.buses_at_a):
            self.buses[bus_id] = BusState(
                bus_id=bus_id,
                home_depot='A',
                current_location='A',
                available_time=0,  # Available from Monday 00:00
                last_trip_day=-1,
                last_trip_route=None,
                total_trips=0,
                total_revenue=0.0
            )
            bus_id += 1
        
        # Buses (buses_at_a + 1) to total start at depot B  
        for i in range(self.buses_at_b):
            self.buses[bus_id] = BusState(
                bus_id=bus_id,
                home_depot='B',
                current_location='B',
                available_time=0,  # Available from Monday 00:00
                last_trip_day=-1,
                last_trip_route=None,
                total_trips=0,
                total_revenue=0.0
            )
            bus_id += 1
    
    def get_available_buses(self, origin: str, global_time: int, 
                           max_slack_minutes: int = None) -> List[int]:
        """
        Get buses available at origin depot at given time (with optional slack)
        
        Args:
            origin: Depot location ('A' or 'B')
            global_time: Time in global minutes
            max_slack_minutes: Maximum slack time to wait for better buses
        
        Returns:
            List of bus IDs available at the origin
        """
        if max_slack_minutes is None:
            max_slack_minutes = int(self.max_slack_h * 60)
        
        available_buses = []
        
        for bus_id, bus in self.buses.items():
            # Bus must be at correct location
            if bus.current_location != origin:
                continue
            
            # Check if bus is available within slack time
            if bus.available_time <= global_time + max_slack_minutes:
                available_buses.append(bus_id)
        
        return available_buses
    
    def find_best_bus_for_trip(self, origin: str, global_time: int, 
                              epk: float, max_slack_minutes: int = None) -> Optional[Tuple[int, int]]:
        """
        Find the best bus for a trip considering EPK optimization and slack time
        
        Returns:
            Tuple of (bus_id, optimal_start_time) or None if no bus available
        """
        available_buses = self.get_available_buses(origin, global_time, max_slack_minutes)
        
        if not available_buses:
            return None
        
        best_bus = None
        best_start_time = None
        best_score = -1
        
        for bus_id in available_buses:
            bus = self.buses[bus_id]
            
            # Calculate earliest possible start time
            earliest_start = max(global_time, bus.available_time)
            
            # If bus needs to wait, calculate slack benefit
            wait_time = earliest_start - global_time
            
            # Score based on: EPK value, utilization, and minimal wait time
            utilization_bonus = 1000 if bus.total_trips < 2 else 500  # Encourage underused buses
            epk_bonus = epk * 10
            wait_penalty = wait_time * 2  # Penalty for delaying trip
            
            score = epk_bonus + utilization_bonus - wait_penalty
            
            if score > best_score:
                best_score = score
                best_bus = bus_id
                best_start_time = earliest_start
        
        return (best_bus, best_start_time) if best_bus else None
    
    def assign_trip(self, bus_id: int, trip_start: int, trip_day: int, 
                   route: str, epk: float) -> bool:
        """
        Assign a trip to a bus and update its state
        
        Returns:
            True if assignment successful, False otherwise
        """
        if bus_id not in self.buses:
            self.logger.error(f"Bus {bus_id} not found in fleet")
            return False
        
        bus = self.buses[bus_id]
        
        # Validate assignment
        origin = route.split('-')[0]
        if bus.current_location != origin:
            self.logger.error(f"Bus {bus_id} not at origin {origin} (currently at {bus.current_location})")
            return False
        
        if not bus.is_available_at(trip_start):
            self.logger.error(f"Bus {bus_id} not available at {trip_start} (available at {bus.available_time})")
            return False
        
        # Update bus state
        bus.update_after_trip(trip_start, trip_day, route, epk, self.travel_h, self.charge_h)
        
        self.logger.info(f"Assigned trip to Bus {bus_id}: {route} at day {trip_day}, "
                        f"now at {bus.current_location}, available at {bus.available_time}")
        
        return True
    
    def get_end_states(self, end_day: int = 3) -> List[Dict]:
        """
        Get end states of all buses after specified day for next phase
        
        Args:
            end_day: Last day of current phase (0-6, where 0=Monday)
        
        Returns:
            List of end state dictionaries compatible with existing solvers
        """
        end_states = []
        end_of_day = (end_day + 1) * 1440  # Start of next day in global minutes
        
        for bus_id in range(1, self.total_buses + 1):
            bus = self.buses[bus_id]
            
            # If bus's last availability is before end of phase, it's ready at start of next phase
            if bus.available_time < end_of_day:
                availability = end_of_day
            else:
                availability = bus.available_time
            
            end_states.append({
                "loc": bus.current_location,
                "avail": availability
            })
        
        return end_states
    
    def reset_for_phase(self, end_states: List[Dict]):
        """Reset bus states for next scheduling phase"""
        for i, state in enumerate(end_states):
            bus_id = i + 1
            if bus_id in self.buses:
                self.buses[bus_id].current_location = state["loc"]
                self.buses[bus_id].available_time = state["avail"]
    
    def get_fleet_summary(self) -> Dict:
        """Get comprehensive fleet status summary"""
        summary = {
            "total_buses": self.total_buses,
            "buses_at_a": self.buses_at_a,
            "buses_at_b": self.buses_at_b,
            "fleet_utilization": {},
            "location_distribution": {"A": 0, "B": 0},
            "total_trips": 0,
            "total_revenue": 0.0
        }
        
        for bus in self.buses.values():
            # Location distribution
            summary["location_distribution"][bus.current_location] += 1
            
            # Fleet stats
            summary["total_trips"] += bus.total_trips
            summary["total_revenue"] += bus.total_revenue
            
            # Individual bus utilization
            summary["fleet_utilization"][f"bus_{bus.bus_id}"] = {
                "home": bus.home_depot,
                "location": bus.current_location,
                "trips": bus.total_trips,
                "revenue": round(bus.total_revenue, 2),
                "available_day": bus.available_time // 1440,
                "available_time": f"{(bus.available_time % 1440)//60:02d}:{(bus.available_time % 1440)%60:02d}"
            }
        
        return summary
    
    def validate_constraints(self) -> List[str]:
        """Validate all buses meet operational constraints"""
        violations = []
        
        for bus in self.buses.values():
            # Check if bus is stranded (can't get home easily)
            if bus.current_location != bus.home_depot:
                violations.append(f"Bus {bus.bus_id} stranded at {bus.current_location} "
                                f"(home: {bus.home_depot})")
            
            # Check for overutilization (more than reasonable trips per week)
            if bus.total_trips > 14:  # Max 2 trips per day * 7 days
                violations.append(f"Bus {bus.bus_id} overutilized: {bus.total_trips} trips")
        
        return violations
    
    def optimize_day_transitions(self, day: int, trips: List[Dict], 
                                epk_threshold: float = 100.0) -> List[Dict]:
        """
        Optimize transitions between days for better EPK capture
        
        Args:
            day: Current day (0-6)
            trips: Available trips for the day
            epk_threshold: Minimum EPK to consider for slack optimization
        
        Returns:
            Optimized trip assignments
        """
        optimized_assignments = []
        high_value_trips = [t for t in trips if t["epk"] >= epk_threshold]
        regular_trips = [t for t in trips if t["epk"] < epk_threshold]
        
        # Process high-value trips first with slack optimization
        for trip in sorted(high_value_trips, key=lambda x: x["epk"], reverse=True):
            global_time = day * 1440 + trip["start"]
            origin = trip["origin"]
            
            # Allow more slack for high-value trips
            max_slack = int(self.max_slack_h * 60) if trip["epk"] >= epk_threshold else 0
            
            best_assignment = self.find_best_bus_for_trip(
                origin, global_time, trip["epk"], max_slack
            )
            
            if best_assignment:
                bus_id, optimal_start = best_assignment
                
                # Adjust trip start time if beneficial
                if optimal_start != global_time:
                    trip["start"] = optimal_start % 1440  # Keep within day
                    trip["optimized"] = True
                
                if self.assign_trip(bus_id, optimal_start, day, trip["route"], trip["epk"]):
                    optimized_assignments.append({
                        "bus": bus_id,
                        "trip": trip,
                        "optimized_start": optimal_start
                    })
        
        # Process regular trips with standard scheduling
        for trip in regular_trips:
            global_time = day * 1440 + trip["start"]
            origin = trip["origin"]
            
            available_buses = self.get_available_buses(origin, global_time, 0)  # No slack
            
            if available_buses:
                # Choose bus with lowest utilization
                best_bus = min(available_buses, 
                             key=lambda b: self.buses[b].total_trips)
                
                if self.assign_trip(best_bus, global_time, day, trip["route"], trip["epk"]):
                    optimized_assignments.append({
                        "bus": best_bus,
                        "trip": trip,
                        "optimized_start": global_time
                    })
        
        return optimized_assignments 

    def analyze_next_day_requirements(self, day: int, all_trips: List[Dict]) -> Dict:
        """
        Analyze next day's trip requirements to determine positioning needs
        
        Args:
            day: Current day (0-6)
            all_trips: All available trips across days
            
        Returns:
            Dict with positioning requirements for each depot
        """
        if day >= 6:  # Sunday
            next_day = 0  # Monday
        else:
            next_day = day + 1
        
        next_day_trips = [t for t in all_trips if t["day"] == next_day]
        
        # Focus on early morning trips (before 08:00) and high EPK trips
        early_trips = [t for t in next_day_trips if t["start"] < 480]  # Before 08:00
        high_epk_trips = [t for t in next_day_trips if t["epk"] >= 100]  # High value
        
        priority_trips = early_trips + high_epk_trips
        
        # Count requirements by depot
        depot_requirements = {"A": 0, "B": 0}
        for trip in priority_trips:
            origin = trip["route"].split('-')[0]
            depot_requirements[origin] += 1
        
        # Current distribution
        current_distribution = {"A": 0, "B": 0}
        for bus in self.buses.values():
            current_distribution[bus.current_location] += 1
        
        # Calculate positioning needs
        positioning_needs = {}
        for depot in ["A", "B"]:
            needed = depot_requirements[depot]
            available = current_distribution[depot]
            shortage = max(0, needed - available)
            
            if shortage > 0:
                positioning_needs[depot] = shortage
                self.logger.info(f"Day {day}: Need {shortage} buses positioned at depot {depot} for next day")
        
        self.next_day_requirements[next_day] = positioning_needs
        return positioning_needs

    def get_available_buses_enhanced(self, origin: str, global_time: int, 
                                   max_slack_minutes: int = None, 
                                   allow_cross_depot: bool = False) -> List[Tuple[int, bool]]:
        """
        Enhanced bus availability that supports cross-depot assignments
        
        Returns:
            List of (bus_id, is_cross_depot) tuples
        """
        if max_slack_minutes is None:
            max_slack_minutes = int(self.max_slack_h * 60)
        
        available_buses = []
        
        for bus_id, bus in self.buses.items():
            # Check availability within slack time
            if bus.available_time <= global_time + max_slack_minutes:
                
                if bus.current_location == origin:
                    # Same depot - direct assignment
                    available_buses.append((bus_id, False))
                elif allow_cross_depot:
                    # Cross depot - check if beneficial
                    # Could add depot-to-depot travel time here
                    available_buses.append((bus_id, True))
        
        return available_buses

    def find_best_bus_for_trip_enhanced(self, origin: str, global_time: int, trip: Dict,
                                      max_slack_minutes: int, constraints: Dict) -> Optional[Tuple[int, int, Dict]]:
        """
        Enhanced bus assignment with strict day continuity validation
        Ensures buses can only start trips from their current location
        """
        available_buses = []
        
        for bus_id, bus in self.buses.items():
            # CRITICAL: Day continuity check - bus must be at the trip's origin
            if bus.current_location != origin:
                self.logger.debug(f"Bus {bus_id}: REJECTED - at {bus.current_location}, trip needs {origin}")
                continue
                
            if bus.is_available_at(global_time):
                slack_needed = max(0, global_time - bus.available_time)
                
                if slack_needed <= max_slack_minutes:
                    # Calculate comprehensive score
                    score = self.calculate_assignment_score(bus_id, trip, global_time, constraints)
                    
                    assignment_info = {
                        "score": score,
                        "slack_needed": slack_needed,
                        "current_location": bus.current_location,
                        "continuity_valid": True,  # Already validated above
                        "route_preference": self.get_route_preference_info(bus_id, trip, constraints),
                        "cross_depot": False,  # Since we only allow same-depot assignments
                    }
                    
                    start_time = max(global_time, bus.available_time)
                    available_buses.append((bus_id, start_time, assignment_info))
                else:
                    self.logger.debug(f"Bus {bus_id}: REJECTED - needs {slack_needed}min slack (max: {max_slack_minutes}min)")
            else:
                self.logger.debug(f"Bus {bus_id}: REJECTED - not available until {bus.available_time} (need: {global_time})")
        
        if not available_buses:
            self.logger.warning(f"No buses available at depot {origin} for trip at {global_time}")
            return None
        
        # Sort by score (highest first) and select best bus
        available_buses.sort(key=lambda x: x[2]["score"], reverse=True)
        best_assignment = available_buses[0]
        
        bus_id, start_time, info = best_assignment
        self.logger.debug(f"Selected Bus {bus_id} (score: {info['score']}, slack: {info['slack_needed']}min)")
        
        return best_assignment

    def _calculate_enhanced_score(self, bus_id: int, trip: Dict, start_time: int, 
                                day: int, is_cross_depot: bool, constraints: Dict) -> float:
        """Calculate enhanced scoring for bus assignments"""
        bus = self.buses[bus_id]
        
        # Base scores
        epk_score = trip["epk"] * constraints.get("epk_weight", 1000)
        utilization_score = constraints.get("utilization_weight", 500000)
        
        # Utilization bonuses
        if bus.total_trips < constraints.get("min_trips_per_bus", 1):
            utilization_score *= 2
        
        # Daily capacity bonus
        if bus.has_capacity_for_day(day):
            utilization_score += 25000
        
        # Route preference scoring
        route_score = 0
        if constraints.get("route_preference_enabled", False):
            route_score = self._get_route_preference_score(bus_id, trip, day, constraints)
        
        # Cross-depot penalty (small to encourage same-depot when possible)
        cross_depot_penalty = 5000 if is_cross_depot else 0
        
        # Day positioning bonus
        positioning_bonus = 0
        if bus.needs_positioning:
            trip_destination = trip["route"].split('-')[1]
            if trip_destination == bus.positioning_target:
                positioning_bonus = constraints.get("day_positioning_weight", 100000)
        
        # Night coverage bonus
        night_bonus = 0
        trip_start = trip["start"]
        if (trip_start >= 1320 or trip_start <= 360):  # Night hours
            night_bonus = constraints.get("night_shift_bonus", 0)
        
        total_score = (epk_score + utilization_score + route_score + 
                      positioning_bonus + night_bonus - cross_depot_penalty)
        
        return total_score

    def _get_route_preference_score(self, bus_id: int, trip: Dict, day: int, constraints: Dict) -> float:
        """Get route preference score for enhanced assignments"""
        bus = self.buses[bus_id]
        route = trip["route"]
        trip_start = trip["start"]
        
        preference_score = 0
        
        # Prefer V-H during high-demand daytime hours (better EPK typically)
        if trip_start > 360 and trip_start < 1200:  # 06:00-20:00
            if route == "B-A":  # V-H direction
                preference_score += constraints.get("route_preference_bonus", 50000)
        
        # Alternating route bonus
        if bus.last_trip_route and bus.last_trip_route != route:
            preference_score += 25000
        
        # EPK-based preference
        if trip["epk"] >= 100:  # Premium trips
            preference_score += trip["epk"] * 500
        
        return preference_score

    def _get_route_preference_info(self, bus_id: int, trip: Dict, day: int) -> Dict:
        """Get route preference information for assignment details"""
        bus = self.buses[bus_id]
        
        return {
            "preferred_route": trip["route"] == "B-A" and 360 < trip["start"] < 1200,
            "alternating_route": bus.last_trip_route != trip["route"] if bus.last_trip_route else False,
            "high_epk": trip["epk"] >= 100,
            "positioning_match": bus.positioning_target == trip["route"].split('-')[1] if bus.needs_positioning else False
        }

    def mark_buses_for_positioning(self, day: int, all_trips: List[Dict]):
        """Mark buses that need specific positioning for next day's early/high-value trips"""
        positioning_needs = self.analyze_next_day_requirements(day, all_trips)
        
        if not positioning_needs:
            return
        
        # Find buses that should be positioned
        buses_needing_positioning = []
        
        for depot, shortage in positioning_needs.items():
            if shortage > 0:
                # Find buses currently not at this depot that could be repositioned
                other_depot = "B" if depot == "A" else "A"
                candidate_buses = [
                    bus for bus in self.buses.values() 
                    if bus.current_location == other_depot and not bus.needs_positioning
                ]
                
                # Sort by lowest utilization (these buses can afford positioning trips)
                candidate_buses.sort(key=lambda b: b.total_trips)
                
                # Mark buses for positioning
                for i in range(min(shortage, len(candidate_buses))):
                    bus = candidate_buses[i]
                    bus.mark_for_positioning(depot, f"early/high-value trips tomorrow")
                    buses_needing_positioning.append(bus.bus_id)
        
        if buses_needing_positioning:
            self.logger.info(f"Day {day}: Marked buses {buses_needing_positioning} for positioning") 