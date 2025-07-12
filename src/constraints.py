from ortools.sat.python import cp_model

def forbid_forbidden_slots(x, trips):
    # we already skipped in loader.py
    pass

def no_same_origin_time(model: cp_model.CpModel, x, trips, B, day_lo, day_hi):
    # For each day in [day_lo, day_hi], no two buses depart same origin/time
    for d in range(day_lo, day_hi+1):
        times = set(t["start"] for t in trips if t["day"]==d)
        for s in times:
            for orig in ("A","B"):
                ids = [t["id"] for t in trips
                       if t["day"]==d and t["start"]==s and t["origin"]==orig]
                if not ids: continue
                model.Add(
                    sum(x[b,i] for b in range(B) for i in ids) <= 1
                )

def max_two_trips_per_day(model: cp_model.CpModel, x, trips, B, day_lo, day_hi):
    # Each bus ≤2 trips/day, and ≤1 per direction
    for b in range(B):
        for d in range(day_lo, day_hi+1):
            day_ids = [t["id"] for t in trips if t["day"]==d]
            model.Add(sum(x[b,i] for i in day_ids) <= 2)
            ab = [t["id"] for t in trips if t["day"]==d and t["route"]=="A-B"]
            ba = [t["id"] for t in trips if t["day"]==d and t["route"]=="B-A"]
            if ab: model.Add(sum(x[b,i] for i in ab) <= 1)
            if ba: model.Add(sum(x[b,i] for i in ba) <= 1)

def minimum_trips_constraint(model: cp_model.CpModel, x, trips, B, day_lo, day_hi):
    """
    Ensure each bus does at least 1 trip per day if possible.
    This is a soft constraint - we try to maximize trips anyway.
    """
    for b in range(B):
        for d in range(day_lo, day_hi+1):
            day_ids = [t["id"] for t in trips if t["day"]==d]
            if day_ids:
                # Encourage at least 1 trip per day per bus
                # This is handled in the objective function primarily
                pass

def single_trip_threshold(model: cp_model.CpModel, x, trips, B, threshold, day_lo, day_hi):
    # If a bus does exactly 1 trip on day d, it must be >= threshold
    # sum_trips and eq1 borrowed from prior code
    sum_trips = {}
    eq1       = {}
    for b in range(B):
        for d in range(day_lo, day_hi+1):
            sum_trips[b,d] = model.NewIntVar(0,2,f"sum[{b},{d}]")
            eq1[b,d]       = model.NewBoolVar(f"eq1[{b},{d}]")
            model.Add(sum_trips[b,d]==1).OnlyEnforceIf(eq1[b,d])
            model.Add(sum_trips[b,d]!=1).OnlyEnforceIf(eq1[b,d].Not())
            # link
            day_ids = [t["id"] for t in trips if t["day"]==d]
            model.Add(sum(x[b,i] for i in day_ids) == sum_trips[b,d])
            # enforce threshold
            high = [i for i,t in enumerate(trips) 
                    if t["day"]==d and t["epk"]>=threshold]
            model.Add(
                sum((1 if t["epk"]>=threshold else 0)*x[b,t["id"]] 
                    for t in trips if t["day"]==d)
                >= eq1[b,d]
            )

def max_idle_time_constraint(model: cp_model.CpModel, x, trips, B, day_lo, day_hi, 
                           travel_h, charge_h, max_idle_h):
    """
    Soft constraint: Limit idle time between consecutive trips within the same day.
    Returns penalty variables that can be added to objective function.
    """
    penalty_vars = []
    cycle = int((travel_h + charge_h) * 60)
    max_idle = int(max_idle_h * 60)
    
    for b in range(B):
        for d in range(day_lo, day_hi+1):
            day_trips = [t for t in trips if t["day"] == d]
            day_trips.sort(key=lambda t: t["start"])
            
            for i in range(len(day_trips)-1):
                t1, t2 = day_trips[i], day_trips[i+1]
                
                # Check if these trips can be done by same bus
                end_t1 = t1["start"] + cycle
                gap = t2["start"] - end_t1
                
                if gap > max_idle:
                    # Add penalty for violating idle time constraint
                    penalty = model.NewBoolVar(f"idle_penalty[{b},{t1['id']},{t2['id']}]")
                    # If both trips are done by same bus, activate penalty
                    model.Add(penalty >= x[b,t1["id"]] + x[b,t2["id"]] - 1)
                    penalty_vars.append(penalty)
                    
    return penalty_vars

def bus_parity_constraint(model: cp_model.CpModel, x, trips, B, day_lo, day_hi):
    """
    Soft constraint: Encourage buses to do A-B followed by B-A or vice versa.
    Returns variables that can be used in objective to encourage this pattern.
    """
    parity_bonus_vars = []
    
    for b in range(B):
        for d in range(day_lo, day_hi+1):
            ab_trips = [t["id"] for t in trips if t["day"]==d and t["route"]=="A-B"]
            ba_trips = [t["id"] for t in trips if t["day"]==d and t["route"]=="B-A"]
            
            if ab_trips and ba_trips:
                # Bonus for doing both A-B and B-A on same day
                parity_bonus = model.NewBoolVar(f"parity_bonus[{b},{d}]")
                
                # Activate bonus if bus does both types of trips
                model.Add(parity_bonus <= sum(x[b,i] for i in ab_trips))
                model.Add(parity_bonus <= sum(x[b,i] for i in ba_trips))
                
                parity_bonus_vars.append(parity_bonus)
                
    return parity_bonus_vars

def no_overlap(model: cp_model.CpModel, x, trips, B, day_lo, day_hi, travel_h, charge_h):
    # Use interval vars per day-range
    cycle = int((travel_h+charge_h)*60)
    for b in range(B):
        ivs=[]
        for t in trips:
            if day_lo <= t["day"] <= day_hi:
                start = t["day"]*1440 + t["start"]
                iv = model.NewOptionalIntervalVar(
                    start, cycle, start+cycle,
                    x[b,t["id"]], f"iv[{b},{t['id']}]"
                )
                ivs.append(iv)
        model.AddNoOverlap(ivs)

def depot_chaining(model, x, trips, B, buses_a):
    """
    For each bus b, forbid any first trip that departs from
    the wrong depot if that bus has no earlier trip to reposition it.
    Only applies to the trips actually in x[b,tid].
    """
    for b in range(B):
        home = "A" if b < buses_a else "B"
        for t in trips:
            # only enforce on the trips this solver actually schedules
            if (b, t["id"]) not in x:
                continue

            if t["origin"] != home:
                # find any earlier trip for this bus that could reposition it
                earlier = [
                    u for u in trips
                    if (u["day"] < t["day"]) or
                       (u["day"] == t["day"] and u["start"] < t["start"])
                ]
                # if there's absolutely no earlier trip var for this bus, forbid
                # also guard the earlier check itself by requiring x[b,u["id"]] exists
                possible = False
                for u in earlier:
                    if (b, u["id"]) in x:
                        possible = True
                        break
                if not possible:
                    model.Add(x[b, t["id"]] == 0)

def depot_chaining_with_state(model, x, trips, B, buses_a, end_states=None):
    """
    Enhanced depot chaining that considers end states from previous solver.
    If end_states is provided, use those locations. Otherwise, use home depots.
    """
    for b in range(B):
        if end_states and b < len(end_states):
            current_location = end_states[b]["loc"]
        else:
            current_location = "A" if b < buses_a else "B"
        
        # Sort trips by day and start time to process in chronological order
        bus_trips = [(t, (b, t["id"])) for t in trips if (b, t["id"]) in x]
        bus_trips.sort(key=lambda item: (item[0]["day"], item[0]["start"]))
        
        # Track location through the sequence
        for i, (t, trip_var_key) in enumerate(bus_trips):
            if t["origin"] != current_location:
                # Check if any earlier trip for this bus could have repositioned it
                repositioned = False
                temp_location = current_location
                
                for j in range(i):
                    prev_t, prev_var_key = bus_trips[j]
                    # If this earlier trip is selected, it changes location
                    if prev_var_key in x:
                        temp_location = 'B' if prev_t["origin"] == 'A' else 'A'
                        if temp_location == t["origin"]:
                            repositioned = True
                            # Add constraint: if this trip is selected, the repositioning trip must also be selected
                            model.Add(x[trip_var_key] <= x[prev_var_key])
                            break
                
                if not repositioned:
                    # No way to be at correct location, forbid this trip
                    model.Add(x[trip_var_key] == 0)

def enhanced_location_continuity(model, x, trips, B, travel_h, charge_h, end_states=None, buses_a=0):
    """
    More sophisticated location continuity constraint that tracks bus positions
    throughout the entire schedule period.
    """
    cycle = int((travel_h + charge_h) * 60)
    
    for b in range(B):
        # Initial location
        if end_states and b < len(end_states):
            initial_location = end_states[b]["loc"]
        else:
            initial_location = "A" if b < buses_a else "B"
        
        # Get all trips for this bus, sorted by global time
        bus_trips = []
        for t in trips:
            if (b, t["id"]) in x:
                global_start = t["day"] * 1440 + t["start"]
                bus_trips.append((t, global_start))
        
        bus_trips.sort(key=lambda item: item[1])  # Sort by global start time
        
        # For each trip, ensure location continuity
        current_location = initial_location
        
        for i, (trip, global_start) in enumerate(bus_trips):
            trip_var = x[b, trip["id"]]
            
            # If this trip requires a different origin than current location
            if trip["origin"] != current_location:
                # Find if there's a previous trip that can reposition the bus
                valid_repositioning = False
                
                for j in range(i):
                    prev_trip, prev_global_start = bus_trips[j]
                    prev_var = x[b, prev_trip["id"]]
                    
                    # Check if previous trip ends at the required location
                    prev_end_location = 'B' if prev_trip["origin"] == 'A' else 'A'
                    prev_end_time = prev_global_start + cycle
                    
                    if (prev_end_location == trip["origin"] and 
                        prev_end_time <= global_start):
                        # This previous trip can provide valid repositioning
                        valid_repositioning = True
                        # Add implication: if current trip is selected, repositioning trip must be selected
                        model.Add(trip_var <= prev_var)
                        break
                
                if not valid_repositioning:
                    # No valid repositioning possible, forbid this trip
                    model.Add(trip_var == 0)
            
            # Update current location (this will be used for next iteration)
            # Note: This is logical tracking, not a constraint
            if trip["origin"] == current_location:
                current_location = 'B' if trip["origin"] == 'A' else 'A'

def cross_day_chain(model, x, trips, B, travel_h, charge_h):
    """
    Enforce that if bus b does trip t1 on day d and trip t2 on day d+1,
    then t2.start_global >= t1.start_global + cycle.
    Only applies to those trips for which x[b,t_id] exists.
    """
    cycle = int((travel_h + charge_h) * 60)

    for b in range(B):
        for t1 in trips:
            # only if this trip was included in the model
            if (b, t1["id"]) not in x:
                continue

            end1 = t1["day"] * 1440 + t1["start"] + cycle
            next_day = (t1["day"] + 1) % 7

            for t2 in trips:
                if t2["day"] != next_day:
                    continue
                # skip pairs not present in this submodel
                if (b, t2["id"]) not in x:
                    continue

                # compute t2 global start in the *following* calendar day
                start2 = ((t1["day"] + 1) * 1440) + t2["start"]

                # if t2 would start before t1 finishes → forbid
                if start2 < end1:
                    model.Add(
                        x[b, t1["id"]] + x[b, t2["id"]] <= 1
                    )

def cross_week_continuity(model, x, trips, B, travel_h, charge_h):
    """
    Ensure continuity from Sunday to Monday (week rollover).
    This handles the constraint that Sunday's last trip should allow
    Monday's first trip to be feasible.
    """
    cycle = int((travel_h + charge_h) * 60)
    
    for b in range(B):
        sunday_trips = [t for t in trips if t["day"] == 6]  # Sunday
        monday_trips = [t for t in trips if t["day"] == 0]  # Monday
        
        for sun_trip in sunday_trips:
            if (b, sun_trip["id"]) not in x:
                continue
                
            # End time of Sunday trip in global minutes
            sun_end = 6 * 1440 + sun_trip["start"] + cycle
            
            for mon_trip in monday_trips:
                if (b, mon_trip["id"]) not in x:
                    continue
                    
                # Monday trip start time (in next week, so add 7*1440)
                mon_start = 7 * 1440 + mon_trip["start"]
                
                # If Monday trip starts before Sunday trip finishes, forbid
                if mon_start < sun_end:
                    model.Add(
                        x[b, sun_trip["id"]] + x[b, mon_trip["id"]] <= 1
                    )

def strict_bus_availability(model, x, trips, B, travel_h, charge_h, end_states=None, buses_a=0):
    """
    CRITICAL CONSTRAINT: Ensure no bus starts a trip before completing 
    previous trip + charging time.
    
    For each bus, if it does trip A and then trip B:
    - Trip A ends at: start_A + travel_time
    - Bus ready for next trip at: start_A + travel_time + charge_time
    - Trip B can only start at or after the ready time
    """
    cycle_minutes = int((travel_h + charge_h) * 60)
    travel_minutes = int(travel_h * 60)
    charge_minutes = int(charge_h * 60)
    
    print(f"🔧 Applying strict availability constraint:")
    print(f"   Travel time: {travel_minutes} minutes ({travel_h}h)")
    print(f"   Charge time: {charge_minutes} minutes ({charge_h}h)")
    print(f"   Total cycle: {cycle_minutes} minutes")
    
    for b in range(B):
        # Get initial availability from end_states if provided
        if end_states and isinstance(end_states, list) and b < len(end_states):
            initial_avail_time = end_states[b]["avail"]
            initial_location = end_states[b]["loc"]
        else:
            initial_avail_time = 0  # Available from start of week
            initial_location = "A" if b < buses_a else "B"
        
        # Get all possible trips for this bus, sorted by global time
        bus_trips = []
        for t in trips:
            if (b, t["id"]) in x:
                global_start = t["day"] * 1440 + t["start"]
                bus_trips.append((t, global_start))
        
        bus_trips.sort(key=lambda item: item[1])  # Sort by global start time
        
        # For each pair of trips, ensure proper timing
        for i in range(len(bus_trips)):
            trip1, global_start1 = bus_trips[i]
            trip1_var = x[b, trip1["id"]]
            
            # Check against initial availability
            if global_start1 < initial_avail_time:
                print(f"   🚫 Bus {b+1}: Trip {trip1['id']} at day {trip1['day']} {trip1['start']//60:02d}:{trip1['start']%60:02d} conflicts with initial availability")
                model.Add(trip1_var == 0)
                continue
            
            # Check location continuity for first trip (only if no previous trips can reposition)
            if i == 0 and trip1["origin"] != initial_location:
                # Check if there are any earlier trips that could have repositioned the bus
                can_reposition = False
                for j in range(len(bus_trips)):
                    if j == i:
                        continue
                    other_trip, other_start = bus_trips[j]
                    if other_start < global_start1:
                        other_end_location = 'B' if other_trip["origin"] == 'A' else 'A'
                        other_end_time = other_start + cycle_minutes
                        if other_end_location == trip1["origin"] and other_end_time <= global_start1:
                            can_reposition = True
                            other_var = x[b, other_trip["id"]]
                            # Add implication: if trip1 is selected, repositioning trip must be selected
                            model.Add(trip1_var <= other_var)
                            break
                
                if not can_reposition:
                    print(f"   🚫 Bus {b+1}: First trip {trip1['id']} starts from {trip1['origin']} but bus is at {initial_location}")
                    model.Add(trip1_var == 0)
                    continue
            
            # Check timing against all later trips
            for j in range(i + 1, len(bus_trips)):
                trip2, global_start2 = bus_trips[j]
                trip2_var = x[b, trip2["id"]]
                
                # Calculate when trip1 ends (ready for next trip)
                trip1_ready_time = global_start1 + cycle_minutes
                
                # If trip2 starts before trip1 is ready, they cannot both be selected
                if global_start2 < trip1_ready_time:
                    model.Add(trip1_var + trip2_var <= 1)
                    
                # Also check location continuity
                trip1_end_location = 'B' if trip1["origin"] == 'A' else 'A'
                if trip2["origin"] != trip1_end_location:
                    # Location mismatch - they cannot be consecutive
                    model.Add(trip1_var + trip2_var <= 1)

def cross_day_chain_enhanced(model, x, trips, B, travel_h, charge_h):
    """
    Enhanced cross-day chaining that properly handles midnight transitions
    and tracks exact timing across day boundaries.
    """
    cycle_minutes = int((travel_h + charge_h) * 60)
    
    for b in range(B):
        # For each day, check transitions to next day
        for d in range(7):
            next_d = (d + 1) % 7
            
            # Get trips for current day and next day
            day_trips = [t for t in trips if t["day"] == d and (b, t["id"]) in x]
            next_day_trips = [t for t in trips if t["day"] == next_d and (b, t["id"]) in x]
            
            for curr_trip in day_trips:
                curr_var = x[b, curr_trip["id"]]
                curr_end_time = d * 1440 + curr_trip["start"] + cycle_minutes
                curr_end_location = 'B' if curr_trip["origin"] == 'A' else 'A'
                
                for next_trip in next_day_trips:
                    next_var = x[b, next_trip["id"]]
                    
                    # Handle day rollover properly
                    if next_d == 0 and d == 6:  # Sunday to Monday
                        next_start_time = 7 * 1440 + next_trip["start"]  # Next week
                    else:
                        next_start_time = next_d * 1440 + next_trip["start"]
                    
                    # Timing constraint
                    if next_start_time < curr_end_time:
                        model.Add(curr_var + next_var <= 1)
                    
                    # Location constraint
                    if next_trip["origin"] != curr_end_location:
                        model.Add(curr_var + next_var <= 1)

def validate_schedule_timing(schedule_data, travel_h=9.0, charge_h=2.0):
    """
    Validate that the generated schedule respects timing constraints.
    Returns detailed analysis of timing violations.
    """
    cycle_minutes = int((travel_h + charge_h) * 60)
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # Organize all trips by bus
    bus_schedules = {}
    
    for day_name, day_data in schedule_data.items():
        day_idx = days_order.index(day_name)
        for assignment in day_data["assignments"]:
            bus_num = assignment["busNumber"]
            if bus_num not in bus_schedules:
                bus_schedules[bus_num] = []
            
            # Parse time
            time_parts = assignment["trip"]["startTime"].split(":")
            h, m = int(time_parts[0]), int(time_parts[1])
            global_start = day_idx * 1440 + h * 60 + m
            
            bus_schedules[bus_num].append({
                "day": day_idx,
                "day_name": day_name,
                "start_minutes": h * 60 + m,
                "global_start": global_start,
                "route": assignment["trip"]["route"],
                "origin": assignment["trip"]["origin"],
                "epk": assignment["trip"]["epk"]
            })
    
    # Sort each bus's trips by global time
    for bus_num in bus_schedules:
        bus_schedules[bus_num].sort(key=lambda x: x["global_start"])
    
    # Check timing violations
    violations = []
    
    for bus_num, trips in bus_schedules.items():
        for i in range(len(trips) - 1):
            curr_trip = trips[i]
            next_trip = trips[i + 1]
            
            # Calculate when current trip ends + charging
            curr_ready_time = curr_trip["global_start"] + cycle_minutes
            
            # Check if next trip starts too early
            if next_trip["global_start"] < curr_ready_time:
                gap_minutes = next_trip["global_start"] - curr_ready_time
                violations.append({
                    "bus": bus_num,
                    "issue": "timing_violation",
                    "trip1": curr_trip,
                    "trip2": next_trip,
                    "gap_minutes": gap_minutes,
                    "description": f"Bus {bus_num}: Trip on {next_trip['day_name']} starts {abs(gap_minutes)} minutes too early"
                })
            
            # Check location continuity
            curr_end_location = 'B' if curr_trip["origin"] == 'A' else 'A'
            if next_trip["origin"] != curr_end_location:
                violations.append({
                    "bus": bus_num,
                    "issue": "location_violation", 
                    "trip1": curr_trip,
                    "trip2": next_trip,
                    "description": f"Bus {bus_num}: Ends at {curr_end_location} but next trip starts from {next_trip['origin']}"
                })
    
    return violations

def global_bus_trip_chaining(model, x, trips, B, travel_h, charge_h):
    """
    For each bus, enforce that all assigned trips form a valid, continuous chain:
    - For every pair of trips (T1, T2), if T2 starts after T1 ends but at the wrong location, or if T2 starts before T1 ends, both cannot be assigned.
    """
    cycle_minutes = int((travel_h + charge_h) * 60)
    for b in range(B):
        # Get all trips for this bus
        bus_trips = [t for t in trips if (b, t["id"]) in x]
        for i in range(len(bus_trips)):
            t1 = bus_trips[i]
            t1_end_time = t1["day"] * 1440 + t1["start"] + cycle_minutes
            t1_end_loc = 'B' if t1["origin"] == 'A' else 'A'
            for j in range(len(bus_trips)):
                if i == j:
                    continue
                t2 = bus_trips[j]
                t2_start_time = t2["day"] * 1440 + t2["start"]
                t2_start_loc = t2["origin"]
                # If t2 starts after t1 ends
                if t2_start_time >= t1_end_time:
                    if t2_start_loc != t1_end_loc:
                        model.Add(x[b, t1["id"]] + x[b, t2["id"]] <= 1)
                # If t2 starts before t1 ends, both cannot be assigned
                if t2_start_time < t1_end_time:
                    model.Add(x[b, t1["id"]] + x[b, t2["id"]] <= 1)

