from nbus_calculator_optimized import generate_epk_curve, load_corridor

epk_map, days = load_corridor("cache/fb-vh-raw-30-days-interpolated.csv", "H-V", "V-H")



route_fwd = input("Enter the forward route: ")
route_rev = input("Enter the reverse route: ")
travel_h = float(input("Enter the travel time: "))
charge_h = float(input("Enter the charge time: "))
N_max = int(input("Enter the maximum number of buses: "))
idle_h = float(input("Enter the idle time: "))

generate_epk_curve(epk_map, days, route_fwd, route_rev, travel_h, charge_h, N_max, idle_h,balanced_window_size=2,time_limit=300)