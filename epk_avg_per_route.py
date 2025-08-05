import csv

csv_path = 'cache/overall_epk_data.csv'

sums = {}
counts = {}

def get_avg_epk_per_route(csv_path):
    with open(csv_path, 'r', encoding='utf-8', newline='') as csvfile:
        results = {}
        reader = csv.DictReader(csvfile)
        route_names = [col for col in reader.fieldnames if col != 'journeyTime']
        print('Route names:', route_names)
        for route in route_names:
            sums[route] = 0.0
            counts[route] = 0
        for row in reader:
            for route in route_names:
                value = row[route]
                if value:
                    try:
                        val = float(value)
                        sums[route] += val
                        counts[route] += 1
                    except ValueError:
                        pass  # skip invalid values
    print('Average EPK per route:')
    for route in route_names:
        avg = sums[route] / counts[route] if counts[route] else 0
        results[route] = avg
    return results


results = get_avg_epk_per_route(csv_path)
print(results)
