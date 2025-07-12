import json

class BusScheduleVisualizer:
    def __init__(self, schedule_data):
        """Initialize the visualizer with schedule data (dict or path to JSON file)."""
        if isinstance(schedule_data, str):
            # If string, treat as file path
            with open(schedule_data, 'r') as file:
                self.data = json.load(file)
        else:
            # If dict, use directly
            self.data = schedule_data
        self.days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    def calculate_statistics(self):
        """Calculate the number of unique buses and various metrics."""
        all_buses = set()
        total_epk = 0
        total_trips = 0
        total_distance = 0
        total_time_minutes = 0
        
        for day in self.data:
            for assignment in self.data[day]["assignments"]:
                all_buses.add(assignment["busNumber"])
                total_epk += assignment["trip"]["epk"]
                total_trips += 1
                # Assume average distance per trip (you can modify this)
                total_distance += 740  # km per trip as shown in image
                # Assume average trip time (modify based on your data)
                total_time_minutes += 210  # 3.5 hours per trip
        
        avg_distance_per_day = total_distance / 7 if total_trips > 0 else 0
        avg_trip_time = total_time_minutes / total_trips if total_trips > 0 else 0
        avg_epk = total_epk / total_trips if total_trips > 0 else 0
        
        return {
            'no_buses': len(all_buses),
            'avg_epk': avg_epk,
            'avg_distance_per_day': avg_distance_per_day,
            'avg_trip_time_hours': avg_trip_time / 60,
            'avg_trip_time_minutes': avg_trip_time % 60
        }

    def generate_html(self, output_file):
        """Generate HTML file matching the user's desired layout."""
        stats = self.calculate_statistics()
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bus Schedule Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        /* Header Stats */
        .stats-header {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}
        
        .stat-item {{
            text-align: center;
        }}
        
        .stat-label {{
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
            font-weight: 500;
        }}
        
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        
        /* Days Grid */
        .days-grid {{
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 15px;
        }}
        
        .day-column {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }}
        
        .day-header {{
            background: #333;
            color: white;
            text-align: center;
            padding: 12px 8px;
            font-weight: bold;
            font-size: 14px;
        }}
        
        .routes-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
        }}
        
        .route-column {{
            min-height: 400px;
        }}
        
        .route-header {{
            background: #f8f9fa;
            text-align: center;
            padding: 8px 4px;
            font-weight: bold;
            font-size: 12px;
            color: #333;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .trip-item {{
            padding: 8px;
            margin: 4px;
            border-radius: 6px;
            font-size: 11px;
            line-height: 1.3;
        }}
        
        /* Trip colors similar to the image */
        .trip-red {{
            background: #ff6b6b;
            color: white;
        }}
        
        .trip-orange {{
            background: #ffa726;
            color: white;
        }}
        
        .trip-yellow {{
            background: #ffeb3b;
            color: #333;
        }}
        
        .trip-green {{
            background: #66bb6a;
            color: white;
        }}
        
        .trip-blue {{
            background: #42a5f5;
            color: white;
        }}
        
        .trip-purple {{
            background: #ab47bc;
            color: white;
        }}
        
        .trip-teal {{
            background: #26c6da;
            color: white;
        }}
        
        .trip-indigo {{
            background: #5c6bc0;
            color: white;
        }}
        
        .bus-number {{
            font-weight: bold;
            font-size: 12px;
        }}
        
        .trip-time {{
            font-size: 10px;
            opacity: 0.9;
        }}
        
        .trip-epk {{
            font-size: 10px;
            opacity: 0.9;
        }}
        
        @media (max-width: 1200px) {{
            .days-grid {{
                grid-template-columns: repeat(4, 1fr);
            }}
        }}
        
        @media (max-width: 800px) {{
            .days-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            .stats-header {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
        
        @media (max-width: 500px) {{
            .days-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header Stats -->
        <div class="stats-header">
            <div class="stat-item">
                <div class="stat-label">No. of Buses</div>
                <div class="stat-value">{stats['no_buses']}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">EPK</div>
                <div class="stat-value">{stats['avg_epk']:.2f}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Avg Distance/bus/day (km)</div>
                <div class="stat-value">{stats['avg_distance_per_day']:.0f}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Avg Trip Time</div>
                <div class="stat-value">{int(stats['avg_trip_time_hours'])}h {int(stats['avg_trip_time_minutes'])}m</div>
            </div>
        </div>
        
        <!-- Days Grid -->
        <div class="days-grid">"""

        # Color classes for different buses
        colors = ['trip-red', 'trip-orange', 'trip-yellow', 'trip-green', 
                 'trip-blue', 'trip-purple', 'trip-teal', 'trip-indigo']

        # Generate each day column
        for day in self.days_order:
            day_data = self.data.get(day, {"assignments": []})
            
            # Separate trips by route
            ab_trips = [t for t in day_data["assignments"] if t["trip"]["route"] == "A-B"]
            ba_trips = [t for t in day_data["assignments"] if t["trip"]["route"] == "B-A"]
            
            # Sort by start time
            ab_trips.sort(key=lambda x: x["trip"]["startTime"])
            ba_trips.sort(key=lambda x: x["trip"]["startTime"])
            
            html_content += f"""
            <div class="day-column">
                <div class="day-header">{day}</div>
                <div class="routes-container">
                    <!-- H → V (A-B) -->
                    <div class="route-column">
                        <div class="route-header">H → V</div>"""
            
            # Add A-B trips
            for i, trip in enumerate(ab_trips):
                color_class = colors[trip["busNumber"] % len(colors)]
                start_time = trip["trip"]["startTime"].split(":")[0] + ":" + trip["trip"]["startTime"].split(":")[1]
                
                html_content += f"""
                        <div class="trip-item {color_class}">
                            <div class="bus-number">Bus {trip["busNumber"]:02d}</div>
                            <div class="trip-time">{start_time}</div>
                            <div class="trip-epk">EPK {trip["trip"]["epk"]:.1f}</div>
                        </div>"""
            
            html_content += """
                    </div>
                    
                    <!-- V → H (B-A) -->
                    <div class="route-column">
                        <div class="route-header">V → H</div>"""
            
            # Add B-A trips
            for i, trip in enumerate(ba_trips):
                color_class = colors[trip["busNumber"] % len(colors)]
                start_time = trip["trip"]["startTime"].split(":")[0] + ":" + trip["trip"]["startTime"].split(":")[1]
                
                html_content += f"""
                        <div class="trip-item {color_class}">
                            <div class="bus-number">Bus {trip["busNumber"]:02d}</div>
                            <div class="trip-time">{start_time}</div>
                            <div class="trip-epk">EPK {trip["trip"]["epk"]:.1f}</div>
                        </div>"""
            
            html_content += """
                    </div>
                </div>
            </div>"""

        html_content += """
        </div>
    </div>
</body>
</html>"""
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(html_content)
        print(f"✅ Schedule visualization saved to {output_file}")

    def run(self):
        """Execute the full process: load data, calculate statistics, generate HTML, and write to file."""
        self.load_data()
        no_buses, avg_epk = self.calculate_statistics()
        html_content = self.generate_html(no_buses, avg_epk)
        self.write_html(html_content)
       
