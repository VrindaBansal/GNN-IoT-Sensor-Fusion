"""
Smart City IoT Sensor Network Simulator
Generates realistic sensor data with correlations and anomalies
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import threading
import queue
import json
import random
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from faker import Faker
import math


@dataclass
class SensorConfig:
    """Configuration for individual sensor simulation"""
    sensor_id: str
    sensor_type: str
    location: Tuple[float, float, float]  # lat, lon, elevation
    sampling_rate: float  # Hz
    noise_level: float = 0.1
    drift_rate: float = 0.001
    failure_probability: float = 0.001
    reading_range: Tuple[float, float] = (0.0, 100.0)


class UrbanEventSimulator:
    """Simulates urban events that affect sensor readings"""

    def __init__(self):
        self.active_events = {}
        self.event_types = {
            'traffic_congestion': {'duration': (600, 3600), 'radius': 200},
            'construction': {'duration': (7200, 86400), 'radius': 100},
            'weather_front': {'duration': (1800, 10800), 'radius': 5000},
            'emergency_vehicle': {'duration': (60, 300), 'radius': 150},
            'public_event': {'duration': (3600, 14400), 'radius': 500},
            'air_quality_spike': {'duration': (1800, 7200), 'radius': 1000}
        }

    def generate_event(self, timestamp: float, area_bounds: Tuple[float, float, float, float]) -> Optional[Dict]:
        """Generate random urban event"""
        if random.random() < 0.01:  # 1% chance per simulation step
            event_type = random.choice(list(self.event_types.keys()))
            config = self.event_types[event_type]

            event = {
                'id': f"{event_type}_{int(timestamp)}",
                'type': event_type,
                'start_time': timestamp,
                'duration': random.uniform(*config['duration']),
                'location': (
                    random.uniform(area_bounds[0], area_bounds[1]),  # lat
                    random.uniform(area_bounds[2], area_bounds[3]),  # lon
                    random.uniform(0, 100)  # elevation
                ),
                'radius': config['radius'],
                'intensity': random.uniform(0.5, 2.0)
            }

            self.active_events[event['id']] = event
            return event

        return None

    def get_active_events(self, timestamp: float) -> List[Dict]:
        """Get currently active events"""
        active = []
        expired_events = []

        for event_id, event in self.active_events.items():
            if timestamp - event['start_time'] > event['duration']:
                expired_events.append(event_id)
            else:
                active.append(event)

        # Remove expired events
        for event_id in expired_events:
            del self.active_events[event_id]

        return active

    def calculate_event_impact(self, sensor_location: Tuple[float, float, float],
                             events: List[Dict]) -> Dict[str, float]:
        """Calculate impact of events on sensor readings"""
        impacts = {
            'temperature': 0.0,
            'humidity': 0.0,
            'noise': 0.0,
            'air_quality': 0.0,
            'vibration': 0.0
        }

        for event in events:
            distance = self._calculate_distance(sensor_location, event['location'])
            if distance <= event['radius']:
                # Impact decreases with distance
                impact_factor = (1.0 - distance / event['radius']) * event['intensity']

                # Event-specific impacts
                if event['type'] == 'traffic_congestion':
                    impacts['noise'] += impact_factor * 0.8
                    impacts['air_quality'] += impact_factor * 0.6
                    impacts['vibration'] += impact_factor * 0.4

                elif event['type'] == 'construction':
                    impacts['noise'] += impact_factor * 1.2
                    impacts['vibration'] += impact_factor * 1.0

                elif event['type'] == 'weather_front':
                    impacts['temperature'] += impact_factor * 0.3
                    impacts['humidity'] += impact_factor * 0.5

                elif event['type'] == 'emergency_vehicle':
                    impacts['noise'] += impact_factor * 1.5
                    impacts['vibration'] += impact_factor * 0.3

                elif event['type'] == 'air_quality_spike':
                    impacts['air_quality'] += impact_factor * 1.0

        return impacts

    def _calculate_distance(self, loc1: Tuple[float, float, float],
                          loc2: Tuple[float, float, float]) -> float:
        """Calculate approximate distance between two locations"""
        lat_diff = (loc1[0] - loc2[0]) * 111000  # Approximate meters per degree
        lon_diff = (loc1[1] - loc2[1]) * 111000 * math.cos(math.radians(loc1[0]))
        elevation_diff = loc1[2] - loc2[2]

        return math.sqrt(lat_diff**2 + lon_diff**2 + elevation_diff**2)


class RealisticNoiseGenerator:
    """Generates realistic noise patterns for sensor readings"""

    def __init__(self):
        self.time_of_day_patterns = {}
        self.seasonal_patterns = {}

    def generate_noise(self, sensor_type: str, base_value: float,
                      timestamp: float, noise_level: float = 0.1) -> float:
        """Generate realistic noise for sensor reading"""
        # Base Gaussian noise
        noise = random.gauss(0, noise_level)

        # Time-of-day variation
        hour = (timestamp % 86400) / 3600  # Hour of day
        daily_variation = self._get_daily_variation(sensor_type, hour)

        # Weekly pattern
        day_of_week = int((timestamp / 86400) % 7)
        weekly_variation = self._get_weekly_variation(sensor_type, day_of_week)

        # Combine variations
        total_variation = daily_variation + weekly_variation + noise

        return base_value * (1.0 + total_variation)

    def _get_daily_variation(self, sensor_type: str, hour: float) -> float:
        """Get daily variation pattern for sensor type"""
        if sensor_type == 'temperature':
            # Peak at 2 PM, minimum at 6 AM
            return 0.3 * math.sin((hour - 6) * math.pi / 12)
        elif sensor_type == 'humidity':
            # Inverse of temperature
            return -0.2 * math.sin((hour - 6) * math.pi / 12)
        elif sensor_type == 'audio':
            # High during day, low at night
            if 6 <= hour <= 22:
                return 0.4 * math.sin((hour - 6) * math.pi / 16)
            else:
                return -0.3
        elif sensor_type == 'air_quality':
            # Rush hour peaks
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                return 0.5
            else:
                return 0.0
        else:
            return 0.0

    def _get_weekly_variation(self, sensor_type: str, day: int) -> float:
        """Get weekly variation pattern"""
        if sensor_type in ['audio', 'air_quality']:
            # Lower on weekends
            if day in [5, 6]:  # Saturday, Sunday
                return -0.2
            else:
                return 0.1
        return 0.0


class SmartCitySensorSimulator:
    """Main simulator for smart city IoT sensor network"""

    def __init__(self, num_sensors: int = 1000, area_km2: float = 10):
        self.num_sensors = num_sensors
        self.area_km2 = area_km2
        self.faker = Faker()

        # Simulation state
        self.sensors = {}
        self.is_running = False
        self.data_queue = queue.Queue()
        self.event_simulator = UrbanEventSimulator()
        self.noise_generator = RealisticNoiseGenerator()

        # Area bounds (roughly New York City coordinates)
        self.area_bounds = (40.7, 40.8, -74.0, -73.9)  # lat_min, lat_max, lon_min, lon_max

        # Initialize sensors
        self._initialize_sensor_network()

    def _initialize_sensor_network(self):
        """Create realistic sensor network layout"""
        sensor_types = ['temperature', 'humidity', 'accelerometer', 'audio', 'visual', 'air_quality']
        type_counts = {
            'temperature': int(self.num_sensors * 0.2),
            'humidity': int(self.num_sensors * 0.15),
            'accelerometer': int(self.num_sensors * 0.15),
            'audio': int(self.num_sensors * 0.2),
            'visual': int(self.num_sensors * 0.2),
            'air_quality': int(self.num_sensors * 0.1)
        }

        sensor_id = 0
        for sensor_type, count in type_counts.items():
            for _ in range(count):
                location = self._generate_realistic_location(sensor_type)
                config = SensorConfig(
                    sensor_id=f"sensor_{sensor_id:04d}",
                    sensor_type=sensor_type,
                    location=location,
                    sampling_rate=self._get_sampling_rate(sensor_type),
                    noise_level=random.uniform(0.05, 0.15),
                    reading_range=self._get_sensor_range(sensor_type)
                )
                self.sensors[config.sensor_id] = config
                sensor_id += 1

    def _generate_realistic_location(self, sensor_type: str) -> Tuple[float, float, float]:
        """Generate realistic sensor placement based on type"""
        lat = random.uniform(self.area_bounds[0], self.area_bounds[1])
        lon = random.uniform(self.area_bounds[2], self.area_bounds[3])

        # Height placement based on sensor type
        if sensor_type == 'visual':
            elevation = random.uniform(3, 10)  # Traffic cameras, security cameras
        elif sensor_type == 'audio':
            elevation = random.uniform(2, 8)   # Noise monitoring
        elif sensor_type == 'air_quality':
            elevation = random.uniform(1, 5)   # Near ground level
        else:
            elevation = random.uniform(1, 15)  # General environmental sensors

        return (lat, lon, elevation)

    def _get_sampling_rate(self, sensor_type: str) -> float:
        """Get appropriate sampling rate for sensor type"""
        rates = {
            'temperature': 0.1,      # 10 seconds
            'humidity': 0.1,         # 10 seconds
            'accelerometer': 10.0,   # 10 Hz
            'audio': 1.0,           # 1 Hz
            'visual': 0.033,        # ~30 seconds
            'air_quality': 0.05     # 20 seconds
        }
        return rates.get(sensor_type, 0.1)

    def _get_sensor_range(self, sensor_type: str) -> Tuple[float, float]:
        """Get measurement range for sensor type"""
        ranges = {
            'temperature': (-20.0, 50.0),     # Celsius
            'humidity': (0.0, 100.0),         # Percentage
            'accelerometer': (0.0, 10.0),     # m/s²
            'audio': (30.0, 120.0),           # dB
            'visual': (0.0, 255.0),           # Brightness/motion index
            'air_quality': (0.0, 500.0)       # AQI
        }
        return ranges.get(sensor_type, (0.0, 100.0))

    def generate_sensor_reading(self, sensor_id: str, timestamp: float) -> Dict:
        """Generate realistic sensor reading"""
        config = self.sensors[sensor_id]
        sensor_type = config.sensor_type

        # Base value generation
        base_value = self._generate_base_value(sensor_type, timestamp, config.location)

        # Apply event impacts
        active_events = self.event_simulator.get_active_events(timestamp)
        event_impacts = self.event_simulator.calculate_event_impact(config.location, active_events)

        # Apply event impact
        impact_multiplier = 1.0
        if sensor_type == 'temperature':
            impact_multiplier += event_impacts['temperature']
        elif sensor_type == 'humidity':
            impact_multiplier += event_impacts['humidity']
        elif sensor_type == 'audio':
            impact_multiplier += event_impacts['noise']
        elif sensor_type == 'air_quality':
            impact_multiplier += event_impacts['air_quality']
        elif sensor_type == 'accelerometer':
            impact_multiplier += event_impacts['vibration']

        base_value *= max(0.1, impact_multiplier)

        # Apply realistic noise
        final_value = self.noise_generator.generate_noise(
            sensor_type, base_value, timestamp, config.noise_level
        )

        # Clamp to sensor range
        min_val, max_val = config.reading_range
        final_value = max(min_val, min(max_val, final_value))

        # Check for sensor failure
        is_failed = random.random() < config.failure_probability

        reading = {
            'sensor_id': sensor_id,
            'sensor_type': sensor_type,
            'timestamp': timestamp,
            'location': config.location,
            'value': final_value if not is_failed else None,
            'quality': 'good' if not is_failed else 'failed',
            'active_events': len(active_events)
        }

        # Add sensor-specific fields
        if sensor_type == 'temperature':
            reading['unit'] = 'celsius'
        elif sensor_type == 'humidity':
            reading['unit'] = 'percent'
        elif sensor_type == 'audio':
            reading['unit'] = 'dB'
            reading['frequency_band'] = random.choice(['low', 'mid', 'high'])
        elif sensor_type == 'air_quality':
            reading['unit'] = 'AQI'
            reading['primary_pollutant'] = random.choice(['PM2.5', 'PM10', 'NO2', 'O3'])

        return reading

    def _generate_base_value(self, sensor_type: str, timestamp: float,
                           location: Tuple[float, float, float]) -> float:
        """Generate base sensor value with realistic patterns"""
        if sensor_type == 'temperature':
            # Seasonal and daily patterns
            day_of_year = int((timestamp / 86400) % 365)
            hour_of_day = (timestamp % 86400) / 3600
            seasonal_temp = 15 + 10 * math.sin((day_of_year - 80) * 2 * math.pi / 365)
            daily_variation = 5 * math.sin((hour_of_day - 6) * math.pi / 12)
            return seasonal_temp + daily_variation

        elif sensor_type == 'humidity':
            # Inverse correlation with temperature
            temp = self._generate_base_value('temperature', timestamp, location)
            return max(20, min(90, 70 - (temp - 20) * 1.5))

        elif sensor_type == 'audio':
            # Traffic and activity patterns
            hour = (timestamp % 86400) / 3600
            if 6 <= hour <= 22:
                base_noise = 55 + 15 * math.sin((hour - 6) * math.pi / 16)
            else:
                base_noise = 35
            return base_noise + random.uniform(-5, 10)

        elif sensor_type == 'air_quality':
            # Pollution patterns with rush hours
            hour = (timestamp % 86400) / 3600
            base_aqi = 50
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                base_aqi += 30  # Rush hour spike
            return base_aqi + random.uniform(-10, 20)

        elif sensor_type == 'accelerometer':
            # Random vibration with traffic influence
            base_vibration = 0.1
            hour = (timestamp % 86400) / 3600
            if 6 <= hour <= 22:
                base_vibration *= 1.5
            return base_vibration + random.uniform(0, 0.5)

        elif sensor_type == 'visual':
            # Brightness/motion detection
            hour = (timestamp % 86400) / 3600
            if 6 <= hour <= 18:
                return random.uniform(100, 255)  # Daylight
            else:
                return random.uniform(0, 80)     # Night/artificial light

        return random.uniform(0, 100)

    def start_simulation(self, duration_hours: float = 24, real_time: bool = False):
        """Start the sensor simulation"""
        self.is_running = True
        start_time = time.time()
        sim_start_time = start_time

        total_seconds = duration_hours * 3600
        sim_time = 0

        print(f"Starting simulation for {duration_hours} hours with {len(self.sensors)} sensors")

        while self.is_running and sim_time < total_seconds:
            current_timestamp = sim_start_time + sim_time

            # Generate events
            event = self.event_simulator.generate_event(current_timestamp, self.area_bounds)
            if event:
                print(f"Event generated: {event['type']} at {event['location']}")

            # Generate sensor readings based on sampling rates
            batch_data = {}
            for sensor_id, config in self.sensors.items():
                # Check if this sensor should generate reading at this time
                if sim_time % (1.0 / config.sampling_rate) < 1.0:
                    reading = self.generate_sensor_reading(sensor_id, current_timestamp)
                    batch_data[sensor_id] = reading

            # Put batch data in queue
            if batch_data:
                self.data_queue.put({
                    'timestamp': current_timestamp,
                    'batch_data': batch_data,
                    'active_events': self.event_simulator.get_active_events(current_timestamp)
                })

            # Time management
            if real_time:
                time.sleep(1.0)  # 1 second simulation steps
                sim_time = time.time() - start_time
            else:
                sim_time += 1.0  # 1 second simulation steps

            # Progress reporting
            if int(sim_time) % 300 == 0:  # Every 5 minutes
                progress = (sim_time / total_seconds) * 100
                print(f"Simulation progress: {progress:.1f}%")

        print("Simulation completed")

    def get_real_time_data(self) -> Optional[Dict]:
        """Get real-time sensor data batch"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None

    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False

    def export_data_to_file(self, filename: str, duration_minutes: int = 60):
        """Export simulated data to file for offline analysis"""
        data_records = []
        start_time = time.time()

        print(f"Generating {duration_minutes} minutes of data...")

        for sim_second in range(duration_minutes * 60):
            timestamp = start_time + sim_second

            # Generate events
            self.event_simulator.generate_event(timestamp, self.area_bounds)

            for sensor_id, config in self.sensors.items():
                if sim_second % int(1.0 / config.sampling_rate) == 0:
                    reading = self.generate_sensor_reading(sensor_id, timestamp)
                    data_records.append(reading)

        # Save to file
        df = pd.DataFrame(data_records)
        df.to_csv(filename, index=False)
        print(f"Data exported to {filename}: {len(data_records)} readings")

        return df

    def get_sensor_network_info(self) -> Dict:
        """Get information about the sensor network"""
        type_counts = {}
        for config in self.sensors.values():
            sensor_type = config.sensor_type
            type_counts[sensor_type] = type_counts.get(sensor_type, 0) + 1

        return {
            'total_sensors': len(self.sensors),
            'sensor_types': type_counts,
            'area_coverage': f"{self.area_km2} km²",
            'area_bounds': self.area_bounds,
            'avg_sampling_rate': np.mean([config.sampling_rate for config in self.sensors.values()])
        }