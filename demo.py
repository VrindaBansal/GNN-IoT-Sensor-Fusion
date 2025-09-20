#!/usr/bin/env python3
"""
UrbanSense Demo: Real-time IoT Sensor Fusion with Graph Neural Networks
Complete demonstration of the UrbanSense system with visualization
"""

import os
import sys
import time
import threading
import argparse
import signal

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import system components
from data.simulators.smart_city_simulator import SmartCitySensorSimulator
from graph.construction.dynamic_graph_builder import DynamicGraphBuilder, SensorNode, GraphConfig

# Try to import TensorFlow components, fallback to simple versions
try:
    from models.gnn.multimodal_gnn import create_multimodal_gnn_model
    from models.fusion.multimodal_fusion import create_fusion_layer
    from applications.anomaly.anomaly_detector import RealTimeAnomalyDetector
    from applications.traffic.traffic_optimizer import TrafficOptimizer
    from applications.emergency.emergency_detector import EmergencyResponseSystem
    import tensorflow as tf
    TF_AVAILABLE = True
    print("🚀 Using TensorFlow-based models")
except ImportError:
    from models.gnn.simple_gnn import create_simple_gnn_model as create_multimodal_gnn_model
    from models.gnn.simple_gnn import SimpleAnomalyDetector as RealTimeAnomalyDetector
    from models.gnn.simple_gnn import SimpleTrafficOptimizer as TrafficOptimizer
    from models.gnn.simple_gnn import SimpleEmergencyDetector as EmergencyResponseSystem
    TF_AVAILABLE = False
    print("⚡ Using simplified models (TensorFlow not available)")

try:
    from graph.visualization.dashboard import create_dashboard
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    print("📊 Dashboard not available (missing dash/plotly)")

import numpy as np


class UrbanSenseDemo:
    """Main demo class for UrbanSense system"""

    def __init__(self, num_sensors=100, area_km2=5, with_dashboard=True):
        self.num_sensors = num_sensors
        self.area_km2 = area_km2
        self.with_dashboard = with_dashboard
        self.running = False

        print("🏙️  Initializing UrbanSense: Real-time IoT Sensor Fusion System")
        print("=" * 70)

        # Initialize components
        self._initialize_simulator()
        self._initialize_graph_builder()
        self._initialize_gnn_model()
        self._initialize_applications()

        if self.with_dashboard and DASH_AVAILABLE:
            self._initialize_dashboard()
        elif self.with_dashboard and not DASH_AVAILABLE:
            print("⚠️  Dashboard disabled - install dash and plotly to enable")

        print("✅ UrbanSense system initialized successfully!")
        print(f"📊 {num_sensors} sensors deployed across {area_km2} km²")
        print()

    def _initialize_simulator(self):
        """Initialize IoT sensor simulator"""
        print("🔧 Initializing IoT sensor simulator...")
        self.simulator = SmartCitySensorSimulator(
            num_sensors=self.num_sensors,
            area_km2=self.area_km2
        )

        # Get network info
        info = self.simulator.get_sensor_network_info()
        print(f"   📍 Sensor network deployed:")
        for sensor_type, count in info['sensor_types'].items():
            print(f"      - {sensor_type}: {count} sensors")

    def _initialize_graph_builder(self):
        """Initialize dynamic graph construction"""
        print("🕸️  Initializing dynamic graph construction...")
        config = GraphConfig(
            spatial_threshold_meters=200.0,
            temporal_window_seconds=300,
            correlation_threshold=0.6,
            max_neighbors=8
        )
        self.graph_builder = DynamicGraphBuilder(config)

        # Add all sensors to graph builder
        for sensor_config in self.simulator.sensors.values():
            sensor_node = SensorNode(
                id=sensor_config.sensor_id,
                sensor_type=sensor_config.sensor_type,
                location=sensor_config.location,
                capabilities={}
            )
            self.graph_builder.add_sensor(sensor_node)

        print(f"   🔗 Graph builder configured with {len(self.graph_builder.sensor_nodes)} nodes")

    def _initialize_gnn_model(self):
        """Initialize Graph Neural Network model"""
        print("🧠 Initializing Graph Neural Network...")
        self.gnn_config = {
            'node_features': 10,
            'edge_features': 5,
            'output_dim': 32,
            'num_sensor_types': 6,
            'hidden_dim': 64,
            'num_heads': 8,
            'num_layers': 3
        }

        self.gnn_model = create_multimodal_gnn_model(self.gnn_config)

        # Initialize fusion layer (if TensorFlow available)
        if TF_AVAILABLE:
            from models.fusion.multimodal_fusion import create_fusion_layer
            self.fusion_layer = create_fusion_layer('adaptive', output_dim=32)
        else:
            self.fusion_layer = None

        print("   🔮 Multimodal GNN model ready")

    def _initialize_applications(self):
        """Initialize smart city applications"""
        print("🚀 Initializing smart city applications...")

        # Anomaly detection
        self.anomaly_detector = RealTimeAnomalyDetector(
            sensor_feature_dim=10,
            window_size=50,
            threshold_percentile=90.0
        )

        # Traffic optimization
        self.traffic_optimizer = TrafficOptimizer(area_grid_size=10)

        # Emergency detection
        self.emergency_detector = EmergencyResponseSystem(
            response_time_threshold=300.0
        )

        print("   🚨 Anomaly detection system active")
        print("   🚦 Traffic optimization system active")
        print("   🆘 Emergency response system active")

    def _initialize_dashboard(self):
        """Initialize visualization dashboard"""
        print("📊 Initializing real-time dashboard...")
        self.dashboard = create_dashboard(
            simulator=self.simulator,
            gnn_model=self.gnn_model,
            graph_builder=self.graph_builder
        )
        print("   🌐 Dashboard ready at http://localhost:8050")

    def run_simulation(self, duration_hours=1.0, real_time=False):
        """Run the complete UrbanSense simulation"""
        print("\n🎬 Starting UrbanSense simulation...")
        print("=" * 50)

        self.running = True
        simulation_thread = None

        try:
            # Start dashboard in separate thread if enabled
            if self.with_dashboard and DASH_AVAILABLE:
                dashboard_thread = threading.Thread(
                    target=self.dashboard.run,
                    kwargs={'host': 'localhost', 'port': 8050, 'debug': False},
                    daemon=True
                )
                dashboard_thread.start()
                print("🌐 Dashboard started at http://localhost:8050")
                time.sleep(2)  # Give dashboard time to start

            # Start sensor simulation
            simulation_thread = threading.Thread(
                target=self._run_sensor_simulation,
                args=(duration_hours, real_time),
                daemon=True
            )
            simulation_thread.start()

            # Main processing loop
            self._run_main_processing_loop()

        except KeyboardInterrupt:
            print("\n⏹️  Simulation stopped by user")
        finally:
            self.running = False
            if simulation_thread:
                simulation_thread.join(timeout=5)

    def _run_sensor_simulation(self, duration_hours, real_time):
        """Run sensor data simulation in background"""
        try:
            self.simulator.start_simulation(duration_hours, real_time)
        except Exception as e:
            print(f"❌ Simulation error: {e}")

    def _run_main_processing_loop(self):
        """Main processing loop for sensor data"""
        print("🔄 Processing real-time sensor data...")
        print("   Press Ctrl+C to stop\n")

        iteration = 0
        last_stats_time = time.time()

        while self.running:
            try:
                # Get sensor data
                data_batch = self.simulator.get_real_time_data()

                if data_batch:
                    iteration += 1
                    timestamp = data_batch['timestamp']
                    sensor_data = data_batch['batch_data']
                    active_events = data_batch['active_events']

                    if sensor_data:
                        # Process data through the pipeline
                        self._process_data_batch(sensor_data, timestamp, iteration)

                        # Update dashboard if available
                        if self.with_dashboard:
                            self.dashboard.add_sensor_data(data_batch)

                    # Print statistics every 30 seconds
                    if time.time() - last_stats_time > 30:
                        self._print_statistics(iteration)
                        last_stats_time = time.time()

                    # Print event notifications
                    if active_events:
                        print(f"🎯 Active events: {len(active_events)}")

                time.sleep(0.5)  # Process every 500ms

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Processing error: {e}")
                time.sleep(1)

    def _process_data_batch(self, sensor_data, timestamp, iteration):
        """Process a batch of sensor data through the complete pipeline"""
        try:
            # 1. Construct dynamic graphs
            graphs = self.graph_builder.construct_graph(sensor_data, timestamp)

            # 2. Extract node features
            node_features = self.graph_builder.get_node_features()

            if len(node_features) == 0:
                return

            # 3. Run GNN model (if we have enough data)
            gnn_features = None
            if len(node_features) >= 3:  # Need minimum nodes for GNN
                try:
                    batch_size = 1
                    inputs = {
                        'node_features': tf.expand_dims(
                            tf.constant(node_features, dtype=tf.float32), 0
                        ),
                        'spatial_adj': tf.expand_dims(
                            tf.constant(graphs['spatial'], dtype=tf.float32), 0
                        ),
                        'semantic_adj': tf.expand_dims(
                            tf.constant(graphs['semantic'], dtype=tf.float32), 0
                        )
                    }

                    gnn_output = self.gnn_model(inputs)
                    gnn_features = gnn_output.numpy()[0]  # Remove batch dimension

                except Exception as e:
                    print(f"⚠️  GNN processing failed: {e}")
                    gnn_features = node_features

            else:
                gnn_features = node_features

            # 4. Run smart city applications
            self._run_applications(sensor_data, gnn_features, timestamp, iteration)

        except Exception as e:
            print(f"❌ Pipeline error: {e}")

    def _run_applications(self, sensor_data, features, timestamp, iteration):
        """Run smart city applications on processed data"""
        # Anomaly detection
        try:
            anomalies = self.anomaly_detector.detect_anomalies(
                sensor_data, features, timestamp
            )

            if anomalies:
                for anomaly in anomalies:
                    print(f"🚨 ANOMALY: {anomaly.anomaly_type} at {anomaly.sensor_id} "
                          f"(score: {anomaly.anomaly_score:.2f})")

                    if self.with_dashboard:
                        for anomaly in anomalies:
                            self.dashboard.add_anomaly(anomaly)

        except Exception as e:
            print(f"⚠️  Anomaly detection failed: {e}")

        # Traffic optimization
        try:
            traffic_recs = self.traffic_optimizer.optimize_traffic(
                sensor_data, features, timestamp
            )

            if traffic_recs:
                for rec in traffic_recs:
                    print(f"🚦 TRAFFIC: {rec.recommendation_type} for {rec.area_id} "
                          f"({rec.current_condition} → {rec.expected_improvement:.1f}% improvement)")

                    if self.with_dashboard:
                        for rec in traffic_recs:
                            self.dashboard.add_traffic_recommendation(rec)

        except Exception as e:
            print(f"⚠️  Traffic optimization failed: {e}")

        # Emergency detection
        try:
            emergency_alerts = self.emergency_detector.detect_emergencies(
                sensor_data, features, timestamp
            )

            if emergency_alerts:
                for alert in emergency_alerts:
                    print(f"🆘 EMERGENCY: {alert.emergency_type.upper()} "
                          f"(severity: {alert.severity}, confidence: {alert.confidence:.2f})")
                    print(f"    Location: {alert.location}")
                    print(f"    Response: {alert.recommended_response}")

                    if self.with_dashboard:
                        for alert in emergency_alerts:
                            self.dashboard.add_emergency_alert(alert)

        except Exception as e:
            print(f"⚠️  Emergency detection failed: {e}")

    def _print_statistics(self, iteration):
        """Print system statistics"""
        print("\n📈 SYSTEM STATISTICS")
        print("-" * 30)

        # Graph statistics
        if hasattr(self.graph_builder, 'graph_history') and self.graph_builder.graph_history:
            graph_stats = self.graph_builder.get_graph_statistics()
            print(f"🕸️  Graph: {graph_stats.get('merged', {}).get('num_edges', 0)} edges, "
                  f"{graph_stats.get('merged', {}).get('density', 0):.3f} density")

        # Application statistics
        anomaly_stats = self.anomaly_detector.get_anomaly_statistics()
        traffic_stats = self.traffic_optimizer.get_traffic_statistics()
        emergency_stats = self.emergency_detector.get_emergency_statistics()

        print(f"🚨 Anomalies: {anomaly_stats['currently_anomalous']}/{anomaly_stats['total_sensors_monitored']} sensors")
        print(f"🚦 Traffic: {traffic_stats['congested_areas']}/{traffic_stats['total_areas_monitored']} areas congested")
        print(f"🆘 Emergencies: {emergency_stats['currently_active_alerts']} active alerts")
        print(f"🔄 Iterations: {iteration}")
        print()

    def run_batch_demo(self, duration_minutes=5):
        """Run a quick batch demonstration"""
        print("\n🎬 Running batch demonstration...")

        # Generate sample data
        print(f"📊 Generating {duration_minutes} minutes of synthetic data...")
        filename = f"demo_data_{int(time.time())}.csv"
        df = self.simulator.export_data_to_file(filename, duration_minutes)

        print(f"✅ Generated {len(df)} sensor readings")
        print(f"📁 Data saved to: {filename}")

        # Process sample of the data
        print("\n🔬 Processing sample data through pipeline...")

        # Group data by timestamp (simulate real-time batches)
        timestamps = sorted(df['timestamp'].unique())[:10]  # First 10 timestamps

        for i, timestamp in enumerate(timestamps):
            batch_df = df[df['timestamp'] == timestamp]
            sensor_data = {}

            for _, row in batch_df.iterrows():
                sensor_data[row['sensor_id']] = {
                    'value': row['value'],
                    'sensor_type': row['sensor_type'],
                    'location': eval(row['location']),  # Convert string back to tuple
                    'timestamp': row['timestamp'],
                    'quality': row['quality']
                }

            if sensor_data:
                print(f"🔄 Processing batch {i+1}/10 ({len(sensor_data)} sensors)...")
                self._process_data_batch(sensor_data, timestamp, i+1)

        print("\n✅ Batch demonstration completed!")

        # Clean up
        if os.path.exists(filename):
            os.remove(filename)
            print(f"🗑️  Cleaned up {filename}")


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n👋 Shutting down UrbanSense...")
    sys.exit(0)


def main():
    """Main entry point"""
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description='UrbanSense: IoT Sensor Fusion Demo')
    parser.add_argument('--sensors', type=int, default=50,
                       help='Number of sensors to simulate (default: 50)')
    parser.add_argument('--area', type=float, default=2.0,
                       help='Area size in km² (default: 2.0)')
    parser.add_argument('--duration', type=float, default=0.5,
                       help='Simulation duration in hours (default: 0.5)')
    parser.add_argument('--no-dashboard', action='store_true',
                       help='Run without web dashboard')
    parser.add_argument('--batch', action='store_true',
                       help='Run batch demo instead of real-time')
    parser.add_argument('--real-time', action='store_true',
                       help='Run in real-time (slower but more realistic)')

    args = parser.parse_args()

    print("🏙️  UrbanSense: Real-time IoT Sensor Fusion with Graph Neural Networks")
    print("🔬 Based on multimodal AI research for smart city applications")
    print("=" * 70)

    try:
        # Create demo instance
        demo = UrbanSenseDemo(
            num_sensors=args.sensors,
            area_km2=args.area,
            with_dashboard=not args.no_dashboard
        )

        if args.batch:
            # Run batch demonstration
            demo.run_batch_demo(duration_minutes=int(args.duration * 60))
        else:
            # Run real-time simulation
            if not args.no_dashboard:
                print("\n🌐 Open http://localhost:8050 in your browser to view the dashboard")
                print("⏰ Dashboard will start in a few seconds...")

            demo.run_simulation(duration_hours=args.duration, real_time=args.real_time)

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)

    print("\n✅ UrbanSense demo completed successfully!")
    print("🙏 Thank you for exploring multimodal IoT sensor fusion!")


if __name__ == '__main__':
    main()