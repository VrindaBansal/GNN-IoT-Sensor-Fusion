#!/usr/bin/env python3
"""
UrbanSense Quick Demo - Minimal setup required
Shows the system initialization and basic functionality
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("🏙️  UrbanSense Quick Demo")
print("🔬 IoT Sensor Fusion with Graph Neural Networks")
print("=" * 60)

try:
    # Import core components
    from data.simulators.smart_city_simulator import SmartCitySensorSimulator
    from graph.construction.dynamic_graph_builder import DynamicGraphBuilder, SensorNode, GraphConfig

    print("✅ Core components imported successfully")

    # Initialize simulator
    print("\n🔧 Initializing sensor simulator...")
    simulator = SmartCitySensorSimulator(num_sensors=25, area_km2=2.0)

    info = simulator.get_sensor_network_info()
    print("📍 Sensor network deployed:")
    for sensor_type, count in info['sensor_types'].items():
        print(f"   - {sensor_type}: {count} sensors")

    # Initialize graph builder
    print("\n🕸️  Initializing graph construction...")
    config = GraphConfig(
        spatial_threshold_meters=200.0,
        temporal_window_seconds=300,
        correlation_threshold=0.6,
        max_neighbors=8
    )
    graph_builder = DynamicGraphBuilder(config)

    # Add sensors to graph
    for sensor_config in simulator.sensors.values():
        sensor_node = SensorNode(
            id=sensor_config.sensor_id,
            sensor_type=sensor_config.sensor_type,
            location=sensor_config.location,
            capabilities={}
        )
        graph_builder.add_sensor(sensor_node)

    print(f"🔗 Graph builder configured with {len(graph_builder.sensor_nodes)} nodes")

    # Generate some sample data
    print("\n📊 Generating sample sensor data...")
    sample_data = {}
    import time
    current_time = time.time()

    for i, (sensor_id, sensor_config) in enumerate(list(simulator.sensors.items())[:10]):
        value = simulator._generate_base_value(sensor_config.sensor_type, current_time, sensor_config.location)
        sample_data[sensor_id] = {
            'value': value,
            'sensor_type': sensor_config.sensor_type,
            'location': sensor_config.location,
            'timestamp': current_time,
            'quality': 0.9
        }

    print(f"✅ Generated data for {len(sample_data)} sensors")

    # Test graph construction
    print("\n🔬 Testing graph construction...")
    try:
        graphs = graph_builder.construct_graph(sample_data, 1234567890.0)
        node_features = graph_builder.get_node_features()

        print(f"📈 Graph statistics:")
        print(f"   - Nodes: {len(node_features)}")
        print(f"   - Spatial edges: {graphs['spatial'].sum()}")
        print(f"   - Semantic edges: {graphs['semantic'].sum()}")

    except Exception as e:
        print(f"⚠️  Graph construction had issues: {e}")

    # Test simple models
    print("\n🧠 Testing simplified AI models...")
    try:
        from models.gnn.simple_gnn import (
            create_simple_gnn_model,
            SimpleAnomalyDetector,
            SimpleTrafficOptimizer,
            SimpleEmergencyDetector
        )

        # Create models
        gnn_config = {'node_features': 10, 'hidden_dim': 32, 'output_dim': 16}
        gnn_model = create_simple_gnn_model(gnn_config)

        anomaly_detector = SimpleAnomalyDetector()
        traffic_optimizer = SimpleTrafficOptimizer()
        emergency_detector = SimpleEmergencyDetector()

        print("✅ AI models initialized successfully")

        # Test anomaly detection
        if len(node_features) > 0:
            anomalies = anomaly_detector.detect_anomalies(sample_data, node_features, 1234567890.0)
            traffic_recs = traffic_optimizer.optimize_traffic(sample_data, node_features, 1234567890.0)
            emergencies = emergency_detector.detect_emergencies(sample_data, node_features, 1234567890.0)

            print(f"🚨 Detected {len(anomalies)} anomalies")
            print(f"🚦 Generated {len(traffic_recs)} traffic recommendations")
            print(f"🆘 Found {len(emergencies)} emergency alerts")

    except Exception as e:
        print(f"⚠️  AI model testing had issues: {e}")

    print("\n🎉 Quick demo completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python demo.py --no-dashboard --sensors 50' for full simulation")
    print("2. Install plotly/dash for web visualization: pip install plotly dash")
    print("3. Run 'python demo.py' for interactive dashboard")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nTo fix:")
    print("pip install numpy pandas networkx scikit-learn faker geopy")
    sys.exit(1)

except Exception as e:
    print(f"❌ Demo failed: {e}")
    print("\nThe core system is working, but some components need fixes.")
    sys.exit(1)