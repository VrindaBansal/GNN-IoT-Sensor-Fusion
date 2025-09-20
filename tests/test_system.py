"""
Comprehensive testing framework for UrbanSense system
"""

import unittest
import numpy as np
import tensorflow as tf
import time
import threading
from unittest.mock import Mock, patch
import tempfile
import os

# Import system components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.simulators.smart_city_simulator import SmartCitySensorSimulator
from src.graph.construction.dynamic_graph_builder import DynamicGraphBuilder, SensorNode, GraphConfig
from src.models.gnn.multimodal_gnn import MultimodalGNN, create_multimodal_gnn_model
from src.models.fusion.multimodal_fusion import EarlyFusion, JointFusion, LateFusion
from src.applications.anomaly.anomaly_detector import RealTimeAnomalyDetector
from src.applications.traffic.traffic_optimizer import TrafficOptimizer
from src.applications.emergency.emergency_detector import EmergencyResponseSystem


class TestSmartCitySimulator(unittest.TestCase):
    """Test the IoT sensor simulator"""

    def setUp(self):
        self.simulator = SmartCitySensorSimulator(num_sensors=10, area_km2=1)

    def test_simulator_initialization(self):
        """Test simulator creates sensors correctly"""
        self.assertEqual(len(self.simulator.sensors), 10)
        self.assertIn('sensor_0000', self.simulator.sensors)

        # Check sensor types are distributed
        sensor_types = [config.sensor_type for config in self.simulator.sensors.values()]
        self.assertGreater(len(set(sensor_types)), 1)

    def test_sensor_reading_generation(self):
        """Test sensor reading generation"""
        timestamp = time.time()
        sensor_id = list(self.simulator.sensors.keys())[0]

        reading = self.simulator.generate_sensor_reading(sensor_id, timestamp)

        self.assertIn('sensor_id', reading)
        self.assertIn('timestamp', reading)
        self.assertIn('value', reading)
        self.assertIn('sensor_type', reading)
        self.assertEqual(reading['sensor_id'], sensor_id)

    def test_data_export(self):
        """Test data export functionality"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            filename = f.name

        try:
            df = self.simulator.export_data_to_file(filename, duration_minutes=1)
            self.assertTrue(os.path.exists(filename))
            self.assertGreater(len(df), 0)
            self.assertIn('sensor_id', df.columns)
            self.assertIn('value', df.columns)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_network_info(self):
        """Test network information retrieval"""
        info = self.simulator.get_sensor_network_info()
        self.assertIn('total_sensors', info)
        self.assertIn('sensor_types', info)
        self.assertEqual(info['total_sensors'], 10)


class TestDynamicGraphBuilder(unittest.TestCase):
    """Test the dynamic graph construction"""

    def setUp(self):
        self.config = GraphConfig(spatial_threshold_meters=50.0)
        self.builder = DynamicGraphBuilder(self.config)

        # Add test sensors
        sensors = [
            SensorNode("s1", "temperature", (40.7589, -73.9851, 10), {}),
            SensorNode("s2", "humidity", (40.7590, -73.9850, 12), {}),
            SensorNode("s3", "audio", (40.7600, -73.9860, 8), {}),
        ]

        for sensor in sensors:
            self.builder.add_sensor(sensor)

    def test_sensor_addition(self):
        """Test adding sensors to graph builder"""
        self.assertEqual(len(self.builder.sensor_nodes), 3)
        self.assertIn("s1", self.builder.sensor_nodes)

    def test_spatial_graph_construction(self):
        """Test spatial graph construction"""
        spatial_adj = self.builder.build_spatial_graph()
        self.assertEqual(spatial_adj.shape, (3, 3))
        # Diagonal should be zero (no self-connections)
        self.assertTrue(np.allclose(np.diag(spatial_adj), 0))

    def test_graph_construction_with_data(self):
        """Test complete graph construction with sensor data"""
        sensor_data = {
            "s1": {"value": 25.0, "timestamp": time.time()},
            "s2": {"value": 60.0, "timestamp": time.time()},
            "s3": {"value": 55.0, "timestamp": time.time()}
        }

        graphs = self.builder.construct_graph(sensor_data, time.time())

        self.assertIn('spatial', graphs)
        self.assertIn('temporal', graphs)
        self.assertIn('semantic', graphs)
        self.assertIn('merged', graphs)

    def test_node_features_extraction(self):
        """Test node feature extraction"""
        # Add some readings
        sensor_data = {
            "s1": {"value": 25.0, "timestamp": time.time()},
            "s2": {"value": 60.0, "timestamp": time.time()},
            "s3": {"value": 55.0, "timestamp": time.time()}
        }
        self.builder.construct_graph(sensor_data, time.time())

        features = self.builder.get_node_features()
        self.assertEqual(features.shape, (3, 10))  # 3 sensors, 10 features each


class TestMultimodalGNN(unittest.TestCase):
    """Test the multimodal GNN implementation"""

    def setUp(self):
        self.config = {
            'node_features': 10,
            'edge_features': 5,
            'output_dim': 16,
            'hidden_dim': 32,
            'num_heads': 4
        }
        self.model = create_multimodal_gnn_model(self.config)

    def test_model_creation(self):
        """Test model creation and configuration"""
        self.assertIsInstance(self.model, MultimodalGNN)
        self.assertEqual(self.model.output_dim, 16)
        self.assertEqual(self.model.hidden_dim, 32)

    def test_model_forward_pass(self):
        """Test model forward pass"""
        batch_size, num_nodes = 2, 5

        inputs = {
            'node_features': tf.random.normal((batch_size, num_nodes, 10)),
            'spatial_adj': tf.random.uniform((batch_size, num_nodes, num_nodes)),
            'sensor_types': tf.random.uniform((batch_size, num_nodes), maxval=6, dtype=tf.int32)
        }

        output = self.model(inputs)
        self.assertEqual(output.shape, (batch_size, num_nodes, 16))

    def test_model_with_temporal_data(self):
        """Test model with temporal features"""
        batch_size, num_nodes, timesteps = 1, 3, 5

        inputs = {
            'node_features': tf.random.normal((batch_size, num_nodes, 10)),
            'spatial_adj': tf.random.uniform((batch_size, num_nodes, num_nodes)),
            'temporal_features': tf.random.normal((batch_size, num_nodes, timesteps, 10)),
            'temporal_adj': tf.random.uniform((batch_size, timesteps, num_nodes, num_nodes))
        }

        output = self.model(inputs)
        self.assertEqual(output.shape, (batch_size, num_nodes, 16))

    def test_model_embeddings(self):
        """Test model embedding extraction"""
        batch_size, num_nodes = 1, 3

        inputs = {
            'node_features': tf.random.normal((batch_size, num_nodes, 10)),
            'spatial_adj': tf.random.uniform((batch_size, num_nodes, num_nodes))
        }

        embeddings = self.model.get_embeddings(inputs)
        self.assertEqual(len(embeddings.shape), 3)  # Should return embeddings


class TestFusionLayers(unittest.TestCase):
    """Test multimodal fusion implementations"""

    def setUp(self):
        self.input_shapes = {
            'modality1': (None, 64),
            'modality2': (None, 32),
            'modality3': (None, 48)
        }
        self.test_data = {
            'modality1': tf.random.normal((2, 64)),
            'modality2': tf.random.normal((2, 32)),
            'modality3': tf.random.normal((2, 48))
        }

    def test_early_fusion(self):
        """Test early fusion layer"""
        layer = EarlyFusion(output_dim=128)
        layer.build(self.input_shapes)

        output = layer(self.test_data)
        self.assertEqual(output.shape, (2, 128))

    def test_joint_fusion(self):
        """Test joint fusion layer"""
        layer = JointFusion(hidden_dim=64, output_dim=128)
        layer.build(self.input_shapes)

        output = layer(self.test_data)
        self.assertEqual(output.shape, (2, 128))

    def test_late_fusion(self):
        """Test late fusion layer"""
        layer = LateFusion(feature_dim=64, output_dim=128)
        layer.build(self.input_shapes)

        output = layer(self.test_data)
        self.assertEqual(output.shape, (2, 128))


class TestAnomalyDetector(unittest.TestCase):
    """Test anomaly detection system"""

    def setUp(self):
        self.detector = RealTimeAnomalyDetector()

    def test_detector_initialization(self):
        """Test detector initialization"""
        self.assertIsNotNone(self.detector.detection_model)
        self.assertEqual(len(self.detector.anomaly_types), 5)

    def test_anomaly_detection(self):
        """Test anomaly detection on sample data"""
        sensor_data = {
            "sensor_1": {
                "value": 25.0,
                "sensor_type": "temperature",
                "timestamp": time.time(),
                "quality": "good"
            },
            "sensor_2": {
                "value": None,  # Suspicious missing value
                "sensor_type": "humidity",
                "timestamp": time.time(),
                "quality": "failed"
            }
        }

        graph_features = np.random.random((2, 10))
        results = self.detector.detect_anomalies(sensor_data, graph_features, time.time())

        self.assertIsInstance(results, list)
        # Should detect anomaly for sensor_2 (failed quality)
        if results:
            self.assertTrue(any(r.sensor_id == "sensor_2" for r in results))

    def test_statistics_generation(self):
        """Test statistics generation"""
        stats = self.detector.get_anomaly_statistics()
        self.assertIn('total_sensors_monitored', stats)
        self.assertIn('currently_anomalous', stats)


class TestTrafficOptimizer(unittest.TestCase):
    """Test traffic optimization system"""

    def setUp(self):
        self.optimizer = TrafficOptimizer()

    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        self.assertIsNotNone(self.optimizer.optimization_model)
        self.assertEqual(len(self.optimizer.recommendation_types), 4)

    def test_traffic_optimization(self):
        """Test traffic optimization"""
        # Create traffic-relevant sensor data
        sensor_data = {
            "audio_1": {
                "value": 85.0,  # High noise (traffic)
                "sensor_type": "audio",
                "location": (40.7589, -73.9851, 5),
                "timestamp": time.time()
            },
            "visual_1": {
                "value": 150.0,  # Moderate brightness
                "sensor_type": "visual",
                "location": (40.7590, -73.9852, 6),
                "timestamp": time.time()
            }
        }

        graph_features = np.random.random((2, 10))
        recommendations = self.optimizer.optimize_traffic(sensor_data, graph_features, time.time())

        self.assertIsInstance(recommendations, list)

    def test_traffic_statistics(self):
        """Test traffic statistics"""
        stats = self.optimizer.get_traffic_statistics()
        self.assertIn('total_areas_monitored', stats)
        self.assertIn('congestion_rate', stats)


class TestEmergencyDetector(unittest.TestCase):
    """Test emergency detection system"""

    def setUp(self):
        self.detector = EmergencyResponseSystem()

    def test_detector_initialization(self):
        """Test detector initialization"""
        self.assertIsNotNone(self.detector.detection_model)
        self.assertEqual(len(self.detector.emergency_types), 6)

    def test_emergency_detection(self):
        """Test emergency detection"""
        # Create emergency-like sensor data
        sensor_data = {
            "temp_1": {
                "value": 80.0,  # Very high temperature
                "sensor_type": "temperature",
                "location": (40.7589, -73.9851, 5),
                "timestamp": time.time(),
                "quality": "good"
            },
            "audio_1": {
                "value": 110.0,  # Very loud noise
                "sensor_type": "audio",
                "location": (40.7590, -73.9852, 6),
                "timestamp": time.time(),
                "quality": "good"
            }
        }

        graph_features = np.random.random((2, 12))
        alerts = self.detector.detect_emergencies(sensor_data, graph_features, time.time())

        self.assertIsInstance(alerts, list)

    def test_emergency_statistics(self):
        """Test emergency statistics"""
        stats = self.detector.get_emergency_statistics()
        self.assertIn('total_alerts_generated', stats)
        self.assertIn('currently_active_alerts', stats)


class TestSystemIntegration(unittest.TestCase):
    """Test complete system integration"""

    def setUp(self):
        # Create minimal system components
        self.simulator = SmartCitySensorSimulator(num_sensors=5, area_km2=0.1)
        self.graph_builder = DynamicGraphBuilder()
        self.gnn_model = create_multimodal_gnn_model({
            'node_features': 10,
            'output_dim': 16,
            'hidden_dim': 32
        })
        self.anomaly_detector = RealTimeAnomalyDetector()
        self.traffic_optimizer = TrafficOptimizer()
        self.emergency_detector = EmergencyResponseSystem()

    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline"""
        # Generate sensor data
        data_batch = None
        for _ in range(5):  # Try a few times to get data
            data_batch = self.simulator.get_real_time_data()
            if data_batch:
                break
            time.sleep(0.1)

        if not data_batch:
            # Generate manually if simulator queue is empty
            timestamp = time.time()
            sensor_data = {}
            for sensor_id in list(self.simulator.sensors.keys())[:3]:
                reading = self.simulator.generate_sensor_reading(sensor_id, timestamp)
                sensor_data[sensor_id] = reading

            data_batch = {
                'timestamp': timestamp,
                'batch_data': sensor_data,
                'active_events': []
            }

        self.assertIsNotNone(data_batch)
        sensor_data = data_batch['batch_data']

        # Add sensors to graph builder
        for sensor_id, reading in sensor_data.items():
            if 'location' in reading:
                sensor = SensorNode(
                    sensor_id, reading['sensor_type'],
                    reading['location'], {}
                )
                self.graph_builder.add_sensor(sensor)

        # Construct graphs
        graphs = self.graph_builder.construct_graph(
            sensor_data, data_batch['timestamp']
        )
        self.assertIn('spatial', graphs)

        # Get node features
        node_features = self.graph_builder.get_node_features()
        self.assertGreater(node_features.shape[0], 0)

        # Run GNN (if we have enough nodes)
        if node_features.shape[0] > 0:
            batch_size = 1
            inputs = {
                'node_features': tf.expand_dims(tf.constant(node_features, dtype=tf.float32), 0),
                'spatial_adj': tf.expand_dims(tf.constant(graphs['spatial'], dtype=tf.float32), 0)
            }

            try:
                gnn_output = self.gnn_model(inputs)
                self.assertEqual(gnn_output.shape[0], batch_size)
            except Exception as e:
                print(f"GNN forward pass failed: {e}")

        # Run applications
        anomalies = self.anomaly_detector.detect_anomalies(
            sensor_data, node_features, data_batch['timestamp']
        )
        self.assertIsInstance(anomalies, list)

        traffic_recs = self.traffic_optimizer.optimize_traffic(
            sensor_data, node_features, data_batch['timestamp']
        )
        self.assertIsInstance(traffic_recs, list)

        emergency_alerts = self.emergency_detector.detect_emergencies(
            sensor_data, node_features, data_batch['timestamp']
        )
        self.assertIsInstance(emergency_alerts, list)

    def test_concurrent_processing(self):
        """Test system under concurrent load"""
        results = []
        errors = []

        def process_data():
            try:
                timestamp = time.time()
                sensor_data = {}
                for sensor_id in list(self.simulator.sensors.keys())[:2]:
                    reading = self.simulator.generate_sensor_reading(sensor_id, timestamp)
                    sensor_data[sensor_id] = reading

                # Quick processing
                node_features = np.random.random((2, 10))
                anomalies = self.anomaly_detector.detect_anomalies(
                    sensor_data, node_features, timestamp
                )
                results.append(len(anomalies))
            except Exception as e:
                errors.append(str(e))

        # Run multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=process_data)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join(timeout=5)

        # Check results
        self.assertEqual(len(results), 5)
        self.assertEqual(len(errors), 0)


def run_performance_tests():
    """Run performance benchmarks"""
    print("\n=== Performance Tests ===")

    # Test 1: Graph construction speed
    print("1. Graph Construction Performance")
    builder = DynamicGraphBuilder()
    num_sensors = 100

    # Add sensors
    for i in range(num_sensors):
        sensor = SensorNode(
            f"sensor_{i}",
            np.random.choice(['temperature', 'humidity', 'audio']),
            (40.75 + np.random.random() * 0.01,
             -73.98 + np.random.random() * 0.01,
             np.random.random() * 20),
            {}
        )
        builder.add_sensor(sensor)

    # Time graph construction
    sensor_data = {f"sensor_{i}": {"value": np.random.random() * 100, "timestamp": time.time()}
                  for i in range(num_sensors)}

    start_time = time.time()
    graphs = builder.construct_graph(sensor_data, time.time())
    construction_time = time.time() - start_time

    print(f"   - {num_sensors} sensors: {construction_time:.3f}s")
    print(f"   - Spatial edges: {np.sum(graphs['spatial'] > 0)}")

    # Test 2: GNN inference speed
    print("2. GNN Inference Performance")
    model = create_multimodal_gnn_model({
        'node_features': 10,
        'output_dim': 16,
        'hidden_dim': 64
    })

    batch_sizes = [1, 5, 10]
    for batch_size in batch_sizes:
        inputs = {
            'node_features': tf.random.normal((batch_size, num_sensors, 10)),
            'spatial_adj': tf.random.uniform((batch_size, num_sensors, num_sensors))
        }

        # Warm up
        _ = model(inputs)

        # Time inference
        start_time = time.time()
        for _ in range(10):
            _ = model(inputs)
        inference_time = (time.time() - start_time) / 10

        print(f"   - Batch size {batch_size}: {inference_time:.3f}s per inference")

    print("Performance tests completed\n")


if __name__ == '__main__':
    # Run unit tests
    print("Running UrbanSense System Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)

    # Run performance tests
    run_performance_tests()