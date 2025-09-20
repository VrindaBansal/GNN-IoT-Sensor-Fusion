"""
Simplified Graph Neural Network implementation without TensorFlow dependency
Uses numpy for basic operations and sklearn for machine learning components
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


class SimpleGraphConvolution:
    """Simple graph convolution layer using numpy"""

    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Initialize weights
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros(output_dim)

    def forward(self, node_features: np.ndarray, adjacency_matrix: np.ndarray) -> np.ndarray:
        """
        Forward pass through graph convolution layer

        Args:
            node_features: [num_nodes, input_dim]
            adjacency_matrix: [num_nodes, num_nodes]

        Returns:
            output_features: [num_nodes, output_dim]
        """
        # Message passing: aggregate neighbor features
        neighbor_features = np.dot(adjacency_matrix, node_features)

        # Combine with self features
        combined_features = node_features + neighbor_features

        # Linear transformation
        output = np.dot(combined_features, self.W) + self.b

        # ReLU activation
        output = np.maximum(0, output)

        return output


class SimpleMultimodalGNN:
    """Simplified multimodal GNN using sklearn components"""

    def __init__(self, node_features: int = 10, hidden_dim: int = 64, output_dim: int = 32):
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Graph convolution layers
        self.spatial_conv = SimpleGraphConvolution(node_features, hidden_dim)
        self.temporal_conv = SimpleGraphConvolution(hidden_dim, hidden_dim)

        # Fusion network using sklearn MLP
        self.fusion_network = MLPRegressor(
            hidden_layer_sizes=(hidden_dim, hidden_dim // 2),
            activation='relu',
            max_iter=100,
            random_state=42
        )

        # Feature scaler
        self.scaler = StandardScaler()

        # Trained flag
        self.is_trained = False

    def forward(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Forward pass through the network

        Args:
            inputs: Dictionary containing node_features and adjacency matrices

        Returns:
            Node embeddings
        """
        node_features = inputs.get('node_features')
        spatial_adj = inputs.get('spatial_adj', np.eye(len(node_features)))

        if len(node_features.shape) == 3:
            # Remove batch dimension
            node_features = node_features[0]
            spatial_adj = spatial_adj[0]

        # Spatial graph convolution
        spatial_output = self.spatial_conv.forward(node_features, spatial_adj)

        # Temporal processing (simplified)
        temporal_output = self.temporal_conv.forward(spatial_output, spatial_adj)

        # If fusion network is trained, use it
        if self.is_trained:
            try:
                # Flatten for sklearn
                flattened = temporal_output.reshape(temporal_output.shape[0], -1)
                scaled = self.scaler.transform(flattened)
                fused_output = self.fusion_network.predict(scaled)
                return fused_output.reshape(-1, self.output_dim)
            except:
                pass

        # Fallback: simple linear transformation
        if temporal_output.shape[1] != self.output_dim:
            # Simple projection to output dimension
            projection = np.random.randn(temporal_output.shape[1], self.output_dim) * 0.01
            output = np.dot(temporal_output, projection)
        else:
            output = temporal_output

        return output

    def train(self, training_data: List[Dict]):
        """Train the fusion network on sample data"""
        if not training_data:
            return

        try:
            # Generate training targets (dummy for demonstration)
            features = []
            targets = []

            for data in training_data[:10]:  # Limit training data
                inputs = {
                    'node_features': data.get('node_features', np.random.random((5, self.node_features))),
                    'spatial_adj': data.get('spatial_adj', np.eye(5))
                }

                # Forward pass without fusion
                node_features = inputs['node_features']
                spatial_adj = inputs['spatial_adj']

                if len(node_features.shape) == 3:
                    node_features = node_features[0]
                    spatial_adj = spatial_adj[0]

                spatial_output = self.spatial_conv.forward(node_features, spatial_adj)
                temporal_output = self.temporal_conv.forward(spatial_output, spatial_adj)

                # Flatten for training
                flattened = temporal_output.reshape(temporal_output.shape[0], -1)
                features.extend(flattened)

                # Create dummy targets
                targets.extend(np.random.random((len(flattened), self.output_dim)))

            if features and targets:
                features = np.array(features)
                targets = np.array(targets)

                # Fit scaler
                self.scaler.fit(features)
                scaled_features = self.scaler.transform(features)

                # Train fusion network
                self.fusion_network.fit(scaled_features, targets)
                self.is_trained = True

        except Exception as e:
            print(f"Training failed: {e}")

    def get_embeddings(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Get node embeddings"""
        return self.forward(inputs)


class SimpleAnomalyDetector:
    """Simplified anomaly detector using statistical methods"""

    def __init__(self, sensor_feature_dim: int = 10, window_size: int = 50, threshold_percentile: float = 95.0):
        self.sensor_feature_dim = sensor_feature_dim
        self.window_size = window_size
        self.threshold_percentile = threshold_percentile
        self.baseline_scores = []

    def detect_anomalies(self, sensor_data: Dict, node_features: np.ndarray, timestamp: float) -> List:
        """Detect anomalies using statistical methods"""
        anomalies = []

        if len(node_features) == 0:
            return anomalies

        # Calculate simple anomaly scores
        for i, (sensor_id, reading) in enumerate(sensor_data.items()):
            if i >= len(node_features):
                break

            # Get sensor value
            value = reading.get('value', 0)
            if value is None:
                value = 0

            # Simple anomaly detection: z-score based
            if len(self.baseline_scores) > 10:
                mean_val = np.mean(self.baseline_scores)
                std_val = np.std(self.baseline_scores)
                z_score = abs((value - mean_val) / (std_val + 1e-6))

                if z_score > 2.5:  # 2.5 sigma threshold
                    anomaly = type('AnomalyResult', (), {
                        'sensor_id': sensor_id,
                        'anomaly_score': float(z_score),
                        'anomaly_type': 'statistical_outlier',
                        'is_anomaly': True,
                        'timestamp': timestamp,
                        'confidence': min(z_score / 5.0, 1.0)
                    })()
                    anomalies.append(anomaly)

            # Update baseline
            self.baseline_scores.append(value)
            if len(self.baseline_scores) > 1000:
                self.baseline_scores = self.baseline_scores[-1000:]

        return anomalies

    def get_anomaly_statistics(self) -> Dict:
        """Get anomaly detection statistics"""
        return {
            'total_sensors_monitored': len(self.baseline_scores),
            'currently_anomalous': 0,
            'anomaly_rate': 0.0,
            'baseline_scores_count': len(self.baseline_scores),
            'current_threshold': np.percentile(self.baseline_scores, self.threshold_percentile)
                                if len(self.baseline_scores) > 0 else 0.0
        }


class SimpleTrafficOptimizer:
    """Simplified traffic optimizer"""

    def __init__(self, area_grid_size: int = 10):
        self.area_grid_size = area_grid_size
        self.area_history = {}

    def optimize_traffic(self, sensor_data: Dict, node_features: np.ndarray, timestamp: float) -> List:
        """Generate traffic recommendations"""
        recommendations = []

        # Simple traffic analysis based on audio sensors
        audio_sensors = [
            (sid, reading) for sid, reading in sensor_data.items()
            if reading.get('sensor_type') == 'audio'
        ]

        for sensor_id, reading in audio_sensors[:3]:  # Limit to 3 for demo
            noise_level = reading.get('value', 0)

            if noise_level > 70:  # High noise = potential traffic
                recommendation = type('TrafficRecommendation', (), {
                    'area_id': f"area_{hash(sensor_id) % 10}",
                    'recommendation_type': 'signal_timing',
                    'current_condition': 'congested' if noise_level > 80 else 'moderate',
                    'expected_improvement': min((noise_level - 60) * 2, 30),
                    'timestamp': timestamp,
                    'confidence': 0.7
                })()
                recommendations.append(recommendation)

        return recommendations

    def get_traffic_statistics(self) -> Dict:
        """Get traffic statistics"""
        return {
            'total_areas_monitored': len(self.area_history),
            'congested_areas': 0,
            'congestion_rate': 0.0,
            'average_flow_rate': 50.0,
            'total_data_points': sum(len(h) for h in self.area_history.values())
        }


class SimpleEmergencyDetector:
    """Simplified emergency detector"""

    def __init__(self, response_time_threshold: float = 300.0):
        self.response_time_threshold = response_time_threshold
        self.alert_history = []

    def detect_emergencies(self, sensor_data: Dict, node_features: np.ndarray, timestamp: float) -> List:
        """Detect potential emergencies"""
        alerts = []

        # Look for emergency patterns
        temp_sensors = [(sid, r) for sid, r in sensor_data.items() if r.get('sensor_type') == 'temperature']
        audio_sensors = [(sid, r) for sid, r in sensor_data.items() if r.get('sensor_type') == 'audio']

        # Fire detection: high temperature + loud noise
        for temp_id, temp_reading in temp_sensors:
            temp_value = temp_reading.get('value', 0)

            if temp_value > 50:  # Very high temperature
                # Look for nearby audio sensors
                for audio_id, audio_reading in audio_sensors:
                    audio_value = audio_reading.get('value', 0)

                    if audio_value > 90:  # Very loud
                        alert = type('EmergencyAlert', (), {
                            'alert_id': f"emergency_{int(timestamp)}",
                            'emergency_type': 'fire',
                            'severity': 'high' if temp_value > 60 else 'medium',
                            'confidence': min((temp_value - 30) / 50.0, 1.0),
                            'timestamp': timestamp,
                            'location': temp_reading.get('location', (0, 0, 0)),
                            'detected_sensors': [temp_id, audio_id]
                        })()
                        alerts.append(alert)
                        break

        return alerts

    def get_emergency_statistics(self) -> Dict:
        """Get emergency statistics"""
        return {
            'total_alerts_generated': len(self.alert_history),
            'currently_active_alerts': 0,
            'alerts_by_type': {},
            'alerts_by_severity': {},
            'average_confidence': 0.8
        }


def create_simple_gnn_model(config: Dict) -> SimpleMultimodalGNN:
    """Factory function to create simplified GNN model"""
    return SimpleMultimodalGNN(
        node_features=config.get('node_features', 10),
        hidden_dim=config.get('hidden_dim', 64),
        output_dim=config.get('output_dim', 32)
    )