"""
Real-time Anomaly Detection for IoT Sensor Networks
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class AnomalyResult:
    """Result of anomaly detection"""
    sensor_id: str
    timestamp: float
    anomaly_score: float
    is_anomaly: bool
    anomaly_type: str
    confidence: float
    affected_neighbors: List[str]


class AnomalyDetectionLayer(tf.keras.layers.Layer):
    """Neural network layer for anomaly detection"""

    def __init__(self, hidden_dim: int = 64, threshold: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.threshold = threshold

        # Autoencoder for reconstruction-based anomaly detection
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim // 2, activation='relu'),
            tf.keras.layers.Dense(hidden_dim // 4, activation='relu')
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim // 2, activation='relu'),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
        ])

        # Anomaly classifier
        self.anomaly_classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs, training=None):
        """
        Detect anomalies in sensor data

        Args:
            inputs: Sensor feature tensor [batch_size, num_sensors, features]

        Returns:
            Dictionary with reconstruction and anomaly scores
        """
        # Autoencoder reconstruction
        encoded = self.encoder(inputs, training=training)
        reconstructed = self.decoder(encoded, training=training)

        # Reconstruction error
        reconstruction_error = tf.reduce_mean(tf.square(inputs - reconstructed), axis=-1)

        # Anomaly classification
        anomaly_scores = self.anomaly_classifier(inputs, training=training)
        anomaly_scores = tf.squeeze(anomaly_scores, axis=-1)

        return {
            'reconstruction_error': reconstruction_error,
            'anomaly_scores': anomaly_scores,
            'reconstructed': reconstructed,
            'encoded': encoded
        }


class RealTimeAnomalyDetector:
    """Real-time anomaly detector for IoT sensor networks"""

    def __init__(self,
                 sensor_feature_dim: int = 10,
                 window_size: int = 50,
                 threshold_percentile: float = 95.0):
        self.sensor_feature_dim = sensor_feature_dim
        self.window_size = window_size
        self.threshold_percentile = threshold_percentile

        # Detection model
        self.detection_model = self._build_detection_model()

        # Historical data for adaptive thresholds
        self.sensor_history = {}
        self.baseline_scores = deque(maxlen=1000)

        # Anomaly types
        self.anomaly_types = [
            'sensor_failure',
            'data_drift',
            'outlier_reading',
            'correlation_break',
            'temporal_anomaly'
        ]

    def _build_detection_model(self) -> tf.keras.Model:
        """Build the anomaly detection model"""
        inputs = tf.keras.layers.Input(shape=(None, self.sensor_feature_dim))

        # Graph neural network features (from previous components)
        gnn_features = tf.keras.layers.Dense(64, activation='relu')(inputs)

        # Anomaly detection layer
        anomaly_layer = AnomalyDetectionLayer(hidden_dim=64)
        anomaly_outputs = anomaly_layer(gnn_features)

        model = tf.keras.Model(inputs=inputs, outputs=anomaly_outputs)
        return model

    def detect_anomalies(self,
                        sensor_data: Dict[str, Dict],
                        graph_features: np.ndarray,
                        timestamp: float) -> List[AnomalyResult]:
        """
        Detect anomalies in real-time sensor data

        Args:
            sensor_data: Dictionary of sensor readings
            graph_features: Node features from GNN
            timestamp: Current timestamp

        Returns:
            List of anomaly detection results
        """
        anomaly_results = []

        # Prepare input data
        input_features = self._prepare_features(sensor_data, graph_features)

        if input_features is None:
            return anomaly_results

        # Run detection model
        predictions = self.detection_model(input_features)

        reconstruction_errors = predictions['reconstruction_error'].numpy()
        anomaly_scores = predictions['anomaly_scores'].numpy()

        # Update baseline scores for adaptive thresholding
        self.baseline_scores.extend(reconstruction_errors.flatten())

        # Calculate adaptive threshold
        if len(self.baseline_scores) > 100:
            threshold = np.percentile(list(self.baseline_scores), self.threshold_percentile)
        else:
            threshold = 0.5  # Default threshold

        # Analyze each sensor
        sensor_ids = list(sensor_data.keys())
        for i, sensor_id in enumerate(sensor_ids):
            if i < len(reconstruction_errors):
                reconstruction_error = reconstruction_errors[i]
                anomaly_score = anomaly_scores[i]

                # Determine if anomalous
                is_anomaly = (reconstruction_error > threshold) or (anomaly_score > 0.7)

                if is_anomaly:
                    # Classify anomaly type
                    anomaly_type = self._classify_anomaly_type(
                        sensor_id, sensor_data[sensor_id], reconstruction_error, anomaly_score
                    )

                    # Find affected neighbors
                    affected_neighbors = self._find_affected_neighbors(
                        sensor_id, sensor_data, reconstruction_errors, sensor_ids
                    )

                    result = AnomalyResult(
                        sensor_id=sensor_id,
                        timestamp=timestamp,
                        anomaly_score=float(max(reconstruction_error / threshold, anomaly_score)),
                        is_anomaly=True,
                        anomaly_type=anomaly_type,
                        confidence=float(anomaly_score),
                        affected_neighbors=affected_neighbors
                    )

                    anomaly_results.append(result)

                # Update sensor history
                self._update_sensor_history(sensor_id, {
                    'timestamp': timestamp,
                    'reconstruction_error': float(reconstruction_error),
                    'anomaly_score': float(anomaly_score),
                    'is_anomaly': is_anomaly
                })

        return anomaly_results

    def _prepare_features(self, sensor_data: Dict[str, Dict],
                         graph_features: np.ndarray) -> Optional[tf.Tensor]:
        """Prepare input features for the detection model"""
        if len(sensor_data) == 0:
            return None

        batch_features = []

        for i, (sensor_id, reading) in enumerate(sensor_data.items()):
            # Extract numeric features from sensor reading
            features = []

            # Basic reading features
            if 'value' in reading and reading['value'] is not None:
                features.append(float(reading['value']))
            else:
                features.append(0.0)

            # Location features
            if 'location' in reading:
                features.extend(reading['location'][:3])  # lat, lon, elevation
            else:
                features.extend([0.0, 0.0, 0.0])

            # Timestamp features
            if 'timestamp' in reading:
                hour_of_day = (reading['timestamp'] % 86400) / 3600
                day_of_week = int((reading['timestamp'] / 86400) % 7)
                features.extend([hour_of_day / 24.0, day_of_week / 7.0])
            else:
                features.extend([0.0, 0.0])

            # Quality indicator
            quality_score = 1.0 if reading.get('quality') == 'good' else 0.0
            features.append(quality_score)

            # Add graph features if available
            if i < len(graph_features):
                graph_feat = graph_features[i]
                if len(graph_feat) > 0:
                    features.extend(graph_feat[:3])  # Take first 3 graph features
                else:
                    features.extend([0.0, 0.0, 0.0])
            else:
                features.extend([0.0, 0.0, 0.0])

            # Pad or truncate to fixed size
            if len(features) < self.sensor_feature_dim:
                features.extend([0.0] * (self.sensor_feature_dim - len(features)))
            else:
                features = features[:self.sensor_feature_dim]

            batch_features.append(features)

        # Convert to tensor
        input_tensor = tf.constant(batch_features, dtype=tf.float32)
        input_tensor = tf.expand_dims(input_tensor, 0)  # Add batch dimension

        return input_tensor

    def _classify_anomaly_type(self, sensor_id: str, reading: Dict,
                              reconstruction_error: float, anomaly_score: float) -> str:
        """Classify the type of anomaly detected"""

        # Sensor failure: Very high reconstruction error and low/missing value
        if reconstruction_error > 2.0 and (reading.get('value') is None or
                                         reading.get('quality') == 'failed'):
            return 'sensor_failure'

        # Check for temporal anomaly
        if sensor_id in self.sensor_history:
            recent_history = list(self.sensor_history[sensor_id])[-10:]
            if len(recent_history) > 5:
                recent_errors = [h['reconstruction_error'] for h in recent_history]
                avg_recent_error = np.mean(recent_errors)

                if reconstruction_error > avg_recent_error * 3:
                    return 'temporal_anomaly'

        # Data drift: Consistent moderate anomaly scores
        if 0.3 < anomaly_score < 0.7:
            return 'data_drift'

        # Outlier reading: High anomaly score but not sensor failure
        if anomaly_score > 0.7:
            return 'outlier_reading'

        # Default
        return 'correlation_break'

    def _find_affected_neighbors(self, sensor_id: str, sensor_data: Dict[str, Dict],
                               reconstruction_errors: np.ndarray,
                               sensor_ids: List[str]) -> List[str]:
        """Find neighboring sensors that might be affected by the anomaly"""
        affected = []

        # Simple heuristic: sensors with above-average reconstruction errors
        if len(reconstruction_errors) > 1:
            avg_error = np.mean(reconstruction_errors)
            threshold = avg_error * 1.5

            for i, other_sensor_id in enumerate(sensor_ids):
                if (other_sensor_id != sensor_id and
                    i < len(reconstruction_errors) and
                    reconstruction_errors[i] > threshold):
                    affected.append(other_sensor_id)

        return affected[:5]  # Limit to 5 neighbors

    def _update_sensor_history(self, sensor_id: str, record: Dict):
        """Update historical data for a sensor"""
        if sensor_id not in self.sensor_history:
            self.sensor_history[sensor_id] = deque(maxlen=self.window_size)

        self.sensor_history[sensor_id].append(record)

    def get_anomaly_statistics(self) -> Dict:
        """Get statistics about detected anomalies"""
        total_sensors = len(self.sensor_history)
        anomalous_sensors = 0
        anomaly_type_counts = {atype: 0 for atype in self.anomaly_types}

        for sensor_id, history in self.sensor_history.items():
            recent_anomalies = [r for r in list(history)[-10:] if r['is_anomaly']]
            if recent_anomalies:
                anomalous_sensors += 1

        return {
            'total_sensors_monitored': total_sensors,
            'currently_anomalous': anomalous_sensors,
            'anomaly_rate': anomalous_sensors / max(total_sensors, 1),
            'baseline_scores_count': len(self.baseline_scores),
            'current_threshold': np.percentile(list(self.baseline_scores), self.threshold_percentile)
                                if len(self.baseline_scores) > 0 else 0.5
        }

    def train_baseline(self, normal_data: List[Dict]):
        """Train the detection model on normal sensor data"""
        print("Training anomaly detection baseline...")

        # Prepare training data
        training_features = []
        for data_batch in normal_data:
            features = self._prepare_features(data_batch.get('sensor_data', {}),
                                            data_batch.get('graph_features', np.array([])))
            if features is not None:
                training_features.append(features)

        if not training_features:
            print("No training data available")
            return

        # Combine all training data
        training_tensor = tf.concat(training_features, axis=0)

        # Compile model
        self.detection_model.compile(
            optimizer='adam',
            loss={
                'reconstruction_error': 'mse',
                'anomaly_scores': 'binary_crossentropy'
            }
        )

        # Train autoencoder on normal data (self-supervised)
        print(f"Training on {len(training_tensor)} samples...")

        # Create reconstruction targets (same as input for autoencoder)
        reconstruction_targets = training_tensor

        # Create anomaly labels (all normal = 0)
        anomaly_targets = tf.zeros((len(training_tensor), training_tensor.shape[1]))

        try:
            self.detection_model.fit(
                training_tensor,
                {
                    'reconstruction_error': reconstruction_targets,
                    'anomaly_scores': anomaly_targets
                },
                epochs=20,
                batch_size=32,
                validation_split=0.2,
                verbose=1
            )
            print("Training completed successfully")
        except Exception as e:
            print(f"Training failed: {e}")

    def set_adaptive_threshold(self, percentile: float):
        """Update the adaptive threshold percentile"""
        self.threshold_percentile = max(50.0, min(99.9, percentile))