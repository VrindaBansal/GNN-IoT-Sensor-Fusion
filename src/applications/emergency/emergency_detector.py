"""
Emergency Detection and Response System for Smart Cities
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class EmergencyAlert:
    """Emergency detection result"""
    alert_id: str
    timestamp: float
    emergency_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    location: Tuple[float, float, float]  # lat, lon, elevation
    confidence: float
    description: str
    affected_area_radius: float  # meters
    detected_sensors: List[str]
    recommended_response: str


class EmergencyDetectionLayer(tf.keras.layers.Layer):
    """Neural network layer for emergency detection"""

    def __init__(self, num_emergency_types: int = 6, hidden_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.num_emergency_types = num_emergency_types
        self.hidden_dim = hidden_dim

        # Multi-class emergency classifier
        self.emergency_classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(hidden_dim // 2, activation='relu'),
            tf.keras.layers.Dense(num_emergency_types, activation='softmax')
        ])

        # Severity estimator
        self.severity_estimator = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim // 2, activation='relu'),
            tf.keras.layers.Dense(4, activation='softmax')  # low, medium, high, critical
        ])

        # Spatial impact predictor
        self.impact_predictor = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Impact radius (normalized)
        ])

    def call(self, inputs, training=None):
        """
        Detect emergencies from sensor data

        Args:
            inputs: Sensor feature tensor [batch_size, num_sensors, features]

        Returns:
            Dictionary with emergency detection results
        """
        # Emergency type classification
        emergency_probs = self.emergency_classifier(inputs, training=training)

        # Severity estimation
        severity_probs = self.severity_estimator(inputs, training=training)

        # Impact radius prediction
        impact_radius = self.impact_predictor(inputs, training=training)
        impact_radius = tf.squeeze(impact_radius, axis=-1)

        return {
            'emergency_probs': emergency_probs,
            'severity_probs': severity_probs,
            'impact_radius': impact_radius
        }


class EmergencyResponseSystem:
    """Real-time emergency detection and response system"""

    def __init__(self, response_time_threshold: float = 300.0):  # 5 minutes
        self.response_time_threshold = response_time_threshold
        self.detection_model = self._build_detection_model()

        # Emergency types and their characteristics
        self.emergency_types = {
            0: 'fire',
            1: 'explosion',
            2: 'structural_collapse',
            3: 'chemical_spill',
            4: 'severe_weather',
            5: 'other'
        }

        self.severity_levels = ['low', 'medium', 'high', 'critical']

        # Active alerts
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)

        # Response protocols
        self.response_protocols = self._initialize_response_protocols()

    def _build_detection_model(self) -> tf.keras.Model:
        """Build the emergency detection model"""
        inputs = tf.keras.layers.Input(shape=(None, 12))  # Emergency-specific features

        # Temporal attention for pattern recognition
        temporal_attention = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)

        # Spatial attention for multi-sensor correlation
        spatial_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=16
        )(temporal_attention, temporal_attention)

        # Emergency detection layer
        detection_layer = EmergencyDetectionLayer(num_emergency_types=6, hidden_dim=64)
        outputs = detection_layer(spatial_attention)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def detect_emergencies(self,
                          sensor_data: Dict[str, Dict],
                          graph_features: np.ndarray,
                          timestamp: float) -> List[EmergencyAlert]:
        """
        Detect emergencies from real-time sensor data

        Args:
            sensor_data: Dictionary of sensor readings
            graph_features: Node features from GNN
            timestamp: Current timestamp

        Returns:
            List of emergency alerts
        """
        alerts = []

        # Prepare emergency-specific features
        input_features = self._prepare_emergency_features(sensor_data, graph_features)

        if input_features is None:
            return alerts

        # Run detection model
        predictions = self.detection_model(input_features)

        emergency_probs = predictions['emergency_probs'].numpy()
        severity_probs = predictions['severity_probs'].numpy()
        impact_radius = predictions['impact_radius'].numpy()

        # Analyze each sensor for emergency patterns
        sensor_ids = list(sensor_data.keys())
        for i, sensor_id in enumerate(sensor_ids):
            if i < len(emergency_probs[0]):
                emergency_prob = emergency_probs[0][i]
                severity_prob = severity_probs[0][i]
                radius = impact_radius[0][i]

                # Check for emergency threshold
                max_emergency_prob = np.max(emergency_prob)
                if max_emergency_prob > 0.7:  # Threshold for emergency detection

                    emergency_type_idx = np.argmax(emergency_prob)
                    emergency_type = self.emergency_types[emergency_type_idx]

                    severity_idx = np.argmax(severity_prob)
                    severity = self.severity_levels[severity_idx]

                    # Check if this is a new or escalating emergency
                    alert = self._create_emergency_alert(
                        sensor_id, sensor_data[sensor_id], emergency_type,
                        severity, radius, max_emergency_prob, timestamp
                    )

                    if alert and self._validate_emergency_alert(alert):
                        alerts.append(alert)

        # Check for multi-sensor emergencies
        multi_sensor_alerts = self._detect_multi_sensor_emergencies(
            sensor_data, emergency_probs, severity_probs, timestamp
        )
        alerts.extend(multi_sensor_alerts)

        # Update active alerts
        self._update_active_alerts(alerts, timestamp)

        return alerts

    def _prepare_emergency_features(self, sensor_data: Dict[str, Dict],
                                  graph_features: np.ndarray) -> Optional[tf.Tensor]:
        """Prepare emergency-specific features from sensor data"""
        if len(sensor_data) == 0:
            return None

        batch_features = []

        for i, (sensor_id, reading) in enumerate(sensor_data.items()):
            features = []

            # Basic sensor reading
            if 'value' in reading and reading['value'] is not None:
                value = float(reading['value'])
            else:
                value = 0.0

            # Emergency-relevant features based on sensor type
            sensor_type = reading.get('sensor_type', 'unknown')

            if sensor_type == 'temperature':
                # High temperature could indicate fire
                temp_anomaly = max(0.0, (value - 25.0) / 50.0)  # Normalize above 25°C
                features.extend([value / 100.0, temp_anomaly])
            elif sensor_type == 'audio':
                # Loud sounds could indicate explosions or emergencies
                noise_level = value / 120.0  # Normalize dB
                loud_anomaly = max(0.0, (value - 80.0) / 40.0)  # Above 80 dB
                features.extend([noise_level, loud_anomaly])
            elif sensor_type == 'air_quality':
                # Poor air quality could indicate chemical spills or fires
                aqi_norm = value / 500.0  # Normalize AQI
                hazardous = 1.0 if value > 300 else 0.0  # Hazardous level
                features.extend([aqi_norm, hazardous])
            elif sensor_type == 'accelerometer':
                # Vibrations could indicate explosions or structural issues
                vibration_norm = value / 10.0  # Normalize acceleration
                high_vibration = 1.0 if value > 2.0 else 0.0
                features.extend([vibration_norm, high_vibration])
            elif sensor_type == 'visual':
                # Brightness changes could indicate fires or explosions
                brightness_norm = value / 255.0
                sudden_bright = 1.0 if value > 200 else 0.0
                features.extend([brightness_norm, sudden_bright])
            else:
                features.extend([value / 100.0, 0.0])

            # Temporal features
            if 'timestamp' in reading:
                hour = (reading['timestamp'] % 86400) / 3600
                features.append(hour / 24.0)
            else:
                features.append(0.0)

            # Location features
            if 'location' in reading:
                lat, lon, elevation = reading['location'][:3]
                features.extend([
                    (lat - 40.7) * 10,   # Normalize to NYC area
                    (lon + 74.0) * 10,
                    elevation / 100.0
                ])
            else:
                features.extend([0.0, 0.0, 0.0])

            # Sensor quality and status
            quality_score = 1.0 if reading.get('quality') == 'good' else 0.0
            features.append(quality_score)

            # Rate of change (if available from history)
            rate_of_change = self._calculate_rate_of_change(sensor_id, value, reading.get('timestamp', 0))
            features.append(rate_of_change)

            # Multi-sensor correlation features
            correlation_score = self._calculate_multi_sensor_correlation(sensor_id, sensor_data)
            features.append(correlation_score)

            # Pad to fixed size
            while len(features) < 12:
                features.append(0.0)

            batch_features.append(features[:12])

        input_tensor = tf.constant(batch_features, dtype=tf.float32)
        input_tensor = tf.expand_dims(input_tensor, 0)

        return input_tensor

    def _create_emergency_alert(self,
                              sensor_id: str,
                              reading: Dict,
                              emergency_type: str,
                              severity: str,
                              radius: float,
                              confidence: float,
                              timestamp: float) -> Optional[EmergencyAlert]:
        """Create an emergency alert from detection results"""

        alert_id = f"alert_{int(timestamp)}_{sensor_id}"

        # Get sensor location
        location = reading.get('location', (0.0, 0.0, 0.0))

        # Calculate affected area radius (denormalize)
        affected_radius = radius * 1000.0  # Convert to meters, max 1km

        # Generate description
        description = self._generate_emergency_description(
            emergency_type, severity, reading
        )

        # Get recommended response
        recommended_response = self.response_protocols.get(
            emergency_type, {}
        ).get(severity, "Contact emergency services immediately")

        return EmergencyAlert(
            alert_id=alert_id,
            timestamp=timestamp,
            emergency_type=emergency_type,
            severity=severity,
            location=location,
            confidence=float(confidence),
            description=description,
            affected_area_radius=affected_radius,
            detected_sensors=[sensor_id],
            recommended_response=recommended_response
        )

    def _generate_emergency_description(self, emergency_type: str,
                                      severity: str, reading: Dict) -> str:
        """Generate human-readable emergency description"""
        sensor_type = reading.get('sensor_type', 'unknown')
        value = reading.get('value', 0)

        descriptions = {
            'fire': f"Potential fire detected by {sensor_type} sensor (reading: {value})",
            'explosion': f"Possible explosion detected by {sensor_type} sensor",
            'structural_collapse': f"Structural damage indicated by vibration sensors",
            'chemical_spill': f"Chemical hazard detected by air quality monitoring",
            'severe_weather': f"Severe weather conditions detected",
            'other': f"Emergency situation detected by {sensor_type} sensor"
        }

        base_description = descriptions.get(emergency_type, "Emergency detected")
        severity_modifier = {
            'low': "Minor incident",
            'medium': "Moderate emergency",
            'high': "Serious emergency",
            'critical': "CRITICAL EMERGENCY"
        }

        return f"{severity_modifier.get(severity, '')}: {base_description}"

    def _validate_emergency_alert(self, alert: EmergencyAlert) -> bool:
        """Validate emergency alert to reduce false positives"""

        # Check confidence threshold
        if alert.confidence < 0.7:
            return False

        # Check for duplicate alerts in same area within time window
        for existing_alert in self.active_alerts.values():
            time_diff = abs(alert.timestamp - existing_alert.timestamp)
            distance = self._calculate_distance(alert.location, existing_alert.location)

            if (time_diff < 300 and  # Within 5 minutes
                distance < alert.affected_area_radius and
                alert.emergency_type == existing_alert.emergency_type):
                return False  # Likely duplicate

        # Critical emergencies always pass
        if alert.severity == 'critical':
            return True

        # Additional validation for other severities
        if alert.severity in ['high', 'medium']:
            return alert.confidence > 0.8

        return alert.confidence > 0.9  # Higher threshold for low severity

    def _detect_multi_sensor_emergencies(self,
                                       sensor_data: Dict[str, Dict],
                                       emergency_probs: np.ndarray,
                                       severity_probs: np.ndarray,
                                       timestamp: float) -> List[EmergencyAlert]:
        """Detect emergencies that span multiple sensors"""
        multi_alerts = []

        # Group sensors by proximity
        sensor_groups = self._group_sensors_by_proximity(sensor_data, radius=200.0)

        for group_sensors in sensor_groups:
            if len(group_sensors) >= 3:  # At least 3 sensors
                group_alert = self._analyze_sensor_group(
                    group_sensors, sensor_data, emergency_probs,
                    severity_probs, timestamp
                )
                if group_alert:
                    multi_alerts.append(group_alert)

        return multi_alerts

    def _analyze_sensor_group(self,
                            group_sensors: List[str],
                            sensor_data: Dict[str, Dict],
                            emergency_probs: np.ndarray,
                            severity_probs: np.ndarray,
                            timestamp: float) -> Optional[EmergencyAlert]:
        """Analyze a group of sensors for coordinated emergency patterns"""

        # Calculate group-level emergency probability
        group_emergency_probs = []
        group_severity_probs = []
        group_locations = []

        sensor_ids = list(sensor_data.keys())

        for sensor_id in group_sensors:
            if sensor_id in sensor_ids:
                sensor_idx = sensor_ids.index(sensor_id)
                if sensor_idx < len(emergency_probs[0]):
                    group_emergency_probs.append(emergency_probs[0][sensor_idx])
                    group_severity_probs.append(severity_probs[0][sensor_idx])

                    location = sensor_data[sensor_id].get('location', (0.0, 0.0, 0.0))
                    group_locations.append(location)

        if len(group_emergency_probs) < 3:
            return None

        # Calculate group consensus
        avg_emergency_prob = np.mean(group_emergency_probs, axis=0)
        avg_severity_prob = np.mean(group_severity_probs, axis=0)

        max_emergency_prob = np.max(avg_emergency_prob)

        # Require higher threshold for group emergencies
        if max_emergency_prob > 0.6:
            emergency_type_idx = np.argmax(avg_emergency_prob)
            emergency_type = self.emergency_types[emergency_type_idx]

            severity_idx = np.argmax(avg_severity_prob)
            severity = self.severity_levels[severity_idx]

            # Calculate center location
            center_location = tuple(np.mean(group_locations, axis=0))

            # Calculate affected radius
            max_distance = max([
                self._calculate_distance(center_location, loc)
                for loc in group_locations
            ])
            affected_radius = max_distance * 1.5  # 50% buffer

            alert_id = f"group_alert_{int(timestamp)}_{len(group_sensors)}"

            return EmergencyAlert(
                alert_id=alert_id,
                timestamp=timestamp,
                emergency_type=emergency_type,
                severity=severity,
                location=center_location,
                confidence=float(max_emergency_prob),
                description=f"Multi-sensor {emergency_type} detected across {len(group_sensors)} sensors",
                affected_area_radius=affected_radius,
                detected_sensors=group_sensors,
                recommended_response=self.response_protocols.get(emergency_type, {}).get(
                    severity, "Coordinate emergency response across affected area"
                )
            )

        return None

    def _group_sensors_by_proximity(self, sensor_data: Dict[str, Dict],
                                   radius: float = 200.0) -> List[List[str]]:
        """Group sensors by geographic proximity"""
        groups = []
        ungrouped_sensors = list(sensor_data.keys())

        while ungrouped_sensors:
            current_sensor = ungrouped_sensors[0]
            current_group = [current_sensor]
            ungrouped_sensors.remove(current_sensor)

            current_location = sensor_data[current_sensor].get('location')
            if not current_location:
                continue

            # Find nearby sensors
            for sensor_id in ungrouped_sensors[:]:
                sensor_location = sensor_data[sensor_id].get('location')
                if sensor_location:
                    distance = self._calculate_distance(current_location, sensor_location)
                    if distance <= radius:
                        current_group.append(sensor_id)
                        ungrouped_sensors.remove(sensor_id)

            groups.append(current_group)

        return [group for group in groups if len(group) > 1]

    def _calculate_distance(self, loc1: Tuple[float, float, float],
                          loc2: Tuple[float, float, float]) -> float:
        """Calculate distance between two locations"""
        if len(loc1) < 2 or len(loc2) < 2:
            return float('inf')

        lat_diff = (loc1[0] - loc2[0]) * 111000  # Approx meters per degree
        lon_diff = (loc1[1] - loc2[1]) * 111000
        elevation_diff = (loc1[2] - loc2[2]) if len(loc1) > 2 and len(loc2) > 2 else 0

        return np.sqrt(lat_diff**2 + lon_diff**2 + elevation_diff**2)

    def _calculate_rate_of_change(self, sensor_id: str, current_value: float,
                                timestamp: float) -> float:
        """Calculate rate of change for sensor value"""
        # Simplified implementation - would use historical data in practice
        return min(abs(current_value) / 100.0, 1.0)

    def _calculate_multi_sensor_correlation(self, sensor_id: str,
                                          sensor_data: Dict[str, Dict]) -> float:
        """Calculate correlation with nearby sensors"""
        # Simplified correlation score
        return 0.5  # Neutral correlation

    def _initialize_response_protocols(self) -> Dict:
        """Initialize emergency response protocols"""
        return {
            'fire': {
                'low': "Monitor situation, prepare fire suppression resources",
                'medium': "Deploy fire department, evacuate immediate area",
                'high': "Full fire department response, evacuate 200m radius",
                'critical': "Multi-unit response, evacuate 500m radius, alert hospitals"
            },
            'explosion': {
                'low': "Investigate source, check for gas leaks",
                'medium': "Bomb squad investigation, evacuate 100m radius",
                'high': "Full emergency response, evacuate 300m radius",
                'critical': "HAZMAT and bomb squad, evacuate 1km radius"
            },
            'structural_collapse': {
                'low': "Building inspection required",
                'medium': "Evacuate building, structural assessment",
                'high': "Evacuate block, urban search and rescue",
                'critical': "Mass evacuation, full rescue operations"
            },
            'chemical_spill': {
                'low': "Environmental assessment, contain spill",
                'medium': "HAZMAT response, evacuate downwind areas",
                'high': "Full HAZMAT response, evacuate 500m radius",
                'critical': "Regional emergency response, mass evacuation"
            },
            'severe_weather': {
                'low': "Weather advisory, monitor conditions",
                'medium': "Weather warning, prepare shelters",
                'high': "Severe weather warning, open emergency shelters",
                'critical': "Disaster declaration, mass evacuation"
            }
        }

    def _update_active_alerts(self, new_alerts: List[EmergencyAlert], timestamp: float):
        """Update active alerts and handle alert lifecycle"""
        # Add new alerts
        for alert in new_alerts:
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)

        # Remove expired alerts (older than response threshold)
        expired_alerts = []
        for alert_id, alert in self.active_alerts.items():
            if timestamp - alert.timestamp > self.response_time_threshold:
                expired_alerts.append(alert_id)

        for alert_id in expired_alerts:
            del self.active_alerts[alert_id]

    def get_emergency_statistics(self) -> Dict:
        """Get emergency detection statistics"""
        total_alerts = len(self.alert_history)
        active_count = len(self.active_alerts)

        # Count by type and severity
        type_counts = {}
        severity_counts = {}

        for alert in self.alert_history:
            type_counts[alert.emergency_type] = type_counts.get(alert.emergency_type, 0) + 1
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1

        return {
            'total_alerts_generated': total_alerts,
            'currently_active_alerts': active_count,
            'alerts_by_type': type_counts,
            'alerts_by_severity': severity_counts,
            'average_confidence': np.mean([alert.confidence for alert in self.alert_history])
                                if self.alert_history else 0.0
        }