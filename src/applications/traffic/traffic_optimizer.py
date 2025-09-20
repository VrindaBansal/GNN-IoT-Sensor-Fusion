"""
Traffic Optimization System using IoT Sensor Data
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import math


@dataclass
class TrafficRecommendation:
    """Traffic optimization recommendation"""
    area_id: str
    timestamp: float
    recommendation_type: str  # 'signal_timing', 'route_diversion', 'speed_limit'
    current_condition: str   # 'congested', 'moderate', 'free_flow'
    recommended_action: str
    expected_improvement: float  # Percentage improvement
    confidence: float
    affected_sensors: List[str]


class TrafficOptimizationLayer(tf.keras.layers.Layer):
    """Neural network layer for traffic optimization"""

    def __init__(self, hidden_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

        # Traffic condition classifier
        self.condition_classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(hidden_dim // 2, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')  # free_flow, moderate, congested
        ])

        # Flow prediction network
        self.flow_predictor = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim // 2, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')  # Predicted flow rate
        ])

        # Optimization recommendation network
        self.recommendation_network = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim // 2, activation='relu'),
            tf.keras.layers.Dense(4, activation='softmax')  # 4 recommendation types
        ])

    def call(self, inputs, training=None):
        """
        Analyze traffic conditions and generate recommendations

        Args:
            inputs: Sensor feature tensor [batch_size, num_sensors, features]

        Returns:
            Dictionary with traffic analysis results
        """
        # Classify traffic conditions
        conditions = self.condition_classifier(inputs, training=training)

        # Predict traffic flow
        flow_predictions = self.flow_predictor(inputs, training=training)

        # Generate recommendations
        recommendations = self.recommendation_network(inputs, training=training)

        return {
            'conditions': conditions,  # [batch_size, num_sensors, 3]
            'flow_predictions': tf.squeeze(flow_predictions, axis=-1),  # [batch_size, num_sensors]
            'recommendations': recommendations  # [batch_size, num_sensors, 4]
        }


class TrafficOptimizer:
    """Real-time traffic optimization system"""

    def __init__(self, area_grid_size: int = 10):
        self.area_grid_size = area_grid_size
        self.optimization_model = self._build_optimization_model()

        # Traffic areas (grid-based)
        self.traffic_areas = {}
        self.area_history = defaultdict(list)

        # Recommendation types
        self.recommendation_types = [
            'signal_timing',     # Adjust traffic light timing
            'route_diversion',   # Suggest alternative routes
            'speed_limit',       # Dynamic speed limit adjustment
            'lane_management'    # Dynamic lane usage
        ]

        # Condition mapping
        self.condition_names = ['free_flow', 'moderate', 'congested']

    def _build_optimization_model(self) -> tf.keras.Model:
        """Build the traffic optimization model"""
        inputs = tf.keras.layers.Input(shape=(None, 10))  # Sensor features

        # Graph attention for spatial relationships
        attention_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=16
        )
        attended_features = attention_layer(inputs, inputs)

        # Traffic optimization layer
        optimization_layer = TrafficOptimizationLayer(hidden_dim=64)
        outputs = optimization_layer(attended_features)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def optimize_traffic(self,
                        sensor_data: Dict[str, Dict],
                        graph_features: np.ndarray,
                        timestamp: float) -> List[TrafficRecommendation]:
        """
        Generate traffic optimization recommendations

        Args:
            sensor_data: Dictionary of sensor readings
            graph_features: Node features from GNN
            timestamp: Current timestamp

        Returns:
            List of traffic optimization recommendations
        """
        recommendations = []

        # Prepare input features
        input_features = self._prepare_traffic_features(sensor_data, graph_features)

        if input_features is None:
            return recommendations

        # Run optimization model
        predictions = self.optimization_model(input_features)

        conditions = predictions['conditions'].numpy()
        flow_predictions = predictions['flow_predictions'].numpy()
        recommendation_scores = predictions['recommendations'].numpy()

        # Group sensors by traffic areas
        area_sensor_map = self._group_sensors_by_area(sensor_data)

        # Generate recommendations for each area
        for area_id, sensor_ids in area_sensor_map.items():
            area_recommendations = self._analyze_area_traffic(
                area_id, sensor_ids, sensor_data, conditions,
                flow_predictions, recommendation_scores, timestamp
            )
            recommendations.extend(area_recommendations)

        return recommendations

    def _prepare_traffic_features(self, sensor_data: Dict[str, Dict],
                                graph_features: np.ndarray) -> Optional[tf.Tensor]:
        """Prepare traffic-specific features"""
        if len(sensor_data) == 0:
            return None

        batch_features = []

        for i, (sensor_id, reading) in enumerate(sensor_data.items()):
            features = []

            # Traffic-relevant sensor data
            if reading.get('sensor_type') in ['audio', 'visual', 'accelerometer']:
                # These sensors can indicate traffic activity
                if 'value' in reading and reading['value'] is not None:
                    features.append(float(reading['value']))
                else:
                    features.append(0.0)

                # Normalize by sensor type
                if reading.get('sensor_type') == 'audio':
                    features[-1] = features[-1] / 120.0  # Normalize dB
                elif reading.get('sensor_type') == 'visual':
                    features[-1] = features[-1] / 255.0  # Normalize brightness

            else:
                features.append(0.0)  # Non-traffic sensor

            # Time-based features (rush hour indicators)
            if 'timestamp' in reading:
                hour_of_day = (reading['timestamp'] % 86400) / 3600
                day_of_week = int((reading['timestamp'] / 86400) % 7)

                # Rush hour indicators
                is_morning_rush = 1.0 if 7 <= hour_of_day <= 9 else 0.0
                is_evening_rush = 1.0 if 17 <= hour_of_day <= 19 else 0.0
                is_weekend = 1.0 if day_of_week >= 5 else 0.0

                features.extend([
                    hour_of_day / 24.0,
                    is_morning_rush,
                    is_evening_rush,
                    is_weekend
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

            # Location-based features
            if 'location' in reading:
                lat, lon = reading['location'][:2]
                # Normalize to area grid
                grid_x = (lat - 40.7) * 100  # Rough NYC coordinates
                grid_y = (lon + 74.0) * 100
                features.extend([grid_x, grid_y])
            else:
                features.extend([0.0, 0.0])

            # Sensor density (number of nearby sensors)
            sensor_density = min(len(sensor_data) / 100.0, 1.0)
            features.append(sensor_density)

            # Historical traffic pattern
            historical_pattern = self._get_historical_pattern(sensor_id, reading.get('timestamp', 0))
            features.append(historical_pattern)

            # Pad to fixed size
            while len(features) < 10:
                features.append(0.0)

            batch_features.append(features[:10])

        input_tensor = tf.constant(batch_features, dtype=tf.float32)
        input_tensor = tf.expand_dims(input_tensor, 0)

        return input_tensor

    def _group_sensors_by_area(self, sensor_data: Dict[str, Dict]) -> Dict[str, List[str]]:
        """Group sensors into traffic management areas"""
        area_sensor_map = defaultdict(list)

        for sensor_id, reading in sensor_data.items():
            if 'location' in reading:
                lat, lon = reading['location'][:2]

                # Simple grid-based area assignment
                grid_x = int((lat - 40.7) * 100) % self.area_grid_size
                grid_y = int((lon + 74.0) * 100) % self.area_grid_size

                area_id = f"area_{grid_x}_{grid_y}"
                area_sensor_map[area_id].append(sensor_id)
            else:
                area_sensor_map['default'].append(sensor_id)

        return area_sensor_map

    def _analyze_area_traffic(self,
                            area_id: str,
                            sensor_ids: List[str],
                            sensor_data: Dict[str, Dict],
                            conditions: np.ndarray,
                            flow_predictions: np.ndarray,
                            recommendation_scores: np.ndarray,
                            timestamp: float) -> List[TrafficRecommendation]:
        """Analyze traffic for a specific area and generate recommendations"""
        recommendations = []

        if not sensor_ids:
            return recommendations

        # Calculate area-level metrics
        traffic_relevant_sensors = []
        area_conditions = []
        area_flows = []
        area_rec_scores = []

        for sensor_id in sensor_ids:
            if sensor_id in sensor_data:
                reading = sensor_data[sensor_id]
                if reading.get('sensor_type') in ['audio', 'visual', 'accelerometer']:
                    traffic_relevant_sensors.append(sensor_id)

                    # Find sensor index
                    sensor_index = list(sensor_data.keys()).index(sensor_id)
                    if sensor_index < len(conditions[0]):
                        area_conditions.append(conditions[0][sensor_index])
                        area_flows.append(flow_predictions[0][sensor_index])
                        area_rec_scores.append(recommendation_scores[0][sensor_index])

        if not area_conditions:
            return recommendations

        # Aggregate area metrics
        avg_conditions = np.mean(area_conditions, axis=0)
        avg_flow = np.mean(area_flows)
        avg_rec_scores = np.mean(area_rec_scores, axis=0)

        # Determine overall area condition
        condition_idx = np.argmax(avg_conditions)
        current_condition = self.condition_names[condition_idx]

        # Generate recommendations based on condition
        if current_condition == 'congested':
            # High priority recommendations for congestion
            rec_type_idx = np.argmax(avg_rec_scores)
            rec_type = self.recommendation_types[rec_type_idx]

            recommendation = self._generate_specific_recommendation(
                area_id, rec_type, current_condition, avg_flow,
                timestamp, traffic_relevant_sensors, avg_conditions[condition_idx]
            )

            if recommendation:
                recommendations.append(recommendation)

        elif current_condition == 'moderate':
            # Preventive recommendations
            if avg_rec_scores[0] > 0.6:  # Signal timing optimization
                recommendation = self._generate_specific_recommendation(
                    area_id, 'signal_timing', current_condition, avg_flow,
                    timestamp, traffic_relevant_sensors, avg_conditions[condition_idx]
                )
                if recommendation:
                    recommendations.append(recommendation)

        # Update area history
        self._update_area_history(area_id, {
            'timestamp': timestamp,
            'condition': current_condition,
            'flow': float(avg_flow),
            'num_sensors': len(traffic_relevant_sensors)
        })

        return recommendations

    def _generate_specific_recommendation(self,
                                        area_id: str,
                                        rec_type: str,
                                        condition: str,
                                        flow: float,
                                        timestamp: float,
                                        sensor_ids: List[str],
                                        confidence: float) -> Optional[TrafficRecommendation]:
        """Generate specific traffic recommendation"""

        recommendations_map = {
            'signal_timing': {
                'congested': 'Extend green light duration on main corridors by 15-20 seconds',
                'moderate': 'Optimize signal coordination to maintain flow',
                'free_flow': 'Reduce cycle times to minimize delays'
            },
            'route_diversion': {
                'congested': 'Activate dynamic message signs to divert traffic to alternative routes',
                'moderate': 'Provide real-time route suggestions via traffic apps',
                'free_flow': 'No diversion needed'
            },
            'speed_limit': {
                'congested': 'Reduce speed limit to 25 mph to improve safety in dense traffic',
                'moderate': 'Maintain current speed limits with advisory messages',
                'free_flow': 'Consider increasing speed limit to 35 mph during off-peak'
            },
            'lane_management': {
                'congested': 'Open additional lanes or implement contra-flow if available',
                'moderate': 'Monitor for potential lane restrictions',
                'free_flow': 'Standard lane configuration'
            }
        }

        if rec_type not in recommendations_map:
            return None

        recommended_action = recommendations_map[rec_type].get(condition, 'Monitor conditions')

        # Calculate expected improvement
        expected_improvement = self._calculate_expected_improvement(rec_type, condition, flow)

        return TrafficRecommendation(
            area_id=area_id,
            timestamp=timestamp,
            recommendation_type=rec_type,
            current_condition=condition,
            recommended_action=recommended_action,
            expected_improvement=expected_improvement,
            confidence=float(confidence),
            affected_sensors=sensor_ids
        )

    def _calculate_expected_improvement(self, rec_type: str, condition: str, flow: float) -> float:
        """Calculate expected improvement percentage"""
        base_improvements = {
            'signal_timing': {'congested': 15, 'moderate': 8, 'free_flow': 3},
            'route_diversion': {'congested': 25, 'moderate': 12, 'free_flow': 5},
            'speed_limit': {'congested': 10, 'moderate': 5, 'free_flow': 2},
            'lane_management': {'congested': 30, 'moderate': 15, 'free_flow': 0}
        }

        base_improvement = base_improvements.get(rec_type, {}).get(condition, 0)

        # Adjust based on flow intensity
        flow_factor = min(flow / 100.0, 1.5)  # Cap at 1.5x improvement
        final_improvement = base_improvement * flow_factor

        return min(final_improvement, 50.0)  # Cap at 50% improvement

    def _get_historical_pattern(self, sensor_id: str, timestamp: float) -> float:
        """Get historical traffic pattern for this sensor/time"""
        hour_of_day = (timestamp % 86400) / 3600

        # Simple pattern based on typical urban traffic
        if 7 <= hour_of_day <= 9 or 17 <= hour_of_day <= 19:
            return 0.8  # Rush hour
        elif 10 <= hour_of_day <= 16:
            return 0.5  # Moderate daytime traffic
        elif 22 <= hour_of_day or hour_of_day <= 6:
            return 0.2  # Low nighttime traffic
        else:
            return 0.4  # Other times

    def _update_area_history(self, area_id: str, record: Dict):
        """Update historical data for a traffic area"""
        self.area_history[area_id].append(record)

        # Keep only last 100 records per area
        if len(self.area_history[area_id]) > 100:
            self.area_history[area_id] = self.area_history[area_id][-100:]

    def get_traffic_statistics(self) -> Dict:
        """Get traffic optimization statistics"""
        total_areas = len(self.area_history)
        congested_areas = 0
        avg_flow = 0.0
        total_records = 0

        for area_id, history in self.area_history.items():
            if history:
                latest = history[-1]
                if latest['condition'] == 'congested':
                    congested_areas += 1

                avg_flow += latest['flow']
                total_records += 1

        return {
            'total_areas_monitored': total_areas,
            'congested_areas': congested_areas,
            'congestion_rate': congested_areas / max(total_areas, 1),
            'average_flow_rate': avg_flow / max(total_records, 1),
            'total_data_points': total_records
        }