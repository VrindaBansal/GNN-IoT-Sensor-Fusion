"""
Dynamic Graph Construction Engine for IoT Sensor Networks
Builds spatial, temporal, semantic, and causal relationship graphs
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union

# Optional TensorFlow import
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
from dataclasses import dataclass
from geopy.distance import geodesic
import pandas as pd
from collections import defaultdict, deque


@dataclass
class SensorNode:
    """Represents an IoT sensor node"""
    id: str
    sensor_type: str  # 'temperature', 'humidity', 'accelerometer', 'audio', 'visual', 'air_quality', 'gps'
    location: Tuple[float, float, float]  # (lat, lon, elevation)
    capabilities: Dict[str, Union[str, float, int]]  # sampling_rate, accuracy, range, etc.
    current_reading: Optional[Dict] = None
    history: Optional[List] = None

    def __post_init__(self):
        if self.history is None:
            self.history = deque(maxlen=1000)


@dataclass
class GraphConfig:
    """Configuration for graph construction parameters"""
    spatial_threshold_meters: float = 100.0
    temporal_window_seconds: int = 300
    correlation_threshold: float = 0.7
    max_neighbors: int = 10
    adaptive_threshold: bool = True
    semantic_similarity_threshold: float = 0.5


class DynamicGraphBuilder:
    """
    Constructs multi-layer graphs from IoT sensor data
    """

    def __init__(self, config: GraphConfig = None):
        self.config = config or GraphConfig()
        self.sensor_nodes = {}
        self.graph_history = deque(maxlen=100)
        self.correlation_cache = {}

    def add_sensor(self, sensor: SensorNode):
        """Add a sensor node to the network"""
        self.sensor_nodes[sensor.id] = sensor

    def construct_graph(self,
                       sensor_data: Dict[str, Dict],
                       timestamp: float) -> Dict[str, np.ndarray]:
        """
        Construct multi-layer graph from current sensor data

        Args:
            sensor_data: {sensor_id: {reading_data}}
            timestamp: Current timestamp

        Returns:
            Dictionary of adjacency matrices for different graph types
        """
        # Update sensor readings
        self._update_sensor_readings(sensor_data, timestamp)

        # Build different graph layers
        spatial_adj = self.build_spatial_graph()
        temporal_adj = self.build_temporal_graph()
        semantic_adj = self.build_semantic_graph()
        causal_adj = self.build_causal_graph()

        graph_data = {
            'spatial': spatial_adj,
            'temporal': temporal_adj,
            'semantic': semantic_adj,
            'causal': causal_adj,
            'timestamp': timestamp
        }

        # Store in history
        self.graph_history.append(graph_data)

        # Merge graphs with learned weights
        merged_adj = self.merge_graph_layers(spatial_adj, temporal_adj, semantic_adj, causal_adj)
        graph_data['merged'] = merged_adj

        return graph_data

    def _update_sensor_readings(self, sensor_data: Dict[str, Dict], timestamp: float):
        """Update sensor nodes with new readings"""
        for sensor_id, reading in sensor_data.items():
            if sensor_id in self.sensor_nodes:
                reading_with_time = {**reading, 'timestamp': timestamp}
                self.sensor_nodes[sensor_id].current_reading = reading_with_time
                self.sensor_nodes[sensor_id].history.append(reading_with_time)

    def build_spatial_graph(self) -> np.ndarray:
        """Build graph based on spatial proximity of sensors"""
        sensor_ids = list(self.sensor_nodes.keys())
        n = len(sensor_ids)
        adjacency = np.zeros((n, n))

        for i, id1 in enumerate(sensor_ids):
            for j, id2 in enumerate(sensor_ids):
                if i != j:
                    sensor1 = self.sensor_nodes[id1]
                    sensor2 = self.sensor_nodes[id2]

                    # Calculate geographic distance
                    distance = geodesic(
                        (sensor1.location[0], sensor1.location[1]),
                        (sensor2.location[0], sensor2.location[1])
                    ).meters

                    # Add elevation difference
                    elevation_diff = abs(sensor1.location[2] - sensor2.location[2])
                    distance = np.sqrt(distance**2 + elevation_diff**2)

                    # Create edge if within threshold
                    if distance <= self.config.spatial_threshold_meters:
                        # Weight inversely proportional to distance
                        weight = 1.0 / (1.0 + distance / self.config.spatial_threshold_meters)
                        adjacency[i, j] = weight

        return self._apply_sparsity_constraint(adjacency)

    def build_temporal_graph(self) -> np.ndarray:
        """Build graph based on temporal correlations between sensors"""
        sensor_ids = list(self.sensor_nodes.keys())
        n = len(sensor_ids)
        adjacency = np.zeros((n, n))

        # Calculate temporal correlations
        for i, id1 in enumerate(sensor_ids):
            for j, id2 in enumerate(sensor_ids):
                if i != j:
                    correlation = self._calculate_temporal_correlation(id1, id2)
                    if correlation > self.config.correlation_threshold:
                        adjacency[i, j] = correlation

        return self._apply_sparsity_constraint(adjacency)

    def build_semantic_graph(self) -> np.ndarray:
        """Build graph based on sensor type similarities"""
        sensor_ids = list(self.sensor_nodes.keys())
        n = len(sensor_ids)
        adjacency = np.zeros((n, n))

        # Define sensor type similarity matrix
        sensor_types = ['temperature', 'humidity', 'accelerometer', 'audio', 'visual', 'air_quality', 'gps']
        type_similarity = {
            'temperature': {'humidity': 0.8, 'air_quality': 0.6},
            'humidity': {'temperature': 0.8, 'air_quality': 0.7},
            'accelerometer': {'audio': 0.5, 'visual': 0.3},
            'audio': {'accelerometer': 0.5, 'visual': 0.4},
            'visual': {'accelerometer': 0.3, 'audio': 0.4},
            'air_quality': {'temperature': 0.6, 'humidity': 0.7},
            'gps': {}  # GPS is unique
        }

        for i, id1 in enumerate(sensor_ids):
            for j, id2 in enumerate(sensor_ids):
                if i != j:
                    type1 = self.sensor_nodes[id1].sensor_type
                    type2 = self.sensor_nodes[id2].sensor_type

                    if type1 == type2:
                        similarity = 1.0
                    else:
                        similarity = type_similarity.get(type1, {}).get(type2, 0.0)

                    if similarity > self.config.semantic_similarity_threshold:
                        adjacency[i, j] = similarity

        return self._apply_sparsity_constraint(adjacency)

    def build_causal_graph(self) -> np.ndarray:
        """Build graph based on causal relationships between sensors"""
        sensor_ids = list(self.sensor_nodes.keys())
        n = len(sensor_ids)
        adjacency = np.zeros((n, n))

        # Define known causal relationships
        causal_relationships = {
            'temperature': ['humidity', 'air_quality'],
            'humidity': ['air_quality'],
            'accelerometer': ['audio'],
            'traffic_density': ['audio', 'air_quality'],
            'wind_speed': ['air_quality', 'audio']
        }

        for i, id1 in enumerate(sensor_ids):
            for j, id2 in enumerate(sensor_ids):
                if i != j:
                    type1 = self.sensor_nodes[id1].sensor_type
                    type2 = self.sensor_nodes[id2].sensor_type

                    # Check if type1 causes type2
                    if type2 in causal_relationships.get(type1, []):
                        # Calculate causal strength based on temporal lag correlation
                        causal_strength = self._calculate_causal_strength(id1, id2)
                        adjacency[i, j] = causal_strength

        return adjacency

    def _calculate_temporal_correlation(self, sensor_id1: str, sensor_id2: str) -> float:
        """Calculate temporal correlation between two sensors"""
        cache_key = f"{sensor_id1}_{sensor_id2}"
        if cache_key in self.correlation_cache:
            return self.correlation_cache[cache_key]

        sensor1 = self.sensor_nodes[sensor_id1]
        sensor2 = self.sensor_nodes[sensor_id2]

        if len(sensor1.history) < 10 or len(sensor2.history) < 10:
            return 0.0

        # Extract time series data
        history1 = list(sensor1.history)[-50:]  # Last 50 readings
        history2 = list(sensor2.history)[-50:]

        # Align timestamps and extract values
        aligned_data = self._align_time_series(history1, history2)
        if len(aligned_data) < 5:
            return 0.0

        values1, values2 = zip(*aligned_data)

        # Calculate correlation
        try:
            correlation = np.corrcoef(values1, values2)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        except:
            correlation = 0.0

        # Cache result
        self.correlation_cache[cache_key] = abs(correlation)
        return abs(correlation)

    def _calculate_causal_strength(self, cause_sensor_id: str, effect_sensor_id: str) -> float:
        """Calculate causal strength using Granger causality approximation"""
        cause_sensor = self.sensor_nodes[cause_sensor_id]
        effect_sensor = self.sensor_nodes[effect_sensor_id]

        if len(cause_sensor.history) < 20 or len(effect_sensor.history) < 20:
            return 0.0

        # Get recent history
        cause_history = list(cause_sensor.history)[-30:]
        effect_history = list(effect_sensor.history)[-30:]

        # Align time series
        aligned_data = self._align_time_series(cause_history, effect_history)
        if len(aligned_data) < 10:
            return 0.0

        cause_values, effect_values = zip(*aligned_data)

        # Simple lag correlation for causal strength
        max_correlation = 0.0
        for lag in range(1, min(5, len(cause_values) - 1)):
            if lag < len(cause_values) and lag < len(effect_values):
                try:
                    # Correlation between cause[t-lag] and effect[t]
                    corr = np.corrcoef(
                        cause_values[:-lag] if lag > 0 else cause_values,
                        effect_values[lag:] if lag > 0 else effect_values
                    )[0, 1]
                    if not np.isnan(corr):
                        max_correlation = max(max_correlation, abs(corr))
                except:
                    continue

        return max_correlation

    def _align_time_series(self, history1: List[Dict], history2: List[Dict]) -> List[Tuple[float, float]]:
        """Align two time series by timestamps and extract values"""
        # Create timestamp to value mapping
        ts_values1 = {reading['timestamp']: self._extract_numeric_value(reading)
                     for reading in history1 if 'timestamp' in reading}
        ts_values2 = {reading['timestamp']: self._extract_numeric_value(reading)
                     for reading in history2 if 'timestamp' in reading}

        # Find common timestamps (within tolerance)
        aligned_data = []
        tolerance = 5.0  # 5 second tolerance

        for ts1, value1 in ts_values1.items():
            for ts2, value2 in ts_values2.items():
                if abs(ts1 - ts2) <= tolerance and value1 is not None and value2 is not None:
                    aligned_data.append((value1, value2))
                    break

        return aligned_data

    def _extract_numeric_value(self, reading: Dict) -> Optional[float]:
        """Extract a numeric value from sensor reading"""
        # Priority order for numeric extraction
        keys_to_try = ['value', 'temperature', 'humidity', 'pressure', 'amplitude',
                      'brightness', 'concentration', 'speed', 'acceleration']

        for key in keys_to_try:
            if key in reading:
                try:
                    return float(reading[key])
                except (ValueError, TypeError):
                    continue

        # If no numeric value found, try to convert any value
        for key, value in reading.items():
            if key != 'timestamp':
                try:
                    return float(value)
                except (ValueError, TypeError):
                    continue

        return None

    def merge_graph_layers(self, *adjacency_matrices: np.ndarray) -> np.ndarray:
        """
        Merge multiple graph layers with learned weights
        """
        if not adjacency_matrices:
            return np.array([])

        # Initialize with equal weights
        weights = np.ones(len(adjacency_matrices)) / len(adjacency_matrices)

        # Simple weighted combination
        merged = np.zeros_like(adjacency_matrices[0])
        for weight, adj_matrix in zip(weights, adjacency_matrices):
            merged += weight * adj_matrix

        # Normalize
        merged = np.clip(merged, 0.0, 1.0)

        return merged

    def _apply_sparsity_constraint(self, adjacency: np.ndarray) -> np.ndarray:
        """Apply sparsity constraint to keep only top-k neighbors"""
        if self.config.max_neighbors <= 0:
            return adjacency

        sparse_adjacency = np.zeros_like(adjacency)

        for i in range(adjacency.shape[0]):
            # Get top-k neighbors for each node
            neighbors = np.argsort(adjacency[i])[-self.config.max_neighbors:]
            for j in neighbors:
                if adjacency[i, j] > 0:
                    sparse_adjacency[i, j] = adjacency[i, j]

        return sparse_adjacency

    def get_node_features(self) -> np.ndarray:
        """Extract node features from sensor data"""
        sensor_ids = list(self.sensor_nodes.keys())
        n = len(sensor_ids)
        feature_dim = 10  # Base feature dimension

        features = np.zeros((n, feature_dim))

        for i, sensor_id in enumerate(sensor_ids):
            sensor = self.sensor_nodes[sensor_id]

            # Location features (normalized)
            features[i, 0] = sensor.location[0] / 90.0  # Normalized latitude
            features[i, 1] = sensor.location[1] / 180.0  # Normalized longitude
            features[i, 2] = sensor.location[2] / 1000.0  # Normalized elevation

            # Sensor type encoding
            type_encoding = self._encode_sensor_type(sensor.sensor_type)
            features[i, 3:6] = type_encoding

            # Current reading features
            if sensor.current_reading:
                value = self._extract_numeric_value(sensor.current_reading)
                if value is not None:
                    features[i, 6] = value

            # Historical statistics
            if len(sensor.history) > 0:
                recent_values = [self._extract_numeric_value(reading)
                               for reading in list(sensor.history)[-10:]]
                recent_values = [v for v in recent_values if v is not None]

                if recent_values:
                    features[i, 7] = np.mean(recent_values)  # Mean
                    features[i, 8] = np.std(recent_values)   # Std
                    features[i, 9] = len(recent_values)      # Count

        return features

    def _encode_sensor_type(self, sensor_type: str) -> np.ndarray:
        """One-hot encode sensor type"""
        types = ['temperature', 'humidity', 'accelerometer', 'audio', 'visual', 'air_quality']
        encoding = np.zeros(3)  # Reduced to 3 dimensions

        if sensor_type in types:
            idx = types.index(sensor_type)
            encoding[idx % 3] = 1.0

        return encoding

    def get_graph_statistics(self) -> Dict:
        """Get statistics about the constructed graphs"""
        if not self.graph_history:
            return {}

        latest_graph = self.graph_history[-1]
        stats = {}

        for graph_type, adj_matrix in latest_graph.items():
            if isinstance(adj_matrix, np.ndarray) and adj_matrix.ndim == 2:
                stats[graph_type] = {
                    'num_nodes': adj_matrix.shape[0],
                    'num_edges': np.sum(adj_matrix > 0),
                    'density': np.sum(adj_matrix > 0) / (adj_matrix.shape[0] ** 2),
                    'avg_degree': np.mean(np.sum(adj_matrix > 0, axis=1)),
                    'max_weight': np.max(adj_matrix),
                    'avg_weight': np.mean(adj_matrix[adj_matrix > 0]) if np.any(adj_matrix > 0) else 0
                }

        return stats