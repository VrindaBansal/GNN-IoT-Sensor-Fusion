"""
Multimodal Fusion Layers for IoT Sensor Data
Implements early, joint, and late fusion techniques inspired by the research paper
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod


class FusionLayer(keras.layers.Layer, ABC):
    """Abstract base class for fusion layers"""

    @abstractmethod
    def call(self, modality_features: Dict[str, tf.Tensor], **kwargs) -> tf.Tensor:
        """Fuse features from multiple modalities"""
        pass


class EarlyFusion(FusionLayer):
    """
    Early fusion: concatenate raw features before processing
    """

    def __init__(self, output_dim: int, dropout_rate: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        self.normalization_layers = {}
        self.projection_layers = {}
        self.fusion_network = None

    def build(self, input_shape):
        # Build normalization and projection layers for each modality
        modality_names = list(input_shape.keys()) if isinstance(input_shape, dict) else ['default']

        for modality in modality_names:
            # Layer normalization for each modality
            self.normalization_layers[modality] = keras.layers.LayerNormalization()

            # Projection to common dimension
            if isinstance(input_shape, dict):
                feature_dim = input_shape[modality][-1] if len(input_shape[modality]) > 1 else input_shape[modality]
            else:
                feature_dim = input_shape[-1]

            self.projection_layers[modality] = keras.layers.Dense(
                self.output_dim // len(modality_names),
                activation='relu',
                name=f'proj_{modality}'
            )

        # Fusion network
        self.fusion_network = keras.Sequential([
            keras.layers.Dense(self.output_dim, activation='relu'),
            keras.layers.Dropout(self.dropout_rate),
            keras.layers.Dense(self.output_dim, activation='relu'),
            keras.layers.LayerNormalization()
        ])

        super().build(input_shape)

    def call(self, modality_features: Dict[str, tf.Tensor], training=None) -> tf.Tensor:
        """
        Early fusion by concatenating normalized and projected features

        Args:
            modality_features: Dictionary of {modality_name: tensor}

        Returns:
            Fused feature tensor
        """
        processed_features = []

        for modality, features in modality_features.items():
            # Normalize features
            normalized = self.normalization_layers[modality](features)

            # Project to common dimension
            projected = self.projection_layers[modality](normalized)

            processed_features.append(projected)

        # Concatenate all modalities
        concatenated = tf.concat(processed_features, axis=-1)

        # Process through fusion network
        fused_features = self.fusion_network(concatenated, training=training)

        return fused_features


class JointFusion(FusionLayer):
    """
    Joint fusion: partial processing before fusion, then continued processing
    """

    def __init__(self, hidden_dim: int, output_dim: int, dropout_rate: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        self.modality_encoders = {}
        self.cross_attention = None
        self.fusion_network = None

    def build(self, input_shape):
        modality_names = list(input_shape.keys()) if isinstance(input_shape, dict) else ['default']

        # Individual modality encoders
        for modality in modality_names:
            self.modality_encoders[modality] = keras.Sequential([
                keras.layers.Dense(self.hidden_dim, activation='relu'),
                keras.layers.LayerNormalization(),
                keras.layers.Dropout(self.dropout_rate),
                keras.layers.Dense(self.hidden_dim, activation='relu')
            ], name=f'encoder_{modality}')

        # Cross-attention for modality interaction
        self.cross_attention = keras.layers.MultiHeadAttention(
            num_heads=8,
            key_dim=self.hidden_dim // 8,
            name='cross_attention'
        )

        # Final fusion network
        self.fusion_network = keras.Sequential([
            keras.layers.Dense(self.output_dim, activation='relu'),
            keras.layers.LayerNormalization(),
            keras.layers.Dropout(self.dropout_rate),
            keras.layers.Dense(self.output_dim)
        ])

        super().build(input_shape)

    def call(self, modality_features: Dict[str, tf.Tensor], training=None) -> tf.Tensor:
        """
        Joint fusion with cross-modal attention

        Args:
            modality_features: Dictionary of {modality_name: tensor}

        Returns:
            Fused feature tensor
        """
        # Step 1: Encode each modality independently
        encoded_modalities = {}
        for modality, features in modality_features.items():
            encoded = self.modality_encoders[modality](features, training=training)
            encoded_modalities[modality] = encoded

        # Step 2: Cross-modal attention
        modality_list = list(encoded_modalities.keys())
        attended_features = []

        for i, query_modality in enumerate(modality_list):
            query = encoded_modalities[query_modality]

            # Attention with all other modalities
            context_features = []
            for j, key_modality in enumerate(modality_list):
                if i != j:
                    key_value = encoded_modalities[key_modality]
                    attended = self.cross_attention(
                        query=query,
                        key=key_value,
                        value=key_value,
                        training=training
                    )
                    context_features.append(attended)

            if context_features:
                # Combine attended features
                combined_context = tf.reduce_mean(tf.stack(context_features), axis=0)
                # Residual connection
                final_feature = query + combined_context
            else:
                final_feature = query

            attended_features.append(final_feature)

        # Step 3: Aggregate and fuse
        aggregated = tf.reduce_mean(tf.stack(attended_features), axis=0)
        fused_output = self.fusion_network(aggregated, training=training)

        return fused_output


class LateFusion(FusionLayer):
    """
    Late fusion: fully process each modality independently, then combine
    """

    def __init__(self, feature_dim: int, output_dim: int,
                 fusion_method: str = 'attention', **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.fusion_method = fusion_method

        self.modality_networks = {}
        self.fusion_layer = None

    def build(self, input_shape):
        modality_names = list(input_shape.keys()) if isinstance(input_shape, dict) else ['default']

        # Independent processing networks for each modality
        for modality in modality_names:
            self.modality_networks[modality] = keras.Sequential([
                keras.layers.Dense(self.feature_dim, activation='relu'),
                keras.layers.LayerNormalization(),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(self.feature_dim, activation='relu'),
                keras.layers.LayerNormalization(),
                keras.layers.Dense(self.feature_dim, activation='tanh')
            ], name=f'network_{modality}')

        # Fusion layer based on method
        if self.fusion_method == 'attention':
            self.fusion_layer = AttentionFusion(self.output_dim)
        elif self.fusion_method == 'weighted_sum':
            self.fusion_layer = WeightedSumFusion(len(modality_names), self.output_dim)
        else:  # 'concatenate'
            self.fusion_layer = keras.Sequential([
                keras.layers.Dense(self.output_dim, activation='relu'),
                keras.layers.Dense(self.output_dim)
            ])

        super().build(input_shape)

    def call(self, modality_features: Dict[str, tf.Tensor], training=None) -> tf.Tensor:
        """
        Late fusion after independent processing

        Args:
            modality_features: Dictionary of {modality_name: tensor}

        Returns:
            Fused feature tensor
        """
        # Process each modality independently
        processed_modalities = {}
        for modality, features in modality_features.items():
            processed = self.modality_networks[modality](features, training=training)
            processed_modalities[modality] = processed

        # Fuse processed features
        if self.fusion_method == 'concatenate':
            concatenated = tf.concat(list(processed_modalities.values()), axis=-1)
            return self.fusion_layer(concatenated, training=training)
        else:
            return self.fusion_layer(processed_modalities, training=training)


class AttentionFusion(keras.layers.Layer):
    """Attention-based fusion of multiple modalities"""

    def __init__(self, output_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

        self.attention_weights = None
        self.output_projection = None

    def build(self, input_shape):
        # Attention mechanism to weight modalities
        self.attention_weights = keras.layers.Dense(1, activation='softmax', name='attention')

        # Output projection
        self.output_projection = keras.layers.Dense(self.output_dim, name='output_proj')

        super().build(input_shape)

    def call(self, modality_features: Dict[str, tf.Tensor], training=None) -> tf.Tensor:
        """
        Attention-weighted fusion of modalities

        Args:
            modality_features: Dictionary of processed modality features

        Returns:
            Attention-weighted fused features
        """
        modality_names = list(modality_features.keys())
        features = list(modality_features.values())

        # Stack features: [batch_size, num_modalities, feature_dim]
        stacked_features = tf.stack(features, axis=1)

        # Compute attention weights for each modality
        attention_scores = []
        for feature in features:
            score = self.attention_weights(feature)
            attention_scores.append(score)

        # Normalize attention weights
        attention_weights = tf.nn.softmax(tf.stack(attention_scores, axis=1), axis=1)

        # Weighted combination
        weighted_features = stacked_features * attention_weights
        fused_features = tf.reduce_sum(weighted_features, axis=1)

        # Final projection
        output = self.output_projection(fused_features, training=training)

        return output


class WeightedSumFusion(keras.layers.Layer):
    """Learnable weighted sum fusion"""

    def __init__(self, num_modalities: int, output_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_modalities = num_modalities
        self.output_dim = output_dim

    def build(self, input_shape):
        # Learnable weights for each modality
        self.modality_weights = self.add_weight(
            name='modality_weights',
            shape=(self.num_modalities,),
            initializer='uniform',
            trainable=True
        )

        # Output projection
        self.output_projection = keras.layers.Dense(self.output_dim)

        super().build(input_shape)

    def call(self, modality_features: Dict[str, tf.Tensor], training=None) -> tf.Tensor:
        """
        Weighted sum fusion with learnable weights

        Args:
            modality_features: Dictionary of processed modality features

        Returns:
            Weighted sum of modality features
        """
        features = list(modality_features.values())

        # Apply softmax to weights
        normalized_weights = tf.nn.softmax(self.modality_weights)

        # Weighted combination
        weighted_sum = tf.zeros_like(features[0])
        for i, feature in enumerate(features):
            weighted_sum += normalized_weights[i] * feature

        # Final projection
        output = self.output_projection(weighted_sum, training=training)

        return output


class AdaptiveFusion(FusionLayer):
    """
    Adaptive fusion that learns to select fusion strategy based on input
    """

    def __init__(self, output_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

        self.fusion_strategies = ['early', 'joint', 'late']
        self.strategy_networks = {}
        self.strategy_selector = None
        self.final_projection = None

    def build(self, input_shape):
        # Create different fusion strategies
        self.strategy_networks['early'] = EarlyFusion(self.output_dim)
        self.strategy_networks['joint'] = JointFusion(self.output_dim // 2, self.output_dim)
        self.strategy_networks['late'] = LateFusion(self.output_dim // 2, self.output_dim)

        # Strategy selector network
        total_input_dim = sum([shape[-1] for shape in input_shape.values()]) if isinstance(input_shape, dict) else input_shape[-1]
        self.strategy_selector = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(len(self.fusion_strategies), activation='softmax')
        ])

        # Final projection
        self.final_projection = keras.layers.Dense(self.output_dim)

        # Build all sub-networks
        for strategy_net in self.strategy_networks.values():
            strategy_net.build(input_shape)

        super().build(input_shape)

    def call(self, modality_features: Dict[str, tf.Tensor], training=None) -> tf.Tensor:
        """
        Adaptive fusion based on learned strategy selection

        Args:
            modality_features: Dictionary of {modality_name: tensor}

        Returns:
            Adaptively fused features
        """
        # Concatenate features for strategy selection
        concat_features = tf.concat(list(modality_features.values()), axis=-1)

        # Select fusion strategy
        strategy_weights = self.strategy_selector(concat_features, training=training)

        # Apply each fusion strategy
        strategy_outputs = []
        for strategy, network in self.strategy_networks.items():
            output = network(modality_features, training=training)
            strategy_outputs.append(output)

        # Weight and combine strategy outputs
        stacked_outputs = tf.stack(strategy_outputs, axis=-1)
        weighted_output = tf.reduce_sum(
            stacked_outputs * tf.expand_dims(strategy_weights, axis=1),
            axis=-1
        )

        # Final projection
        final_output = self.final_projection(weighted_output, training=training)

        return final_output


def create_fusion_layer(fusion_type: str, **kwargs) -> FusionLayer:
    """
    Factory function to create fusion layers

    Args:
        fusion_type: Type of fusion ('early', 'joint', 'late', 'adaptive')
        **kwargs: Additional parameters for the fusion layer

    Returns:
        Configured fusion layer
    """
    fusion_classes = {
        'early': EarlyFusion,
        'joint': JointFusion,
        'late': LateFusion,
        'adaptive': AdaptiveFusion
    }

    if fusion_type not in fusion_classes:
        raise ValueError(f"Unknown fusion type: {fusion_type}. "
                        f"Available types: {list(fusion_classes.keys())}")

    return fusion_classes[fusion_type](**kwargs)