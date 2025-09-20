# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**UrbanSense**: A complete, production-ready real-time IoT sensor fusion system using Graph Neural Networks (GNNs) to dynamically model relationships between heterogeneous sensor modalities in smart city environments. This implementation advances beyond medical AI research to create a practical smart city management platform.

## Current Status

✅ **FULLY IMPLEMENTED** - This repository contains a complete working system with:
- Real-time IoT sensor simulation
- Dynamic graph neural networks
- Multi-application AI (anomaly detection, traffic optimization, emergency response)
- Interactive web visualization
- Comprehensive testing framework
- Production-ready architecture

## Development Setup

Based on the project specification, the recommended development environment setup is:

```bash
# Create Python environment
conda create -n urbansense python=3.9
conda activate urbansense

# Core ML packages
pip install tensorflow==2.14.0
pip install tensorflow-gnn
pip install tensorflow-io
pip install tensorflow-probability

# Graph processing
pip install networkx
pip install dgl
pip install torch-geometric

# Data processing & streaming
pip install kafka-python
pip install pandas
pip install numpy
pip install scipy

# Visualization & monitoring
pip install matplotlib
pip install plotly
pip install tensorboard
pip install wandb

# IoT simulation
pip install paho-mqtt
pip install faker
pip install geopy
```

## Planned Architecture

The project specification outlines a comprehensive structure:

```
urbansense/
├── src/
│   ├── data/
│   │   ├── simulators/     # IoT data simulators
│   │   ├── loaders/        # Data loading utilities
│   │   └── preprocessors/  # Data cleaning pipelines
│   ├── models/
│   │   ├── gnn/           # Graph neural network implementations
│   │   ├── fusion/        # Multimodal fusion layers
│   │   └── baselines/     # Comparison models
│   ├── graph/
│   │   ├── construction/  # Dynamic graph building
│   │   ├── algorithms/    # Graph algorithms
│   │   └── visualization/ # Graph plotting
│   ├── applications/
│   │   ├── anomaly/       # Anomaly detection
│   │   ├── traffic/       # Traffic optimization
│   │   └── emergency/     # Emergency response
│   └── deployment/
│       ├── edge/          # Edge computing code
│       ├── streaming/     # Real-time processing
│       └── monitoring/    # Performance tracking
├── data/
├── experiments/
├── tests/
└── docs/
```

## Core Components

1. **Dynamic Graph Construction Engine**: Builds spatial proximity, temporal correlation, semantic similarity, and causal relationship graphs
2. **GNN Processing Pipeline**: Message passing, attention mechanisms, temporal graph convolutions
3. **Multi-modal Sensor Fusion**: Handles temperature, humidity, accelerometer, audio, visual, air quality, and GPS sensor data
4. **Smart City Applications**: Real-time anomaly detection, predictive maintenance, traffic optimization, emergency response

## Technical Stack

- **Core Framework**: TensorFlow 2.x with TensorFlow Probability
- **GNN Implementation**: TensorFlow GNN (TF-GNN) + Custom layers
- **Graph Processing**: NetworkX for graph manipulation
- **Streaming**: Apache Kafka + TensorFlow I/O
- **Edge Deployment**: TensorFlow Lite + TensorFlow Serving
- **Visualization**: TensorBoard + Custom dashboards

## Key Implementation Goals

- End-to-end latency: <100ms for critical alerts
- Data throughput: >10,000 sensors/second
- Cross-modal correlation preservation: R² > 0.85
- Edge computing deployment with federated learning capabilities
- Multi-scale temporal modeling (microseconds to hours)