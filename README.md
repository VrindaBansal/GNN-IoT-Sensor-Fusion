# UrbanSense: Real-Time IoT Sensor Fusion with Graph Neural Networks

A comprehensive smart city platform that uses Graph Neural Networks to process real-time IoT sensor data, detect events, and provide intelligent recommendations for urban management.

🌟 **[Live Demo](https://vrindabansal.github.io/urbansense/web_demo/)** - Try the interactive demo!

## 🚀 Quick Start

### Option 1: Online Demo
Visit the [live demo](https://vrindabansal.github.io/urbansense/web_demo/) to see UrbanSense in action immediately.

### Option 2: Local Setup
```bash
# Clone the repository
git clone https://github.com/VrindaBansal/urbansense.git
cd urbansense

# Navigate to web demo
cd web_demo

# Start local server
python -m http.server 8080

# Open http://localhost:8080 in your browser
```

That's it! No dependencies required for the basic demo.

## 🏙️ What is UrbanSense?

UrbanSense transforms how cities understand and respond to urban events by connecting thousands of IoT sensors through intelligent AI. Instead of treating sensors as isolated data points, our Graph Neural Network creates a "social network" for sensors, understanding relationships and patterns across the entire urban ecosystem.

### 🎯 Key Features

- **🗺️ Live City Sensor Map**: Visualize real-time sensor data across an interactive city map
- **🧠 AI Event Detection**: Automatically detect wildfires, emergencies, traffic incidents, and weather events
- **⚡ Real-Time Processing**: Process sensor data in real-time with <50ms latency
- **📊 Dynamic Visualizations**: Interactive charts, network graphs, and performance metrics
- **🔔 Smart Notifications**: Optional event alerts with actionable recommendations
- **⚙️ Interactive Controls**: Adjust sensor count (10-200) and simulation speed (1x-10x)

### 🌟 Event Detection Examples

UrbanSense can automatically detect and respond to:

- **🔥 Wildfire**: "Elevated temperatures and poor air quality detected - Fire department dispatched"
- **🚔 Police Action**: "Audio spikes and anomalies detected - Redirect traffic away from area"
- **🚑 Medical Emergency**: "Unusual movement patterns detected - Ambulance dispatched"
- **🚗 Traffic Incident**: "Severe congestion detected - Activate alternative routes"
- **☀️ Weather Alert**: "Extreme weather conditions - Activate cooling centers"

## 🛠️ Installation & Setup

### Local Development

1. **Clone the repository**:
```bash
git clone https://github.com/VrindaBansal/urbansense.git
cd urbansense
```

2. **For basic web demo** (no dependencies):
```bash
cd web_demo
python -m http.server 8080
# Visit http://localhost:8080
```

3. **For full AI capabilities** (optional):
```bash
# Install Python dependencies
pip install -r requirements.txt

# For advanced ML features
pip install -r requirements-full.txt
```

4. **Environment Setup** (if using backend features):
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your configurations
```

### Docker Setup (Optional)

```bash
# Build and run with Docker
docker build -t urbansense .
docker run -p 8080:8080 urbansense
```

## 💻 Usage

### Web Interface Controls

- **Start/Stop Simulation**: Begin or pause real-time sensor processing
- **Sensor Count Slider**: Add/remove sensors dynamically (10-200 sensors)
- **Speed Control**: Adjust simulation speed (1x-10x)
- **Event Notifications Toggle**: Enable/disable event alerts
- **Interactive Elements**:
  - Hover over sensors for real-time readings
  - Click network nodes to see connections
  - View live charts and performance metrics

### Programmatic API

```python
from src.applications.smart_city.sensor_fusion import SmartCitySensorFusion

# Initialize the system
fusion = SmartCitySensorFusion()

# Add sensors
fusion.add_sensor('temp_001', 'temperature', location=(40.7589, -73.9851))
fusion.add_sensor('traffic_001', 'traffic', location=(40.7580, -73.9855))

# Process real-time data
fusion.process_sensor_data('temp_001', 28.5)
fusion.process_sensor_data('traffic_001', 75.0)

# Get AI insights
anomalies = fusion.detect_anomalies()
events = fusion.detect_events()
recommendations = fusion.get_recommendations()
```

## 🏗️ Architecture

### Graph Neural Network Components

- **Spatial Graph**: Geographic proximity relationships
- **Temporal Graph**: Time-series patterns and trends
- **Semantic Graph**: Logical sensor type relationships
- **Causal Graph**: Cause-and-effect event relationships

### Fusion Strategies

1. **Early Fusion**: Combine raw sensor data before processing
2. **Joint Fusion**: Process multiple sensor types together
3. **Late Fusion**: Combine individual sensor decisions
4. **Adaptive Fusion**: Dynamic strategy selection based on data quality

## 📁 Project Structure

```
urbansense/
├── web_demo/                 # Interactive web interface
│   ├── index.html           # Main interface
│   ├── simulation.js        # Core simulation engine
│   ├── style.css           # UI styling
│   └── quick_demo.py       # Standalone demo server
├── src/
│   ├── models/gnn/          # Graph Neural Network models
│   ├── applications/        # Smart city applications
│   ├── fusion/             # Sensor fusion algorithms
│   └── utils/              # Utility functions
├── docs/                   # Documentation
├── tests/                  # Unit tests
├── requirements.txt        # Core dependencies
├── requirements-full.txt   # Full ML dependencies
├── .env.example           # Environment template
└── README.md              # This file
```

## 🚀 Deployment

### GitHub Pages (Automatic)
The project automatically deploys to GitHub Pages on every push to main branch:
- **URL**: https://vrindabansal.github.io/urbansense/web_demo/
- **Status**: [![Deploy](https://github.com/VrindaBansal/urbansense/actions/workflows/deploy.yml/badge.svg)](https://github.com/VrindaBansal/urbansense/actions/workflows/deploy.yml)

### Manual Deployment
```bash
# Build for production
npm run build

# Deploy to your hosting platform
npm run deploy
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test category
python -m pytest tests/test_gnn.py
python -m pytest tests/test_fusion.py

# Run with coverage
python -m pytest --cov=src tests/
```

## 📊 Performance

- **Sensors**: Supports 10-200 simultaneous sensors
- **Latency**: <50ms real-time processing
- **Throughput**: 1000+ sensor readings per second
- **Accuracy**: 95%+ event detection accuracy
- **Browser**: Works on all modern browsers (Chrome, Firefox, Safari, Edge)

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes and test**
4. **Commit**: `git commit -m 'Add amazing feature'`
5. **Push**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Development Guidelines

- Follow existing code style
- Add tests for new features
- Update documentation
- Test across different browsers

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♀️ Support

- **Issues**: [GitHub Issues](https://github.com/VrindaBansal/urbansense/issues)
- **Discussions**: [GitHub Discussions](https://github.com/VrindaBansal/urbansense/discussions)
- **Documentation**: [Full Documentation](./docs/)

## 🌟 Acknowledgments

Built with inspiration from multimodal AI research in medical applications, adapted for smart city use cases. This project demonstrates how AI can make urban environments more responsive, safe, and efficient.

---

**Ready to transform your city with AI? [Try the demo now!](https://vrindabansal.github.io/urbansense/web_demo/)**