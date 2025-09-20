// UrbanSense Simulation Engine
class UrbanSenseSimulation {
    constructor() {
        this.isRunning = false;
        this.sensors = [];
        this.sensorCount = 50;
        this.simulationSpeed = 1;
        this.currentTime = 0;
        this.alerts = {
            anomalies: [],
            traffic: [],
            emergencies: []
        };

        // Event notifications system
        this.eventNotifications = {
            enabled: true,
            queue: [],
            currentNotification: null
        };

        // Performance metrics
        this.metrics = {
            latency: 0,
            throughput: 0,
            accuracy: 0,
            edges: 0
        };

        // Chart instances
        this.sensorChart = null;
        this.networkGraph = null;

        this.initializeInterface();
        this.createSensorNetwork();
        this.setupNetworkGraph();
        this.setupSensorChart();
        this.setupCityMap();
    }

    initializeInterface() {
        // Control buttons
        document.getElementById('startBtn').addEventListener('click', () => this.startSimulation());
        document.getElementById('stopBtn').addEventListener('click', () => this.stopSimulation());

        // Sliders
        document.getElementById('sensorSlider').addEventListener('input', (e) => {
            this.sensorCount = parseInt(e.target.value);
            document.getElementById('sensorValue').textContent = this.sensorCount;

            // Update city map sensors in real-time
            this.updateCityMapSensorCount();

            if (!this.isRunning) {
                this.createSensorNetwork();
                this.updateNetworkGraph();
            }
        });

        document.getElementById('speedSlider').addEventListener('input', (e) => {
            this.simulationSpeed = parseInt(e.target.value);
            document.getElementById('speedValue').textContent = `${this.simulationSpeed}x`;

            // Update city animation speed
            this.updateCityAnimationSpeed();
        });

        // Event notifications toggle
        document.getElementById('notificationsToggle').addEventListener('change', (e) => {
            this.eventNotifications.enabled = e.target.checked;
            if (!this.eventNotifications.enabled && this.eventNotifications.currentNotification) {
                this.hideNotification();
            }
        });

        // Notification close button
        document.querySelector('.notification-close').addEventListener('click', () => {
            this.hideNotification();
        });
    }

    createSensorNetwork() {
        this.sensors = [];
        const sensorTypes = ['temperature', 'humidity', 'audio', 'visual', 'air_quality', 'accelerometer'];
        const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3'];

        // NYC-like coordinates
        const bounds = {
            lat: [40.7, 40.8],
            lon: [-74.0, -73.9]
        };

        for (let i = 0; i < this.sensorCount; i++) {
            const type = sensorTypes[Math.floor(Math.random() * sensorTypes.length)];
            const sensor = {
                id: `sensor_${i.toString().padStart(4, '0')}`,
                type: type,
                location: {
                    lat: bounds.lat[0] + Math.random() * (bounds.lat[1] - bounds.lat[0]),
                    lon: bounds.lon[0] + Math.random() * (bounds.lon[1] - bounds.lon[0]),
                    elevation: Math.random() * 20
                },
                value: this.generateSensorValue(type),
                lastUpdate: Date.now(),
                history: [],
                connections: [],
                color: colors[sensorTypes.indexOf(type)],
                status: 'normal' // normal, anomaly, offline
            };

            this.sensors.push(sensor);
        }

        // Create connections (spatial proximity)
        this.createSpatialConnections();

        // Update UI
        document.getElementById('sensorCount').textContent = this.sensorCount;
    }

    createSpatialConnections() {
        const maxDistance = 0.01; // ~1km in degrees

        this.sensors.forEach(sensor => {
            sensor.connections = [];

            this.sensors.forEach(otherSensor => {
                if (sensor.id !== otherSensor.id) {
                    const distance = this.calculateDistance(sensor.location, otherSensor.location);
                    if (distance < maxDistance) {
                        sensor.connections.push({
                            to: otherSensor.id,
                            weight: 1 - (distance / maxDistance),
                            type: 'spatial'
                        });
                    }
                }
            });
        });
    }

    calculateDistance(loc1, loc2) {
        const latDiff = loc1.lat - loc2.lat;
        const lonDiff = loc1.lon - loc2.lon;
        return Math.sqrt(latDiff * latDiff + lonDiff * lonDiff);
    }

    generateSensorValue(type, time = Date.now()) {
        const hour = (time / 1000 / 3600) % 24;
        let baseValue;

        switch (type) {
            case 'temperature':
                baseValue = 20 + 10 * Math.sin((hour - 6) * Math.PI / 12) + Math.random() * 5;
                break;
            case 'humidity':
                baseValue = 60 + 20 * Math.sin((hour - 12) * Math.PI / 12) + Math.random() * 10;
                break;
            case 'audio':
                baseValue = (hour >= 6 && hour <= 22) ?
                    50 + 20 * Math.sin((hour - 6) * Math.PI / 16) + Math.random() * 15 :
                    30 + Math.random() * 10;
                break;
            case 'visual':
                baseValue = (hour >= 6 && hour <= 18) ?
                    100 + Math.random() * 155 : Math.random() * 50;
                break;
            case 'air_quality':
                const rushHour = (hour >= 7 && hour <= 9) || (hour >= 17 && hour <= 19);
                baseValue = rushHour ? 80 + Math.random() * 40 : 40 + Math.random() * 30;
                break;
            case 'accelerometer':
                baseValue = 0.1 + Math.random() * 0.5;
                break;
            default:
                baseValue = Math.random() * 100;
        }

        return Math.max(0, baseValue);
    }

    setupNetworkGraph() {
        const container = document.getElementById('networkGraph');
        const options = {
            nodes: {
                borderWidth: 3,
                size: 35,
                font: {
                    color: 'white',
                    size: 12,
                    strokeWidth: 2,
                    strokeColor: '#000000'
                },
                shadow: {
                    enabled: true,
                    color: 'rgba(0,212,255,0.3)',
                    size: 15,
                    x: 0,
                    y: 0
                }
            },
            edges: {
                width: 2,
                color: {
                    color: 'rgba(0, 212, 255, 0.6)',
                    highlight: 'rgba(0, 212, 255, 1)'
                },
                smooth: {
                    enabled: true,
                    type: 'continuous'
                }
            },
            physics: {
                enabled: true,
                stabilization: { iterations: 200 },
                barnesHut: {
                    gravitationalConstant: -2000,
                    centralGravity: 0.1,
                    springLength: 100,
                    springConstant: 0.04,
                    damping: 0.09
                }
            },
            interaction: {
                hover: true,
                tooltipDelay: 300
            }
        };

        this.networkGraph = new vis.Network(container, {}, options);
        this.updateNetworkGraph();
    }

    updateNetworkGraph() {
        if (!this.networkGraph) return;

        const nodes = this.sensors.map(sensor => ({
            id: sensor.id,
            label: sensor.id.split('_')[1],
            color: {
                background: sensor.color,
                border: sensor.status === 'anomaly' ? '#ff4757' :
                        sensor.status === 'offline' ? '#666' : 'white'
            },
            title: `${sensor.type}: ${sensor.value.toFixed(2)}`
        }));

        const edges = [];
        this.sensors.forEach(sensor => {
            sensor.connections.slice(0, 3).forEach(conn => { // Limit connections for clarity
                edges.push({
                    from: sensor.id,
                    to: conn.to,
                    width: Math.max(1, conn.weight * 4),
                    opacity: conn.weight
                });
            });
        });

        this.networkGraph.setData({ nodes, edges });
        this.metrics.edges = edges.length;
    }

    setupSensorChart() {
        const ctx = document.getElementById('sensorChart').getContext('2d');

        this.sensorChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Temperature',
                        data: [],
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Humidity',
                        data: [],
                        borderColor: '#4ecdc4',
                        backgroundColor: 'rgba(78, 205, 196, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Audio (dB)',
                        data: [],
                        borderColor: '#45b7d1',
                        backgroundColor: 'rgba(69, 183, 209, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Air Quality',
                        data: [],
                        borderColor: '#feca57',
                        backgroundColor: 'rgba(254, 202, 87, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        display: true,
                        title: {
                            display: true,
                            text: 'Time',
                            color: 'white'
                        },
                        ticks: { color: 'white' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Sensor Value',
                            color: 'white'
                        },
                        ticks: { color: 'white' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: 'white' }
                    }
                }
            }
        });
    }

    startSimulation() {
        this.isRunning = true;
        document.getElementById('startBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;
        document.getElementById('systemStatus').innerHTML =
            '<i class="fas fa-circle status-active"></i> ACTIVE';

        this.simulationLoop();
    }

    stopSimulation() {
        this.isRunning = false;
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
        document.getElementById('systemStatus').innerHTML =
            '<i class="fas fa-circle" style="color: #666;"></i> STOPPED';

        // Stop city animations when simulation stops
        this.stopCityAnimations();
    }

    simulationLoop() {
        if (!this.isRunning) return;

        const startTime = performance.now();

        // Update sensor values
        this.updateSensors();

        // Run AI applications
        this.runAnomalyDetection();
        this.runTrafficOptimization();
        this.runEmergencyDetection();
        this.runEventDetection();

        // Update visualizations
        this.updateChart();
        this.updateNetworkGraph();
        this.updateMetrics();
        this.updateAlerts();

        // Calculate processing time
        const endTime = performance.now();
        this.metrics.latency = Math.round(endTime - startTime);
        this.metrics.throughput = Math.round(this.sensorCount / (this.simulationSpeed * 2));
        this.metrics.accuracy = Math.min(95 + Math.random() * 5, 100);

        // Continue simulation
        setTimeout(() => this.simulationLoop(), 2000 / this.simulationSpeed);
    }

    updateSensors() {
        const now = Date.now();

        this.sensors.forEach(sensor => {
            sensor.value = this.generateSensorValue(sensor.type, now);
            sensor.lastUpdate = now;

            // Keep history for chart
            sensor.history.push({
                time: this.currentTime,
                value: sensor.value
            });

            if (sensor.history.length > 20) {
                sensor.history.shift();
            }
        });

        this.currentTime += 1;
    }

    runAnomalyDetection() {
        // Simulate anomaly detection
        this.sensors.forEach(sensor => {
            sensor.status = 'normal';

            // Random anomaly generation (5% chance)
            if (Math.random() < 0.05) {
                sensor.status = 'anomaly';

                const anomaly = {
                    id: `anomaly_${Date.now()}_${sensor.id}`,
                    sensor: sensor.id,
                    type: this.getAnomalyType(sensor),
                    severity: Math.random() > 0.7 ? 'high' : 'medium',
                    timestamp: new Date(),
                    description: `${sensor.type} sensor showing unusual readings`
                };

                this.alerts.anomalies.unshift(anomaly);
                if (this.alerts.anomalies.length > 5) {
                    this.alerts.anomalies.pop();
                }
            }

            // Sensor failure simulation (1% chance)
            if (Math.random() < 0.01) {
                sensor.status = 'offline';
            }
        });
    }

    getAnomalyType(sensor) {
        const types = ['sensor_failure', 'data_drift', 'outlier_reading', 'correlation_break'];
        return types[Math.floor(Math.random() * types.length)];
    }

    runTrafficOptimization() {
        // Simulate traffic analysis
        if (Math.random() < 0.3) { // 30% chance of traffic recommendation
            const areas = ['area_1_2', 'area_3_4', 'area_5_6', 'area_2_3'];
            const conditions = ['free_flow', 'moderate', 'congested'];
            const recommendations = ['signal_timing', 'route_diversion', 'speed_limit', 'lane_management'];

            const traffic = {
                id: `traffic_${Date.now()}`,
                area: areas[Math.floor(Math.random() * areas.length)],
                condition: conditions[Math.floor(Math.random() * conditions.length)],
                recommendation: recommendations[Math.floor(Math.random() * recommendations.length)],
                improvement: Math.round(Math.random() * 30 + 5),
                timestamp: new Date()
            };

            this.alerts.traffic.unshift(traffic);
            if (this.alerts.traffic.length > 3) {
                this.alerts.traffic.pop();
            }
        }
    }

    runEmergencyDetection() {
        // Simulate emergency detection
        if (Math.random() < 0.1) { // 10% chance of emergency
            const types = ['fire', 'explosion', 'structural_collapse', 'chemical_spill', 'severe_weather'];
            const severities = ['low', 'medium', 'high', 'critical'];

            const emergency = {
                id: `emergency_${Date.now()}`,
                type: types[Math.floor(Math.random() * types.length)],
                severity: severities[Math.floor(Math.random() * severities.length)],
                confidence: Math.random() * 0.3 + 0.7,
                location: this.sensors[Math.floor(Math.random() * this.sensors.length)].location,
                timestamp: new Date(),
                sensors: Math.floor(Math.random() * 5) + 1
            };

            this.alerts.emergencies.unshift(emergency);
            if (this.alerts.emergencies.length > 3) {
                this.alerts.emergencies.pop();
            }
        }
    }

    updateChart() {
        if (!this.sensorChart) return;

        const sensorTypes = ['temperature', 'humidity', 'audio', 'air_quality'];

        // Get average values by type
        const averages = sensorTypes.map(type => {
            const sensorsOfType = this.sensors.filter(s => s.type === type);
            if (sensorsOfType.length === 0) return 0;
            return sensorsOfType.reduce((sum, s) => sum + s.value, 0) / sensorsOfType.length;
        });

        // Update chart data
        this.sensorChart.data.labels.push(this.currentTime);
        this.sensorChart.data.datasets.forEach((dataset, index) => {
            dataset.data.push(averages[index] || 0);

            // Keep only last 20 points
            if (dataset.data.length > 20) {
                dataset.data.shift();
            }
        });

        // Remove old labels
        if (this.sensorChart.data.labels.length > 20) {
            this.sensorChart.data.labels.shift();
        }

        this.sensorChart.update('none');
    }

    updateMetrics() {
        document.getElementById('latencyMetric').textContent = `${this.metrics.latency}ms`;
        document.getElementById('throughputMetric').textContent = this.metrics.throughput;
        document.getElementById('accuracyMetric').textContent = `${Math.round(this.metrics.accuracy)}%`;
        document.getElementById('edgesMetric').textContent = this.metrics.edges;

        // Update alert count
        const totalAlerts = this.alerts.anomalies.length + this.alerts.traffic.length + this.alerts.emergencies.length;
        document.getElementById('alertCount').textContent = totalAlerts;
    }

    updateAlerts() {
        // Update anomaly alerts
        const anomalyList = document.getElementById('anomalyList');
        if (this.alerts.anomalies.length === 0) {
            anomalyList.innerHTML = '<div class="no-alerts">No anomalies detected</div>';
        } else {
            anomalyList.innerHTML = this.alerts.anomalies.map(anomaly => `
                <div class="alert-item anomaly">
                    <div class="alert-title">⚠️ ${anomaly.type.replace('_', ' ').toUpperCase()}</div>
                    <div class="alert-details">
                        Sensor: ${anomaly.sensor} | Severity: ${anomaly.severity}<br>
                        ${anomaly.description}
                    </div>
                </div>
            `).join('');
        }

        // Update traffic alerts
        const trafficList = document.getElementById('trafficList');
        if (this.alerts.traffic.length === 0) {
            trafficList.innerHTML = '<div class="no-alerts">Traffic flowing normally</div>';
        } else {
            trafficList.innerHTML = this.alerts.traffic.map(traffic => `
                <div class="alert-item traffic">
                    <div class="alert-title">🚦 ${traffic.area.toUpperCase()}</div>
                    <div class="alert-details">
                        Condition: ${traffic.condition} | Action: ${traffic.recommendation}<br>
                        Expected improvement: ${traffic.improvement}%
                    </div>
                </div>
            `).join('');
        }

        // Update emergency alerts
        const emergencyList = document.getElementById('emergencyList');
        if (this.alerts.emergencies.length === 0) {
            emergencyList.innerHTML = '<div class="no-alerts">No emergencies detected</div>';
        } else {
            emergencyList.innerHTML = this.alerts.emergencies.map(emergency => `
                <div class="alert-item emergency">
                    <div class="alert-title">🚨 ${emergency.type.toUpperCase()}</div>
                    <div class="alert-details">
                        Severity: ${emergency.severity} | Confidence: ${Math.round(emergency.confidence * 100)}%<br>
                        Sensors involved: ${emergency.sensors}
                    </div>
                </div>
            `).join('');
        }
    }

    setupCityMap() {
        const mapContainer = document.getElementById('cityMap');
        console.log('Setting up city map, container:', mapContainer);

        if (!mapContainer) {
            console.error('City map container not found!');
            return;
        }

        // Clear existing content
        mapContainer.innerHTML = '';
        console.log('Cleared city map container');

        // Create city infrastructure
        this.createCityInfrastructure(mapContainer);
        console.log('Created city infrastructure');

        // Create sensors
        this.createCitySensors(mapContainer);
        console.log('Created city sensors');

        // Start animations
        this.startCityAnimations();
        console.log('Started city animations');

        // Initialize sensor readings immediately
        this.updateCitySensors();
    }

    createCityInfrastructure(container) {
        const mapWidth = container.offsetWidth;
        const mapHeight = 600;
        console.log('Creating infrastructure for container:', mapWidth, 'x', mapHeight);

        // Create roads
        // Horizontal roads
        for (let i = 1; i < 4; i++) {
            const road = document.createElement('div');
            road.className = 'city-road horizontal';
            road.style.top = `${(mapHeight / 4) * i - 15}px`;
            road.style.left = '0px';
            road.style.width = '100%';
            road.style.height = '30px';
            road.style.background = '#1a1a1a';
            road.style.position = 'absolute';
            container.appendChild(road);
            console.log('Created horizontal road', i);
        }

        // Vertical roads
        for (let i = 1; i < 5; i++) {
            const road = document.createElement('div');
            road.className = 'city-road vertical';
            road.style.left = `${(mapWidth / 5) * i - 15}px`;
            road.style.top = '0px';
            road.style.width = '30px';
            road.style.height = '100%';
            road.style.background = '#1a1a1a';
            road.style.position = 'absolute';
            container.appendChild(road);
            console.log('Created vertical road', i);
        }

        // Create buildings
        this.buildings = [];
        for (let i = 0; i < 20; i++) {
            const building = document.createElement('div');
            building.className = 'city-building';

            const width = 60 + Math.random() * 80;
            const height = 80 + Math.random() * 120;
            const x = Math.random() * (mapWidth - width);
            const y = Math.random() * (mapHeight - height);

            building.style.width = `${width}px`;
            building.style.height = `${height}px`;
            building.style.left = `${x}px`;
            building.style.top = `${y}px`;
            building.style.position = 'absolute';
            building.style.background = '#2d3748';
            building.style.border = '2px solid #4a5568';
            building.style.borderRadius = '2px';

            this.buildings.push(building);
            container.appendChild(building);
            console.log('Created building', i, 'at', x, y);
        }

        // Create traffic lights
        this.trafficLights = [];
        for (let i = 1; i < 4; i++) {
            for (let j = 1; j < 5; j++) {
                const light = document.createElement('div');
                light.className = 'traffic-light';
                light.style.left = `${(mapWidth / 5) * j - 4}px`;
                light.style.top = `${(mapHeight / 4) * i - 10}px`;
                this.trafficLights.push(light);
                container.appendChild(light);
            }
        }
    }

    createCitySensors(container) {
        const mapWidth = container.offsetWidth;
        const mapHeight = 600;
        this.citySensors = [];

        // Create different types of sensors
        const sensorTypes = [
            { type: 'temperature', count: 15, color: '#ff6b6b' },
            { type: 'light', count: 12, color: '#ffd93d' },
            { type: 'traffic', count: 10, color: '#74c0fc' }
        ];

        sensorTypes.forEach(sensorType => {
            for (let i = 0; i < sensorType.count; i++) {
                const sensor = document.createElement('div');
                sensor.className = `sensor-point ${sensorType.type}`;

                const x = Math.random() * (mapWidth - 20);
                const y = Math.random() * (mapHeight - 20);

                sensor.style.left = `${x}px`;
                sensor.style.top = `${y}px`;
                sensor.style.position = 'absolute';
                sensor.style.width = '12px';
                sensor.style.height = '12px';
                sensor.style.borderRadius = '50%';
                sensor.style.background = sensorType.color;
                sensor.style.zIndex = '10';
                sensor.style.cursor = 'pointer';
                sensor.style.boxShadow = `0 0 15px ${sensorType.color}80`;

                // Add reading display
                const reading = document.createElement('div');
                reading.className = 'sensor-reading';
                reading.style.position = 'absolute';
                reading.style.top = '-25px';
                reading.style.left = '15px';
                reading.style.background = 'rgba(0,0,0,0.8)';
                reading.style.color = 'white';
                reading.style.padding = '2px 6px';
                reading.style.borderRadius = '3px';
                reading.style.fontSize = '10px';
                reading.style.whiteSpace = 'nowrap';
                reading.style.opacity = '0';
                reading.style.transition = 'opacity 0.3s';
                sensor.appendChild(reading);

                // Add hover events
                sensor.addEventListener('mouseenter', () => {
                    reading.style.opacity = '1';
                });
                sensor.addEventListener('mouseleave', () => {
                    reading.style.opacity = '0';
                });

                const sensorData = {
                    element: sensor,
                    reading: reading,
                    type: sensorType.type,
                    x: x,
                    y: y,
                    value: 0,
                    lastUpdate: Date.now()
                };

                this.citySensors.push(sensorData);
                container.appendChild(sensor);
                console.log('Created sensor', sensorType.type, 'at', x, y);
            }
        });
    }

    startCityAnimations() {
        // Start sensor data updates
        this.cityAnimationInterval = setInterval(() => {
            this.updateCitySensors();
            this.updateBuildingLights();
            this.updateTrafficLights();
        }, 1000);
    }

    updateCitySensors() {
        if (!this.citySensors) return;

        const currentTime = Date.now();
        const hour = (currentTime / 1000 / 3600) % 24;

        this.citySensors.forEach(sensor => {
            let value;

            switch (sensor.type) {
                case 'temperature':
                    // Temperature varies with time of day + some randomness
                    value = 15 + 10 * Math.sin((hour - 6) * Math.PI / 12) + (Math.random() - 0.5) * 5;
                    value = Math.max(0, Math.round(value));
                    sensor.reading.textContent = `${value}°C`;
                    break;

                case 'light':
                    // Light varies dramatically with time + some randomness
                    const isDay = hour >= 6 && hour <= 18;
                    value = isDay ?
                        700 + Math.sin((hour - 6) * Math.PI / 12) * 300 + Math.random() * 100 :
                        50 + Math.random() * 100;
                    value = Math.round(value);
                    sensor.reading.textContent = `${value} lux`;
                    break;

                case 'traffic':
                    // Traffic varies with rush hours + randomness
                    const isRushHour = (hour >= 7 && hour <= 9) || (hour >= 17 && hour <= 19);
                    value = isRushHour ?
                        70 + Math.random() * 30 :
                        20 + Math.random() * 40;
                    value = Math.round(value);
                    sensor.reading.textContent = `${value} cars/min`;
                    break;
            }

            sensor.value = value;

            // Animate sensor intensity based on value
            const intensity = sensor.type === 'light' ? value / 1000 :
                            sensor.type === 'temperature' ? (value - 5) / 25 :
                            value / 100;

            sensor.element.style.opacity = Math.max(0.3, Math.min(1, intensity));
        });
    }

    updateBuildingLights() {
        if (!this.buildings) return;

        const currentTime = Date.now();
        const hour = (currentTime / 1000 / 3600) % 24;
        const isNight = hour < 6 || hour > 20;

        this.buildings.forEach(building => {
            // Buildings are more likely to be lit at night
            const shouldBeLit = isNight ? Math.random() > 0.3 : Math.random() > 0.8;

            if (shouldBeLit) {
                building.classList.add('lit');
            } else {
                building.classList.remove('lit');
            }
        });
    }

    updateTrafficLights() {
        if (!this.trafficLights) return;

        this.trafficLights.forEach((light, index) => {
            // Cycle traffic lights with different phases
            const cycle = (Date.now() / 1000 + index * 10) % 20;
            const isActive = cycle < 10;

            if (isActive) {
                light.classList.add('active');
            } else {
                light.classList.remove('active');
            }
        });
    }

    stopCityAnimations() {
        if (this.cityAnimationInterval) {
            clearInterval(this.cityAnimationInterval);
            this.cityAnimationInterval = null;
        }
    }

    updateCityMapSensorCount() {
        // Adjust number of sensors on the city map based on slider
        const mapContainer = document.getElementById('cityMap');
        if (!mapContainer || !this.citySensors) return;

        const currentTotal = this.citySensors.length;
        const targetSensorRatio = this.sensorCount / 50; // Base of 50 sensors
        const targetTotal = Math.round(37 * targetSensorRatio); // Scale from base 37 city sensors

        if (targetTotal > currentTotal) {
            // Add more sensors
            this.addCitySensors(mapContainer, targetTotal - currentTotal);
        } else if (targetTotal < currentTotal) {
            // Remove sensors
            this.removeCitySensors(currentTotal - targetTotal);
        }
    }

    addCitySensors(container, count) {
        const mapWidth = container.offsetWidth;
        const mapHeight = 600;
        const sensorTypes = [
            { type: 'temperature', color: '#ff6b6b', weight: 0.4 },
            { type: 'light', color: '#ffd93d', weight: 0.32 },
            { type: 'traffic', color: '#74c0fc', weight: 0.28 }
        ];

        for (let i = 0; i < count; i++) {
            // Randomly select sensor type based on weights
            const rand = Math.random();
            let cumulativeWeight = 0;
            let selectedType = sensorTypes[0];

            for (const sensorType of sensorTypes) {
                cumulativeWeight += sensorType.weight;
                if (rand <= cumulativeWeight) {
                    selectedType = sensorType;
                    break;
                }
            }

            const sensor = document.createElement('div');
            sensor.className = `sensor-point ${selectedType.type}`;

            const x = Math.random() * (mapWidth - 20);
            const y = Math.random() * (mapHeight - 20);

            sensor.style.left = `${x}px`;
            sensor.style.top = `${y}px`;
            sensor.style.position = 'absolute';
            sensor.style.width = '12px';
            sensor.style.height = '12px';
            sensor.style.borderRadius = '50%';
            sensor.style.background = selectedType.color;
            sensor.style.zIndex = '10';
            sensor.style.cursor = 'pointer';
            sensor.style.boxShadow = `0 0 15px ${selectedType.color}80`;

            const reading = document.createElement('div');
            reading.className = 'sensor-reading';
            reading.style.position = 'absolute';
            reading.style.top = '-25px';
            reading.style.left = '15px';
            reading.style.background = 'rgba(0,0,0,0.8)';
            reading.style.color = 'white';
            reading.style.padding = '2px 6px';
            reading.style.borderRadius = '3px';
            reading.style.fontSize = '10px';
            reading.style.whiteSpace = 'nowrap';
            reading.style.opacity = '0';
            reading.style.transition = 'opacity 0.3s';
            sensor.appendChild(reading);

            sensor.addEventListener('mouseenter', () => reading.style.opacity = '1');
            sensor.addEventListener('mouseleave', () => reading.style.opacity = '0');

            const sensorData = {
                element: sensor,
                reading: reading,
                type: selectedType.type,
                x: x,
                y: y,
                value: 0,
                lastUpdate: Date.now()
            };

            this.citySensors.push(sensorData);
            container.appendChild(sensor);
        }
    }

    removeCitySensors(count) {
        for (let i = 0; i < count && this.citySensors.length > 0; i++) {
            const sensorData = this.citySensors.pop();
            if (sensorData.element.parentNode) {
                sensorData.element.parentNode.removeChild(sensorData.element);
            }
        }
    }

    updateCityAnimationSpeed() {
        // Restart city animations with new speed
        if (this.cityAnimationInterval) {
            clearInterval(this.cityAnimationInterval);
            const baseInterval = 1000; // Base 1 second
            const newInterval = baseInterval / this.simulationSpeed;

            this.cityAnimationInterval = setInterval(() => {
                this.updateCitySensors();
                this.updateBuildingLights();
                this.updateTrafficLights();
            }, newInterval);
        }
    }

    runEventDetection() {
        if (!this.eventNotifications.enabled) return;

        // Generate significant events based on sensor patterns
        const events = [];

        // Wildfire detection (high temperature + air quality)
        const hotSensors = this.sensors.filter(s => s.type === 'temperature' && s.value > 35);
        const badAirSensors = this.sensors.filter(s => s.type === 'air_quality' && s.value > 100);

        if (hotSensors.length > 3 && badAirSensors.length > 2 && Math.random() < 0.05) {
            events.push({
                type: 'wildfire',
                title: 'Wildfire Detected',
                message: `Elevated temperatures and poor air quality detected in sector ${Math.floor(Math.random() * 10) + 1}`,
                recommendation: 'Fire department dispatched. Evacuation protocols initiated. Monitor wind direction.',
                icon: '🔥',
                priority: 'critical'
            });
        }

        // Police action (multiple anomalies + audio spikes)
        const audioSpikes = this.sensors.filter(s => s.type === 'audio' && s.value > 80);
        const anomalies = this.sensors.filter(s => s.status === 'anomaly');

        if (audioSpikes.length > 2 && anomalies.length > 1 && Math.random() < 0.03) {
            events.push({
                type: 'police',
                title: 'Police Action in Progress',
                message: `Elevated audio levels and sensor anomalies detected near intersection ${Math.floor(Math.random() * 20) + 1}`,
                recommendation: 'Redirect traffic away from area. Monitor situation via nearby cameras.',
                icon: '🚔',
                priority: 'high'
            });
        }

        // Medical emergency (accelerometer + movement patterns)
        const accelSensors = this.sensors.filter(s => s.type === 'accelerometer' && s.value > 0.8);

        if (accelSensors.length > 1 && Math.random() < 0.02) {
            events.push({
                type: 'medical',
                title: 'Medical Emergency Detected',
                message: `Unusual movement patterns detected. Possible accident at ${['Park Ave', 'Main St', 'Broadway', 'Oak St'][Math.floor(Math.random() * 4)]}`,
                recommendation: 'Ambulance dispatched. Clear emergency lanes on nearby routes.',
                icon: '🚑',
                priority: 'high'
            });
        }

        // Traffic incident (high traffic + audio)
        const trafficSensors = this.citySensors?.filter(s => s.type === 'traffic' && s.value > 80) || [];

        if (trafficSensors.length > 3 && audioSpikes.length > 1 && Math.random() < 0.04) {
            events.push({
                type: 'traffic',
                title: 'Major Traffic Incident',
                message: `Severe congestion with elevated noise levels. Possible accident on Highway ${Math.floor(Math.random() * 10) + 1}`,
                recommendation: 'Activate alternative routes. Deploy traffic management team.',
                icon: '🚗',
                priority: 'medium'
            });
        }

        // Weather event (multiple sensor types affected)
        const tempExtremes = this.sensors.filter(s => s.type === 'temperature' && (s.value < 5 || s.value > 40));
        const humidityHigh = this.sensors.filter(s => s.type === 'humidity' && s.value > 90);

        if (tempExtremes.length > 4 && humidityHigh.length > 3 && Math.random() < 0.02) {
            events.push({
                type: 'weather',
                title: 'Severe Weather Alert',
                message: `Extreme weather conditions detected. ${tempExtremes[0]?.value > 30 ? 'Heat wave' : 'Cold snap'} with high humidity`,
                recommendation: 'Activate cooling/warming centers. Issue public safety advisory.',
                icon: tempExtremes[0]?.value > 30 ? '☀️' : '❄️',
                priority: 'medium'
            });
        }

        // Show most critical event
        if (events.length > 0) {
            const priorityOrder = { critical: 3, high: 2, medium: 1 };
            const mostCritical = events.sort((a, b) => priorityOrder[b.priority] - priorityOrder[a.priority])[0];
            this.showNotification(mostCritical);
        }
    }

    showNotification(event) {
        if (this.eventNotifications.currentNotification) return; // Don't overlap notifications

        const notification = document.getElementById('eventNotification');
        const icon = notification.querySelector('.notification-icon');
        const title = notification.querySelector('.notification-title');
        const message = notification.querySelector('.notification-message');
        const recommendation = notification.querySelector('.notification-recommendation');

        // Set content
        icon.textContent = event.icon;
        icon.className = `notification-icon ${event.type}`;
        title.textContent = event.title;
        message.textContent = event.message;
        recommendation.textContent = `Recommendation: ${event.recommendation}`;

        // Show notification
        notification.classList.remove('hidden');
        notification.classList.add('show');

        this.eventNotifications.currentNotification = event;

        // Auto-hide after 8 seconds
        setTimeout(() => {
            if (this.eventNotifications.currentNotification === event) {
                this.hideNotification();
            }
        }, 8000);
    }

    hideNotification() {
        const notification = document.getElementById('eventNotification');
        notification.classList.remove('show');
        notification.classList.add('hidden');
        this.eventNotifications.currentNotification = null;
    }
}

// Initialize simulation when page loads
document.addEventListener('DOMContentLoaded', () => {
    const simulation = new UrbanSenseSimulation();

    // Auto-start simulation after 2 seconds
    setTimeout(() => {
        simulation.startSimulation();
    }, 2000);
});