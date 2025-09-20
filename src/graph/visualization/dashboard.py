"""
Real-time Visualization Dashboard for UrbanSense IoT Network
Interactive dashboard showing sensor data, graph relationships, and AI insights
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
import threading
import time
import queue
from typing import Dict, List, Optional


class UrbanSenseDashboard:
    """Real-time dashboard for IoT sensor network visualization"""

    def __init__(self, simulator=None, gnn_model=None, graph_builder=None):
        self.app = dash.Dash(__name__)
        self.simulator = simulator
        self.gnn_model = gnn_model
        self.graph_builder = graph_builder

        # Data queues for real-time updates
        self.sensor_data_queue = queue.Queue(maxsize=1000)
        self.anomaly_queue = queue.Queue(maxsize=100)
        self.traffic_queue = queue.Queue(maxsize=100)
        self.emergency_queue = queue.Queue(maxsize=100)

        # Dashboard state
        self.current_data = {}
        self.sensor_history = {}
        self.graph_data = None
        self.anomalies = []
        self.traffic_recommendations = []
        self.emergency_alerts = []

        # Colors and styling
        self.colors = {
            'background': '#1e1e1e',
            'paper': '#2d2d2d',
            'text': '#ffffff',
            'primary': '#00b4d8',
            'secondary': '#0077b6',
            'success': '#06d6a0',
            'warning': '#f77f00',
            'danger': '#d62828',
            'nodes': {
                'temperature': '#ff6b6b',
                'humidity': '#4ecdc4',
                'audio': '#45b7d1',
                'visual': '#96ceb4',
                'air_quality': '#feca57',
                'accelerometer': '#ff9ff3'
            }
        }

        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("UrbanSense: Real-time IoT Sensor Fusion",
                       style={'color': self.colors['text'], 'textAlign': 'center'}),
                html.P("Dynamic Graph Neural Networks for Smart City Applications",
                      style={'color': self.colors['text'], 'textAlign': 'center'})
            ], style={'backgroundColor': self.colors['background'], 'padding': '20px'}),

            # Control Panel
            html.Div([
                html.Div([
                    html.Label("Update Interval (seconds):", style={'color': self.colors['text']}),
                    dcc.Slider(id='update-interval-slider', min=1, max=10, value=2, step=1,
                              marks={i: str(i) for i in range(1, 11)})
                ], style={'width': '30%', 'display': 'inline-block'}),

                html.Div([
                    html.Label("Graph Type:", style={'color': self.colors['text']}),
                    dcc.Dropdown(
                        id='graph-type-dropdown',
                        options=[
                            {'label': 'Spatial Relationships', 'value': 'spatial'},
                            {'label': 'Temporal Correlations', 'value': 'temporal'},
                            {'label': 'Semantic Similarities', 'value': 'semantic'},
                            {'label': 'Merged Graph', 'value': 'merged'}
                        ],
                        value='merged',
                        style={'color': '#000000'}
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '5%'}),

                html.Div([
                    html.Button('Start Simulation', id='start-button', n_clicks=0,
                               style={'backgroundColor': self.colors['success'],
                                     'color': 'white', 'border': 'none', 'padding': '10px'}),
                    html.Button('Stop Simulation', id='stop-button', n_clicks=0,
                               style={'backgroundColor': self.colors['danger'],
                                     'color': 'white', 'border': 'none', 'padding': '10px', 'marginLeft': '10px'})
                ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '5%'})
            ], style={'backgroundColor': self.colors['paper'], 'padding': '20px', 'margin': '10px'}),

            # Main Dashboard Grid
            html.Div([
                # Left Column - Network Graph
                html.Div([
                    html.H3("Sensor Network Graph", style={'color': self.colors['text']}),
                    dcc.Graph(id='network-graph', style={'height': '500px'})
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top',
                         'backgroundColor': self.colors['paper'], 'margin': '1%', 'padding': '10px'}),

                # Right Column - Sensor Data
                html.Div([
                    html.H3("Real-time Sensor Readings", style={'color': self.colors['text']}),
                    dcc.Graph(id='sensor-timeseries', style={'height': '500px'})
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top',
                         'backgroundColor': self.colors['paper'], 'margin': '1%', 'padding': '10px'})
            ]),

            # Second Row - Applications
            html.Div([
                # Anomaly Detection
                html.Div([
                    html.H3("Anomaly Detection", style={'color': self.colors['text']}),
                    html.Div(id='anomaly-display')
                ], style={'width': '31%', 'display': 'inline-block', 'verticalAlign': 'top',
                         'backgroundColor': self.colors['paper'], 'margin': '1%', 'padding': '10px'}),

                # Traffic Optimization
                html.Div([
                    html.H3("Traffic Optimization", style={'color': self.colors['text']}),
                    html.Div(id='traffic-display')
                ], style={'width': '31%', 'display': 'inline-block', 'verticalAlign': 'top',
                         'backgroundColor': self.colors['paper'], 'margin': '1%', 'padding': '10px'}),

                # Emergency Detection
                html.Div([
                    html.H3("Emergency Alerts", style={'color': self.colors['text']}),
                    html.Div(id='emergency-display')
                ], style={'width': '31%', 'display': 'inline-block', 'verticalAlign': 'top',
                         'backgroundColor': self.colors['paper'], 'margin': '1%', 'padding': '10px'})
            ]),

            # Statistics Row
            html.Div([
                html.H3("System Statistics", style={'color': self.colors['text']}),
                html.Div(id='statistics-display')
            ], style={'backgroundColor': self.colors['paper'], 'margin': '10px', 'padding': '20px'}),

            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=2000,  # Update every 2 seconds
                n_intervals=0
            ),

            # Data stores
            dcc.Store(id='sensor-data-store'),
            dcc.Store(id='graph-data-store')

        ], style={'backgroundColor': self.colors['background'], 'minHeight': '100vh'})

    def _setup_callbacks(self):
        """Setup dashboard callbacks for interactivity"""

        @self.app.callback(
            [Output('sensor-data-store', 'data'),
             Output('graph-data-store', 'data')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_data_stores(n):
            """Update data stores with latest sensor data"""
            try:
                # Get latest data from queues
                latest_sensor_data = None
                while not self.sensor_data_queue.empty():
                    latest_sensor_data = self.sensor_data_queue.get_nowait()

                if latest_sensor_data:
                    self.current_data = latest_sensor_data
                    # Update sensor history
                    for sensor_id, reading in latest_sensor_data.get('batch_data', {}).items():
                        if sensor_id not in self.sensor_history:
                            self.sensor_history[sensor_id] = []
                        self.sensor_history[sensor_id].append(reading)
                        # Keep last 100 readings
                        if len(self.sensor_history[sensor_id]) > 100:
                            self.sensor_history[sensor_id] = self.sensor_history[sensor_id][-100:]

                return self.current_data, self.graph_data

            except Exception as e:
                print(f"Error updating data stores: {e}")
                return {}, {}

        @self.app.callback(
            Output('network-graph', 'figure'),
            [Input('graph-data-store', 'data'),
             Input('graph-type-dropdown', 'value')]
        )
        def update_network_graph(graph_data, graph_type):
            """Update the network graph visualization"""
            if not graph_data or not self.graph_builder:
                return self._create_empty_graph()

            try:
                # Get node positions and features
                node_features = self.graph_builder.get_node_features()
                sensors = list(self.graph_builder.sensor_nodes.values())

                if len(sensors) == 0:
                    return self._create_empty_graph()

                # Create networkx graph
                G = nx.Graph()

                # Add nodes
                for i, sensor in enumerate(sensors):
                    G.add_node(sensor.id,
                              pos=(sensor.location[1], sensor.location[0]),  # lon, lat
                              sensor_type=sensor.sensor_type,
                              reading=sensor.current_reading.get('value', 0) if sensor.current_reading else 0)

                # Add edges based on selected graph type
                if hasattr(self.graph_builder, 'graph_history') and self.graph_builder.graph_history:
                    latest_graphs = self.graph_builder.graph_history[-1]
                    adjacency = latest_graphs.get(graph_type, latest_graphs.get('merged'))

                    if adjacency is not None:
                        for i, sensor_i in enumerate(sensors):
                            for j, sensor_j in enumerate(sensors):
                                if i < len(adjacency) and j < len(adjacency[0]) and adjacency[i][j] > 0.1:
                                    G.add_edge(sensor_i.id, sensor_j.id, weight=adjacency[i][j])

                return self._create_network_figure(G)

            except Exception as e:
                print(f"Error creating network graph: {e}")
                return self._create_empty_graph()

        @self.app.callback(
            Output('sensor-timeseries', 'figure'),
            [Input('sensor-data-store', 'data')]
        )
        def update_sensor_timeseries(sensor_data):
            """Update sensor time series plot"""
            if not self.sensor_history:
                return self._create_empty_timeseries()

            try:
                fig = go.Figure()

                # Plot recent data for each sensor type
                sensor_types = ['temperature', 'humidity', 'audio', 'air_quality']
                colors = [self.colors['nodes'][st] for st in sensor_types]

                for sensor_type, color in zip(sensor_types, colors):
                    type_data = []
                    timestamps = []

                    for sensor_id, history in self.sensor_history.items():
                        if len(history) > 0 and history[0].get('sensor_type') == sensor_type:
                            for reading in history[-20:]:  # Last 20 readings
                                if 'timestamp' in reading and 'value' in reading and reading['value'] is not None:
                                    timestamps.append(datetime.fromtimestamp(reading['timestamp']))
                                    type_data.append(reading['value'])

                    if timestamps and type_data:
                        fig.add_trace(go.Scatter(
                            x=timestamps,
                            y=type_data,
                            mode='lines+markers',
                            name=sensor_type.title(),
                            line=dict(color=color),
                            marker=dict(size=4)
                        ))

                fig.update_layout(
                    title="Recent Sensor Readings",
                    xaxis_title="Time",
                    yaxis_title="Sensor Value",
                    plot_bgcolor=self.colors['background'],
                    paper_bgcolor=self.colors['paper'],
                    font=dict(color=self.colors['text']),
                    showlegend=True
                )

                return fig

            except Exception as e:
                print(f"Error creating timeseries: {e}")
                return self._create_empty_timeseries()

        @self.app.callback(
            Output('anomaly-display', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_anomaly_display(n):
            """Update anomaly detection display"""
            # Get latest anomalies from queue
            while not self.anomaly_queue.empty():
                try:
                    anomaly = self.anomaly_queue.get_nowait()
                    self.anomalies.append(anomaly)
                    if len(self.anomalies) > 10:
                        self.anomalies = self.anomalies[-10:]
                except queue.Empty:
                    break

            if not self.anomalies:
                return html.P("No anomalies detected", style={'color': self.colors['success']})

            anomaly_items = []
            for anomaly in self.anomalies[-5:]:  # Show last 5
                severity_color = {
                    'low': self.colors['warning'],
                    'medium': self.colors['warning'],
                    'high': self.colors['danger'],
                    'critical': self.colors['danger']
                }.get(getattr(anomaly, 'severity', 'low'), self.colors['warning'])

                anomaly_items.append(
                    html.Div([
                        html.P(f"⚠️ {getattr(anomaly, 'anomaly_type', 'Unknown')}",
                              style={'color': severity_color, 'fontWeight': 'bold', 'margin': '5px 0'}),
                        html.P(f"Sensor: {getattr(anomaly, 'sensor_id', 'Unknown')}",
                              style={'color': self.colors['text'], 'fontSize': '12px', 'margin': '2px 0'}),
                        html.P(f"Score: {getattr(anomaly, 'anomaly_score', 0):.2f}",
                              style={'color': self.colors['text'], 'fontSize': '12px', 'margin': '2px 0'})
                    ], style={'borderLeft': f'4px solid {severity_color}', 'paddingLeft': '10px', 'marginBottom': '10px'})
                )

            return html.Div(anomaly_items)

        @self.app.callback(
            Output('traffic-display', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_traffic_display(n):
            """Update traffic optimization display"""
            # Get latest traffic recommendations
            while not self.traffic_queue.empty():
                try:
                    recommendation = self.traffic_queue.get_nowait()
                    self.traffic_recommendations.append(recommendation)
                    if len(self.traffic_recommendations) > 10:
                        self.traffic_recommendations = self.traffic_recommendations[-10:]
                except queue.Empty:
                    break

            if not self.traffic_recommendations:
                return html.P("No traffic recommendations", style={'color': self.colors['text']})

            traffic_items = []
            for rec in self.traffic_recommendations[-3:]:  # Show last 3
                condition_color = {
                    'free_flow': self.colors['success'],
                    'moderate': self.colors['warning'],
                    'congested': self.colors['danger']
                }.get(getattr(rec, 'current_condition', 'moderate'), self.colors['warning'])

                traffic_items.append(
                    html.Div([
                        html.P(f"🚦 {getattr(rec, 'area_id', 'Unknown Area')}",
                              style={'color': self.colors['text'], 'fontWeight': 'bold', 'margin': '5px 0'}),
                        html.P(f"Condition: {getattr(rec, 'current_condition', 'Unknown')}",
                              style={'color': condition_color, 'fontSize': '12px', 'margin': '2px 0'}),
                        html.P(f"Action: {getattr(rec, 'recommendation_type', 'None')}",
                              style={'color': self.colors['text'], 'fontSize': '12px', 'margin': '2px 0'}),
                        html.P(f"Improvement: {getattr(rec, 'expected_improvement', 0):.1f}%",
                              style={'color': self.colors['success'], 'fontSize': '12px', 'margin': '2px 0'})
                    ], style={'borderLeft': f'4px solid {condition_color}', 'paddingLeft': '10px', 'marginBottom': '10px'})
                )

            return html.Div(traffic_items)

        @self.app.callback(
            Output('emergency-display', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_emergency_display(n):
            """Update emergency alerts display"""
            # Get latest emergency alerts
            while not self.emergency_queue.empty():
                try:
                    alert = self.emergency_queue.get_nowait()
                    self.emergency_alerts.append(alert)
                    if len(self.emergency_alerts) > 10:
                        self.emergency_alerts = self.emergency_alerts[-10:]
                except queue.Empty:
                    break

            if not self.emergency_alerts:
                return html.P("No emergency alerts", style={'color': self.colors['success']})

            alert_items = []
            for alert in self.emergency_alerts[-3:]:  # Show last 3
                severity_color = {
                    'low': self.colors['warning'],
                    'medium': self.colors['warning'],
                    'high': self.colors['danger'],
                    'critical': self.colors['danger']
                }.get(getattr(alert, 'severity', 'low'), self.colors['danger'])

                alert_items.append(
                    html.Div([
                        html.P(f"🚨 {getattr(alert, 'emergency_type', 'Unknown').upper()}",
                              style={'color': severity_color, 'fontWeight': 'bold', 'margin': '5px 0'}),
                        html.P(f"Severity: {getattr(alert, 'severity', 'Unknown')}",
                              style={'color': severity_color, 'fontSize': '12px', 'margin': '2px 0'}),
                        html.P(f"Confidence: {getattr(alert, 'confidence', 0):.2f}",
                              style={'color': self.colors['text'], 'fontSize': '12px', 'margin': '2px 0'}),
                        html.P(f"Sensors: {len(getattr(alert, 'detected_sensors', []))}",
                              style={'color': self.colors['text'], 'fontSize': '12px', 'margin': '2px 0'})
                    ], style={'borderLeft': f'4px solid {severity_color}', 'paddingLeft': '10px', 'marginBottom': '10px'})
                )

            return html.Div(alert_items)

        @self.app.callback(
            Output('statistics-display', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_statistics(n):
            """Update system statistics"""
            stats = []

            # Sensor statistics
            total_sensors = len(self.sensor_history)
            active_sensors = sum(1 for history in self.sensor_history.values()
                               if len(history) > 0 and
                               (time.time() - history[-1].get('timestamp', 0)) < 30)

            stats.extend([
                html.Div([
                    html.H4(f"{total_sensors}", style={'color': self.colors['primary'], 'margin': '0'}),
                    html.P("Total Sensors", style={'color': self.colors['text'], 'margin': '0'})
                ], style={'display': 'inline-block', 'textAlign': 'center', 'margin': '0 20px'}),

                html.Div([
                    html.H4(f"{active_sensors}", style={'color': self.colors['success'], 'margin': '0'}),
                    html.P("Active Sensors", style={'color': self.colors['text'], 'margin': '0'})
                ], style={'display': 'inline-block', 'textAlign': 'center', 'margin': '0 20px'}),

                html.Div([
                    html.H4(f"{len(self.anomalies)}", style={'color': self.colors['warning'], 'margin': '0'}),
                    html.P("Anomalies", style={'color': self.colors['text'], 'margin': '0'})
                ], style={'display': 'inline-block', 'textAlign': 'center', 'margin': '0 20px'}),

                html.Div([
                    html.H4(f"{len(self.emergency_alerts)}", style={'color': self.colors['danger'], 'margin': '0'}),
                    html.P("Emergency Alerts", style={'color': self.colors['text'], 'margin': '0'})
                ], style={'display': 'inline-block', 'textAlign': 'center', 'margin': '0 20px'})
            ])

            return html.Div(stats, style={'textAlign': 'center'})

    def _create_network_figure(self, G):
        """Create network graph figure from NetworkX graph"""
        if len(G.nodes()) == 0:
            return self._create_empty_graph()

        # Get node positions
        pos = nx.get_node_attributes(G, 'pos')
        if not pos:
            pos = nx.spring_layout(G)

        # Create edge traces
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color=self.colors['secondary']),
            hoverinfo='none',
            mode='lines'
        )

        # Create node traces by sensor type
        node_traces = []
        sensor_types = set(nx.get_node_attributes(G, 'sensor_type').values())

        for sensor_type in sensor_types:
            node_x, node_y, node_text = [], [], []
            nodes_of_type = [node for node, attr in G.nodes(data=True)
                           if attr.get('sensor_type') == sensor_type]

            for node in nodes_of_type:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                reading = G.nodes[node].get('reading', 0)
                node_text.append(f"{node}<br>Type: {sensor_type}<br>Reading: {reading:.2f}")

            if node_x:
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    hoverinfo='text',
                    text=node_text,
                    name=sensor_type.title(),
                    marker=dict(
                        size=10,
                        color=self.colors['nodes'].get(sensor_type, self.colors['primary']),
                        line=dict(width=2, color='white')
                    )
                )
                node_traces.append(node_trace)

        # Create figure
        fig = go.Figure(data=[edge_trace] + node_traces,
                       layout=go.Layout(
                           title='IoT Sensor Network Graph',
                           titlefont_size=16,
                           showlegend=True,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           annotations=[dict(
                               text="Node colors represent sensor types",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor="left", yanchor="bottom",
                               font=dict(color=self.colors['text'], size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor=self.colors['background'],
                           paper_bgcolor=self.colors['paper'],
                           font=dict(color=self.colors['text'])
                       ))

        return fig

    def _create_empty_graph(self):
        """Create empty graph placeholder"""
        return go.Figure(
            layout=go.Layout(
                title="Waiting for sensor data...",
                plot_bgcolor=self.colors['background'],
                paper_bgcolor=self.colors['paper'],
                font=dict(color=self.colors['text'])
            )
        )

    def _create_empty_timeseries(self):
        """Create empty timeseries placeholder"""
        return go.Figure(
            layout=go.Layout(
                title="Waiting for sensor readings...",
                plot_bgcolor=self.colors['background'],
                paper_bgcolor=self.colors['paper'],
                font=dict(color=self.colors['text'])
            )
        )

    def add_sensor_data(self, data):
        """Add sensor data to the dashboard queue"""
        try:
            self.sensor_data_queue.put_nowait(data)
        except queue.Full:
            # Remove oldest item and add new one
            try:
                self.sensor_data_queue.get_nowait()
                self.sensor_data_queue.put_nowait(data)
            except queue.Empty:
                pass

    def add_anomaly(self, anomaly):
        """Add anomaly to the dashboard queue"""
        try:
            self.anomaly_queue.put_nowait(anomaly)
        except queue.Full:
            pass

    def add_traffic_recommendation(self, recommendation):
        """Add traffic recommendation to the dashboard queue"""
        try:
            self.traffic_queue.put_nowait(recommendation)
        except queue.Full:
            pass

    def add_emergency_alert(self, alert):
        """Add emergency alert to the dashboard queue"""
        try:
            self.emergency_queue.put_nowait(alert)
        except queue.Full:
            pass

    def run(self, host='localhost', port=8050, debug=False):
        """Run the dashboard server"""
        print(f"Starting UrbanSense Dashboard at http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)


def create_dashboard(simulator=None, gnn_model=None, graph_builder=None):
    """Factory function to create dashboard instance"""
    return UrbanSenseDashboard(simulator, gnn_model, graph_builder)