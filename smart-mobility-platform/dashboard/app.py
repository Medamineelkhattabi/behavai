"""
Smart Mobility Platform Dashboard
Interactive Streamlit dashboard for operators and stakeholders
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import requests
import asyncio
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Any, Optional

# Dashboard configuration
st.set_page_config(
    page_title="Smart Mobility Platform",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
API_HEADERS = {
    "Authorization": "Bearer demo_token",  # In production, use proper authentication
    "Content-Type": "application/json"
}

class DashboardAPI:
    """API client for dashboard data"""
    
    @staticmethod
    @st.cache_data(ttl=60)  # Cache for 1 minute
    def get_system_health():
        """Get system health status"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", headers=API_HEADERS, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": "API unavailable"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @staticmethod
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_vehicle_status(route_id=None):
        """Get real-time vehicle status"""
        try:
            params = {"route_id": route_id} if route_id else {}
            response = requests.get(f"{API_BASE_URL}/realtime/vehicles", 
                                  headers=API_HEADERS, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return []
        except Exception as e:
            st.error(f"Failed to fetch vehicle data: {e}")
            return []
    
    @staticmethod
    @st.cache_data(ttl=300)
    def get_congestion_overview():
        """Get congestion overview"""
        try:
            response = requests.get(f"{API_BASE_URL}/realtime/congestion", 
                                  headers=API_HEADERS, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"overall_congestion_level": 0, "congested_routes": [], 
                       "average_speed": 0, "incidents": []}
        except Exception as e:
            st.error(f"Failed to fetch congestion data: {e}")
            return {"overall_congestion_level": 0, "congested_routes": [], 
                   "average_speed": 0, "incidents": []}
    
    @staticmethod
    @st.cache_data(ttl=60)
    def get_recent_anomalies(hours=24):
        """Get recent anomalies"""
        try:
            params = {"hours": hours}
            response = requests.get(f"{API_BASE_URL}/anomalies/recent", 
                                  headers=API_HEADERS, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return []
        except Exception as e:
            st.error(f"Failed to fetch anomaly data: {e}")
            return []
    
    @staticmethod
    def predict_demand(locations, time_horizon=24):
        """Get demand predictions"""
        try:
            payload = {
                "locations": locations,
                "time_horizon": time_horizon
            }
            response = requests.post(f"{API_BASE_URL}/predictions/demand", 
                                   headers=API_HEADERS, json=payload, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"predictions": []}
        except Exception as e:
            st.error(f"Failed to get demand predictions: {e}")
            return {"predictions": []}

def create_sample_data():
    """Create sample data for demo purposes"""
    
    # Sample vehicle data
    vehicles = []
    for i in range(20):
        vehicles.append({
            "vehicle_id": f"vehicle_{i:03d}",
            "route_id": f"route_{i % 5}",
            "transport_mode": "bus",
            "current_location": {
                "latitude": 25.2048 + np.random.uniform(-0.1, 0.1),
                "longitude": 55.2708 + np.random.uniform(-0.1, 0.1)
            },
            "occupancy_level": np.random.randint(20, 90),
            "speed": np.random.uniform(15, 45),
            "status": np.random.choice(["active", "maintenance"], p=[0.9, 0.1]),
            "last_updated": datetime.now().isoformat()
        })
    
    # Sample congestion data
    congestion = {
        "overall_congestion_level": np.random.randint(40, 80),
        "congested_routes": ["route_001", "route_003"],
        "average_speed": np.random.uniform(25, 35),
        "incidents": [
            {"location": "Downtown", "type": "accident", "severity": "medium"},
            {"location": "Airport Road", "type": "construction", "severity": "low"}
        ],
        "timestamp": datetime.now().isoformat()
    }
    
    # Sample anomalies
    anomalies = []
    for i in range(5):
        anomalies.append({
            "alert_id": f"alert_{i:03d}",
            "timestamp": (datetime.now() - timedelta(hours=np.random.randint(1, 24))).isoformat(),
            "location_id": f"stop_{np.random.randint(1, 50):03d}",
            "anomaly_type": np.random.choice(["demand_surge", "congestion_anomaly", "vehicle_breakdown"]),
            "severity": np.random.choice(["low", "medium", "high"]),
            "description": f"Anomaly detected at location with severity level",
            "status": "active"
        })
    
    return vehicles, congestion, anomalies

def render_header():
    """Render dashboard header"""
    st.markdown('<h1 class="main-header">🚌 Smart Mobility Platform</h1>', unsafe_allow_html=True)
    
    # System status indicator
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Status", "Operational", "98.5% Uptime")
    
    with col2:
        st.metric("Active Vehicles", "187", "+5 from yesterday")
    
    with col3:
        st.metric("Total Passengers", "12,456", "+8.2% today")
    
    with col4:
        st.metric("Avg Response Time", "1.2s", "-0.3s improvement")

def render_real_time_overview():
    """Render real-time system overview"""
    st.subheader("📊 Real-Time System Overview")
    
    # Get data (use sample data for demo)
    vehicles, congestion, anomalies = create_sample_data()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Congestion heatmap
        st.subheader("Congestion Levels")
        
        # Create sample congestion data for visualization
        routes = [f"Route {i}" for i in range(1, 11)]
        congestion_levels = np.random.randint(20, 90, 10)
        colors = ['green' if x < 40 else 'yellow' if x < 70 else 'red' for x in congestion_levels]
        
        fig = go.Figure(data=go.Bar(
            x=routes,
            y=congestion_levels,
            marker_color=colors,
            text=[f"{x}%" for x in congestion_levels],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Current Congestion by Route",
            xaxis_title="Routes",
            yaxis_title="Congestion Level (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Key metrics
        st.subheader("System Metrics")
        
        # Overall congestion
        congestion_level = congestion["overall_congestion_level"]
        congestion_color = "red" if congestion_level > 70 else "orange" if congestion_level > 40 else "green"
        st.metric("Overall Congestion", f"{congestion_level}%", delta_color="inverse")
        
        # Average speed
        avg_speed = congestion["average_speed"]
        st.metric("Average Speed", f"{avg_speed:.1f} km/h")
        
        # Active incidents
        incidents_count = len(congestion["incidents"])
        st.metric("Active Incidents", incidents_count)
        
        # Fleet utilization
        active_vehicles = len([v for v in vehicles if v["status"] == "active"])
        utilization = (active_vehicles / len(vehicles)) * 100
        st.metric("Fleet Utilization", f"{utilization:.1f}%")

def render_vehicle_tracking():
    """Render vehicle tracking map"""
    st.subheader("🗺️ Live Vehicle Tracking")
    
    # Get vehicle data
    vehicles, _, _ = create_sample_data()
    
    # Create map centered on Dubai
    m = folium.Map(location=[25.2048, 55.2708], zoom_start=11)
    
    # Add vehicle markers
    for vehicle in vehicles:
        lat = vehicle["current_location"]["latitude"]
        lng = vehicle["current_location"]["longitude"]
        occupancy = vehicle["occupancy_level"]
        
        # Color based on occupancy
        if occupancy < 50:
            color = 'green'
        elif occupancy < 80:
            color = 'orange'
        else:
            color = 'red'
        
        # Create popup info
        popup_info = f"""
        <b>Vehicle:</b> {vehicle['vehicle_id']}<br>
        <b>Route:</b> {vehicle['route_id']}<br>
        <b>Occupancy:</b> {occupancy}%<br>
        <b>Speed:</b> {vehicle['speed']:.1f} km/h<br>
        <b>Status:</b> {vehicle['status']}
        """
        
        folium.CircleMarker(
            location=[lat, lng],
            radius=8,
            popup=folium.Popup(popup_info, max_width=200),
            color=color,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    # Display map
    st_folium(m, width=700, height=500)
    
    # Vehicle status table
    st.subheader("Vehicle Status Summary")
    
    # Create DataFrame for display
    df_vehicles = pd.DataFrame([{
        'Vehicle ID': v['vehicle_id'],
        'Route': v['route_id'],
        'Occupancy (%)': v['occupancy_level'],
        'Speed (km/h)': f"{v['speed']:.1f}",
        'Status': v['status']
    } for v in vehicles])
    
    st.dataframe(df_vehicles, use_container_width=True)

def render_predictions():
    """Render prediction dashboard"""
    st.subheader("🔮 Predictive Analytics")
    
    tab1, tab2, tab3 = st.tabs(["Demand Forecast", "Congestion Prediction", "Passenger Flow"])
    
    with tab1:
        st.subheader("Passenger Demand Forecast")
        
        # Time horizon selector
        time_horizon = st.selectbox("Forecast Horizon", [6, 12, 24, 48], index=2)
        
        # Generate sample prediction data
        hours = list(range(time_horizon))
        current_hour = datetime.now().hour
        
        # Simulate demand patterns
        base_demand = 100
        demand_data = []
        
        for i in hours:
            hour = (current_hour + i) % 24
            
            # Rush hour pattern
            if hour in [7, 8, 9]:  # Morning rush
                multiplier = 2.0
            elif hour in [17, 18, 19]:  # Evening rush
                multiplier = 2.5
            elif hour in [10, 11, 12, 13, 14, 15, 16]:  # Daytime
                multiplier = 1.2
            else:  # Night
                multiplier = 0.4
            
            predicted_demand = base_demand * multiplier + np.random.normal(0, 10)
            demand_data.append(max(0, predicted_demand))
        
        # Create time labels
        time_labels = [(datetime.now() + timedelta(hours=i)).strftime("%H:%M") for i in hours]
        
        # Plot demand forecast
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_labels,
            y=demand_data,
            mode='lines+markers',
            name='Predicted Demand',
            line=dict(color='blue', width=3),
            marker=dict(size=6)
        ))
        
        # Add confidence intervals
        upper_bound = [d * 1.2 for d in demand_data]
        lower_bound = [d * 0.8 for d in demand_data]
        
        fig.add_trace(go.Scatter(
            x=time_labels + time_labels[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))
        
        fig.update_layout(
            title=f"Demand Forecast - Next {time_horizon} Hours",
            xaxis_title="Time",
            yaxis_title="Passenger Count",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Peak demand insights
        peak_hour_idx = np.argmax(demand_data)
        peak_time = time_labels[peak_hour_idx]
        peak_demand = demand_data[peak_hour_idx]
        
        st.info(f"📈 **Peak Demand**: {peak_demand:.0f} passengers expected at {peak_time}")
    
    with tab2:
        st.subheader("Congestion Prediction")
        
        # Route selector
        selected_routes = st.multiselect(
            "Select Routes",
            [f"Route {i}" for i in range(1, 11)],
            default=["Route 1", "Route 2", "Route 3"]
        )
        
        if selected_routes:
            # Generate congestion predictions
            fig = go.Figure()
            
            for route in selected_routes:
                # Simulate congestion pattern
                congestion_data = []
                for i in range(24):
                    hour = (datetime.now().hour + i) % 24
                    
                    if hour in [7, 8, 9, 17, 18, 19]:  # Rush hours
                        congestion = np.random.uniform(60, 90)
                    elif hour in [22, 23, 0, 1, 2, 3, 4, 5]:  # Night
                        congestion = np.random.uniform(10, 30)
                    else:
                        congestion = np.random.uniform(30, 60)
                    
                    congestion_data.append(congestion)
                
                time_labels = [(datetime.now() + timedelta(hours=i)).strftime("%H:%M") for i in range(24)]
                
                fig.add_trace(go.Scatter(
                    x=time_labels,
                    y=congestion_data,
                    mode='lines+markers',
                    name=route,
                    line=dict(width=2),
                    marker=dict(size=4)
                ))
            
            fig.update_layout(
                title="24-Hour Congestion Forecast by Route",
                xaxis_title="Time",
                yaxis_title="Congestion Level (%)",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Passenger Flow Prediction")
        
        # Origin-destination flow matrix
        st.write("**Top Origin-Destination Flows (Next 6 Hours)**")
        
        # Sample OD data
        origins = ["Central Station", "Airport", "Mall", "University", "Business District"]
        destinations = ["Central Station", "Airport", "Mall", "University", "Business District"]
        
        # Create flow matrix
        flow_matrix = np.random.randint(10, 100, (len(origins), len(destinations)))
        np.fill_diagonal(flow_matrix, 0)  # No self-flows
        
        # Create heatmap
        fig = px.imshow(
            flow_matrix,
            x=destinations,
            y=origins,
            color_continuous_scale="Blues",
            title="Predicted Passenger Flows"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def render_anomaly_alerts():
    """Render anomaly detection alerts"""
    st.subheader("🚨 Anomaly Detection & Alerts")
    
    # Get anomaly data
    _, _, anomalies = create_sample_data()
    
    # Alert summary
    col1, col2, col3 = st.columns(3)
    
    high_alerts = len([a for a in anomalies if a["severity"] == "high"])
    medium_alerts = len([a for a in anomalies if a["severity"] == "medium"])
    low_alerts = len([a for a in anomalies if a["severity"] == "low"])
    
    with col1:
        st.metric("🔴 High Priority", high_alerts)
    
    with col2:
        st.metric("🟡 Medium Priority", medium_alerts)
    
    with col3:
        st.metric("🟢 Low Priority", low_alerts)
    
    # Display alerts
    st.subheader("Recent Alerts")
    
    for alert in sorted(anomalies, key=lambda x: x["timestamp"], reverse=True):
        severity_class = f"alert-{alert['severity']}"
        
        alert_time = datetime.fromisoformat(alert["timestamp"]).strftime("%Y-%m-%d %H:%M")
        
        st.markdown(f"""
        <div class="{severity_class}">
            <strong>{alert['anomaly_type'].replace('_', ' ').title()}</strong> - {alert['severity'].upper()}<br>
            <strong>Location:</strong> {alert['location_id']}<br>
            <strong>Time:</strong> {alert_time}<br>
            <strong>Description:</strong> {alert['description']}<br>
            <strong>Status:</strong> {alert['status']}
        </div>
        """, unsafe_allow_html=True)

def render_optimization_recommendations():
    """Render optimization recommendations"""
    st.subheader("⚡ Optimization Recommendations")
    
    # Current optimization opportunities
    st.write("**Current Optimization Opportunities**")
    
    recommendations = [
        {
            "type": "Schedule Adjustment",
            "priority": "High",
            "description": "Increase frequency on Route 3 during 17:00-19:00 to reduce wait times by 25%",
            "expected_benefit": "25% reduction in passenger wait time",
            "implementation": "Immediate"
        },
        {
            "type": "Vehicle Dispatch",
            "priority": "Medium",
            "description": "Deploy additional vehicle to Central Station to handle demand surge",
            "expected_benefit": "15% improvement in service level",
            "implementation": "Within 30 minutes"
        },
        {
            "type": "Route Optimization",
            "priority": "Medium",
            "description": "Reroute vehicles from Route 1 to avoid construction zone",
            "expected_benefit": "10 minutes average time saving",
            "implementation": "Next service cycle"
        }
    ]
    
    for i, rec in enumerate(recommendations):
        priority_color = "🔴" if rec["priority"] == "High" else "🟡" if rec["priority"] == "Medium" else "🟢"
        
        with st.expander(f"{priority_color} {rec['type']} - {rec['priority']} Priority"):
            st.write(f"**Description:** {rec['description']}")
            st.write(f"**Expected Benefit:** {rec['expected_benefit']}")
            st.write(f"**Implementation:** {rec['implementation']}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Implement", key=f"impl_{i}"):
                    st.success("Optimization request submitted!")
            
            with col2:
                if st.button(f"Dismiss", key=f"dismiss_{i}"):
                    st.info("Recommendation dismissed")

def render_performance_kpis():
    """Render performance KPIs"""
    st.subheader("📈 Performance KPIs")
    
    # Time period selector
    period = st.selectbox("Time Period", ["Last 24 Hours", "Last 7 Days", "Last 30 Days"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Service reliability
        st.subheader("Service Reliability")
        
        reliability_data = {
            "On-time Performance": 87.5,
            "Service Availability": 98.2,
            "Fleet Utilization": 82.1,
            "Route Coverage": 95.8
        }
        
        for metric, value in reliability_data.items():
            st.metric(metric, f"{value}%")
        
        # Reliability trend chart
        days = list(range(7))
        reliability_trend = [85 + np.random.uniform(-3, 3) for _ in days]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[f"Day {i+1}" for i in days],
            y=reliability_trend,
            mode='lines+markers',
            name='On-time Performance',
            line=dict(color='green', width=3)
        ))
        
        fig.update_layout(
            title="7-Day Reliability Trend",
            yaxis_title="Percentage (%)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Passenger satisfaction
        st.subheader("Passenger Experience")
        
        satisfaction_data = {
            "Overall Satisfaction": 4.2,
            "Wait Time Satisfaction": 3.8,
            "Comfort Rating": 4.1,
            "Information Quality": 4.0
        }
        
        for metric, value in satisfaction_data.items():
            st.metric(metric, f"{value}/5.0")
        
        # Satisfaction breakdown
        categories = list(satisfaction_data.keys())
        values = list(satisfaction_data.values())
        
        fig = go.Figure(data=go.Bar(
            x=categories,
            y=values,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        ))
        
        fig.update_layout(
            title="Satisfaction Ratings",
            yaxis_title="Rating (1-5)",
            yaxis_range=[0, 5],
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_ai_assistant():
    """Render AI assistant chat interface"""
    st.subheader("🤖 AI Assistant")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat interface
    with st.container():
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.write(f"**You:** {message['content']}")
            else:
                st.write(f"**AI Assistant:** {message['content']}")
        
        # Chat input
        user_input = st.text_input("Ask me anything about the mobility system...", key="chat_input")
        
        if st.button("Send") and user_input:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Generate AI response (mock for demo)
            if "status" in user_input.lower():
                ai_response = "The system is currently operating normally with 98.5% uptime. All key components are functional and there are no critical alerts at this time."
            elif "predict" in user_input.lower() or "forecast" in user_input.lower():
                ai_response = "Based on current patterns, I predict moderate congestion during evening rush hour (5-7 PM) with peak demand on Route A. Passenger volume is expected to increase by 15% compared to yesterday."
            elif "anomal" in user_input.lower() or "alert" in user_input.lower():
                ai_response = "I've detected 2 minor anomalies in the past hour: unusual passenger surge at Central Station (+40% above normal) and slight delay on Route B (5 minutes behind schedule). Both are being monitored."
            elif "optimize" in user_input.lower():
                ai_response = "To optimize current operations, I recommend: 1) Deploy additional vehicle on Route A, 2) Adjust schedule frequency during peak hours, 3) Implement dynamic pricing to distribute demand."
            else:
                ai_response = "I'm here to help with your smart mobility platform questions. I can provide information about system status, predictions, anomalies, optimization recommendations, and operational guidance."
            
            # Add AI response
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            
            # Rerun to update chat display
            st.rerun()
        
        # Quick action buttons
        st.write("**Quick Actions:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("System Status"):
                st.session_state.chat_history.append({"role": "user", "content": "What's the current system status?"})
                st.rerun()
        
        with col2:
            if st.button("Predictions"):
                st.session_state.chat_history.append({"role": "user", "content": "Show me demand predictions"})
                st.rerun()
        
        with col3:
            if st.button("Anomalies"):
                st.session_state.chat_history.append({"role": "user", "content": "Are there any anomalies?"})
                st.rerun()
        
        with col4:
            if st.button("Optimize"):
                st.session_state.chat_history.append({"role": "user", "content": "How can I optimize operations?"})
                st.rerun()

def main():
    """Main dashboard application"""
    
    # Render header
    render_header()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Overview", "Vehicle Tracking", "Predictions", "Anomaly Detection", 
         "Optimization", "Performance KPIs", "AI Assistant"]
    )
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # User role selector
    user_role = st.sidebar.selectbox("User Role", ["Operator", "Manager", "Analyst"])
    
    # Render selected page
    if page == "Overview":
        render_real_time_overview()
    elif page == "Vehicle Tracking":
        render_vehicle_tracking()
    elif page == "Predictions":
        render_predictions()
    elif page == "Anomaly Detection":
        render_anomaly_alerts()
    elif page == "Optimization":
        render_optimization_recommendations()
    elif page == "Performance KPIs":
        render_performance_kpis()
    elif page == "AI Assistant":
        render_ai_assistant()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Smart Mobility Platform v1.0**")
    st.sidebar.markdown("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()