"""
Smart Mobility Platform - Simplified Windows Version
Main application entry point
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
import sys

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration
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
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import plotly
    except ImportError:
        missing_deps.append("plotly")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    return missing_deps

def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    
    # Sample vehicle data
    vehicles = []
    for i in range(20):
        vehicles.append({
            "vehicle_id": f"BUS-{i:03d}",
            "route": f"Route {(i % 5) + 1}",
            "latitude": 25.2048 + np.random.uniform(-0.05, 0.05),
            "longitude": 55.2708 + np.random.uniform(-0.05, 0.05),
            "occupancy": np.random.randint(20, 90),
            "speed": np.random.uniform(15, 45),
            "status": np.random.choice(["Active", "Maintenance"], p=[0.9, 0.1])
        })
    
    # Sample demand data
    hours = list(range(24))
    demand_data = []
    for hour in hours:
        if hour in [7, 8, 9]:  # Morning rush
            base_demand = np.random.randint(150, 200)
        elif hour in [17, 18, 19]:  # Evening rush
            base_demand = np.random.randint(180, 220)
        elif hour in [10, 11, 12, 13, 14, 15, 16]:  # Daytime
            base_demand = np.random.randint(80, 120)
        else:  # Night
            base_demand = np.random.randint(20, 50)
        
        demand_data.append({
            "hour": hour,
            "demand": base_demand,
            "route_1": base_demand * np.random.uniform(0.8, 1.2),
            "route_2": base_demand * np.random.uniform(0.7, 1.1),
            "route_3": base_demand * np.random.uniform(0.9, 1.3)
        })
    
    return vehicles, demand_data

def render_header():
    """Render dashboard header"""
    st.markdown('<h1 class="main-header">🚌 Smart Mobility Platform</h1>', unsafe_allow_html=True)
    
    # Check system status
    missing_deps = check_dependencies()
    
    if not missing_deps:
        st.markdown("""
        <div class="success-box">
            ✅ <strong>System Status:</strong> All dependencies loaded successfully!
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="info-box">
            ⚠️ <strong>Missing Dependencies:</strong> {', '.join(missing_deps)}
            <br>Please run: <code>pip install {' '.join(missing_deps)}</code>
        </div>
        """, unsafe_allow_html=True)
    
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Status", "Operational", "✅ Online")
    
    with col2:
        st.metric("Active Vehicles", "20", "+2 since last hour")
    
    with col3:
        st.metric("Total Passengers", "1,247", "+15% today")
    
    with col4:
        st.metric("Avg Response Time", "0.8s", "-0.2s improvement")

def render_vehicle_overview():
    """Render vehicle overview section"""
    st.subheader("🚌 Vehicle Fleet Overview")
    
    vehicles, _ = create_sample_data()
    
    # Vehicle status summary
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create vehicle status chart
        df_vehicles = pd.DataFrame(vehicles)
        
        fig = px.scatter_mapbox(
            df_vehicles,
            lat="latitude",
            lon="longitude",
            color="occupancy",
            size="occupancy",
            hover_data=["vehicle_id", "route", "speed", "status"],
            color_continuous_scale="RdYlGn_r",
            size_max=15,
            zoom=11,
            title="Real-time Vehicle Locations"
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            height=500,
            margin={"r": 0, "t": 30, "l": 0, "b": 0}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Fleet Statistics")
        
        # Status distribution
        status_counts = pd.DataFrame(vehicles)['status'].value_counts()
        fig_status = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Vehicle Status Distribution"
        )
        st.plotly_chart(fig_status, use_container_width=True)
        
        # Average occupancy by route
        df_vehicles = pd.DataFrame(vehicles)
        route_occupancy = df_vehicles.groupby('route')['occupancy'].mean().sort_values(ascending=False)
        
        st.subheader("Occupancy by Route")
        for route, occupancy in route_occupancy.items():
            st.metric(route, f"{occupancy:.1f}%")

def render_demand_forecast():
    """Render demand forecasting section"""
    st.subheader("📈 Demand Forecasting")
    
    _, demand_data = create_sample_data()
    df_demand = pd.DataFrame(demand_data)
    
    # Time period selector
    forecast_period = st.selectbox("Forecast Period", ["Next 6 Hours", "Next 12 Hours", "Next 24 Hours"])
    
    if forecast_period == "Next 6 Hours":
        hours_to_show = 6
    elif forecast_period == "Next 12 Hours":
        hours_to_show = 12
    else:
        hours_to_show = 24
    
    # Create demand forecast chart
    current_hour = datetime.now().hour
    future_hours = [(current_hour + i) % 24 for i in range(hours_to_show)]
    
    forecast_data = df_demand[df_demand['hour'].isin(future_hours)].head(hours_to_show)
    
    fig = go.Figure()
    
    # Add demand line
    fig.add_trace(go.Scatter(
        x=[f"{h:02d}:00" for h in forecast_data['hour']],
        y=forecast_data['demand'],
        mode='lines+markers',
        name='Total Demand',
        line=dict(color='blue', width=3)
    ))
    
    # Add route-specific lines
    fig.add_trace(go.Scatter(
        x=[f"{h:02d}:00" for h in forecast_data['hour']],
        y=forecast_data['route_1'],
        mode='lines+markers',
        name='Route 1',
        line=dict(dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=[f"{h:02d}:00" for h in forecast_data['hour']],
        y=forecast_data['route_2'],
        mode='lines+markers',
        name='Route 2',
        line=dict(dash='dash')
    ))
    
    fig.update_layout(
        title=f"Passenger Demand Forecast - {forecast_period}",
        xaxis_title="Time",
        yaxis_title="Passenger Count",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Peak demand insights
    peak_hour = forecast_data.loc[forecast_data['demand'].idxmax(), 'hour']
    peak_demand = forecast_data['demand'].max()
    
    st.info(f"📊 **Peak Demand**: {peak_demand:.0f} passengers expected at {peak_hour:02d}:00")

def render_system_performance():
    """Render system performance metrics"""
    st.subheader("⚡ System Performance")
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Service Reliability")
        reliability_metrics = {
            "On-time Performance": 87.5,
            "Service Availability": 98.2,
            "Fleet Utilization": 82.1
        }
        
        for metric, value in reliability_metrics.items():
            st.metric(metric, f"{value}%")
    
    with col2:
        st.subheader("Passenger Experience")
        experience_metrics = {
            "Satisfaction Score": 4.2,
            "Avg Wait Time": "3.5 min",
            "Complaint Rate": "0.8%"
        }
        
        for metric, value in experience_metrics.items():
            st.metric(metric, value)
    
    with col3:
        st.subheader("Operational Efficiency")
        efficiency_metrics = {
            "Fuel Efficiency": "85%",
            "Route Optimization": "92%",
            "Cost per Passenger": "$2.40"
        }
        
        for metric, value in efficiency_metrics.items():
            st.metric(metric, value)
    
    # Performance trend chart
    st.subheader("Performance Trends (Last 7 Days)")
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    on_time = [85, 87, 89, 86, 88, 82, 84]
    satisfaction = [4.1, 4.2, 4.3, 4.0, 4.2, 4.0, 4.1]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=days,
        y=on_time,
        mode='lines+markers',
        name='On-time Performance (%)',
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=days,
        y=[s * 20 for s in satisfaction],  # Scale for dual axis
        mode='lines+markers',
        name='Satisfaction Score',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Weekly Performance Trends",
        xaxis_title="Day",
        yaxis=dict(title="On-time Performance (%)", side="left"),
        yaxis2=dict(title="Satisfaction Score", side="right", overlaying="y", range=[60, 100]),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_ai_assistant():
    """Render AI assistant interface"""
    st.subheader("🤖 AI Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your Smart Mobility AI Assistant. I can help you with system status, predictions, and operational insights. How can I assist you today?"}
        ]
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about the mobility system..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response (simplified for demo)
        if "status" in prompt.lower():
            response = "The system is currently operating normally with 98.2% uptime. All 20 vehicles are active, and we're serving approximately 1,247 passengers today with an average satisfaction score of 4.2/5."
        elif "predict" in prompt.lower() or "forecast" in prompt.lower():
            response = "Based on current patterns, I predict peak demand will occur at 18:00 with approximately 220 passengers. Route 1 will have the highest utilization at 89%. I recommend deploying an additional vehicle on Route 1 during this period."
        elif "problem" in prompt.lower() or "issue" in prompt.lower():
            response = "I've detected minor delays on Route 2 (average 2.3 minutes behind schedule) due to traffic congestion. No critical issues are currently active. All systems are performing within normal parameters."
        elif "optimize" in prompt.lower():
            response = "Current optimization recommendations: 1) Increase Route 1 frequency by 15% during 17:00-19:00, 2) Redistribute 2 vehicles from Route 3 to Route 1, 3) Implement dynamic pricing to encourage off-peak travel. Expected improvement: 12% reduction in wait times."
        else:
            response = "I can help you with system status, demand predictions, performance optimization, and operational insights. Try asking about current status, forecasts, or optimization recommendations!"
        
        # Add AI response
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
    
    # Quick action buttons
    st.subheader("Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("System Status", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What's the current system status?"})
            st.rerun()
    
    with col2:
        if st.button("Demand Forecast", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Show me demand predictions"})
            st.rerun()
    
    with col3:
        if st.button("Check Issues", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Are there any current problems?"})
            st.rerun()
    
    with col4:
        if st.button("Optimize Routes", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "How can I optimize operations?"})
            st.rerun()

def main():
    """Main application"""
    
    # Render header
    render_header()
    
    # Sidebar navigation
    st.sidebar.title("🚌 Navigation")
    page = st.sidebar.selectbox(
        "Select View",
        ["Dashboard Overview", "Vehicle Fleet", "Demand Forecasting", "Performance Metrics", "AI Assistant"]
    )
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)")
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # System info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Information**")
    st.sidebar.markdown(f"**Status**: 🟢 Online")
    st.sidebar.markdown(f"**Last Updated**: {datetime.now().strftime('%H:%M:%S')}")
    st.sidebar.markdown(f"**Version**: v1.0.0")
    
    # Render selected page
    if page == "Dashboard Overview":
        col1, col2 = st.columns(2)
        with col1:
            render_vehicle_overview()
        with col2:
            render_demand_forecast()
    elif page == "Vehicle Fleet":
        render_vehicle_overview()
    elif page == "Demand Forecasting":
        render_demand_forecast()
    elif page == "Performance Metrics":
        render_system_performance()
    elif page == "AI Assistant":
        render_ai_assistant()
    
    # Footer
    st.markdown("---")
    st.markdown("**Smart Mobility Platform** - Powered by AI | Windows Demo Version")

if __name__ == "__main__":
    main()