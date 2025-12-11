"""
LeanNLP: Manufacturing Analytics Dashboard
SpaceX-inspired dark theme design with SVG icons.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_generator import SyntheticDataGenerator
from data.cmapss_loader import CMAPSSDataLoader, CMAPSSFeatureEngineer, generate_synthetic_cmapss
from models.nlp_pipeline import InsightExtractor, NaturalLanguageGenerator
from models.knowledge_graph import ManufacturingKnowledgeGraph
from models.predictive_analytics import PredictiveAnalyticsEngine
from utils.icons import ICONS, get_icon, icon_html, LOGO_SVG

# Page configuration
st.set_page_config(
    page_title="LEANNLP",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# SpaceX-inspired dark theme CSS
DARK_THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --bg-primary: #000000;
        --bg-secondary: #0a0a0a;
        --bg-tertiary: #111111;
        --bg-card: #161616;
        --border-color: #222222;
        --text-primary: #ffffff;
        --text-secondary: #a0a0a0;
        --text-muted: #666666;
        --accent-blue: #0066ff;
        --accent-green: #00ff88;
        --accent-red: #ff3333;
        --accent-yellow: #ffaa00;
    }
    
    .stApp {
        background-color: var(--bg-primary) !important;
    }
    
    .main .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }
    
    /* Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary);
    }
    
    h1, h2, h3 {
        font-weight: 500 !important;
        letter-spacing: 0.5px;
        color: var(--text-primary) !important;
    }
    
    p, span, div {
        color: var(--text-secondary);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }
    
    /* Header styles */
    .main-header {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 1rem 0 2rem 0;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 1.5rem;
        font-weight: 600;
        letter-spacing: 4px;
        margin: 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 4px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .metric-label {
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 300;
        color: var(--text-primary);
        line-height: 1.2;
    }
    
    .metric-delta {
        font-size: 0.875rem;
        margin-top: 0.5rem;
    }
    
    .metric-delta.positive { color: var(--accent-green); }
    .metric-delta.negative { color: var(--accent-red); }
    .metric-delta.neutral { color: var(--text-muted); }
    
    /* Cards */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 4px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .card-title {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Alert boxes */
    .alert {
        padding: 1rem 1.25rem;
        border-radius: 4px;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }
    
    .alert-high {
        background: rgba(255, 51, 51, 0.1);
        border-left: 3px solid var(--accent-red);
    }
    
    .alert-medium {
        background: rgba(255, 170, 0, 0.1);
        border-left: 3px solid var(--accent-yellow);
    }
    
    .alert-low {
        background: rgba(0, 255, 136, 0.1);
        border-left: 3px solid var(--accent-green);
    }
    
    .alert-info {
        background: rgba(0, 102, 255, 0.1);
        border-left: 3px solid var(--accent-blue);
    }
    
    .alert-content {
        flex: 1;
    }
    
    .alert-title {
        font-weight: 500;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }
    
    .alert-text {
        font-size: 0.875rem;
        color: var(--text-secondary);
    }
    
    /* Navigation */
    .nav-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 4px;
        cursor: pointer;
        transition: background 0.2s;
        color: var(--text-secondary);
    }
    
    .nav-item:hover {
        background: var(--bg-tertiary);
    }
    
    .nav-item.active {
        background: var(--bg-tertiary);
        color: var(--text-primary);
    }
    
    /* Data table */
    .dataframe {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
    }
    
    .dataframe th {
        background: var(--bg-tertiary) !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        font-size: 0.75rem !important;
        letter-spacing: 1px !important;
    }
    
    .dataframe td {
        border-color: var(--border-color) !important;
    }
    
    /* Streamlit overrides */
    .stSelectbox > div > div {
        background: var(--bg-card) !important;
        border-color: var(--border-color) !important;
    }
    
    .stTextInput > div > div {
        background: var(--bg-card) !important;
        border-color: var(--border-color) !important;
    }
    
    .stButton > button {
        background: transparent !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        letter-spacing: 1px !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.2s !important;
    }
    
    .stButton > button:hover {
        background: var(--bg-tertiary) !important;
        border-color: var(--text-muted) !important;
    }
    
    .stFileUploader > div {
        background: var(--bg-card) !important;
        border: 1px dashed var(--border-color) !important;
    }
    
    /* Radio buttons */
    .stRadio > div {
        flex-direction: row !important;
        gap: 1rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Chat */
    .stChatMessage {
        background: var(--bg-card) !important;
    }
    
    /* Plotly chart backgrounds */
    .js-plotly-plot .plotly .bg {
        fill: var(--bg-primary) !important;
    }
    
    /* Section divider */
    .section-divider {
        height: 1px;
        background: var(--border-color);
        margin: 2rem 0;
    }
    
    /* Status indicator */
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    
    .status-dot.online { background: var(--accent-green); }
    .status-dot.warning { background: var(--accent-yellow); }
    .status-dot.offline { background: var(--accent-red); }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: var(--text-muted);
        font-size: 0.75rem;
        letter-spacing: 1px;
        border-top: 1px solid var(--border-color);
        margin-top: 3rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""


def apply_plotly_theme(fig):
    """Apply dark theme to plotly figure."""
    fig.update_layout(
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        font=dict(color='#a0a0a0', family='Inter'),
        xaxis=dict(gridcolor='#222222', linecolor='#222222', zerolinecolor='#222222'),
        yaxis=dict(gridcolor='#222222', linecolor='#222222', zerolinecolor='#222222'),
        margin=dict(t=40, b=40, l=40, r=20),
        legend=dict(bgcolor='rgba(0,0,0,0)')
    )
    return fig


@st.cache_resource
def load_manufacturing_data():
    """Load or generate manufacturing data."""
    generator = SyntheticDataGenerator(seed=42)
    return generator.generate_all_data()


@st.cache_resource
def load_cmapss_data():
    """Load synthetic CMAPSS data for demo."""
    return generate_synthetic_cmapss(n_units=20, avg_cycles=200)


@st.cache_resource
def build_knowledge_graph(_data):
    """Build knowledge graph from data."""
    kg = ManufacturingKnowledgeGraph()
    kg.build_from_dataframes(_data)
    return kg


@st.cache_resource
def train_analytics_engine(_data):
    """Train predictive analytics engine."""
    engine = PredictiveAnalyticsEngine()
    engine.train_all_models(_data)
    return engine


@st.cache_resource
def extract_insights(_data):
    """Extract insights from data."""
    extractor = InsightExtractor()
    
    all_insights = []
    all_insights.extend(extractor.analyze_maintenance_patterns(_data["maintenance"]))
    all_insights.extend(extractor.analyze_supplier_performance(_data["deliveries"]))
    all_insights.extend(extractor.analyze_production_efficiency(_data["production"]))
    all_insights.extend(extractor.analyze_financial_trends(_data["financials"]))
    
    recommendations = extractor.generate_recommendations(all_insights)
    
    return all_insights, recommendations


def render_metric_card(label: str, value: str, delta: str = None, 
                       delta_type: str = "neutral"):
    """Render a metric card with label, value, and optional delta."""
    delta_html = ""
    if delta:
        delta_class = delta_type
        delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>'
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_alert(severity: str, title: str, text: str, icon_name: str = None):
    """Render an alert box."""
    icon = get_icon(icon_name or "alert_high", size=20, color="#ffffff")
    st.markdown(f"""
    <div class="alert alert-{severity}">
        <div>{icon}</div>
        <div class="alert-content">
            <div class="alert-title">{title}</div>
            <div class="alert-text">{text}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_header():
    """Render page header."""
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="main-header">
        {LOGO_SVG}
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar navigation."""
    with st.sidebar:
        st.markdown(f"""
        <div style="padding: 1rem 0; margin-bottom: 1rem;">
            {LOGO_SVG}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="card-title">NAVIGATION</div>', unsafe_allow_html=True)
        
        pages = {
            "Dashboard": "dashboard",
            "Maintenance": "maintenance",
            "Suppliers": "supplier",
            "Production": "production",
            "Financials": "financial",
            "Turbofan Analysis": "turbine",
            "Data Upload": "upload",
            "Consultant": "chat"
        }
        
        selected = st.radio(
            "Select View",
            list(pages.keys()),
            label_visibility="collapsed"
        )
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card-title">SYSTEM STATUS</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size: 0.875rem; color: #a0a0a0;">
            <div style="margin: 0.5rem 0;"><span class="status-dot online"></span>Analytics Engine</div>
            <div style="margin: 0.5rem 0;"><span class="status-dot online"></span>Knowledge Graph</div>
            <div style="margin: 0.5rem 0;"><span class="status-dot online"></span>Data Pipeline</div>
        </div>
        """, unsafe_allow_html=True)
        
        return selected


def render_dashboard(data, insights, recommendations):
    """Render main dashboard."""
    financials = data["financials"]
    recent = financials.tail(3)
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_margin = recent["profit_margin"].mean()
        delta = recent["profit_margin"].iloc[-1] - financials["profit_margin"].iloc[0]
        delta_type = "positive" if delta > 0 else "negative"
        render_metric_card("PROFIT MARGIN", f"{avg_margin:.1f}%", 
                          f"{delta:+.1f}%", delta_type)
    
    with col2:
        total_downtime = data["maintenance"]["downtime_impact_hours"].sum()
        render_metric_card("TOTAL DOWNTIME", f"{total_downtime:.0f} HRS", "YTD", "neutral")
    
    with col3:
        on_time_rate = data["deliveries"]["on_time"].mean() * 100
        delta_type = "positive" if on_time_rate > 80 else "negative"
        render_metric_card("ON-TIME DELIVERY", f"{on_time_rate:.1f}%", 
                          "TARGET: 80%", delta_type)
    
    with col4:
        high_priority = len([i for i in insights if i.severity in ["high", "critical"]])
        delta_type = "negative" if high_priority > 0 else "positive"
        render_metric_card("PRIORITY ISSUES", str(high_priority), 
                          "REQUIRES ATTENTION" if high_priority > 0 else "ALL CLEAR", 
                          delta_type)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Insights and Recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="card-title">
            {get_icon("alert_high", 16)} KEY FINDINGS
        </div>
        """, unsafe_allow_html=True)
        
        high_severity = [i for i in insights if i.severity in ["high", "critical"]][:5]
        for insight in high_severity:
            render_alert(insight.severity, insight.category.replace("_", " ").upper(),
                        insight.description, "alert_high" if insight.severity == "high" else "alert_medium")
    
    with col2:
        st.markdown(f"""
        <div class="card-title">
            {get_icon("target", 16)} RECOMMENDATIONS
        </div>
        """, unsafe_allow_html=True)
        
        for rec in recommendations[:5]:
            savings = rec.get("estimated_savings", 0)
            savings_text = f" | Est. savings: ${savings:,.0f}" if savings else ""
            render_alert("info", f"PRIORITY {rec['priority']}", 
                        f"{rec['recommendation']}{savings_text}", "check")


def render_maintenance_analysis(data, kg):
    """Render maintenance analysis."""
    maintenance = data["maintenance"]
    
    st.markdown(f'<div class="card-title">{get_icon("maintenance", 16)} MAINTENANCE ANALYSIS</div>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        machine_costs = maintenance.groupby("machine_id")["cost"].sum().reset_index()
        fig = px.bar(
            machine_costs.nlargest(10, "cost"),
            x="machine_id",
            y="cost",
            title="MAINTENANCE COST BY MACHINE"
        )
        fig = apply_plotly_theme(fig)
        fig.update_traces(marker_color='#0066ff')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        event_counts = maintenance["event_type"].value_counts()
        fig = px.pie(
            values=event_counts.values,
            names=event_counts.index,
            title="EVENT DISTRIBUTION",
            color_discrete_sequence=['#0066ff', '#00ff88', '#ff3333']
        )
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    # Downtime trend
    maintenance["month"] = pd.to_datetime(maintenance["start_time"]).dt.to_period("M")
    monthly_downtime = maintenance.groupby("month")["downtime_impact_hours"].sum().reset_index()
    monthly_downtime["month"] = monthly_downtime["month"].astype(str)
    
    fig = px.line(
        monthly_downtime,
        x="month",
        y="downtime_impact_hours",
        title="MONTHLY DOWNTIME TREND"
    )
    fig = apply_plotly_theme(fig)
    fig.update_traces(line_color='#ff3333')
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk assessment
    st.markdown(f'<div class="card-title">{get_icon("alert_high", 16)} RISK ASSESSMENT</div>', 
                unsafe_allow_html=True)
    
    risks = kg.find_risk_patterns()
    machine_risks = [r for r in risks if r["type"] == "high_maintenance_machine"][:5]
    
    for risk in machine_risks:
        render_alert(risk["severity"], risk["node_id"], risk["details"], "machine")


def render_turbofan_analysis(cmapss_data=None):
    """Render NASA CMAPSS turbofan analysis."""
    st.markdown(f'<div class="card-title">{get_icon("turbine", 16)} TURBOFAN ENGINE ANALYSIS</div>', 
                unsafe_allow_html=True)
    
    if cmapss_data is None:
        cmapss_data = load_cmapss_data()
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    n_units = cmapss_data["unit_id"].nunique()
    avg_cycles = cmapss_data.groupby("unit_id")["cycle"].max().mean()
    min_rul = cmapss_data.groupby("unit_id")["rul"].min().min()
    critical_units = (cmapss_data.groupby("unit_id")["rul"].min() < 50).sum()
    
    with col1:
        render_metric_card("ENGINE UNITS", str(n_units), "IN FLEET", "neutral")
    with col2:
        render_metric_card("AVG CYCLES", f"{avg_cycles:.0f}", "PER UNIT", "neutral")
    with col3:
        render_metric_card("MIN RUL", f"{min_rul:.0f}", "CYCLES", 
                          "negative" if min_rul < 30 else "neutral")
    with col4:
        render_metric_card("CRITICAL UNITS", str(critical_units), 
                          "RUL < 50", "negative" if critical_units > 0 else "positive")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Unit selector
    selected_unit = st.selectbox("SELECT ENGINE UNIT", 
                                 sorted(cmapss_data["unit_id"].unique()))
    
    unit_data = cmapss_data[cmapss_data["unit_id"] == selected_unit]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RUL over time
        fig = px.line(unit_data, x="cycle", y="rul", 
                     title=f"REMAINING USEFUL LIFE - UNIT {selected_unit}")
        fig = apply_plotly_theme(fig)
        fig.update_traces(line_color='#00ff88')
        fig.add_hline(y=30, line_dash="dash", line_color="#ff3333",
                     annotation_text="CRITICAL THRESHOLD")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Key sensor trends
        sensor_cols = ['sensor_2', 'sensor_7', 'sensor_11', 'sensor_15']
        fig = go.Figure()
        colors = ['#0066ff', '#00ff88', '#ffaa00', '#ff00ff']
        
        for sensor, color in zip(sensor_cols, colors):
            if sensor in unit_data.columns:
                # Normalize for comparison
                normalized = (unit_data[sensor] - unit_data[sensor].min()) / \
                            (unit_data[sensor].max() - unit_data[sensor].min() + 0.001)
                fig.add_trace(go.Scatter(
                    x=unit_data["cycle"],
                    y=normalized,
                    name=sensor.upper(),
                    line=dict(color=color)
                ))
        
        fig.update_layout(title=f"SENSOR TRENDS - UNIT {selected_unit}")
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    # Fleet overview
    st.markdown(f'<div class="card-title">{get_icon("analytics", 16)} FLEET RUL DISTRIBUTION</div>', 
                unsafe_allow_html=True)
    
    unit_rul = cmapss_data.groupby("unit_id")["rul"].min().reset_index()
    unit_rul.columns = ["unit_id", "min_rul"]
    unit_rul["status"] = unit_rul["min_rul"].apply(
        lambda x: "CRITICAL" if x < 30 else ("WARNING" if x < 70 else "NOMINAL")
    )
    
    fig = px.bar(unit_rul.sort_values("min_rul"), x="unit_id", y="min_rul",
                 color="status", title="MINIMUM RUL BY UNIT",
                 color_discrete_map={
                     "CRITICAL": "#ff3333",
                     "WARNING": "#ffaa00", 
                     "NOMINAL": "#00ff88"
                 })
    fig = apply_plotly_theme(fig)
    fig.add_hline(y=30, line_dash="dash", line_color="#ff3333")
    fig.add_hline(y=70, line_dash="dash", line_color="#ffaa00")
    st.plotly_chart(fig, use_container_width=True)


def render_data_upload():
    """Render data upload interface."""
    st.markdown(f'<div class="card-title">{get_icon("upload", 16)} DATA UPLOAD</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <p style="color: #a0a0a0;">
            Upload your manufacturing or sensor data for analysis. 
            Supported formats include NASA CMAPSS datasets and custom CSV files.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    data_type = st.radio(
        "DATA TYPE",
        ["NASA CMAPSS Dataset", "Manufacturing Data (CSV)", "Custom Sensor Data"],
        horizontal=True
    )
    
    if data_type == "NASA CMAPSS Dataset":
        st.markdown("""
        <div class="alert alert-info">
            <div class="alert-content">
                <div class="alert-title">CMAPSS FORMAT</div>
                <div class="alert-text">
                    Space-separated values with columns: unit_id, cycle, op_setting_1-3, sensor_1-21
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            train_file = st.file_uploader("TRAINING DATA", type=["txt", "csv"])
        with col2:
            test_file = st.file_uploader("TEST DATA (OPTIONAL)", type=["txt", "csv"])
        
        rul_file = st.file_uploader("RUL LABELS (OPTIONAL)", type=["txt", "csv"])
        
        if train_file is not None:
            if st.button("PROCESS DATA"):
                with st.spinner("Processing..."):
                    loader = CMAPSSDataLoader()
                    try:
                        data = loader.load_from_file(
                            train_file, 
                            test_file if test_file else None,
                            rul_file if rul_file else None
                        )
                        st.session_state["cmapss_data"] = data["train"]
                        
                        render_alert("low", "SUCCESS", 
                                    f"Loaded {len(data['train'])} samples from "
                                    f"{data['train']['unit_id'].nunique()} units", "check")
                        
                        # Show preview
                        st.markdown('<div class="card-title">DATA PREVIEW</div>', 
                                   unsafe_allow_html=True)
                        st.dataframe(data["train"].head(10))
                        
                    except Exception as e:
                        render_alert("high", "ERROR", str(e), "alert_high")
    
    elif data_type == "Manufacturing Data (CSV)":
        uploaded_files = st.file_uploader(
            "UPLOAD CSV FILES",
            type=["csv"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for file in uploaded_files:
                df = pd.read_csv(file)
                st.markdown(f'<div class="card-title">{file.name}</div>', 
                           unsafe_allow_html=True)
                st.dataframe(df.head())
    
    else:
        st.markdown("""
        <div class="card">
            <p style="color: #a0a0a0;">
                Upload sensor data with the following columns:<br>
                - unit_id or machine_id: Equipment identifier<br>
                - cycle or timestamp: Time indicator<br>
                - sensor_1, sensor_2, ...: Sensor readings
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        sensor_file = st.file_uploader("SENSOR DATA", type=["csv"])
        
        if sensor_file is not None:
            df = pd.read_csv(sensor_file)
            st.markdown('<div class="card-title">DATA PREVIEW</div>', unsafe_allow_html=True)
            st.dataframe(df.head())
            
            # Try to process
            loader = CMAPSSDataLoader()
            try:
                data = loader.load_from_dataframe(df)
                st.session_state["custom_sensor_data"] = data["train"]
                render_alert("low", "SUCCESS", 
                            f"Processed {len(data['train'])} samples", "check")
            except Exception as e:
                render_alert("medium", "WARNING", 
                            f"Could not auto-process: {str(e)}", "alert_medium")


def render_consultant(data, insights, nlg):
    """Render chatbot interface."""
    st.markdown(f'<div class="card-title">{get_icon("chat", 16)} MANUFACTURING CONSULTANT</div>', 
                unsafe_allow_html=True)
    
    # Initialize chat
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Systems online. I can provide analysis on maintenance, suppliers, production efficiency, and predictive insights. What would you like to know?"
        }]
    
    # Display chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input
    if prompt := st.chat_input("Enter query..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        response = nlg.generate_chatbot_response(prompt, insights, data)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
    
    # Quick actions
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">QUICK QUERIES</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("SYSTEM SUMMARY"):
            st.session_state.messages.append({"role": "user", "content": "Give me a summary"})
            response = nlg.generate_chatbot_response("summary", insights, data)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    with col2:
        if st.button("MAINTENANCE STATUS"):
            st.session_state.messages.append({"role": "user", "content": "Maintenance issues"})
            response = nlg.generate_chatbot_response("maintenance", insights, data)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    with col3:
        if st.button("RECOMMENDATIONS"):
            st.session_state.messages.append({"role": "user", "content": "Recommendations"})
            response = nlg.generate_chatbot_response("recommend", insights, data)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()


def render_supplier_analysis(data):
    """Render supplier analysis."""
    deliveries = data["deliveries"]
    
    st.markdown(f'<div class="card-title">{get_icon("supplier", 16)} SUPPLIER PERFORMANCE</div>', 
                unsafe_allow_html=True)
    
    supplier_stats = deliveries.groupby("supplier_name").agg({
        "delivery_id": "count",
        "on_time": "mean",
        "days_late": "mean",
        "quality_score": "mean",
        "total_cost": "sum"
    }).reset_index()
    supplier_stats.columns = ["Supplier", "Deliveries", "On-Time Rate", 
                              "Avg Days Late", "Avg Quality", "Total Spend"]
    supplier_stats["On-Time Rate"] = supplier_stats["On-Time Rate"] * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            supplier_stats.sort_values("On-Time Rate"),
            x="Supplier",
            y="On-Time Rate",
            title="ON-TIME DELIVERY RATE",
            color="On-Time Rate",
            color_continuous_scale=[[0, '#ff3333'], [0.5, '#ffaa00'], [1, '#00ff88']]
        )
        fig = apply_plotly_theme(fig)
        fig.add_hline(y=80, line_dash="dash", line_color="#ffffff", 
                     annotation_text="TARGET")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            supplier_stats,
            x="On-Time Rate",
            y="Avg Quality",
            size="Total Spend",
            hover_name="Supplier",
            title="QUALITY VS RELIABILITY"
        )
        fig = apply_plotly_theme(fig)
        fig.add_hline(y=4.0, line_dash="dash", line_color="#00ff88")
        fig.add_vline(x=80, line_dash="dash", line_color="#00ff88")
        st.plotly_chart(fig, use_container_width=True)
    
    # Table
    st.markdown('<div class="card-title">SUPPLIER SCORECARD</div>', unsafe_allow_html=True)
    formatted = supplier_stats.copy()
    formatted["Total Spend"] = formatted["Total Spend"].apply(lambda x: f"${x:,.0f}")
    formatted["On-Time Rate"] = formatted["On-Time Rate"].apply(lambda x: f"{x:.1f}%")
    formatted["Avg Quality"] = formatted["Avg Quality"].apply(lambda x: f"{x:.1f}/5")
    st.dataframe(formatted, use_container_width=True)


def render_production_analysis(data, engine):
    """Render production analysis."""
    production = data["production"]
    
    st.markdown(f'<div class="card-title">{get_icon("production", 16)} PRODUCTION EFFICIENCY</div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    avg_eff = production["efficiency_pct"].mean()
    avg_defect = production["defect_rate"].mean()
    total_units = production["units_produced"].sum()
    
    with col1:
        render_metric_card("AVG EFFICIENCY", f"{avg_eff:.1f}%", 
                          "ABOVE TARGET" if avg_eff > 85 else "BELOW TARGET",
                          "positive" if avg_eff > 85 else "negative")
    with col2:
        render_metric_card("DEFECT RATE", f"{avg_defect:.2f}%",
                          "ACCEPTABLE" if avg_defect < 2 else "HIGH",
                          "positive" if avg_defect < 2 else "negative")
    with col3:
        render_metric_card("TOTAL OUTPUT", f"{total_units:,}", "UNITS", "neutral")
    
    col1, col2 = st.columns(2)
    
    with col1:
        machine_eff = production.groupby("machine_id")["efficiency_pct"].mean().reset_index()
        fig = px.bar(
            machine_eff.sort_values("efficiency_pct"),
            x="machine_id",
            y="efficiency_pct",
            title="EFFICIENCY BY MACHINE",
            color="efficiency_pct",
            color_continuous_scale=[[0, '#ff3333'], [0.5, '#ffaa00'], [1, '#00ff88']]
        )
        fig = apply_plotly_theme(fig)
        fig.add_hline(y=avg_eff, line_dash="dash", line_color="#ffffff")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        operator_eff = production.groupby("operator_id")["efficiency_pct"].mean().reset_index()
        fig = px.bar(
            operator_eff.nlargest(10, "efficiency_pct"),
            x="operator_id",
            y="efficiency_pct",
            title="TOP OPERATORS"
        )
        fig = apply_plotly_theme(fig)
        fig.update_traces(marker_color='#0066ff')
        st.plotly_chart(fig, use_container_width=True)


def render_financial_analysis(data, engine):
    """Render financial analysis."""
    financials = data["financials"]
    
    st.markdown(f'<div class="card-title">{get_icon("financial", 16)} FINANCIAL ANALYSIS</div>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(x=financials["month"], y=financials["revenue"], 
                  name="Revenue", marker_color='#0066ff'),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=financials["month"], y=financials["profit_margin"], 
                      name="Margin %", line=dict(color='#00ff88')),
            secondary_y=True
        )
        
        fig.update_layout(title="REVENUE & MARGIN")
        fig = apply_plotly_theme(fig)
        fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
        fig.update_yaxes(title_text="Margin (%)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        cost_cols = ["material_cost", "labor_cost", "maintenance_cost", 
                    "scrap_cost", "overhead"]
        cost_totals = financials[cost_cols].sum()
        
        fig = px.pie(
            values=cost_totals.values,
            names=[c.replace("_", " ").upper() for c in cost_cols],
            title="COST BREAKDOWN",
            color_discrete_sequence=['#0066ff', '#00ff88', '#ff3333', 
                                    '#ffaa00', '#ff00ff']
        )
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    # Cost trends
    fig = go.Figure()
    colors = {'maintenance_cost': '#ff3333', 'scrap_cost': '#ffaa00', 
              'material_cost': '#0066ff'}
    for col, color in colors.items():
        fig.add_trace(go.Scatter(
            x=financials["month"],
            y=financials[col],
            name=col.replace("_", " ").upper(),
            line=dict(color=color)
        ))
    fig.update_layout(title="COST TRENDS")
    fig = apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application entry point."""
    # Load data and models
    with st.spinner("Initializing systems..."):
        data = load_manufacturing_data()
        kg = build_knowledge_graph(data)
        engine = train_analytics_engine(data)
        insights, recommendations = extract_insights(data)
        nlg = NaturalLanguageGenerator()
    
    # Render header
    render_header()
    
    # Sidebar navigation
    selected_page = render_sidebar()
    
    # Render selected page
    if selected_page == "Dashboard":
        render_dashboard(data, insights, recommendations)
    elif selected_page == "Maintenance":
        render_maintenance_analysis(data, kg)
    elif selected_page == "Suppliers":
        render_supplier_analysis(data)
    elif selected_page == "Production":
        render_production_analysis(data, engine)
    elif selected_page == "Financials":
        render_financial_analysis(data, engine)
    elif selected_page == "Turbofan Analysis":
        cmapss_data = st.session_state.get("cmapss_data", None)
        render_turbofan_analysis(cmapss_data)
    elif selected_page == "Data Upload":
        render_data_upload()
    elif selected_page == "Consultant":
        render_consultant(data, insights, nlg)
    
    # Footer
    st.markdown("""
    <div class="footer">
        LEANNLP MANUFACTURING ANALYTICS | MALIK MASHIGO & KAM WILLIAMS | 2025
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
