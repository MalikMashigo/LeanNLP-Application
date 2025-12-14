"""
LeanNLP Manufacturing Analytics Dashboard
SpaceX-inspired dark theme with SVG icons
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import json
import os
import re
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="LEANNLP",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONSTANTS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "demo_data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "trained_models")

CMAPSS_COLUMNS = [
    'unit_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20', 'sensor_21'
]

USEFUL_SENSORS = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_9',
                  'sensor_11', 'sensor_12', 'sensor_14', 'sensor_15', 
                  'sensor_17', 'sensor_20', 'sensor_21']

# ============================================================
# SVG ICONS
# ============================================================
ICONS = {
    "dashboard": '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="7" height="9"/><rect x="14" y="3" width="7" height="5"/><rect x="14" y="12" width="7" height="9"/><rect x="3" y="16" width="7" height="5"/></svg>''',
    
    "turbine": '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="3"/><path d="M12 2v4"/><path d="M12 18v4"/><path d="m4.93 4.93 2.83 2.83"/><path d="m16.24 16.24 2.83 2.83"/><path d="M2 12h4"/><path d="M18 12h4"/><path d="m4.93 19.07 2.83-2.83"/><path d="m16.24 7.76 2.83-2.83"/></svg>''',
    
    "maintenance": '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>''',
    
    "supplier": '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M16 16h6"/><path d="M16 20h6"/><path d="M22 12H2"/><path d="M22 12v8a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-8"/><path d="m22 12-3-9H5L2 12"/></svg>''',
    
    "upload": '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>''',
    
    "train": '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>''',
    
    "alert": '''<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>''',
    
    "check": '''<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>''',
    
    "x": '''<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>''',
    
    "analytics": '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>''',
    
    "file": '''<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>''',
    
    "rocket": '''<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/></svg>'''
}

LOGO_SVG = '''<svg width="160" height="28" viewBox="0 0 160 28" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M4 7L11 4L18 7V21L11 24L4 21V7Z" stroke="white" stroke-width="1.5" fill="none"/>
  <path d="M11 4V24" stroke="white" stroke-width="1" opacity="0.4"/>
  <path d="M4 7L18 7" stroke="white" stroke-width="1" opacity="0.4"/>
  <circle cx="11" cy="11" r="2" fill="white"/>
  <text x="28" y="18" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="white" letter-spacing="3">LEANNLP</text>
</svg>'''

# ============================================================
# DARK THEME CSS
# ============================================================
DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #000000;
    --bg-secondary: #0a0a0a;
    --bg-card: #111111;
    --border: #222222;
    --text-primary: #ffffff;
    --text-secondary: #888888;
    --text-muted: #555555;
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
    padding-bottom: 2rem;
    max-width: 1400px;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary);
}

h1, h2, h3, h4 {
    font-weight: 500 !important;
    letter-spacing: 0.5px;
    color: var(--text-primary) !important;
}

p, span, label {
    color: var(--text-secondary);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] .stRadio > label {
    color: var(--text-secondary) !important;
}

/* Cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

.metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 1.75rem;
    font-weight: 300;
    color: var(--text-primary);
    line-height: 1.2;
}

.metric-delta {
    font-size: 0.8rem;
    margin-top: 0.5rem;
    color: var(--text-muted);
}

.metric-delta.positive { color: var(--accent-green); }
.metric-delta.negative { color: var(--accent-red); }

/* Section title */
.section-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Status indicators */
.status-row {
    display: flex;
    align-items: center;
    padding: 0.75rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
}

.status-row:last-child {
    border-bottom: none;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 10px;
}

.status-dot.green { background: var(--accent-green); }
.status-dot.red { background: var(--accent-red); }
.status-dot.yellow { background: var(--accent-yellow); }

/* Alerts */
.alert-box {
    padding: 1rem;
    border-radius: 4px;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: flex-start;
    gap: 10px;
}

.alert-box.high {
    background: rgba(255, 51, 51, 0.1);
    border-left: 3px solid var(--accent-red);
}

.alert-box.medium {
    background: rgba(255, 170, 0, 0.1);
    border-left: 3px solid var(--accent-yellow);
}

.alert-box.low {
    background: rgba(0, 255, 136, 0.1);
    border-left: 3px solid var(--accent-green);
}

.alert-box.info {
    background: rgba(0, 102, 255, 0.1);
    border-left: 3px solid var(--accent-blue);
}

.alert-title {
    font-weight: 500;
    color: var(--text-primary);
    font-size: 0.85rem;
}

.alert-text {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-top: 2px;
}

/* Divider */
.divider {
    height: 1px;
    background: var(--border);
    margin: 1.5rem 0;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: #111 !important;
    border-color: #444 !important;
}

.stButton > button[kind="primary"] {
    background: var(--accent-blue) !important;
    border-color: var(--accent-blue) !important;
}

.stButton > button[kind="primary"]:hover {
    background: #0055dd !important;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--bg-card);
    border: 1px dashed var(--border);
    border-radius: 4px;
    padding: 1rem;
}

/* Dataframe */
.stDataFrame {
    background: var(--bg-card) !important;
}

/* Select box */
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
}

/* Progress bar */
.stProgress > div > div {
    background: var(--accent-blue) !important;
}

/* Footer */
.footer {
    text-align: center;
    padding: 2rem 0;
    color: var(--text-muted);
    font-size: 0.7rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
}
</style>
"""

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def icon(name):
    """Return SVG icon HTML."""
    return ICONS.get(name, "")


def metric_card(label, value, delta=None, delta_type="neutral"):
    """Render a metric card."""
    delta_class = delta_type if delta_type in ["positive", "negative"] else ""
    delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>' if delta else ""
    
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    ''', unsafe_allow_html=True)


def section_title(text, icon_name=None):
    """Render a section title with optional icon."""
    icon_html = f'<span style="color: #555;">{icon(icon_name)}</span>' if icon_name else ""
    st.markdown(f'<div class="section-title">{icon_html} {text}</div>', unsafe_allow_html=True)


def alert_box(level, title, text):
    """Render an alert box."""
    st.markdown(f'''
    <div class="alert-box {level}">
        <div>
            <div class="alert-title">{title}</div>
            <div class="alert-text">{text}</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)


def status_item(label, is_ok, detail=""):
    """Render a status item."""
    color = "green" if is_ok else "red"
    icon_svg = ICONS["check"] if is_ok else ICONS["x"]
    icon_color = "#00ff88" if is_ok else "#ff3333"
    
    st.markdown(f'''
    <div class="status-row">
        <span class="status-dot {color}"></span>
        <span style="color: #fff; flex: 1;">{label}</span>
        <span style="color: #555; font-size: 0.8rem;">{detail}</span>
    </div>
    ''', unsafe_allow_html=True)


def divider():
    """Render a divider."""
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


def apply_plotly_theme(fig):
    """Apply dark theme to Plotly figure."""
    fig.update_layout(
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        font=dict(color='#888888', family='Inter'),
        title_font=dict(color='#ffffff', size=14),
        xaxis=dict(
            gridcolor='#222222',
            linecolor='#222222',
            zerolinecolor='#222222',
            tickfont=dict(color='#666666')
        ),
        yaxis=dict(
            gridcolor='#222222',
            linecolor='#222222',
            zerolinecolor='#222222',
            tickfont=dict(color='#666666')
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='#888888')
        ),
        margin=dict(t=50, b=50, l=50, r=30)
    )
    return fig

# ============================================================
# DATA LOADING
# ============================================================

def load_cmapss_data():
    """Load CMAPSS turbofan data."""
    train_path = os.path.join(DATA_DIR, "train_FD001.txt")
    test_path = os.path.join(DATA_DIR, "test_FD001.txt")
    rul_path = os.path.join(DATA_DIR, "RUL_FD001.txt")
    
    if not os.path.exists(train_path):
        return None, None, None
    
    try:
        train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=CMAPSS_COLUMNS, engine='python')
        
        max_cycles = train_df.groupby('unit_id')['cycle'].max()
        train_df = train_df.merge(max_cycles.rename('max_cycle').reset_index(), on='unit_id')
        train_df['rul'] = train_df['max_cycle'] - train_df['cycle']
        train_df.drop('max_cycle', axis=1, inplace=True)
        
        test_df = None
        rul_df = None
        
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=CMAPSS_COLUMNS, engine='python')
        
        if os.path.exists(rul_path):
            rul_df = pd.read_csv(rul_path, header=None, names=['rul'])
        
        return train_df, test_df, rul_df
    except Exception as e:
        return None, None, None


def load_maintenance_data():
    """Load maintenance logs."""
    path = os.path.join(DATA_DIR, "maintenance_logs.csv")
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except:
            pass
    return None


def load_supplier_data():
    """Load supplier and delivery data."""
    suppliers_path = os.path.join(DATA_DIR, "suppliers.csv")
    deliveries_path = os.path.join(DATA_DIR, "deliveries.csv")
    
    suppliers_df = None
    deliveries_df = None
    
    try:
        if os.path.exists(suppliers_path):
            suppliers_df = pd.read_csv(suppliers_path)
        if os.path.exists(deliveries_path):
            deliveries_df = pd.read_csv(deliveries_path)
    except:
        pass
    
    return suppliers_df, deliveries_df


def load_models():
    """Load trained models."""
    models = {}
    
    rul_path = os.path.join(MODELS_DIR, "rul_model.pkl")
    if os.path.exists(rul_path):
        try:
            with open(rul_path, 'rb') as f:
                models['rul'] = pickle.load(f)
        except:
            pass
    
    cost_path = os.path.join(MODELS_DIR, "cost_model.pkl")
    if os.path.exists(cost_path):
        try:
            with open(cost_path, 'rb') as f:
                models['cost'] = pickle.load(f)
        except:
            pass
    
    results_path = os.path.join(MODELS_DIR, "training_results.json")
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                models['results'] = json.load(f)
        except:
            pass
    
    return models

# ============================================================
# NLP FUNCTIONS
# ============================================================

def extract_failure_type(text):
    """Extract failure type from text."""
    text_lower = text.lower()
    
    if 'motor' in text_lower:
        return 'MOTOR'
    elif 'bearing' in text_lower:
        return 'BEARING'
    elif 'hydraulic' in text_lower:
        return 'HYDRAULIC'
    elif 'electric' in text_lower:
        return 'ELECTRICAL'
    elif 'software' in text_lower or 'plc' in text_lower:
        return 'SOFTWARE'
    elif 'calibrat' in text_lower:
        return 'CALIBRATION'
    else:
        return 'OTHER'

# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def engineer_features(train_df):
    """Create features for RUL prediction."""
    features_list = []
    
    for unit_id in train_df['unit_id'].unique():
        unit_data = train_df[train_df['unit_id'] == unit_id].sort_values('cycle')
        
        for idx, row in unit_data.iterrows():
            features = {'unit_id': unit_id, 'cycle': row['cycle'], 'rul': row['rul']}
            
            for sensor in USEFUL_SENSORS:
                features[sensor] = row[sensor]
            
            cycle = row['cycle']
            history = unit_data[unit_data['cycle'] <= cycle]
            
            for sensor in USEFUL_SENSORS[:6]:
                if len(history) >= 5:
                    features[f'{sensor}_rolling_mean'] = history[sensor].tail(5).mean()
                    features[f'{sensor}_rolling_std'] = history[sensor].tail(5).std()
                else:
                    features[f'{sensor}_rolling_mean'] = row[sensor]
                    features[f'{sensor}_rolling_std'] = 0
                features[f'{sensor}_delta'] = row[sensor] - history[sensor].iloc[0]
            
            features_list.append(features)
    
    return pd.DataFrame(features_list)


def train_rul_model(train_df, progress_bar=None):
    """Train RUL model."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    if progress_bar:
        progress_bar.progress(10, "Engineering features...")
    
    features_df = engineer_features(train_df)
    features_df['rul'] = features_df['rul'].clip(upper=125)
    
    if progress_bar:
        progress_bar.progress(50, "Training model...")
    
    feature_cols = [c for c in features_df.columns if c not in ['unit_id', 'cycle', 'rul']]
    X = features_df[feature_cols].values
    y = features_df['rul'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    if progress_bar:
        progress_bar.progress(70, "Fitting Random Forest...")
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    if progress_bar:
        progress_bar.progress(90, "Evaluating...")
    
    y_pred = model.predict(X_val)
    metrics = {
        'mae': float(mean_absolute_error(y_val, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_val, y_pred))),
        'r2': float(r2_score(y_val, y_pred))
    }
    
    importance = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)
    top_features = importance[:10]
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, 'rul_model.pkl'), 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'feature_names': feature_cols, 'metrics': metrics, 'top_features': top_features}, f)
    
    save_results('rul_predictor', metrics)
    
    if progress_bar:
        progress_bar.progress(100, "Complete")
    
    return metrics, top_features


def train_cost_model(maintenance_df, progress_bar=None):
    """Train cost model."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    if progress_bar:
        progress_bar.progress(20, "Preparing features...")
    
    df = maintenance_df.copy()
    df['is_planned'] = (df['event_type'] == 'planned').astype(int)
    df['is_emergency'] = (df['event_type'] == 'emergency').astype(int)
    df['machine_num'] = df['machine_id'].str.extract(r'(\d+)').astype(int)
    df['has_motor'] = df['description'].str.lower().str.contains('motor').astype(int)
    df['has_bearing'] = df['description'].str.lower().str.contains('bearing').astype(int)
    df['has_hydraulic'] = df['description'].str.lower().str.contains('hydraulic').astype(int)
    df['has_electrical'] = df['description'].str.lower().str.contains('electric').astype(int)
    df['desc_length'] = df['description'].str.len()
    
    feature_cols = ['duration_hours', 'is_planned', 'is_emergency', 'machine_num',
                    'has_motor', 'has_bearing', 'has_hydraulic', 'has_electrical', 'desc_length']
    
    X = df[feature_cols].values
    y = df['cost'].values
    
    if progress_bar:
        progress_bar.progress(50, "Training model...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    if progress_bar:
        progress_bar.progress(90, "Evaluating...")
    
    y_pred = model.predict(X_val)
    metrics = {
        'mae': float(mean_absolute_error(y_val, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_val, y_pred))),
        'r2': float(r2_score(y_val, y_pred))
    }
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, 'cost_model.pkl'), 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'feature_names': feature_cols, 'metrics': metrics}, f)
    
    save_results('cost_predictor', metrics)
    
    if progress_bar:
        progress_bar.progress(100, "Complete")
    
    return metrics


def save_results(model_name, metrics):
    """Save training results."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    results_path = os.path.join(MODELS_DIR, 'training_results.json')
    
    results = {}
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
        except:
            pass
    
    if 'models' not in results:
        results['models'] = {}
    
    results['models'][model_name] = metrics
    results['last_updated'] = datetime.now().isoformat()
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

# ============================================================
# PAGES
# ============================================================

def page_dashboard():
    """Dashboard page."""
    section_title("OVERVIEW", "dashboard")
    
    train_df, test_df, rul_df = load_cmapss_data()
    maintenance_df = load_maintenance_data()
    suppliers_df, deliveries_df = load_supplier_data()
    models = load_models()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if train_df is not None:
            metric_card("ENGINE UNITS", str(train_df['unit_id'].nunique()), "in dataset")
        else:
            metric_card("ENGINE UNITS", "-", "no data")
    
    with col2:
        if 'rul' in models:
            r2 = models['rul']['metrics']['r2']
            metric_card("RUL MODEL R2", f"{r2:.3f}", "trained", "positive")
        else:
            metric_card("RUL MODEL", "-", "not trained", "negative")
    
    with col3:
        if maintenance_df is not None:
            emergency = len(maintenance_df[maintenance_df['event_type'] == 'emergency'])
            metric_card("EMERGENCY EVENTS", str(emergency), "in logs")
        else:
            metric_card("MAINTENANCE", "-", "no data")
    
    with col4:
        if deliveries_df is not None:
            on_time = deliveries_df['on_time'].mean() * 100
            delta_type = "positive" if on_time > 80 else "negative"
            metric_card("ON-TIME DELIVERY", f"{on_time:.1f}%", "average", delta_type)
        else:
            metric_card("DELIVERIES", "-", "no data")
    
    divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        section_title("DATA STATUS", "file")
        
        if train_df is not None:
            status_item("CMAPSS Training Data", True, f"{len(train_df):,} records")
        else:
            status_item("CMAPSS Training Data", False, "not found")
        
        if maintenance_df is not None:
            status_item("Maintenance Logs", True, f"{len(maintenance_df)} records")
        else:
            status_item("Maintenance Logs", False, "not found")
        
        if deliveries_df is not None:
            status_item("Supplier Deliveries", True, f"{len(deliveries_df)} records")
        else:
            status_item("Supplier Deliveries", False, "not found")
    
    with col2:
        section_title("MODEL STATUS", "analytics")
        
        if 'rul' in models:
            m = models['rul']['metrics']
            status_item("RUL Predictor", True, f"MAE: {m['mae']:.2f}")
        else:
            status_item("RUL Predictor", False, "not trained")
        
        if 'cost' in models:
            m = models['cost']['metrics']
            status_item("Cost Predictor", True, f"MAE: ${m['mae']:.2f}")
        else:
            status_item("Cost Predictor", False, "not trained")


def page_turbofan():
    """Turbofan analysis page."""
    section_title("TURBOFAN ENGINE ANALYSIS", "turbine")
    
    train_df, test_df, rul_df = load_cmapss_data()
    
    if train_df is None:
        alert_box("high", "NO DATA", "Upload CMAPSS data first")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        metric_card("TRAINING ENGINES", str(train_df['unit_id'].nunique()))
    with col2:
        avg_life = train_df.groupby('unit_id')['cycle'].max().mean()
        metric_card("AVG LIFETIME", f"{avg_life:.0f} cycles")
    with col3:
        if test_df is not None:
            metric_card("TEST ENGINES", str(test_df['unit_id'].nunique()))
        else:
            metric_card("TEST ENGINES", "-")
    
    divider()
    
    selected_unit = st.selectbox("SELECT ENGINE UNIT", sorted(train_df['unit_id'].unique()))
    unit_data = train_df[train_df['unit_id'] == selected_unit]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(unit_data, x='cycle', y='rul')
        fig.add_hline(y=30, line_dash="dash", line_color="#ff3333", annotation_text="CRITICAL")
        fig.update_traces(line_color='#00ff88')
        fig.update_layout(title=f'REMAINING USEFUL LIFE - UNIT {selected_unit}')
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        colors = ['#0066ff', '#00ff88', '#ffaa00', '#ff00ff']
        for sensor, color in zip(['sensor_2', 'sensor_7', 'sensor_11', 'sensor_15'], colors):
            norm = (unit_data[sensor] - unit_data[sensor].min()) / (unit_data[sensor].max() - unit_data[sensor].min() + 0.001)
            fig.add_trace(go.Scatter(x=unit_data['cycle'], y=norm, name=sensor.upper(), line=dict(color=color)))
        fig.update_layout(title=f'SENSOR DEGRADATION - UNIT {selected_unit}')
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)


def page_maintenance():
    """Maintenance page."""
    section_title("MAINTENANCE ANALYSIS", "maintenance")
    
    maintenance_df = load_maintenance_data()
    
    if maintenance_df is None:
        alert_box("high", "NO DATA", "Upload maintenance data first")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        metric_card("TOTAL RECORDS", str(len(maintenance_df)))
    with col2:
        total_cost = maintenance_df['cost'].sum()
        metric_card("TOTAL COST", f"${total_cost:,.0f}")
    with col3:
        avg_duration = maintenance_df['duration_hours'].mean()
        metric_card("AVG DURATION", f"{avg_duration:.1f}h")
    
    divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        cost_by_type = maintenance_df.groupby('event_type')['cost'].sum().reset_index()
        fig = px.bar(cost_by_type, x='event_type', y='cost')
        fig.update_traces(marker_color='#0066ff')
        fig.update_layout(title='COST BY EVENT TYPE')
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        maintenance_df['failure_type'] = maintenance_df['description'].apply(extract_failure_type)
        failure_counts = maintenance_df['failure_type'].value_counts().reset_index()
        failure_counts.columns = ['type', 'count']
        
        fig = px.bar(failure_counts, x='count', y='type', orientation='h')
        fig.update_traces(marker_color='#ff3333')
        fig.update_layout(title='FAILURE TYPES (NLP EXTRACTED)')
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    divider()
    section_title("SAMPLE NLP EXTRACTIONS")
    
    samples = maintenance_df.sample(min(3, len(maintenance_df)), random_state=42)
    for _, row in samples.iterrows():
        with st.expander(f"{row['event_id']} - {row['failure_type']}"):
            st.markdown(f"**Description:** {row['description']}")
            st.markdown(f"**Extracted Type:** `{row['failure_type']}`")
            st.markdown(f"**Cost:** ${row['cost']:,.2f}")


def page_suppliers():
    """Suppliers page."""
    section_title("SUPPLIER PERFORMANCE", "supplier")
    
    suppliers_df, deliveries_df = load_supplier_data()
    
    if deliveries_df is None:
        alert_box("high", "NO DATA", "Upload supplier data first")
        return
    
    supplier_stats = deliveries_df.groupby('supplier_name').agg({
        'delivery_id': 'count',
        'on_time': 'mean',
        'days_late': 'mean',
        'quality_score': 'mean',
        'total_cost': 'sum'
    }).reset_index()
    supplier_stats['on_time_pct'] = supplier_stats['on_time'] * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        metric_card("TOTAL SUPPLIERS", str(len(supplier_stats)))
    with col2:
        avg = supplier_stats['on_time_pct'].mean()
        metric_card("AVG ON-TIME", f"{avg:.1f}%")
    with col3:
        at_risk = len(supplier_stats[supplier_stats['on_time_pct'] < 75])
        metric_card("AT-RISK", str(at_risk), "below 75%", "negative" if at_risk > 0 else "positive")
    
    divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(supplier_stats.sort_values('on_time_pct'), x='supplier_name', y='on_time_pct',
                    color='on_time_pct', color_continuous_scale=['#ff3333', '#ffaa00', '#00ff88'])
        fig.add_hline(y=75, line_dash="dash", line_color="#ffffff", annotation_text="TARGET")
        fig.update_layout(title='ON-TIME DELIVERY RATE', xaxis_tickangle=-45)
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(supplier_stats, x='on_time_pct', y='quality_score',
                        size='total_cost', hover_name='supplier_name', color_discrete_sequence=['#0066ff'])
        fig.add_vline(x=75, line_dash="dash", line_color="#ff3333")
        fig.add_hline(y=4.0, line_dash="dash", line_color="#00ff88")
        fig.update_layout(title='QUALITY VS RELIABILITY')
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    at_risk_df = supplier_stats[supplier_stats['on_time_pct'] < 75]
    if len(at_risk_df) > 0:
        divider()
        section_title("AT-RISK SUPPLIERS", "alert")
        for _, row in at_risk_df.iterrows():
            alert_box("high", row['supplier_name'], f"On-time: {row['on_time_pct']:.1f}% | Avg late: {row['days_late']:.1f} days")


def page_upload():
    """Upload page."""
    section_title("UPLOAD DATA", "upload")
    
    st.markdown('<p style="color: #666; margin-bottom: 1.5rem;">Upload data files. File types are auto-detected from filenames.</p>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Drop files here", type=['txt', 'csv'], accept_multiple_files=True)
    
    if uploaded_files:
        section_title("DETECTED FILES")
        
        file_mapping = {}
        
        for f in uploaded_files:
            name = f.name.lower()
            
            if 'train' in name and name.endswith('.txt'):
                file_mapping['train_FD001.txt'] = f
                alert_box("info", "CMAPSS Training Data", f.name)
            elif 'test' in name and name.endswith('.txt'):
                file_mapping['test_FD001.txt'] = f
                alert_box("info", "CMAPSS Test Data", f.name)
            elif 'rul' in name and name.endswith('.txt'):
                file_mapping['RUL_FD001.txt'] = f
                alert_box("info", "RUL Labels", f.name)
            elif 'maintenance' in name:
                file_mapping['maintenance_logs.csv'] = f
                alert_box("info", "Maintenance Logs", f.name)
            elif 'supplier' in name:
                file_mapping['suppliers.csv'] = f
                alert_box("info", "Suppliers", f.name)
            elif 'deliver' in name:
                file_mapping['deliveries.csv'] = f
                alert_box("info", "Deliveries", f.name)
            elif 'production' in name:
                file_mapping['production_runs.csv'] = f
                alert_box("info", "Production", f.name)
            else:
                file_mapping[f.name] = f
                alert_box("medium", "Unknown", f.name)
        
        if st.button("SAVE ALL FILES", type="primary"):
            os.makedirs(DATA_DIR, exist_ok=True)
            for filename, file_obj in file_mapping.items():
                with open(os.path.join(DATA_DIR, filename), 'wb') as out_f:
                    out_f.write(file_obj.getvalue())
            alert_box("low", "SUCCESS", f"Saved {len(file_mapping)} files")
    
    divider()
    section_title("CURRENT DATA FILES")
    
    files = [
        ("train_FD001.txt", "CMAPSS Training"),
        ("test_FD001.txt", "CMAPSS Test"),
        ("RUL_FD001.txt", "RUL Labels"),
        ("maintenance_logs.csv", "Maintenance"),
        ("suppliers.csv", "Suppliers"),
        ("deliveries.csv", "Deliveries"),
    ]
    
    for filename, label in files:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            size = os.path.getsize(path)
            size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/1024/1024:.1f} MB"
            status_item(label, True, size_str)
        else:
            status_item(label, False, "not found")


def page_train():
    """Training page."""
    section_title("TRAIN MODELS", "train")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### RUL PREDICTOR")
        st.markdown('<p style="color: #666;">Predicts Remaining Useful Life for turbofan engines.</p>', unsafe_allow_html=True)
        
        train_path = os.path.join(DATA_DIR, "train_FD001.txt")
        
        if os.path.exists(train_path):
            alert_box("low", "DATA READY", "train_FD001.txt found")
            
            if st.button("TRAIN RUL MODEL", type="primary", key="train_rul"):
                train_df, _, _ = load_cmapss_data()
                if train_df is not None:
                    progress = st.progress(0)
                    try:
                        metrics, top_features = train_rul_model(train_df, progress)
                        alert_box("low", "TRAINING COMPLETE", f"MAE: {metrics['mae']:.2f} cycles | R2: {metrics['r2']:.3f}")
                        
                        st.markdown("**TOP FEATURES:**")
                        for feat, imp in top_features[:5]:
                            st.markdown(f"- `{feat}`: {imp:.4f}")
                    except Exception as e:
                        alert_box("high", "TRAINING FAILED", str(e))
        else:
            alert_box("high", "NO DATA", "Upload train_FD001.txt first")
    
    with col2:
        st.markdown("### COST PREDICTOR")
        st.markdown('<p style="color: #666;">Predicts maintenance costs from event features.</p>', unsafe_allow_html=True)
        
        maint_path = os.path.join(DATA_DIR, "maintenance_logs.csv")
        
        if os.path.exists(maint_path):
            alert_box("low", "DATA READY", "maintenance_logs.csv found")
            
            if st.button("TRAIN COST MODEL", type="primary", key="train_cost"):
                maintenance_df = load_maintenance_data()
                if maintenance_df is not None:
                    progress = st.progress(0)
                    try:
                        metrics = train_cost_model(maintenance_df, progress)
                        alert_box("low", "TRAINING COMPLETE", f"MAE: ${metrics['mae']:.2f} | R2: {metrics['r2']:.3f}")
                    except Exception as e:
                        alert_box("high", "TRAINING FAILED", str(e))
        else:
            alert_box("high", "NO DATA", "Upload maintenance_logs.csv first")
    
    divider()
    
    st.markdown("### TRAIN ALL")
    
    if st.button("TRAIN ALL AVAILABLE MODELS"):
        results = []
        
        if os.path.exists(os.path.join(DATA_DIR, "train_FD001.txt")):
            train_df, _, _ = load_cmapss_data()
            if train_df is not None:
                try:
                    metrics, _ = train_rul_model(train_df)
                    results.append(f"RUL Model: MAE={metrics['mae']:.2f}, R2={metrics['r2']:.3f}")
                except Exception as e:
                    results.append(f"RUL Model: FAILED - {e}")
        
        if os.path.exists(os.path.join(DATA_DIR, "maintenance_logs.csv")):
            maintenance_df = load_maintenance_data()
            if maintenance_df is not None:
                try:
                    metrics = train_cost_model(maintenance_df)
                    results.append(f"Cost Model: MAE=${metrics['mae']:.2f}, R2={metrics['r2']:.3f}")
                except Exception as e:
                    results.append(f"Cost Model: FAILED - {e}")
        
        for r in results:
            if "FAILED" in r:
                alert_box("high", "ERROR", r)
            else:
                alert_box("low", "SUCCESS", r)

# ============================================================
# MAIN
# ============================================================

def main():
    st.markdown(DARK_CSS, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown(f'<div style="padding: 1rem 0 1.5rem 0;">{LOGO_SVG}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">NAVIGATION</div>', unsafe_allow_html=True)
        
        page = st.radio(
            "nav",
            ["Dashboard", "Turbofan Analysis", "Maintenance", "Suppliers", "Upload Data", "Train Models"],
            label_visibility="collapsed"
        )
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">MODEL STATUS</div>', unsafe_allow_html=True)
        
        models = load_models()
        
        rul_status = "green" if 'rul' in models else "red"
        cost_status = "green" if 'cost' in models else "red"
        
        st.markdown(f'''
        <div style="font-size: 0.85rem;">
            <div style="display: flex; align-items: center; margin: 0.5rem 0;">
                <span class="status-dot {rul_status}"></span>
                <span style="color: #888;">RUL Predictor</span>
            </div>
            <div style="display: flex; align-items: center; margin: 0.5rem 0;">
                <span class="status-dot {cost_status}"></span>
                <span style="color: #888;">Cost Predictor</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<p style="color: #444; font-size: 0.7rem; letter-spacing: 1px;">LEANNLP V2.0<br/>MASHIGO & WILLIAMS</p>', unsafe_allow_html=True)
    
    # Main content
    if page == "Dashboard":
        page_dashboard()
    elif page == "Turbofan Analysis":
        page_turbofan()
    elif page == "Maintenance":
        page_maintenance()
    elif page == "Suppliers":
        page_suppliers()
    elif page == "Upload Data":
        page_upload()
    elif page == "Train Models":
        page_train()
    
    st.markdown('<div class="footer">LEANNLP MANUFACTURING ANALYTICS | 2025</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
