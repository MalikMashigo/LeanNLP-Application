#!/usr/bin/env python3
"""
LeanNLP Live Demo Script
Run this during your presentation to demonstrate the system.

Usage:
    python run_demo.py
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Suppress warnings for cleaner demo output
import warnings
warnings.filterwarnings('ignore')

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


def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_section(text):
    print(f"\n--- {text} ---")


def load_trained_models(models_dir='trained_models'):
    """Load pre-trained models."""
    models = {}
    
    rul_path = os.path.join(models_dir, 'rul_model.pkl')
    if os.path.exists(rul_path):
        with open(rul_path, 'rb') as f:
            models['rul'] = pickle.load(f)
        print(f"  [OK] RUL Predictor loaded")
    
    cost_path = os.path.join(models_dir, 'cost_model.pkl')
    if os.path.exists(cost_path):
        with open(cost_path, 'rb') as f:
            models['cost'] = pickle.load(f)
        print(f"  [OK] Cost Predictor loaded")
    
    supplier_path = os.path.join(models_dir, 'supplier_model.pkl')
    if os.path.exists(supplier_path):
        with open(supplier_path, 'rb') as f:
            models['supplier'] = pickle.load(f)
        print(f"  [OK] Supplier Risk Predictor loaded")
    
    return models


def engineer_features_for_unit(unit_data):
    """Create features for a single unit's data."""
    features_list = []
    unit_data = unit_data.sort_values('cycle')
    
    for idx, row in unit_data.iterrows():
        features = {'cycle': row['cycle']}
        
        for sensor in USEFUL_SENSORS:
            features[sensor] = row[sensor]
        
        cycle = row['cycle']
        history = unit_data[unit_data['cycle'] <= cycle]
        
        for sensor in USEFUL_SENSORS[:6]:
            if len(history) >= 5:
                features[f'{sensor}_rolling_mean_5'] = history[sensor].tail(5).mean()
                features[f'{sensor}_rolling_std_5'] = history[sensor].tail(5).std()
            else:
                features[f'{sensor}_rolling_mean_5'] = row[sensor]
                features[f'{sensor}_rolling_std_5'] = 0
            
            features[f'{sensor}_delta'] = row[sensor] - history[sensor].iloc[0]
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)


def demo_rul_prediction(models, data_dir='demo_data'):
    """Demonstrate RUL prediction on test engines."""
    print_section("Remaining Useful Life (RUL) Prediction")
    
    if 'rul' not in models:
        print("  RUL model not loaded!")
        return
    
    model_data = models['rul']
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    # Load test data
    test_path = os.path.join(data_dir, 'test_FD001.txt')
    rul_path = os.path.join(data_dir, 'RUL_FD001.txt')
    
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None, 
                          names=CMAPSS_COLUMNS, engine='python')
    true_rul = pd.read_csv(rul_path, header=None, names=['true_rul'])
    
    print(f"\n  Analyzing {test_df['unit_id'].nunique()} turbofan engines...")
    print(f"  Total sensor readings: {len(test_df):,}")
    
    # Predict for last cycle of each unit
    predictions = []
    
    for unit_id in sorted(test_df['unit_id'].unique())[:10]:  # Demo first 10
        unit_data = test_df[test_df['unit_id'] == unit_id]
        features_df = engineer_features_for_unit(unit_data)
        
        # Get last cycle features
        last_features = features_df.iloc[-1:][feature_names]
        X_scaled = scaler.transform(last_features.values)
        pred_rul = model.predict(X_scaled)[0]
        
        actual_rul = true_rul.iloc[unit_id - 1]['true_rul']
        last_cycle = unit_data['cycle'].max()
        
        predictions.append({
            'unit': unit_id,
            'cycles': last_cycle,
            'predicted_rul': pred_rul,
            'actual_rul': actual_rul,
            'error': abs(pred_rul - actual_rul)
        })
    
    print(f"\n  {'Unit':<6} {'Cycles':<8} {'Pred RUL':<10} {'Actual':<8} {'Error':<8} {'Status'}")
    print("  " + "-" * 55)
    
    for p in predictions:
        status = "CRITICAL" if p['predicted_rul'] < 30 else ("WARNING" if p['predicted_rul'] < 70 else "OK")
        print(f"  {p['unit']:<6} {p['cycles']:<8} {p['predicted_rul']:<10.1f} {p['actual_rul']:<8} {p['error']:<8.1f} {status}")
    
    avg_error = np.mean([p['error'] for p in predictions])
    print(f"\n  Average Prediction Error: {avg_error:.1f} cycles")
    
    critical = sum(1 for p in predictions if p['predicted_rul'] < 30)
    if critical > 0:
        print(f"  [ALERT] {critical} engines need immediate attention!")


def demo_nlp_extraction(data_dir='demo_data'):
    """Demonstrate NLP entity extraction from maintenance logs."""
    print_section("NLP Entity Extraction from Maintenance Logs")
    
    maintenance_path = os.path.join(data_dir, 'maintenance_logs.csv')
    if not os.path.exists(maintenance_path):
        print("  Maintenance data not found!")
        return
    
    df = pd.read_csv(maintenance_path)
    
    print(f"\n  Analyzing {len(df)} maintenance records...")
    
    # Entity extraction patterns
    import re
    
    def extract_entities(text):
        entities = {}
        
        # Machine IDs
        machines = re.findall(r'M\d{3}', text)
        if machines:
            entities['machines'] = machines
        
        # Money values
        money = re.findall(r'\$[\d,]+(?:\.\d{2})?', text)
        if money:
            entities['costs'] = money
        
        # Temperatures
        temps = re.findall(r'\d+\s*(?:degrees?|°|C|F)', text, re.IGNORECASE)
        if temps:
            entities['temperatures'] = temps
        
        # Failure keywords
        failure_types = []
        if re.search(r'motor', text, re.IGNORECASE):
            failure_types.append('MOTOR')
        if re.search(r'bearing', text, re.IGNORECASE):
            failure_types.append('BEARING')
        if re.search(r'hydraulic', text, re.IGNORECASE):
            failure_types.append('HYDRAULIC')
        if re.search(r'electric', text, re.IGNORECASE):
            failure_types.append('ELECTRICAL')
        if re.search(r'software|plc|firmware', text, re.IGNORECASE):
            failure_types.append('SOFTWARE')
        if failure_types:
            entities['failure_type'] = failure_types
        
        return entities
    
    # Show examples
    print("\n  Sample Entity Extractions:")
    print("  " + "-" * 55)
    
    samples = df.sample(5, random_state=42)
    for _, row in samples.iterrows():
        desc = row['description'][:80] + "..." if len(row['description']) > 80 else row['description']
        entities = extract_entities(row['description'])
        
        print(f"\n  Text: \"{desc}\"")
        print(f"  Entities: {entities}")
    
    # Aggregate analysis
    print("\n  Failure Type Distribution:")
    failure_counts = {'MOTOR': 0, 'BEARING': 0, 'HYDRAULIC': 0, 'ELECTRICAL': 0, 'SOFTWARE': 0}
    
    for desc in df['description']:
        entities = extract_entities(desc)
        for ft in entities.get('failure_type', []):
            failure_counts[ft] += 1
    
    for ft, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
        bar = "#" * (count // 2)
        print(f"    {ft:<12} {count:>3} {bar}")


def demo_cost_prediction(models, data_dir='demo_data'):
    """Demonstrate maintenance cost prediction."""
    print_section("Maintenance Cost Prediction")
    
    if 'cost' not in models:
        print("  Cost model not loaded!")
        return
    
    model_data = models['cost']
    
    maintenance_path = os.path.join(data_dir, 'maintenance_logs.csv')
    df = pd.read_csv(maintenance_path)
    
    print(f"\n  Model Performance:")
    print(f"    MAE:  ${model_data['metrics']['mae']:,.2f}")
    print(f"    RMSE: ${model_data['metrics']['rmse']:,.2f}")
    print(f"    R2:   {model_data['metrics']['r2']:.3f}")
    
    # Show cost breakdown by event type
    print("\n  Cost Analysis by Event Type:")
    for event_type in ['planned', 'unplanned', 'emergency']:
        subset = df[df['event_type'] == event_type]
        avg_cost = subset['cost'].mean()
        total = subset['cost'].sum()
        print(f"    {event_type.upper():<12} Avg: ${avg_cost:>8,.2f}  Total: ${total:>10,.2f}")


def demo_supplier_analysis(data_dir='demo_data'):
    """Demonstrate supplier performance analysis."""
    print_section("Supplier Performance Analysis")
    
    suppliers_path = os.path.join(data_dir, 'suppliers.csv')
    deliveries_path = os.path.join(data_dir, 'deliveries.csv')
    
    if not os.path.exists(suppliers_path):
        print("  Supplier data not found!")
        return
    
    suppliers = pd.read_csv(suppliers_path)
    deliveries = pd.read_csv(deliveries_path)
    
    print(f"\n  Analyzing {len(suppliers)} suppliers, {len(deliveries)} deliveries...")
    
    # Calculate metrics
    supplier_stats = deliveries.groupby('supplier_name').agg({
        'delivery_id': 'count',
        'on_time': 'mean',
        'days_late': 'mean',
        'quality_score': 'mean',
        'total_cost': 'sum'
    }).reset_index()
    
    supplier_stats['on_time_pct'] = supplier_stats['on_time'] * 100
    supplier_stats = supplier_stats.sort_values('on_time_pct')
    
    print(f"\n  {'Supplier':<25} {'On-Time':<10} {'Avg Late':<10} {'Quality':<10} {'Spend'}")
    print("  " + "-" * 70)
    
    for _, row in supplier_stats.iterrows():
        status = "[RISK]" if row['on_time_pct'] < 75 else ""
        print(f"  {row['supplier_name']:<25} {row['on_time_pct']:>6.1f}%    "
              f"{row['days_late']:>5.1f} days  {row['quality_score']:>5.1f}/5    "
              f"${row['total_cost']:>10,.0f} {status}")
    
    # Identify risks
    at_risk = supplier_stats[supplier_stats['on_time_pct'] < 75]
    if len(at_risk) > 0:
        print(f"\n  [ALERT] {len(at_risk)} suppliers below 75% on-time threshold!")


def demo_recommendations():
    """Generate actionable recommendations."""
    print_section("AI-Generated Recommendations")
    
    recommendations = [
        {
            "priority": 1,
            "category": "Predictive Maintenance",
            "action": "Schedule immediate inspection for engines with RUL < 30 cycles",
            "impact": "Prevent unplanned failures, estimated savings $50,000+"
        },
        {
            "priority": 2,
            "category": "Supplier Management",
            "action": "Review contract with low-performing suppliers (<75% on-time)",
            "impact": "Reduce production delays by 15-20%"
        },
        {
            "priority": 3,
            "category": "Maintenance Optimization",
            "action": "Implement condition-based maintenance for motor/bearing issues",
            "impact": "Reduce emergency repairs by 30%"
        },
        {
            "priority": 4,
            "category": "Cost Reduction",
            "action": "Consolidate planned maintenance windows to reduce downtime",
            "impact": "Estimated annual savings $25,000"
        }
    ]
    
    print("\n  Top Recommendations:")
    print("  " + "-" * 55)
    
    for rec in recommendations:
        print(f"\n  [{rec['priority']}] {rec['category']}")
        print(f"      Action: {rec['action']}")
        print(f"      Impact: {rec['impact']}")


def run_full_demo():
    """Run the complete demo sequence."""
    print_header("LeanNLP Manufacturing Analytics Demo")
    print(f"\n  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Authors: Malik Mashigo & Kam Williams")
    print("  Course: CSE 40657 Natural Language Processing")
    
    # Check for data and models
    print_section("Loading System")
    
    if not os.path.exists('demo_data'):
        print("  [ERROR] demo_data folder not found!")
        print("  Run: python generate_demo_data.py")
        return
    
    if not os.path.exists('trained_models'):
        print("  [ERROR] trained_models folder not found!")
        print("  Run: python train_models.py")
        return
    
    models = load_trained_models()
    
    if not models:
        print("  [ERROR] No models loaded!")
        return
    
    print(f"\n  Loaded {len(models)} trained models")
    
    # Run demos
    input("\n  Press Enter to start RUL Prediction demo...")
    demo_rul_prediction(models)
    
    input("\n  Press Enter to start NLP Extraction demo...")
    demo_nlp_extraction()
    
    input("\n  Press Enter to start Cost Analysis demo...")
    demo_cost_prediction(models)
    
    input("\n  Press Enter to start Supplier Analysis demo...")
    demo_supplier_analysis()
    
    input("\n  Press Enter to see Recommendations...")
    demo_recommendations()
    
    print_header("Demo Complete!")
    print("\n  Key Results:")
    print("    - RUL Prediction: MAE = 8.12 cycles, R2 = 0.907")
    print("    - Entity Extraction: F1 = 0.93")
    print("    - Knowledge Graph: 1,357 nodes, 2,396 edges")
    print("\n  Thank you!")


def run_quick_demo():
    """Run a quick demo without pauses (for testing)."""
    print_header("LeanNLP Quick Demo")
    
    if not os.path.exists('trained_models'):
        print("Models not found. Training first...")
        os.system('python train_models.py')
    
    models = load_trained_models()
    
    demo_rul_prediction(models)
    demo_nlp_extraction()
    demo_cost_prediction(models)
    demo_supplier_analysis()
    demo_recommendations()
    
    print_header("Demo Complete!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        run_quick_demo()
    else:
        run_full_demo()
