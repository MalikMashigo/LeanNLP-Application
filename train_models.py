"""
LeanNLP Training Script
Trains all models on the demo data and saves trained model weights.

Usage:
    python train_models.py --data_dir demo_data --output_dir trained_models
"""

import os
import sys
import argparse
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Column names for CMAPSS data
CMAPSS_COLUMNS = [
    'unit_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20', 'sensor_21'
]

# Sensors that show degradation patterns (remove constant ones)
USEFUL_SENSORS = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_9',
                  'sensor_11', 'sensor_12', 'sensor_14', 'sensor_15', 
                  'sensor_17', 'sensor_20', 'sensor_21']


class RULPredictor:
    """Remaining Useful Life prediction model for turbofan engines."""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.metrics = {}
        self.is_trained = False
        
    def load_cmapss_data(self, train_path, test_path=None, rul_path=None):
        """Load CMAPSS format data files."""
        # Load training data
        train_df = pd.read_csv(train_path, sep=r'\s+', header=None, 
                               names=CMAPSS_COLUMNS, engine='python')
        
        # Calculate RUL for training data (run-to-failure)
        max_cycles = train_df.groupby('unit_id')['cycle'].max()
        train_df = train_df.merge(
            max_cycles.rename('max_cycle').reset_index(), on='unit_id'
        )
        train_df['rul'] = train_df['max_cycle'] - train_df['cycle']
        train_df.drop('max_cycle', axis=1, inplace=True)
        
        test_df = None
        if test_path and rul_path:
            test_df = pd.read_csv(test_path, sep=r'\s+', header=None,
                                  names=CMAPSS_COLUMNS, engine='python')
            rul_df = pd.read_csv(rul_path, header=None, names=['rul'])
            rul_df['unit_id'] = rul_df.index + 1
            
            # Add RUL to test data
            max_cycles = test_df.groupby('unit_id')['cycle'].max().reset_index()
            max_cycles.columns = ['unit_id', 'max_cycle']
            max_cycles = max_cycles.merge(rul_df, on='unit_id')
            
            test_df = test_df.merge(max_cycles, on='unit_id')
            test_df['rul'] = test_df['rul'] + (test_df['max_cycle'] - test_df['cycle'])
            test_df.drop('max_cycle', axis=1, inplace=True)
        
        return train_df, test_df
    
    def engineer_features(self, df):
        """Create features for RUL prediction."""
        features_list = []
        
        for unit_id in df['unit_id'].unique():
            unit_data = df[df['unit_id'] == unit_id].copy()
            unit_data = unit_data.sort_values('cycle')
            
            for _, row in unit_data.iterrows():
                features = {'unit_id': unit_id, 'cycle': row['cycle'], 'rul': row['rul']}
                
                # Current sensor values
                for sensor in USEFUL_SENSORS:
                    features[sensor] = row[sensor]
                
                # Rolling statistics (using data up to current point)
                cycle = row['cycle']
                history = unit_data[unit_data['cycle'] <= cycle]
                
                for sensor in USEFUL_SENSORS[:6]:  # Top 6 most important sensors
                    if len(history) >= 5:
                        features[f'{sensor}_rolling_mean_5'] = history[sensor].tail(5).mean()
                        features[f'{sensor}_rolling_std_5'] = history[sensor].tail(5).std()
                    else:
                        features[f'{sensor}_rolling_mean_5'] = row[sensor]
                        features[f'{sensor}_rolling_std_5'] = 0
                    
                    # Delta from start
                    features[f'{sensor}_delta'] = row[sensor] - history[sensor].iloc[0]
                
                features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def train(self, train_df, clip_rul=125):
        """Train the RUL prediction model."""
        print("Engineering features...")
        features_df = self.engineer_features(train_df)
        
        # Clip RUL at threshold (common practice for CMAPSS)
        features_df['rul'] = features_df['rul'].clip(upper=clip_rul)
        
        # Prepare training data
        self.feature_names = [c for c in features_df.columns 
                             if c not in ['unit_id', 'cycle', 'rul']]
        
        X = features_df[self.feature_names].values
        y = features_df['rul'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split for evaluation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_val)
        
        self.metrics = {
            'mae': mean_absolute_error(y_val, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'r2': r2_score(y_val, y_pred)
        }
        
        self.is_trained = True
        
        # Feature importance
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        self.top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return self.metrics
    
    def predict(self, features_df):
        """Predict RUL for new data."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X = features_df[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, path):
        """Save trained model to disk."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'top_features': self.top_features
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"RUL model saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load trained model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        predictor = cls()
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']
        predictor.feature_names = model_data['feature_names']
        predictor.metrics = model_data['metrics']
        predictor.top_features = model_data['top_features']
        predictor.is_trained = True
        
        return predictor


class MaintenanceCostPredictor:
    """Predicts maintenance costs based on machine and event features."""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.metrics = {}
        self.is_trained = False
        
    def prepare_features(self, maintenance_df):
        """Prepare features from maintenance logs."""
        df = maintenance_df.copy()
        
        # Encode event type
        df['is_planned'] = (df['event_type'] == 'planned').astype(int)
        df['is_emergency'] = (df['event_type'] == 'emergency').astype(int)
        
        # Machine features
        df['machine_num'] = df['machine_id'].str.extract(r'(\d+)').astype(int)
        
        # Text-based features (simplified NLP)
        df['has_motor'] = df['description'].str.lower().str.contains('motor').astype(int)
        df['has_bearing'] = df['description'].str.lower().str.contains('bearing').astype(int)
        df['has_hydraulic'] = df['description'].str.lower().str.contains('hydraulic').astype(int)
        df['has_electrical'] = df['description'].str.lower().str.contains('electric').astype(int)
        df['has_failure'] = df['description'].str.lower().str.contains('fail|failure').astype(int)
        df['desc_length'] = df['description'].str.len()
        
        return df
    
    def train(self, maintenance_df):
        """Train the cost prediction model."""
        df = self.prepare_features(maintenance_df)
        
        self.feature_names = [
            'duration_hours', 'is_planned', 'is_emergency', 'machine_num',
            'has_motor', 'has_bearing', 'has_hydraulic', 'has_electrical',
            'has_failure', 'desc_length'
        ]
        
        X = df[self.feature_names].values
        y = df['cost'].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        print("Training Maintenance Cost model...")
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_val)
        
        self.metrics = {
            'mae': mean_absolute_error(y_val, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'r2': r2_score(y_val, y_pred)
        }
        
        self.is_trained = True
        return self.metrics
    
    def predict(self, maintenance_df):
        """Predict costs for new maintenance events."""
        df = self.prepare_features(maintenance_df)
        X = df[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, path):
        """Save trained model."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Maintenance cost model saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load trained model."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        predictor = cls()
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']
        predictor.feature_names = model_data['feature_names']
        predictor.metrics = model_data['metrics']
        predictor.is_trained = True
        
        return predictor


class SupplierRiskPredictor:
    """Predicts supplier delivery delays."""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=50,
            max_depth=6,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.metrics = {}
        self.is_trained = False
        
    def prepare_features(self, deliveries_df, suppliers_df):
        """Prepare features for delivery prediction."""
        df = deliveries_df.merge(
            suppliers_df[['supplier_id', 'reliability_score', 'avg_lead_time_days', 'quality_rating']],
            on='supplier_id'
        )
        
        df['quantity_log'] = np.log1p(df['quantity'])
        df['total_cost_log'] = np.log1p(df['total_cost'])
        
        return df
    
    def train(self, deliveries_df, suppliers_df):
        """Train delivery delay predictor."""
        df = self.prepare_features(deliveries_df, suppliers_df)
        
        self.feature_names = [
            'reliability_score', 'avg_lead_time_days', 'quality_rating',
            'quantity_log', 'total_cost_log'
        ]
        
        X = df[self.feature_names].values
        y = df['days_late'].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        print("Training Supplier Risk model...")
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_val)
        
        self.metrics = {
            'mae': mean_absolute_error(y_val, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'r2': r2_score(y_val, y_pred)
        }
        
        self.is_trained = True
        return self.metrics
    
    def save(self, path):
        """Save trained model."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Supplier risk model saved to {path}")


def train_all_models(data_dir, output_dir):
    """Train all models and save to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'training_date': datetime.now().isoformat(),
        'data_dir': data_dir,
        'models': {}
    }
    
    print("=" * 60)
    print("LeanNLP Model Training")
    print("=" * 60)
    
    # 1. Train RUL Predictor on CMAPSS data
    print("\n[1/3] Training RUL Predictor on Turbofan Data")
    print("-" * 40)
    
    train_path = os.path.join(data_dir, 'train_FD001.txt')
    test_path = os.path.join(data_dir, 'test_FD001.txt')
    rul_path = os.path.join(data_dir, 'RUL_FD001.txt')
    
    if os.path.exists(train_path):
        rul_predictor = RULPredictor()
        train_df, test_df = rul_predictor.load_cmapss_data(train_path, test_path, rul_path)
        
        print(f"  Loaded {len(train_df)} training records from {train_df['unit_id'].nunique()} units")
        
        metrics = rul_predictor.train(train_df)
        print(f"  Results: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.3f}")
        
        print(f"  Top features: {[f[0] for f in rul_predictor.top_features[:5]]}")
        
        rul_predictor.save(os.path.join(output_dir, 'rul_model.pkl'))
        results['models']['rul_predictor'] = metrics
    else:
        print(f"  WARNING: {train_path} not found, skipping RUL training")
    
    # 2. Train Maintenance Cost Predictor
    print("\n[2/3] Training Maintenance Cost Predictor")
    print("-" * 40)
    
    maintenance_path = os.path.join(data_dir, 'maintenance_logs.csv')
    
    if os.path.exists(maintenance_path):
        maintenance_df = pd.read_csv(maintenance_path)
        print(f"  Loaded {len(maintenance_df)} maintenance records")
        
        cost_predictor = MaintenanceCostPredictor()
        metrics = cost_predictor.train(maintenance_df)
        print(f"  Results: MAE=${metrics['mae']:.2f}, RMSE=${metrics['rmse']:.2f}, R2={metrics['r2']:.3f}")
        
        cost_predictor.save(os.path.join(output_dir, 'cost_model.pkl'))
        results['models']['cost_predictor'] = metrics
    else:
        print(f"  WARNING: {maintenance_path} not found, skipping cost training")
    
    # 3. Train Supplier Risk Predictor
    print("\n[3/3] Training Supplier Risk Predictor")
    print("-" * 40)
    
    suppliers_path = os.path.join(data_dir, 'suppliers.csv')
    deliveries_path = os.path.join(data_dir, 'deliveries.csv')
    
    if os.path.exists(suppliers_path) and os.path.exists(deliveries_path):
        suppliers_df = pd.read_csv(suppliers_path)
        deliveries_df = pd.read_csv(deliveries_path)
        print(f"  Loaded {len(deliveries_df)} deliveries from {len(suppliers_df)} suppliers")
        
        supplier_predictor = SupplierRiskPredictor()
        metrics = supplier_predictor.train(deliveries_df, suppliers_df)
        print(f"  Results: MAE={metrics['mae']:.2f} days, RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.3f}")
        
        supplier_predictor.save(os.path.join(output_dir, 'supplier_model.pkl'))
        results['models']['supplier_predictor'] = metrics
    else:
        print(f"  WARNING: Supplier data not found, skipping supplier training")
    
    # Save training summary
    with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModels saved to: {output_dir}")
    print(f"Training summary saved to: {os.path.join(output_dir, 'training_results.json')}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LeanNLP models")
    parser.add_argument('--data_dir', default='demo_data', 
                       help='Directory containing training data')
    parser.add_argument('--output_dir', default='trained_models',
                       help='Directory to save trained models')
    
    args = parser.parse_args()
    
    train_all_models(args.data_dir, args.output_dir)
