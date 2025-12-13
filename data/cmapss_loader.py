"""
LeanNLP: NASA CMAPSS Dataset Loader
Handles turbofan engine degradation data for predictive maintenance.

The CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset
contains run-to-failure data for turbofan engines with multiple sensor readings.

Dataset structure:
- unit_id: Engine unit number
- cycle: Time in cycles
- op_setting_1-3: Operational settings
- sensor_1-21: Sensor measurements
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import io


# Column names for CMAPSS dataset
CMAPSS_COLUMNS = [
    'unit_id', 'cycle',
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
    'sensor_21'
]

# Sensor descriptions for CMAPSS
SENSOR_DESCRIPTIONS = {
    'sensor_1': 'Total temperature at fan inlet (R)',
    'sensor_2': 'Total temperature at LPC outlet (R)',
    'sensor_3': 'Total temperature at HPC outlet (R)',
    'sensor_4': 'Total temperature at LPT outlet (R)',
    'sensor_5': 'Pressure at fan inlet (psia)',
    'sensor_6': 'Total pressure in bypass-duct (psia)',
    'sensor_7': 'Total pressure at HPC outlet (psia)',
    'sensor_8': 'Physical fan speed (rpm)',
    'sensor_9': 'Physical core speed (rpm)',
    'sensor_10': 'Engine pressure ratio',
    'sensor_11': 'Static pressure at HPC outlet (psia)',
    'sensor_12': 'Ratio of fuel flow to Ps30 (pps/psi)',
    'sensor_13': 'Corrected fan speed (rpm)',
    'sensor_14': 'Corrected core speed (rpm)',
    'sensor_15': 'Bypass Ratio',
    'sensor_16': 'Burner fuel-air ratio',
    'sensor_17': 'Bleed Enthalpy',
    'sensor_18': 'Demanded fan speed (rpm)',
    'sensor_19': 'Demanded corrected fan speed (rpm)',
    'sensor_20': 'HPT coolant bleed (lbm/s)',
    'sensor_21': 'LPT coolant bleed (lbm/s)'
}

# Key sensors that typically show degradation patterns
KEY_SENSORS = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 
               'sensor_11', 'sensor_12', 'sensor_15', 'sensor_17',
               'sensor_20', 'sensor_21']


class CMAPSSDataLoader:
    """Loader for NASA CMAPSS turbofan engine degradation dataset."""
    
    def __init__(self):
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.rul_data: Optional[pd.DataFrame] = None
        self.is_loaded = False
        self.dataset_info = {}
    
    def load_from_file(self, train_file: Union[str, Path, io.BytesIO],
                       test_file: Optional[Union[str, Path, io.BytesIO]] = None,
                       rul_file: Optional[Union[str, Path, io.BytesIO]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load CMAPSS data from files.
        
        Args:
            train_file: Path or file object for training data
            test_file: Optional path or file object for test data
            rul_file: Optional path or file object for RUL labels
        
        Returns:
            Dictionary containing loaded dataframes
        """
        # Load training data
        self.train_data = self._load_single_file(train_file)
        self._add_rul_to_train()
        
        # Load test data if provided
        if test_file is not None:
            self.test_data = self._load_single_file(test_file)
        
        # Load RUL data if provided
        if rul_file is not None:
            self.rul_data = self._load_rul_file(rul_file)
            if self.test_data is not None:
                self._add_rul_to_test()
        
        self.is_loaded = True
        self._compute_dataset_info()
        
        return self.get_data()
    
    def load_from_dataframe(self, df: pd.DataFrame, 
                            is_train: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load data from an existing DataFrame.
        
        Expects columns to match CMAPSS format or similar sensor data.
        """
        # Check if columns match CMAPSS format
        if set(CMAPSS_COLUMNS).issubset(set(df.columns)):
            processed_df = df.copy()
        else:
            # Try to infer structure
            processed_df = self._infer_structure(df)
        
        if is_train:
            self.train_data = processed_df
            self._add_rul_to_train()
        else:
            self.test_data = processed_df
        
        self.is_loaded = True
        self._compute_dataset_info()
        
        return self.get_data()
    
    def _load_single_file(self, file_source: Union[str, Path, io.BytesIO]) -> pd.DataFrame:
        """Load a single CMAPSS data file."""
        df = pd.read_csv(
            file_source,
            sep=r'\s+',
            header=None,
            names=CMAPSS_COLUMNS,
            engine='python'
        )
        return df
    
    def _load_rul_file(self, file_source: Union[str, Path, io.BytesIO]) -> pd.DataFrame:
        """Load RUL labels file."""
        rul = pd.read_csv(file_source, header=None, names=['rul'])
        rul['unit_id'] = rul.index + 1
        return rul
    
    def _add_rul_to_train(self):
        """Calculate RUL for training data (run-to-failure)."""
        if self.train_data is None:
            return
        
        # For training data, RUL = max_cycle - current_cycle for each unit
        max_cycles = self.train_data.groupby('unit_id')['cycle'].max()
        self.train_data = self.train_data.merge(
            max_cycles.rename('max_cycle').reset_index(),
            on='unit_id'
        )
        self.train_data['rul'] = self.train_data['max_cycle'] - self.train_data['cycle']
        self.train_data.drop('max_cycle', axis=1, inplace=True)
    
    def _add_rul_to_test(self):
        """Add RUL labels to test data."""
        if self.test_data is None or self.rul_data is None:
            return
        
        # Get max cycle for each unit in test data
        max_cycles = self.test_data.groupby('unit_id')['cycle'].max().reset_index()
        max_cycles.columns = ['unit_id', 'max_cycle']
        
        # Merge RUL labels
        max_cycles = max_cycles.merge(self.rul_data, on='unit_id')
        
        self.test_data = self.test_data.merge(max_cycles, on='unit_id')
        self.test_data['rul'] = self.test_data['rul'] + (
            self.test_data['max_cycle'] - self.test_data['cycle']
        )
        self.test_data.drop('max_cycle', axis=1, inplace=True)
    
    def _infer_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Infer CMAPSS-like structure from arbitrary sensor data."""
        processed = df.copy()
        
        # Check for common column patterns
        if 'unit' in processed.columns.str.lower().tolist():
            unit_col = [c for c in processed.columns if 'unit' in c.lower()][0]
            processed = processed.rename(columns={unit_col: 'unit_id'})
        elif 'engine' in processed.columns.str.lower().tolist():
            engine_col = [c for c in processed.columns if 'engine' in c.lower()][0]
            processed = processed.rename(columns={engine_col: 'unit_id'})
        elif 'unit_id' not in processed.columns:
            processed['unit_id'] = 1
        
        if 'cycle' not in processed.columns:
            if 'time' in processed.columns.str.lower().tolist():
                time_col = [c for c in processed.columns if 'time' in c.lower()][0]
                processed = processed.rename(columns={time_col: 'cycle'})
            else:
                processed['cycle'] = range(1, len(processed) + 1)
        
        # Identify sensor columns
        sensor_cols = [c for c in processed.columns 
                      if c not in ['unit_id', 'cycle', 'rul'] and 
                      processed[c].dtype in ['float64', 'int64', 'float32', 'int32']]
        
        # Rename sensor columns
        for i, col in enumerate(sensor_cols[:21], 1):
            if col != f'sensor_{i}':
                processed = processed.rename(columns={col: f'sensor_{i}'})
        
        return processed
    
    def _compute_dataset_info(self):
        """Compute dataset statistics."""
        self.dataset_info = {
            'train_samples': len(self.train_data) if self.train_data is not None else 0,
            'test_samples': len(self.test_data) if self.test_data is not None else 0,
            'train_units': self.train_data['unit_id'].nunique() if self.train_data is not None else 0,
            'test_units': self.test_data['unit_id'].nunique() if self.test_data is not None else 0,
        }
        
        if self.train_data is not None:
            self.dataset_info['avg_cycles'] = self.train_data.groupby('unit_id')['cycle'].max().mean()
            self.dataset_info['sensor_count'] = len([c for c in self.train_data.columns 
                                                     if c.startswith('sensor_')])
    
    def get_data(self) -> Dict[str, pd.DataFrame]:
        """Get all loaded data."""
        return {
            'train': self.train_data,
            'test': self.test_data,
            'rul': self.rul_data
        }
    
    def get_unit_data(self, unit_id: int, dataset: str = 'train') -> pd.DataFrame:
        """Get data for a specific engine unit."""
        if dataset == 'train' and self.train_data is not None:
            return self.train_data[self.train_data['unit_id'] == unit_id].copy()
        elif dataset == 'test' and self.test_data is not None:
            return self.test_data[self.test_data['unit_id'] == unit_id].copy()
        return pd.DataFrame()
    
    def get_sensor_statistics(self, dataset: str = 'train') -> pd.DataFrame:
        """Calculate statistics for all sensors."""
        df = self.train_data if dataset == 'train' else self.test_data
        if df is None:
            return pd.DataFrame()
        
        sensor_cols = [c for c in df.columns if c.startswith('sensor_')]
        
        stats = []
        for col in sensor_cols:
            stats.append({
                'sensor': col,
                'description': SENSOR_DESCRIPTIONS.get(col, 'Unknown'),
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'variance': df[col].var()
            })
        
        return pd.DataFrame(stats)
    
    def normalize_sensors(self, method: str = 'minmax') -> Tuple[pd.DataFrame, Dict]:
        """Normalize sensor readings."""
        if self.train_data is None:
            return pd.DataFrame(), {}
        
        df = self.train_data.copy()
        sensor_cols = [c for c in df.columns if c.startswith('sensor_')]
        
        normalization_params = {}
        
        if method == 'minmax':
            for col in sensor_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                normalization_params[col] = {'min': min_val, 'max': max_val}
        
        elif method == 'zscore':
            for col in sensor_cols:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df[col] = (df[col] - mean_val) / std_val
                normalization_params[col] = {'mean': mean_val, 'std': std_val}
        
        return df, normalization_params


class CMAPSSFeatureEngineer:
    """Feature engineering for CMAPSS data."""
    
    def __init__(self, window_sizes: List[int] = [5, 10, 20]):
        self.window_sizes = window_sizes
    
    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics as features."""
        result = df.copy()
        sensor_cols = [c for c in df.columns if c.startswith('sensor_')]
        
        for unit_id in df['unit_id'].unique():
            mask = df['unit_id'] == unit_id
            unit_data = df[mask].copy()
            
            for sensor in sensor_cols:
                for window in self.window_sizes:
                    # Rolling mean
                    result.loc[mask, f'{sensor}_rolling_mean_{window}'] = \
                        unit_data[sensor].rolling(window=window, min_periods=1).mean()
                    
                    # Rolling std
                    result.loc[mask, f'{sensor}_rolling_std_{window}'] = \
                        unit_data[sensor].rolling(window=window, min_periods=1).std().fillna(0)
        
        return result
    
    def add_degradation_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features that indicate degradation trends."""
        result = df.copy()
        sensor_cols = [c for c in df.columns if c.startswith('sensor_') and 
                      'rolling' not in c]
        
        for unit_id in df['unit_id'].unique():
            mask = df['unit_id'] == unit_id
            unit_data = df[mask].copy()
            
            for sensor in sensor_cols:
                # Sensor change from start
                initial_value = unit_data[sensor].iloc[0]
                result.loc[mask, f'{sensor}_delta_from_start'] = \
                    unit_data[sensor] - initial_value
                
                # Rate of change
                result.loc[mask, f'{sensor}_rate_of_change'] = \
                    unit_data[sensor].diff().fillna(0)
        
        return result
    
    def add_health_index(self, df: pd.DataFrame, 
                         key_sensors: List[str] = None) -> pd.DataFrame:
        """Calculate a composite health index from sensor readings."""
        result = df.copy()
        
        if key_sensors is None:
            key_sensors = KEY_SENSORS
        
        # Filter to available sensors
        available_sensors = [s for s in key_sensors if s in df.columns]
        
        if not available_sensors:
            return result
        
        # Normalize each sensor
        normalized = pd.DataFrame()
        for sensor in available_sensors:
            min_val = df[sensor].min()
            max_val = df[sensor].max()
            if max_val > min_val:
                normalized[sensor] = (df[sensor] - min_val) / (max_val - min_val)
            else:
                normalized[sensor] = 0
        
        # Calculate health index (inverse of degradation)
        result['health_index'] = 1 - normalized.mean(axis=1)
        
        return result


def generate_synthetic_cmapss(n_units: int = 10, 
                               avg_cycles: int = 200,
                               seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic CMAPSS-like data for testing.
    
    Args:
        n_units: Number of engine units
        avg_cycles: Average number of cycles per unit
        seed: Random seed
    
    Returns:
        DataFrame with synthetic sensor data
    """
    np.random.seed(seed)
    
    data = []
    
    for unit in range(1, n_units + 1):
        # Random number of cycles for this unit
        n_cycles = int(np.random.normal(avg_cycles, avg_cycles * 0.2))
        n_cycles = max(50, n_cycles)  # Minimum 50 cycles
        
        for cycle in range(1, n_cycles + 1):
            # Operational settings
            op1 = np.random.uniform(-0.1, 0.1)
            op2 = np.random.uniform(-0.1, 0.1)
            op3 = np.random.uniform(0, 100)
            
            # Progress towards failure (0 to 1)
            degradation = cycle / n_cycles
            
            # Base sensor values with degradation trends
            row = {
                'unit_id': unit,
                'cycle': cycle,
                'op_setting_1': op1,
                'op_setting_2': op2,
                'op_setting_3': op3,
            }
            
            # Generate sensor readings with degradation patterns
            for i in range(1, 22):
                base = 500 + i * 10
                noise = np.random.normal(0, base * 0.01)
                
                # Different sensors degrade differently
                if i in [2, 3, 4, 7, 11, 12]:
                    # These sensors increase with degradation
                    trend = degradation * base * 0.1
                elif i in [15, 17, 20, 21]:
                    # These sensors decrease with degradation
                    trend = -degradation * base * 0.05
                else:
                    # Minimal degradation
                    trend = degradation * base * 0.01 * np.random.choice([-1, 1])
                
                row[f'sensor_{i}'] = base + trend + noise
            
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # Add RUL
    max_cycles = df.groupby('unit_id')['cycle'].max()
    df = df.merge(max_cycles.rename('max_cycle').reset_index(), on='unit_id')
    df['rul'] = df['max_cycle'] - df['cycle']
    df.drop('max_cycle', axis=1, inplace=True)
    
    return df


if __name__ == "__main__":
    # Test with synthetic data
    print("Generating synthetic CMAPSS data...")
    synthetic_data = generate_synthetic_cmapss(n_units=10, avg_cycles=200)
    print(f"Generated {len(synthetic_data)} samples for {synthetic_data['unit_id'].nunique()} units")
    print(f"\nColumns: {list(synthetic_data.columns)}")
    print(f"\nSample data:\n{synthetic_data.head()}")
    
    # Test feature engineering
    print("\nTesting feature engineering...")
    engineer = CMAPSSFeatureEngineer(window_sizes=[5, 10])
    featured_data = engineer.add_rolling_features(synthetic_data)
    featured_data = engineer.add_health_index(featured_data)
    print(f"Features after engineering: {len(featured_data.columns)} columns")
