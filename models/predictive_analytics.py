"""
LeanNLP: Predictive Analytics Module
Implements forecasting for costs, downtime, and efficiency trends.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Prediction:
    """Represents a model prediction with metadata."""
    target: str
    value: float
    confidence_low: float
    confidence_high: float
    horizon: str
    model_used: str
    features_used: List[str]
    accuracy_metrics: Dict[str, float]


@dataclass
class Forecast:
    """Represents a time series forecast."""
    target: str
    predictions: pd.DataFrame
    model_used: str
    accuracy_metrics: Dict[str, float]
    trend: str  # increasing, decreasing, stable
    seasonality: Optional[str]


class MaintenancePredictor:
    """Predicts maintenance costs and failure probabilities."""
    
    def __init__(self):
        self.cost_model = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, random_state=42
        )
        self.failure_model = RandomForestRegressor(
            n_estimators=100, max_depth=6, random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
        self.metrics = {}
    
    def prepare_features(self, maintenance_df: pd.DataFrame, 
                         machines_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for maintenance prediction."""
        # Aggregate maintenance history by machine
        machine_stats = maintenance_df.groupby("machine_id").agg({
            "cost": ["sum", "mean", "count"],
            "duration_hours": ["sum", "mean"],
            "downtime_impact_hours": "sum"
        })
        machine_stats.columns = ["total_cost", "avg_cost", "event_count",
                                 "total_duration", "avg_duration", "total_downtime"]
        machine_stats = machine_stats.reset_index()
        
        # Calculate unplanned event ratio
        unplanned_ratio = maintenance_df.groupby("machine_id").apply(
            lambda x: (x["event_type"] != "planned").mean()
        ).reset_index()
        unplanned_ratio.columns = ["machine_id", "unplanned_ratio"]
        
        # Merge with machine data
        features = machines_df.merge(machine_stats, on="machine_id", how="left")
        features = features.merge(unplanned_ratio, on="machine_id", how="left")
        
        # Fill NaN values
        features = features.fillna(0)
        
        # Create derived features
        features["cost_per_event"] = features["total_cost"] / (features["event_count"] + 1)
        features["downtime_per_event"] = features["total_downtime"] / (features["event_count"] + 1)
        
        return features
    
    def fit(self, maintenance_df: pd.DataFrame, machines_df: pd.DataFrame) -> Dict:
        """Train maintenance prediction models."""
        features = self.prepare_features(maintenance_df, machines_df)
        
        # Select numeric features for modeling
        self.feature_names = [
            "age_years", "hourly_rate", "event_count", "avg_cost",
            "avg_duration", "unplanned_ratio", "cost_per_event"
        ]
        
        X = features[self.feature_names].values
        y_cost = features["total_cost"].values
        y_events = features["event_count"].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_cost_train, y_cost_test = train_test_split(
            X_scaled, y_cost, test_size=0.2, random_state=42
        )
        _, _, y_events_train, y_events_test = train_test_split(
            X_scaled, y_events, test_size=0.2, random_state=42
        )
        
        # Train cost model
        self.cost_model.fit(X_train, y_cost_train)
        cost_pred = self.cost_model.predict(X_test)
        
        # Train failure/event model
        self.failure_model.fit(X_train, y_events_train)
        events_pred = self.failure_model.predict(X_test)
        
        # Calculate metrics
        self.metrics = {
            "cost_model": {
                "mae": mean_absolute_error(y_cost_test, cost_pred),
                "rmse": np.sqrt(mean_squared_error(y_cost_test, cost_pred)),
                "r2": r2_score(y_cost_test, cost_pred)
            },
            "event_model": {
                "mae": mean_absolute_error(y_events_test, events_pred),
                "rmse": np.sqrt(mean_squared_error(y_events_test, events_pred)),
                "r2": r2_score(y_events_test, events_pred)
            }
        }
        
        self.is_fitted = True
        return self.metrics
    
    def predict_maintenance_cost(self, machine_features: Dict) -> Prediction:
        """Predict maintenance cost for a machine."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prepare input
        X = np.array([[machine_features.get(f, 0) for f in self.feature_names]])
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        pred = self.cost_model.predict(X_scaled)[0]
        
        # Estimate confidence interval (simplified)
        std_dev = self.metrics["cost_model"]["rmse"]
        
        return Prediction(
            target="maintenance_cost",
            value=max(0, pred),
            confidence_low=max(0, pred - 1.96 * std_dev),
            confidence_high=pred + 1.96 * std_dev,
            horizon="12_months",
            model_used="GradientBoostingRegressor",
            features_used=self.feature_names,
            accuracy_metrics=self.metrics["cost_model"]
        )
    
    def predict_failure_probability(self, machine_features: Dict, 
                                    time_horizon_months: int = 6) -> Dict:
        """Estimate probability of failure within time horizon."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prepare input
        X = np.array([[machine_features.get(f, 0) for f in self.feature_names]])
        X_scaled = self.scaler.transform(X)
        
        # Predict expected events
        expected_events = self.failure_model.predict(X_scaled)[0]
        
        # Adjust for time horizon (assuming events are annualized)
        expected_events_horizon = expected_events * (time_horizon_months / 12)
        
        # Convert to probability using Poisson approximation
        # P(at least one event) = 1 - P(zero events) = 1 - e^(-lambda)
        prob_failure = 1 - np.exp(-expected_events_horizon)
        
        # Risk category
        if prob_failure > 0.7:
            risk_level = "high"
        elif prob_failure > 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "expected_events": expected_events_horizon,
            "failure_probability": prob_failure,
            "risk_level": risk_level,
            "time_horizon_months": time_horizon_months
        }


class CostForecaster:
    """Forecasts costs and financial metrics."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.metrics = {}
    
    def prepare_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features for forecasting."""
        df = df.copy()
        
        if "month" in df.columns:
            df["month_dt"] = pd.to_datetime(df["month"])
        else:
            df["month_dt"] = pd.to_datetime(df.index)
        
        df["month_num"] = df["month_dt"].dt.month
        df["quarter"] = df["month_dt"].dt.quarter
        df["year"] = df["month_dt"].dt.year
        df["trend"] = range(len(df))
        
        # Cyclical encoding
        df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
        
        return df
    
    def fit(self, financials_df: pd.DataFrame, target_cols: List[str] = None) -> Dict:
        """Fit forecasting models for financial metrics."""
        if target_cols is None:
            target_cols = ["maintenance_cost", "scrap_cost", "material_cost", 
                          "profit_margin", "revenue"]
        
        df = self.prepare_time_features(financials_df)
        feature_cols = ["trend", "month_sin", "month_cos", "quarter"]
        
        for target in target_cols:
            if target not in df.columns:
                continue
            
            X = df[feature_cols].values
            y = df[target].values
            
            # Use Ridge regression for stability
            model = Ridge(alpha=1.0)
            scaler = StandardScaler()
            
            X_scaled = scaler.fit_transform(X)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=min(5, len(y)-1), 
                                       scoring='neg_mean_absolute_error')
            
            # Fit on all data
            model.fit(X_scaled, y)
            
            self.models[target] = model
            self.scalers[target] = scaler
            self.metrics[target] = {
                "cv_mae": -cv_scores.mean(),
                "cv_std": cv_scores.std()
            }
        
        return self.metrics
    
    def forecast(self, target: str, periods: int = 6, 
                 last_date: datetime = None) -> Forecast:
        """Generate forecast for specified periods ahead."""
        if target not in self.models:
            raise ValueError(f"No model fitted for {target}")
        
        model = self.models[target]
        scaler = self.scalers[target]
        
        if last_date is None:
            last_date = datetime.now()
        
        # Generate future dates
        future_dates = pd.date_range(
            last_date + timedelta(days=30),
            periods=periods,
            freq='M'
        )
        
        # Create future features
        future_df = pd.DataFrame({"month_dt": future_dates})
        future_df["month_num"] = future_df["month_dt"].dt.month
        future_df["quarter"] = future_df["month_dt"].dt.quarter
        future_df["trend"] = range(12, 12 + periods)  # Continue trend
        future_df["month_sin"] = np.sin(2 * np.pi * future_df["month_num"] / 12)
        future_df["month_cos"] = np.cos(2 * np.pi * future_df["month_num"] / 12)
        
        feature_cols = ["trend", "month_sin", "month_cos", "quarter"]
        X_future = future_df[feature_cols].values
        X_future_scaled = scaler.transform(X_future)
        
        # Make predictions
        predictions = model.predict(X_future_scaled)
        
        # Calculate confidence intervals
        mae = self.metrics[target]["cv_mae"]
        future_df["prediction"] = predictions
        future_df["lower_bound"] = predictions - 1.96 * mae
        future_df["upper_bound"] = predictions + 1.96 * mae
        
        # Determine trend
        if len(predictions) > 1:
            slope = np.polyfit(range(len(predictions)), predictions, 1)[0]
            if slope > mae * 0.1:
                trend = "increasing"
            elif slope < -mae * 0.1:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return Forecast(
            target=target,
            predictions=future_df[["month_dt", "prediction", "lower_bound", "upper_bound"]],
            model_used="Ridge Regression",
            accuracy_metrics=self.metrics[target],
            trend=trend,
            seasonality="monthly"
        )


class EfficiencyPredictor:
    """Predicts production efficiency and identifies optimization opportunities."""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_importance = {}
        self.metrics = {}
    
    def prepare_features(self, production_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for efficiency prediction."""
        df = production_df.copy()
        
        # Time features
        df["start_time"] = pd.to_datetime(df["start_time"])
        df["hour"] = df["start_time"].dt.hour
        df["day_of_week"] = df["start_time"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        
        # Production features
        df["throughput"] = df["units_produced"] / df["duration_hours"]
        
        return df
    
    def fit(self, production_df: pd.DataFrame) -> Dict:
        """Fit efficiency prediction model."""
        df = self.prepare_features(production_df)
        
        feature_cols = ["duration_hours", "hour", "day_of_week", 
                       "is_weekend", "cycle_time_seconds"]
        
        X = df[feature_cols].values
        y = df["efficiency_pct"].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        self.metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred)
        }
        
        # Feature importance
        self.feature_importance = dict(zip(
            feature_cols, self.model.feature_importances_
        ))
        
        self.feature_cols = feature_cols
        self.is_fitted = True
        
        return self.metrics
    
    def predict_efficiency(self, run_params: Dict) -> Prediction:
        """Predict efficiency for given production parameters."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = np.array([[run_params.get(f, 0) for f in self.feature_cols]])
        X_scaled = self.scaler.transform(X)
        
        pred = self.model.predict(X_scaled)[0]
        std = self.metrics["rmse"]
        
        return Prediction(
            target="efficiency_pct",
            value=min(100, max(0, pred)),
            confidence_low=max(0, pred - 1.96 * std),
            confidence_high=min(100, pred + 1.96 * std),
            horizon="single_run",
            model_used="RandomForestRegressor",
            features_used=self.feature_cols,
            accuracy_metrics=self.metrics
        )
    
    def identify_optimization_opportunities(self, production_df: pd.DataFrame) -> List[Dict]:
        """Identify opportunities to improve efficiency."""
        opportunities = []
        df = self.prepare_features(production_df)
        
        # Analyze by time of day
        hourly_eff = df.groupby("hour")["efficiency_pct"].mean()
        best_hours = hourly_eff.nlargest(3).index.tolist()
        worst_hours = hourly_eff.nsmallest(3).index.tolist()
        
        efficiency_gap = hourly_eff.max() - hourly_eff.min()
        if efficiency_gap > 5:
            opportunities.append({
                "category": "scheduling",
                "finding": f"Efficiency varies by {efficiency_gap:.1f}% across hours",
                "recommendation": f"Schedule critical runs during peak hours ({best_hours})",
                "potential_improvement": efficiency_gap * 0.5,
                "confidence": 0.85
            })
        
        # Analyze by day of week
        daily_eff = df.groupby("day_of_week")["efficiency_pct"].mean()
        day_gap = daily_eff.max() - daily_eff.min()
        if day_gap > 3:
            best_day = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][daily_eff.idxmax()]
            opportunities.append({
                "category": "scheduling",
                "finding": f"Efficiency varies by {day_gap:.1f}% across days",
                "recommendation": f"Plan maintenance on low-efficiency days, maximize production on {best_day}",
                "potential_improvement": day_gap * 0.3,
                "confidence": 0.75
            })
        
        # Analyze by machine
        machine_eff = df.groupby("machine_id")["efficiency_pct"].mean()
        machine_gap = machine_eff.max() - machine_eff.min()
        if machine_gap > 5:
            worst_machines = machine_eff.nsmallest(3).index.tolist()
            opportunities.append({
                "category": "machine_optimization",
                "finding": f"Machine efficiency varies by {machine_gap:.1f}%",
                "recommendation": f"Investigate underperforming machines: {', '.join(worst_machines)}",
                "potential_improvement": machine_gap * 0.4,
                "confidence": 0.8
            })
        
        # Analyze by operator
        operator_eff = df.groupby("operator_id")["efficiency_pct"].mean()
        operator_gap = operator_eff.max() - operator_eff.min()
        if operator_gap > 5:
            top_operators = operator_eff.nlargest(3).index.tolist()
            opportunities.append({
                "category": "training",
                "finding": f"Operator efficiency varies by {operator_gap:.1f}%",
                "recommendation": f"Study practices of top operators ({', '.join(top_operators)}) for training",
                "potential_improvement": operator_gap * 0.3,
                "confidence": 0.75
            })
        
        return opportunities


class PredictiveAnalyticsEngine:
    """Main engine coordinating all predictive analytics."""
    
    def __init__(self):
        self.maintenance_predictor = MaintenancePredictor()
        self.cost_forecaster = CostForecaster()
        self.efficiency_predictor = EfficiencyPredictor()
        self.is_trained = False
    
    def train_all_models(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Train all predictive models."""
        results = {}
        
        print("Training predictive models...")
        
        # Train maintenance predictor
        if "maintenance" in data and "machines" in data:
            print("  Training maintenance predictor...")
            results["maintenance"] = self.maintenance_predictor.fit(
                data["maintenance"], data["machines"]
            )
        
        # Train cost forecaster
        if "financials" in data:
            print("  Training cost forecaster...")
            results["costs"] = self.cost_forecaster.fit(data["financials"])
        
        # Train efficiency predictor
        if "production" in data:
            print("  Training efficiency predictor...")
            results["efficiency"] = self.efficiency_predictor.fit(data["production"])
        
        self.is_trained = True
        print("  ✓ All models trained")
        
        return results
    
    def generate_comprehensive_forecast(self, data: Dict[str, pd.DataFrame],
                                         horizon_months: int = 6) -> Dict:
        """Generate comprehensive forecasts for all metrics."""
        if not self.is_trained:
            self.train_all_models(data)
        
        forecasts = {}
        
        # Cost forecasts
        for metric in ["maintenance_cost", "scrap_cost", "revenue", "profit_margin"]:
            try:
                forecast = self.cost_forecaster.forecast(metric, periods=horizon_months)
                forecasts[metric] = {
                    "predictions": forecast.predictions.to_dict("records"),
                    "trend": forecast.trend,
                    "accuracy": forecast.accuracy_metrics
                }
            except (ValueError, KeyError):
                pass
        
        # Machine risk assessments
        if "machines" in data and "maintenance" in data:
            machine_risks = []
            features = self.maintenance_predictor.prepare_features(
                data["maintenance"], data["machines"]
            )
            
            for _, row in features.iterrows():
                machine_features = row.to_dict()
                
                cost_pred = self.maintenance_predictor.predict_maintenance_cost(machine_features)
                failure_pred = self.maintenance_predictor.predict_failure_probability(
                    machine_features, horizon_months
                )
                
                machine_risks.append({
                    "machine_id": row["machine_id"],
                    "predicted_cost": cost_pred.value,
                    "failure_probability": failure_pred["failure_probability"],
                    "risk_level": failure_pred["risk_level"]
                })
            
            forecasts["machine_risks"] = sorted(
                machine_risks, 
                key=lambda x: x["failure_probability"], 
                reverse=True
            )
        
        # Efficiency opportunities
        if "production" in data:
            forecasts["efficiency_opportunities"] = \
                self.efficiency_predictor.identify_optimization_opportunities(data["production"])
        
        return forecasts
    
    def generate_summary_report(self, forecasts: Dict) -> str:
        """Generate natural language summary of forecasts."""
        lines = ["Predictive Analytics Summary", "=" * 30, ""]
        
        # Cost trends
        lines.append("Cost Trends:")
        for metric in ["maintenance_cost", "scrap_cost"]:
            if metric in forecasts:
                trend = forecasts[metric]["trend"]
                lines.append(f"  • {metric.replace('_', ' ').title()}: {trend}")
        
        # Machine risks
        if "machine_risks" in forecasts:
            high_risk = [m for m in forecasts["machine_risks"] if m["risk_level"] == "high"]
            lines.append(f"\nMachine Risk Assessment:")
            lines.append(f"  • {len(high_risk)} machines identified as high risk")
            for machine in high_risk[:3]:
                lines.append(f"    - {machine['machine_id']}: {machine['failure_probability']:.1%} failure probability")
        
        # Efficiency opportunities
        if "efficiency_opportunities" in forecasts:
            opps = forecasts["efficiency_opportunities"]
            total_potential = sum(o.get("potential_improvement", 0) for o in opps)
            lines.append(f"\nEfficiency Opportunities:")
            lines.append(f"  • {len(opps)} opportunities identified")
            lines.append(f"  • Total potential improvement: {total_potential:.1f}%")
            for opp in opps[:3]:
                lines.append(f"    - {opp['recommendation']}")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test predictive analytics
    import sys
    sys.path.insert(0, "/home/claude/leannlp")
    from data.data_generator import SyntheticDataGenerator
    
    print("Testing Predictive Analytics...")
    
    # Generate test data
    generator = SyntheticDataGenerator(seed=42)
    data = generator.generate_all_data()
    
    # Initialize and train engine
    engine = PredictiveAnalyticsEngine()
    training_results = engine.train_all_models(data)
    
    print("\nTraining Results:")
    for model, metrics in training_results.items():
        print(f"  {model}: {metrics}")
    
    # Generate forecasts
    print("\nGenerating Comprehensive Forecast...")
    forecasts = engine.generate_comprehensive_forecast(data, horizon_months=6)
    
    # Print summary
    print("\n" + engine.generate_summary_report(forecasts))
