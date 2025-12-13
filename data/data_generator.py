"""
LeanNLP Data Models and Synthetic Data Generator
Generates realistic manufacturing data for testing and demonstration.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import random
import json


@dataclass
class Machine:
    machine_id: str
    name: str
    type: str
    installation_date: datetime
    hourly_rate: float
    status: str = "operational"
    last_maintenance: Optional[datetime] = None


@dataclass
class Supplier:
    supplier_id: str
    name: str
    category: str
    reliability_score: float
    avg_lead_time_days: int
    contract_value: float


@dataclass
class MaintenanceEvent:
    event_id: str
    machine_id: str
    event_type: str  # planned, unplanned, emergency
    description: str
    start_time: datetime
    duration_hours: float
    cost: float
    parts_replaced: List[str]
    root_cause: Optional[str] = None


@dataclass
class ProductionRun:
    run_id: str
    machine_id: str
    product_code: str
    start_time: datetime
    end_time: datetime
    units_produced: int
    units_defective: int
    cycle_time_seconds: float
    operator_id: str


@dataclass
class SupplierDelivery:
    delivery_id: str
    supplier_id: str
    po_number: str
    expected_date: datetime
    actual_date: datetime
    material_code: str
    quantity: float
    unit_cost: float
    quality_score: float


class SyntheticDataGenerator:
    """Generates realistic manufacturing data for LeanNLP testing."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.base_date = datetime(2024, 1, 1)
        
        # Machine types and their characteristics
        self.machine_types = {
            "CNC Mill": {"hourly_rate": 85, "mtbf_hours": 2000, "repair_cost_range": (500, 5000)},
            "Lathe": {"hourly_rate": 65, "mtbf_hours": 2500, "repair_cost_range": (300, 3000)},
            "Press": {"hourly_rate": 55, "mtbf_hours": 3000, "repair_cost_range": (400, 4000)},
            "Welding Robot": {"hourly_rate": 120, "mtbf_hours": 1800, "repair_cost_range": (800, 8000)},
            "Assembly Line": {"hourly_rate": 95, "mtbf_hours": 2200, "repair_cost_range": (600, 6000)},
            "Injection Molder": {"hourly_rate": 75, "mtbf_hours": 1500, "repair_cost_range": (700, 7000)},
        }
        
        self.supplier_names = [
            "SteelCorp Industries", "FastParts Supply", "PrecisionMetal Co",
            "GlobalComponents Ltd", "QualityAlloys Inc", "TechMaterials Group",
            "ReliableSupply Partners", "IndustrialSource LLC", "MetalWorks Direct",
            "ComponentPro Solutions"
        ]
        
        self.material_categories = [
            "Raw Steel", "Aluminum Stock", "Fasteners", "Bearings",
            "Electronic Components", "Lubricants", "Tooling", "Packaging"
        ]
        
        self.failure_modes = [
            "Motor overheating", "Bearing failure", "Hydraulic leak",
            "Electrical fault", "Calibration drift", "Tool wear",
            "Sensor malfunction", "Software error", "Pneumatic failure"
        ]
        
        self.root_causes = [
            "Normal wear and tear", "Operator error", "Material defect",
            "Environmental conditions", "Insufficient maintenance",
            "Design flaw", "Power surge", "Contamination"
        ]
        
    def generate_machines(self, n: int = 15) -> pd.DataFrame:
        """Generate machine inventory data."""
        machines = []
        for i in range(n):
            machine_type = random.choice(list(self.machine_types.keys()))
            specs = self.machine_types[machine_type]
            
            install_date = self.base_date - timedelta(days=random.randint(365, 365*8))
            last_maint = self.base_date - timedelta(days=random.randint(7, 180))
            
            machines.append({
                "machine_id": f"M{str(i+1).zfill(3)}",
                "name": f"{machine_type} #{i+1}",
                "type": machine_type,
                "installation_date": install_date,
                "hourly_rate": specs["hourly_rate"] * (1 + random.uniform(-0.1, 0.1)),
                "status": random.choices(
                    ["operational", "maintenance", "offline"],
                    weights=[0.85, 0.10, 0.05]
                )[0],
                "last_maintenance": last_maint,
                "age_years": (self.base_date - install_date).days / 365
            })
        
        return pd.DataFrame(machines)
    
    def generate_suppliers(self, n: int = 10) -> pd.DataFrame:
        """Generate supplier master data."""
        suppliers = []
        for i in range(min(n, len(self.supplier_names))):
            suppliers.append({
                "supplier_id": f"S{str(i+1).zfill(3)}",
                "name": self.supplier_names[i],
                "category": random.choice(self.material_categories),
                "reliability_score": round(random.uniform(0.65, 0.99), 2),
                "avg_lead_time_days": random.randint(3, 21),
                "contract_value": round(random.uniform(50000, 500000), 2),
                "payment_terms": random.choice(["Net 30", "Net 45", "Net 60"]),
                "quality_rating": round(random.uniform(3.5, 5.0), 1)
            })
        
        return pd.DataFrame(suppliers)
    
    def generate_maintenance_logs(self, machines_df: pd.DataFrame, 
                                   n_events: int = 200) -> pd.DataFrame:
        """Generate maintenance event logs with realistic patterns."""
        events = []
        machine_ids = machines_df["machine_id"].tolist()
        
        for i in range(n_events):
            machine_id = random.choice(machine_ids)
            machine_row = machines_df[machines_df["machine_id"] == machine_id].iloc[0]
            machine_type = machine_row["type"]
            specs = self.machine_types[machine_type]
            
            event_type = random.choices(
                ["planned", "unplanned", "emergency"],
                weights=[0.5, 0.35, 0.15]
            )[0]
            
            event_date = self.base_date - timedelta(days=random.randint(0, 365))
            
            if event_type == "planned":
                duration = random.uniform(2, 8)
                cost_mult = 0.5
                description = f"Scheduled maintenance for {machine_type}"
            elif event_type == "unplanned":
                duration = random.uniform(4, 24)
                cost_mult = 1.0
                description = f"Unplanned repair: {random.choice(self.failure_modes)}"
            else:
                duration = random.uniform(8, 48)
                cost_mult = 1.5
                description = f"Emergency repair: Critical {random.choice(self.failure_modes)}"
            
            base_cost = random.uniform(*specs["repair_cost_range"])
            
            events.append({
                "event_id": f"ME{str(i+1).zfill(4)}",
                "machine_id": machine_id,
                "machine_name": machine_row["name"],
                "machine_type": machine_type,
                "event_type": event_type,
                "description": description,
                "start_time": event_date,
                "duration_hours": round(duration, 1),
                "cost": round(base_cost * cost_mult, 2),
                "labor_hours": round(duration * random.uniform(1, 2), 1),
                "parts_cost": round(base_cost * cost_mult * random.uniform(0.3, 0.6), 2),
                "root_cause": random.choice(self.root_causes) if event_type != "planned" else "N/A",
                "downtime_impact_hours": round(duration * random.uniform(1.2, 2.0), 1) if event_type != "planned" else 0
            })
        
        df = pd.DataFrame(events)
        return df.sort_values("start_time").reset_index(drop=True)
    
    def generate_production_data(self, machines_df: pd.DataFrame,
                                  n_runs: int = 500) -> pd.DataFrame:
        """Generate production run data."""
        runs = []
        production_machines = machines_df[
            machines_df["type"].isin(["CNC Mill", "Lathe", "Press", "Injection Molder"])
        ]["machine_id"].tolist()
        
        product_codes = [f"PART-{i:04d}" for i in range(1, 51)]
        operators = [f"OP{i:03d}" for i in range(1, 21)]
        
        for i in range(n_runs):
            machine_id = random.choice(production_machines)
            start = self.base_date - timedelta(
                days=random.randint(0, 365),
                hours=random.randint(0, 23)
            )
            duration_hours = random.uniform(2, 12)
            end = start + timedelta(hours=duration_hours)
            
            base_rate = random.randint(50, 200)  # units per hour
            units = int(base_rate * duration_hours * random.uniform(0.8, 1.1))
            defect_rate = random.uniform(0.005, 0.05)
            
            runs.append({
                "run_id": f"PR{str(i+1).zfill(5)}",
                "machine_id": machine_id,
                "product_code": random.choice(product_codes),
                "start_time": start,
                "end_time": end,
                "duration_hours": round(duration_hours, 2),
                "units_produced": units,
                "units_defective": int(units * defect_rate),
                "defect_rate": round(defect_rate * 100, 2),
                "cycle_time_seconds": round(3600 / base_rate, 1),
                "operator_id": random.choice(operators),
                "efficiency_pct": round(random.uniform(75, 98), 1),
                "scrap_cost": round(int(units * defect_rate) * random.uniform(5, 25), 2)
            })
        
        df = pd.DataFrame(runs)
        return df.sort_values("start_time").reset_index(drop=True)
    
    def generate_supplier_deliveries(self, suppliers_df: pd.DataFrame,
                                      n_deliveries: int = 300) -> pd.DataFrame:
        """Generate supplier delivery records with late delivery patterns."""
        deliveries = []
        
        for i in range(n_deliveries):
            supplier = suppliers_df.sample(1).iloc[0]
            
            expected = self.base_date - timedelta(days=random.randint(0, 365))
            
            # Late delivery probability based on reliability score
            if random.random() > supplier["reliability_score"]:
                delay = random.randint(1, 14)
            else:
                delay = random.randint(-2, 1)  # Could be early or on time
            
            actual = expected + timedelta(days=delay)
            
            deliveries.append({
                "delivery_id": f"DEL{str(i+1).zfill(5)}",
                "supplier_id": supplier["supplier_id"],
                "supplier_name": supplier["name"],
                "po_number": f"PO-{random.randint(10000, 99999)}",
                "expected_date": expected,
                "actual_date": actual,
                "days_late": max(0, delay),
                "on_time": delay <= 0,
                "material_code": f"MAT-{random.randint(100, 999)}",
                "material_category": supplier["category"],
                "quantity": random.randint(100, 10000),
                "unit_cost": round(random.uniform(0.5, 50), 2),
                "total_cost": 0,  # Calculated below
                "quality_score": round(random.uniform(
                    max(0.7, supplier["quality_rating"] - 1),
                    min(5.0, supplier["quality_rating"] + 0.5)
                ), 1)
            })
        
        df = pd.DataFrame(deliveries)
        df["total_cost"] = (df["quantity"] * df["unit_cost"]).round(2)
        return df.sort_values("expected_date").reset_index(drop=True)
    
    def generate_financial_summary(self, maintenance_df: pd.DataFrame,
                                    production_df: pd.DataFrame,
                                    deliveries_df: pd.DataFrame) -> pd.DataFrame:
        """Generate monthly financial summaries."""
        months = pd.date_range(self.base_date - timedelta(days=365), 
                               self.base_date, freq='M')
        
        summaries = []
        for month in months:
            month_start = month.replace(day=1)
            month_end = month
            
            # Maintenance costs
            maint_mask = (maintenance_df["start_time"] >= month_start) & \
                        (maintenance_df["start_time"] <= month_end)
            maint_cost = maintenance_df[maint_mask]["cost"].sum()
            
            # Production metrics
            prod_mask = (production_df["start_time"] >= month_start) & \
                       (production_df["start_time"] <= month_end)
            prod_data = production_df[prod_mask]
            units = prod_data["units_produced"].sum()
            scrap_cost = prod_data["scrap_cost"].sum()
            
            # Supplier costs
            del_mask = (deliveries_df["expected_date"] >= month_start) & \
                      (deliveries_df["expected_date"] <= month_end)
            material_cost = deliveries_df[del_mask]["total_cost"].sum()
            late_deliveries = deliveries_df[del_mask]["days_late"].sum()
            
            # Estimated revenue and labor
            revenue = units * random.uniform(45, 65)
            labor_cost = random.uniform(80000, 120000)
            overhead = random.uniform(30000, 50000)
            
            summaries.append({
                "month": month,
                "revenue": round(revenue, 2),
                "material_cost": round(material_cost, 2),
                "labor_cost": round(labor_cost, 2),
                "maintenance_cost": round(maint_cost, 2),
                "scrap_cost": round(scrap_cost, 2),
                "overhead": round(overhead, 2),
                "units_produced": units,
                "late_delivery_days": late_deliveries,
                "gross_profit": 0,  # Calculated below
                "profit_margin": 0
            })
        
        df = pd.DataFrame(summaries)
        df["total_cost"] = df["material_cost"] + df["labor_cost"] + \
                          df["maintenance_cost"] + df["scrap_cost"] + df["overhead"]
        df["gross_profit"] = df["revenue"] - df["total_cost"]
        df["profit_margin"] = (df["gross_profit"] / df["revenue"] * 100).round(2)
        
        return df
    
    def generate_all_data(self) -> Dict[str, pd.DataFrame]:
        """Generate complete dataset for LeanNLP."""
        print("Generating synthetic manufacturing data...")
        
        machines = self.generate_machines(15)
        print(f"  ✓ Generated {len(machines)} machines")
        
        suppliers = self.generate_suppliers(10)
        print(f"  ✓ Generated {len(suppliers)} suppliers")
        
        maintenance = self.generate_maintenance_logs(machines, 200)
        print(f"  ✓ Generated {len(maintenance)} maintenance events")
        
        production = self.generate_production_data(machines, 500)
        print(f"  ✓ Generated {len(production)} production runs")
        
        deliveries = self.generate_supplier_deliveries(suppliers, 300)
        print(f"  ✓ Generated {len(deliveries)} supplier deliveries")
        
        financials = self.generate_financial_summary(maintenance, production, deliveries)
        print(f"  ✓ Generated {len(financials)} monthly financial summaries")
        
        return {
            "machines": machines,
            "suppliers": suppliers,
            "maintenance": maintenance,
            "production": production,
            "deliveries": deliveries,
            "financials": financials
        }
    
    def save_to_csv(self, data: Dict[str, pd.DataFrame], output_dir: str = "data"):
        """Save all generated data to CSV files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in data.items():
            filepath = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(filepath, index=False)
            print(f"  Saved {filepath}")


if __name__ == "__main__":
    generator = SyntheticDataGenerator(seed=42)
    data = generator.generate_all_data()
    generator.save_to_csv(data)
    print("\nData generation complete!")
