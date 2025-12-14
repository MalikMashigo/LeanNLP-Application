"""
Generate realistic NASA CMAPSS-format data for demo purposes.
This creates data files that match the exact format of the real NASA dataset.

NASA CMAPSS Dataset Format:
- Space-separated values
- Columns: unit_id, cycle, op1, op2, op3, sensor1-21 (26 columns total)
- Training: Run-to-failure data
- Test: Partial data (ends before failure)
- RUL: Remaining cycles for test data
"""

import numpy as np
import pandas as pd
import os

def generate_cmapss_training_data(n_units=100, seed=42):
    """
    Generate training data matching NASA CMAPSS FD001 format.
    Each unit runs until failure with degrading sensor readings.
    """
    np.random.seed(seed)
    
    all_data = []
    
    for unit in range(1, n_units + 1):
        # Random lifetime between 128 and 362 cycles (matching real FD001 distribution)
        max_cycles = np.random.randint(128, 362)
        
        for cycle in range(1, max_cycles + 1):
            # Operational settings (matching FD001 which has single operating condition)
            op1 = -0.0007 + np.random.normal(0, 0.0001)
            op2 = -0.0004 + np.random.normal(0, 0.0001)
            op3 = 100.0 + np.random.normal(0, 0.01)
            
            # Degradation factor (0 at start, 1 at failure)
            degradation = cycle / max_cycles
            
            # Sensor readings with realistic degradation patterns
            # Based on actual CMAPSS sensor behavior
            sensors = []
            
            # Sensor 1: Total temp at fan inlet - relatively constant
            sensors.append(518.67 + np.random.normal(0, 0.1))
            
            # Sensor 2: Total temp at LPC outlet - increases with degradation
            sensors.append(642.15 + degradation * 10 + np.random.normal(0, 0.5))
            
            # Sensor 3: Total temp at HPC outlet - increases with degradation
            sensors.append(1580.87 + degradation * 30 + np.random.normal(0, 2))
            
            # Sensor 4: Total temp at LPT outlet - increases with degradation
            sensors.append(1406.59 + degradation * 20 + np.random.normal(0, 2))
            
            # Sensor 5: Pressure at fan inlet - constant
            sensors.append(14.62 + np.random.normal(0, 0.001))
            
            # Sensor 6: Total pressure in bypass-duct - slight variation
            sensors.append(21.61 + np.random.normal(0, 0.01))
            
            # Sensor 7: Total pressure at HPC outlet - decreases with degradation
            sensors.append(553.36 - degradation * 20 + np.random.normal(0, 1))
            
            # Sensor 8: Physical fan speed - relatively constant
            sensors.append(2388.01 + np.random.normal(0, 0.5))
            
            # Sensor 9: Physical core speed - increases slightly
            sensors.append(9050.17 + degradation * 50 + np.random.normal(0, 5))
            
            # Sensor 10: Engine pressure ratio - constant
            sensors.append(1.3 + np.random.normal(0, 0.001))
            
            # Sensor 11: Static pressure at HPC outlet - decreases
            sensors.append(47.20 - degradation * 3 + np.random.normal(0, 0.2))
            
            # Sensor 12: Fuel flow ratio - increases with degradation
            sensors.append(520.72 + degradation * 30 + np.random.normal(0, 1))
            
            # Sensor 13: Corrected fan speed - relatively constant
            sensors.append(2388.03 + np.random.normal(0, 0.5))
            
            # Sensor 14: Corrected core speed - increases
            sensors.append(8131.49 + degradation * 40 + np.random.normal(0, 3))
            
            # Sensor 15: Bypass ratio - decreases with degradation
            sensors.append(8.4052 - degradation * 0.3 + np.random.normal(0, 0.01))
            
            # Sensor 16: Burner fuel-air ratio - constant
            sensors.append(0.03 + np.random.normal(0, 0.0001))
            
            # Sensor 17: Bleed enthalpy - increases with degradation
            sensors.append(392.0 + degradation * 20 + np.random.normal(0, 1))
            
            # Sensor 18: Demanded fan speed - constant
            sensors.append(2388.0 + np.random.normal(0, 0.01))
            
            # Sensor 19: Demanded corrected fan speed - constant
            sensors.append(100.0 + np.random.normal(0, 0.01))
            
            # Sensor 20: HPT coolant bleed - increases with degradation
            sensors.append(38.86 + degradation * 2 + np.random.normal(0, 0.1))
            
            # Sensor 21: LPT coolant bleed - increases with degradation
            sensors.append(23.3190 + degradation * 1.5 + np.random.normal(0, 0.05))
            
            row = [unit, cycle, op1, op2, op3] + sensors
            all_data.append(row)
    
    return np.array(all_data)


def generate_cmapss_test_data(n_units=100, seed=43):
    """
    Generate test data - partial trajectories that end before failure.
    """
    np.random.seed(seed)
    
    all_data = []
    rul_values = []
    
    for unit in range(1, n_units + 1):
        # Total lifetime
        max_cycles = np.random.randint(128, 362)
        
        # Test data ends at random point before failure
        test_end = np.random.randint(max(31, max_cycles - 150), max_cycles - 10)
        rul = max_cycles - test_end
        rul_values.append(rul)
        
        for cycle in range(1, test_end + 1):
            op1 = -0.0007 + np.random.normal(0, 0.0001)
            op2 = -0.0004 + np.random.normal(0, 0.0001)
            op3 = 100.0 + np.random.normal(0, 0.01)
            
            # Use same degradation pattern based on true max_cycles
            degradation = cycle / max_cycles
            
            sensors = []
            sensors.append(518.67 + np.random.normal(0, 0.1))
            sensors.append(642.15 + degradation * 10 + np.random.normal(0, 0.5))
            sensors.append(1580.87 + degradation * 30 + np.random.normal(0, 2))
            sensors.append(1406.59 + degradation * 20 + np.random.normal(0, 2))
            sensors.append(14.62 + np.random.normal(0, 0.001))
            sensors.append(21.61 + np.random.normal(0, 0.01))
            sensors.append(553.36 - degradation * 20 + np.random.normal(0, 1))
            sensors.append(2388.01 + np.random.normal(0, 0.5))
            sensors.append(9050.17 + degradation * 50 + np.random.normal(0, 5))
            sensors.append(1.3 + np.random.normal(0, 0.001))
            sensors.append(47.20 - degradation * 3 + np.random.normal(0, 0.2))
            sensors.append(520.72 + degradation * 30 + np.random.normal(0, 1))
            sensors.append(2388.03 + np.random.normal(0, 0.5))
            sensors.append(8131.49 + degradation * 40 + np.random.normal(0, 3))
            sensors.append(8.4052 - degradation * 0.3 + np.random.normal(0, 0.01))
            sensors.append(0.03 + np.random.normal(0, 0.0001))
            sensors.append(392.0 + degradation * 20 + np.random.normal(0, 1))
            sensors.append(2388.0 + np.random.normal(0, 0.01))
            sensors.append(100.0 + np.random.normal(0, 0.01))
            sensors.append(38.86 + degradation * 2 + np.random.normal(0, 0.1))
            sensors.append(23.3190 + degradation * 1.5 + np.random.normal(0, 0.05))
            
            row = [unit, cycle, op1, op2, op3] + sensors
            all_data.append(row)
    
    return np.array(all_data), np.array(rul_values)


def save_cmapss_format(data, filepath):
    """Save data in NASA CMAPSS format (space-separated, no header)."""
    # Format: each value with appropriate precision
    with open(filepath, 'w') as f:
        for row in data:
            # Unit and cycle as integers, rest as floats
            formatted = []
            formatted.append(f"{int(row[0]):3d}")  # unit
            formatted.append(f"{int(row[1]):3d}")  # cycle
            for val in row[2:]:
                formatted.append(f"{val:12.4f}")
            f.write(" ".join(formatted) + "\n")


def save_rul_file(rul_values, filepath):
    """Save RUL values (one per line)."""
    with open(filepath, 'w') as f:
        for rul in rul_values:
            f.write(f"{int(rul)}\n")


def generate_maintenance_logs(n_records=200, seed=42):
    """Generate realistic maintenance log data with NLP-relevant text."""
    np.random.seed(seed)
    
    machines = [f"M{i:03d}" for i in range(1, 16)]
    machine_types = ["CNC Mill", "Lathe", "Press", "Welding Robot", "Assembly Line", "Injection Molder"]
    
    event_types = ["planned", "unplanned", "emergency"]
    event_weights = [0.5, 0.35, 0.15]
    
    failure_descriptions = {
        "motor": [
            "Motor overheating detected during operation. Temperature exceeded threshold by 15 degrees.",
            "Motor bearing noise increasing. Vibration analysis shows wear pattern.",
            "Motor failed to start. Electrical diagnostics indicate winding issue.",
            "Motor running hot. Cooling system inspection required.",
            "Motor tripped on overload protection during high-demand cycle."
        ],
        "hydraulic": [
            "Hydraulic leak found at main cylinder seal. Pressure dropping gradually.",
            "Hydraulic system pressure unstable. Pump showing signs of wear.",
            "Hydraulic fluid contamination detected. Filter bypass triggered.",
            "Hydraulic line burst near actuator. Emergency shutdown initiated.",
            "Hydraulic reservoir level low. Possible internal leak in circuit."
        ],
        "bearing": [
            "Bearing failure on main spindle. Metal particles in lubricant.",
            "Bearing noise detected during startup. Replacement recommended.",
            "Bearing temperature elevated. Lubrication schedule accelerated.",
            "Bearing seized on drive shaft. Production halted for repair.",
            "Bearing wear pattern abnormal. Misalignment suspected."
        ],
        "electrical": [
            "Electrical fault in control panel. Circuit breaker tripped repeatedly.",
            "Electrical short detected in sensor wiring. Intermittent signal loss.",
            "Electrical connector corrosion causing connection issues.",
            "Power supply unit failing. Voltage fluctuations recorded.",
            "Electrical ground fault detected. Safety system activated."
        ],
        "calibration": [
            "Calibration drift detected on position sensor. Accuracy below spec.",
            "Tool calibration required after collision incident.",
            "Calibration offset accumulating over time. Adjustment performed.",
            "Calibration verification failed on quality check.",
            "Dimensional accuracy degraded. Full recalibration scheduled."
        ],
        "software": [
            "Software error causing unexpected machine behavior. Firmware update applied.",
            "PLC communication timeout with HMI. Network diagnostics performed.",
            "Software crash during cycle. Memory leak identified.",
            "Control software showing erratic behavior after power fluctuation.",
            "Software update required to address known bug in motion control."
        ]
    }
    
    root_causes = [
        "Normal wear and tear",
        "Operator error during setup",
        "Material defect in replacement part",
        "Environmental conditions (humidity/temperature)",
        "Insufficient preventive maintenance",
        "Design limitation under high load",
        "Power quality issue from grid",
        "Contamination from adjacent process"
    ]
    
    records = []
    base_date = pd.Timestamp("2024-01-01")
    
    for i in range(n_records):
        machine = np.random.choice(machines)
        machine_idx = int(machine[1:]) - 1
        machine_type = machine_types[machine_idx % len(machine_types)]
        
        event_type = np.random.choice(event_types, p=event_weights)
        
        if event_type == "planned":
            description = f"Scheduled preventive maintenance for {machine_type}. Routine inspection and lubrication."
            root_cause = "N/A"
            duration = np.random.uniform(2, 6)
            cost_mult = 0.5
        else:
            failure_type = np.random.choice(list(failure_descriptions.keys()))
            description = np.random.choice(failure_descriptions[failure_type])
            root_cause = np.random.choice(root_causes)
            duration = np.random.uniform(4, 24) if event_type == "unplanned" else np.random.uniform(8, 48)
            cost_mult = 1.0 if event_type == "unplanned" else 1.5
        
        base_cost = np.random.uniform(500, 5000)
        cost = base_cost * cost_mult
        
        event_date = base_date + pd.Timedelta(days=np.random.randint(0, 365))
        
        records.append({
            "event_id": f"ME{i+1:04d}",
            "machine_id": machine,
            "machine_type": machine_type,
            "event_type": event_type,
            "description": description,
            "start_time": event_date.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_hours": round(duration, 1),
            "cost": round(cost, 2),
            "labor_hours": round(duration * np.random.uniform(1, 2), 1),
            "parts_cost": round(cost * np.random.uniform(0.3, 0.6), 2),
            "root_cause": root_cause,
            "downtime_impact_hours": round(duration * np.random.uniform(1.2, 2.0), 1) if event_type != "planned" else 0
        })
    
    return pd.DataFrame(records)


def generate_supplier_data(n_suppliers=10, n_deliveries=300, seed=42):
    """Generate supplier and delivery data."""
    np.random.seed(seed)
    
    supplier_names = [
        "SteelCorp Industries", "FastParts Supply", "PrecisionMetal Co",
        "GlobalComponents Ltd", "QualityAlloys Inc", "TechMaterials Group",
        "ReliableSupply Partners", "IndustrialSource LLC", "MetalWorks Direct",
        "ComponentPro Solutions"
    ]
    
    categories = ["Raw Steel", "Aluminum Stock", "Fasteners", "Bearings",
                  "Electronic Components", "Lubricants", "Tooling", "Packaging"]
    
    # Generate suppliers
    suppliers = []
    for i in range(min(n_suppliers, len(supplier_names))):
        reliability = np.random.uniform(0.65, 0.98)
        suppliers.append({
            "supplier_id": f"S{i+1:03d}",
            "name": supplier_names[i],
            "category": np.random.choice(categories),
            "reliability_score": round(reliability, 2),
            "avg_lead_time_days": np.random.randint(3, 21),
            "contract_value": round(np.random.uniform(50000, 500000), 2),
            "payment_terms": np.random.choice(["Net 30", "Net 45", "Net 60"]),
            "quality_rating": round(np.random.uniform(3.5, 5.0), 1)
        })
    
    suppliers_df = pd.DataFrame(suppliers)
    
    # Generate deliveries
    base_date = pd.Timestamp("2024-01-01")
    deliveries = []
    
    for i in range(n_deliveries):
        supplier = suppliers_df.sample(1).iloc[0]
        
        expected_date = base_date + pd.Timedelta(days=np.random.randint(0, 365))
        
        # Late delivery based on reliability
        if np.random.random() > supplier["reliability_score"]:
            delay = np.random.randint(1, 14)
        else:
            delay = np.random.randint(-2, 1)
        
        actual_date = expected_date + pd.Timedelta(days=delay)
        quantity = np.random.randint(100, 10000)
        unit_cost = round(np.random.uniform(0.5, 50), 2)
        
        deliveries.append({
            "delivery_id": f"DEL{i+1:05d}",
            "supplier_id": supplier["supplier_id"],
            "supplier_name": supplier["name"],
            "po_number": f"PO-{np.random.randint(10000, 99999)}",
            "expected_date": expected_date.strftime("%Y-%m-%d"),
            "actual_date": actual_date.strftime("%Y-%m-%d"),
            "days_late": max(0, delay),
            "on_time": delay <= 0,
            "material_code": f"MAT-{np.random.randint(100, 999)}",
            "material_category": supplier["category"],
            "quantity": quantity,
            "unit_cost": unit_cost,
            "total_cost": round(quantity * unit_cost, 2),
            "quality_score": round(np.random.uniform(
                max(0.7, supplier["quality_rating"] - 1),
                min(5.0, supplier["quality_rating"] + 0.5)
            ), 1)
        })
    
    return suppliers_df, pd.DataFrame(deliveries)


def generate_production_data(n_runs=500, seed=42):
    """Generate production run data."""
    np.random.seed(seed)
    
    machines = [f"M{i:03d}" for i in range(1, 16)]
    production_machines = machines[:8]  # First 8 are production machines
    products = [f"PART-{i:04d}" for i in range(1, 51)]
    operators = [f"OP{i:03d}" for i in range(1, 21)]
    
    base_date = pd.Timestamp("2024-01-01")
    runs = []
    
    for i in range(n_runs):
        machine = np.random.choice(production_machines)
        start = base_date + pd.Timedelta(
            days=np.random.randint(0, 365),
            hours=np.random.randint(0, 23)
        )
        duration = np.random.uniform(2, 12)
        end = start + pd.Timedelta(hours=duration)
        
        base_rate = np.random.randint(50, 200)
        units = int(base_rate * duration * np.random.uniform(0.8, 1.1))
        defect_rate = np.random.uniform(0.005, 0.05)
        defects = int(units * defect_rate)
        
        runs.append({
            "run_id": f"PR{i+1:05d}",
            "machine_id": machine,
            "product_code": np.random.choice(products),
            "start_time": start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_hours": round(duration, 2),
            "units_produced": units,
            "units_defective": defects,
            "defect_rate": round(defect_rate * 100, 2),
            "cycle_time_seconds": round(3600 / base_rate, 1),
            "operator_id": np.random.choice(operators),
            "efficiency_pct": round(np.random.uniform(75, 98), 1),
            "scrap_cost": round(defects * np.random.uniform(5, 25), 2)
        })
    
    return pd.DataFrame(runs)


if __name__ == "__main__":
    # Use relative path (same directory as script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "demo_data")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating NASA CMAPSS-format turbofan data...")
    
    # Generate CMAPSS data
    train_data = generate_cmapss_training_data(n_units=100)
    save_cmapss_format(train_data, f"{output_dir}/train_FD001.txt")
    print(f"  Created train_FD001.txt: {len(train_data)} records, 100 units")
    
    test_data, rul_values = generate_cmapss_test_data(n_units=100)
    save_cmapss_format(test_data, f"{output_dir}/test_FD001.txt")
    save_rul_file(rul_values, f"{output_dir}/RUL_FD001.txt")
    print(f"  Created test_FD001.txt: {len(test_data)} records, 100 units")
    print(f"  Created RUL_FD001.txt: {len(rul_values)} RUL values")
    
    print("\nGenerating manufacturing data...")
    
    # Maintenance logs
    maintenance_df = generate_maintenance_logs(n_records=200)
    maintenance_df.to_csv(f"{output_dir}/maintenance_logs.csv", index=False)
    print(f"  Created maintenance_logs.csv: {len(maintenance_df)} records")
    
    # Supplier data
    suppliers_df, deliveries_df = generate_supplier_data()
    suppliers_df.to_csv(f"{output_dir}/suppliers.csv", index=False)
    deliveries_df.to_csv(f"{output_dir}/deliveries.csv", index=False)
    print(f"  Created suppliers.csv: {len(suppliers_df)} suppliers")
    print(f"  Created deliveries.csv: {len(deliveries_df)} deliveries")
    
    # Production data
    production_df = generate_production_data()
    production_df.to_csv(f"{output_dir}/production_runs.csv", index=False)
    print(f"  Created production_runs.csv: {len(production_df)} runs")
    
    print("\nDemo data generation complete!")
    print(f"Files saved to: {output_dir}")
