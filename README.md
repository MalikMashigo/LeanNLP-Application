# LeanNLP Live Demo Package

This folder contains everything you need for a live demonstration of the LeanNLP manufacturing analytics system.

## Quick Start

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn streamlit plotly

# 2. Run the web dashboard
streamlit run app.py
```

This will open the interactive dashboard in your browser at http://localhost:8501

**Important:** You need to train models before using predictions. Go to "Data Upload" > "TRAIN MODELS" tab and click "TRAIN ALL AVAILABLE MODELS".

## Alternative: Command Line Demo

```bash
# Generate demo data (if needed)
python generate_demo_data.py

# Train models on the data
python train_models.py

# Run interactive CLI demo
python run_demo.py
```

## Files

| File | Description |
|------|-------------|
| `app.py` | **Streamlit web dashboard** (main interface) |
| `generate_demo_data.py` | Creates realistic manufacturing and sensor data |
| `train_models.py` | Trains ML models on the generated data |
| `run_demo.py` | Interactive CLI demo script for presentations |

## Demo Data (demo_data/)

| File | Records | Description |
|------|---------|-------------|
| `train_FD001.txt` | 24,068 | NASA CMAPSS format turbofan training data (100 engines) |
| `test_FD001.txt` | 17,202 | Partial engine trajectories for prediction |
| `RUL_FD001.txt` | 100 | True remaining useful life for test engines |
| `maintenance_logs.csv` | 200 | Maintenance records with NLP-rich descriptions |
| `suppliers.csv` | 10 | Supplier master data |
| `deliveries.csv` | 300 | Delivery records with on-time metrics |
| `production_runs.csv` | 500 | Production run data |

## Trained Models (trained_models/)

Models are trained on your machine to ensure compatibility. After training:

| Model | Task | Expected Performance |
|-------|------|-------------|
| `rul_model.pkl` | RUL Prediction | MAE ~8-12, R2 ~0.85-0.92 |
| `cost_model.pkl` | Cost Prediction | MAE ~$1,000-1,500 |

## Demo Walkthrough

The `run_demo.py` script demonstrates:

1. **RUL Prediction**: Predicts remaining useful life for turbofan engines
   - Shows predictions vs actual for 10 engines
   - Highlights critical engines (RUL < 30 cycles)

2. **NLP Entity Extraction**: Extracts entities from maintenance logs
   - Machine IDs, costs, temperatures
   - Failure type classification (motor, bearing, hydraulic, etc.)

3. **Cost Analysis**: Breaks down maintenance costs by event type
   - Planned vs unplanned vs emergency

4. **Supplier Analysis**: Evaluates supplier performance
   - On-time delivery rates
   - Quality scores
   - Risk identification

5. **Recommendations**: AI-generated actionable insights

## Data Format

### NASA CMAPSS Format (train_FD001.txt, test_FD001.txt)

Space-separated values, 26 columns:
```
unit_id  cycle  op1  op2  op3  sensor_1 ... sensor_21
```

Example:
```
  1   1    -0.0007    -0.0004   100.0000     518.6700   642.1500 ...
```

### Maintenance Logs Format

CSV with columns:
- `event_id`: Unique identifier
- `machine_id`: Machine (M001-M015)
- `event_type`: planned/unplanned/emergency
- `description`: Natural language description (for NLP)
- `cost`: Repair cost
- `root_cause`: Identified cause

## For Your Presentation

1. Open terminal in this directory
2. Run `python run_demo.py`
3. Press Enter to advance through each section
4. Key talking points at each step:
   - RUL: "Our model predicts engine failure within 8 cycles on average"
   - NLP: "We extract failure types from unstructured maintenance text"
   - Suppliers: "The system identifies at-risk suppliers automatically"

## Troubleshooting

**"Models not found"**: Run `python train_models.py` first

**"Demo data not found"**: Run `python generate_demo_data.py` first

**Import errors**: Install dependencies with `pip install pandas numpy scikit-learn`
