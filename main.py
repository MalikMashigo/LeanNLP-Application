#!/usr/bin/env python3
"""
LeanNLP: Manufacturing Analytics Application
Main entry point for running the complete system.

Usage:
    python main.py --demo          Run demo analysis with synthetic data
    python main.py --dashboard     Launch Streamlit dashboard
    python main.py --generate      Generate synthetic data files
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_generator import SyntheticDataGenerator
from models.nlp_pipeline import InsightExtractor, NaturalLanguageGenerator
from models.knowledge_graph import ManufacturingKnowledgeGraph
from models.predictive_analytics import PredictiveAnalyticsEngine


def run_demo():
    """Run a complete demo analysis."""
    print("=" * 60)
    print("LeanNLP Manufacturing Analytics - Demo")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n[1/5] Generating synthetic manufacturing data...")
    generator = SyntheticDataGenerator(seed=42)
    data = generator.generate_all_data()
    
    # Build knowledge graph
    print("\n[2/5] Building knowledge graph...")
    kg = ManufacturingKnowledgeGraph()
    kg.build_from_dataframes(data)
    
    stats = kg.get_graph_statistics()
    print(f"  Graph contains {stats['total_nodes']} entities and {stats['total_edges']} relationships")
    
    # Extract insights
    print("\n[3/5] Extracting insights from data...")
    extractor = InsightExtractor()
    
    all_insights = []
    all_insights.extend(extractor.analyze_maintenance_patterns(data["maintenance"]))
    all_insights.extend(extractor.analyze_supplier_performance(data["deliveries"]))
    all_insights.extend(extractor.analyze_production_efficiency(data["production"]))
    all_insights.extend(extractor.analyze_financial_trends(data["financials"]))
    
    high_priority = [i for i in all_insights if i.severity in ["high", "critical"]]
    print(f"  Found {len(all_insights)} insights ({len(high_priority)} high priority)")
    
    # Train predictive models
    print("\n[4/5] Training predictive models...")
    engine = PredictiveAnalyticsEngine()
    engine.train_all_models(data)
    
    # Generate forecasts
    print("\n[5/5] Generating forecasts and recommendations...")
    forecasts = engine.generate_comprehensive_forecast(data, horizon_months=6)
    recommendations = extractor.generate_recommendations(all_insights)
    
    # Print results
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    
    # Key insights
    print("\n[INFO] KEY INSIGHTS:")
    print("-" * 40)
    for i, insight in enumerate(high_priority[:5], 1):
        print(f"\n{i}. [{insight.severity.upper()}] {insight.category}")
        print(f"   {insight.description}")
        if insight.recommendation:
            print(f"   → Recommendation: {insight.recommendation}")
    
    # Risk assessment
    print("\n[WARN]  RISK ASSESSMENT:")
    print("-" * 40)
    risks = kg.find_risk_patterns()
    for risk in risks[:5]:
        print(f"  [{risk['severity'].upper()}] {risk['type']}: {risk['details']}")
    
    # Machine risks from predictive models
    if "machine_risks" in forecasts:
        print("\n[MAINT] MACHINE FAILURE RISKS (Next 6 Months):")
        print("-" * 40)
        for machine in forecasts["machine_risks"][:5]:
            prob = machine["failure_probability"]
            level = machine["risk_level"]
            print(f"  {machine['machine_id']}: {prob:.1%} failure probability ({level} risk)")
    
    # Recommendations
    print("\n[REC] TOP RECOMMENDATIONS:")
    print("-" * 40)
    total_savings = 0
    for rec in recommendations[:5]:
        savings = rec.get("estimated_savings", 0)
        if savings:
            total_savings += savings
            print(f"\n{rec['priority']}. {rec['recommendation']}")
            print(f"   Est. savings: ${savings:,.2f}")
        else:
            print(f"\n{rec['priority']}. {rec['recommendation']}")
    
    print(f"\n[TREND] TOTAL POTENTIAL SAVINGS: ${total_savings:,.2f}")
    
    # Financial forecast summary
    print("\n[TREND] FINANCIAL FORECAST (6-Month Trends):")
    print("-" * 40)
    for metric in ["maintenance_cost", "revenue", "profit_margin"]:
        if metric in forecasts:
            trend = forecasts[metric]["trend"]
            print(f"  {metric.replace('_', ' ').title()}: {trend}")
    
    # Efficiency opportunities
    if "efficiency_opportunities" in forecasts:
        print("\n[EFF]  EFFICIENCY OPPORTUNITIES:")
        print("-" * 40)
        for opp in forecasts["efficiency_opportunities"]:
            print(f"  • {opp['recommendation']}")
            print(f"    Potential improvement: {opp['potential_improvement']:.1f}%")
    
    print("\n" + "=" * 60)
    print("Demo complete! Run 'python main.py --dashboard' to launch the interactive dashboard.")
    print("=" * 60)


def generate_data_files():
    """Generate and save synthetic data to CSV files."""
    print("Generating synthetic data files...")
    generator = SyntheticDataGenerator(seed=42)
    data = generator.generate_all_data()
    generator.save_to_csv(data, "data")
    print("\nData files saved to ./data/")


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print("Launching LeanNLP Dashboard...")
    print("Open your browser to http://localhost:8501")
    os.system("streamlit run app.py --server.headless true")


def main():
    parser = argparse.ArgumentParser(
        description="LeanNLP Manufacturing Analytics Application"
    )
    parser.add_argument(
        "--demo", 
        action="store_true",
        help="Run demo analysis with synthetic data"
    )
    parser.add_argument(
        "--dashboard",
        action="store_true", 
        help="Launch Streamlit dashboard"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate synthetic data files"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    elif args.dashboard:
        launch_dashboard()
    elif args.generate:
        generate_data_files()
    else:
        # Default to demo
        run_demo()


if __name__ == "__main__":
    main()
