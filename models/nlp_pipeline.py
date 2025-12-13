"""
LeanNLP: Natural Language Processing Pipeline
Processes maintenance logs, financial documents, and operational reports.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import Counter, defaultdict
from datetime import datetime


@dataclass
class Entity:
    """Represents an extracted entity from text."""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class Insight:
    """Represents an extracted insight or pattern."""
    category: str
    description: str
    severity: str  # low, medium, high, critical
    confidence: float
    supporting_data: Dict[str, Any]
    recommendation: Optional[str] = None


class ManufacturingNLP:
    """NLP pipeline optimized for manufacturing domain."""
    
    def __init__(self):
        # Domain-specific patterns
        self.failure_patterns = {
            r"motor\s+(overheating|failure|burnout)": "motor_failure",
            r"bearing\s+(failure|wear|noise)": "bearing_issue",
            r"hydraulic\s+(leak|failure|pressure)": "hydraulic_issue",
            r"electrical\s+(fault|short|failure)": "electrical_issue",
            r"calibration\s+(drift|error|needed)": "calibration_issue",
            r"tool\s+(wear|breakage|damage)": "tooling_issue",
            r"sensor\s+(malfunction|failure|error)": "sensor_issue",
            r"pneumatic\s+(failure|leak|pressure)": "pneumatic_issue",
            r"software\s+(error|crash|bug)": "software_issue",
        }
        
        self.cost_keywords = [
            "cost", "expense", "spending", "budget", "invoice",
            "payment", "price", "rate", "fee", "charge"
        ]
        
        self.efficiency_keywords = [
            "efficiency", "throughput", "productivity", "output",
            "performance", "utilization", "capacity", "yield"
        ]
        
        self.risk_keywords = [
            "risk", "failure", "delay", "issue", "problem",
            "critical", "urgent", "emergency", "warning"
        ]
        
        # Severity indicators
        self.severity_patterns = {
            "critical": ["critical", "emergency", "immediate", "severe", "catastrophic"],
            "high": ["urgent", "serious", "significant", "major", "important"],
            "medium": ["moderate", "notable", "concerning", "attention"],
            "low": ["minor", "routine", "scheduled", "normal", "planned"]
        }
        
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract manufacturing-domain entities from text."""
        entities = []
        text_lower = text.lower()
        
        # Extract machine references
        machine_pattern = r'\b(machine|unit|equipment)\s*[#]?\s*(\d+)\b'
        for match in re.finditer(machine_pattern, text_lower):
            entities.append(Entity(
                text=match.group(0),
                label="MACHINE",
                start=match.start(),
                end=match.end()
            ))
        
        # Extract machine IDs (M001 format)
        machine_id_pattern = r'\bM\d{3}\b'
        for match in re.finditer(machine_id_pattern, text, re.IGNORECASE):
            entities.append(Entity(
                text=match.group(0),
                label="MACHINE_ID",
                start=match.start(),
                end=match.end()
            ))
        
        # Extract monetary values
        money_pattern = r'\$[\d,]+(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars|USD)'
        for match in re.finditer(money_pattern, text):
            entities.append(Entity(
                text=match.group(0),
                label="MONEY",
                start=match.start(),
                end=match.end()
            ))
        
        # Extract percentages
        pct_pattern = r'\d+(?:\.\d+)?%'
        for match in re.finditer(pct_pattern, text):
            entities.append(Entity(
                text=match.group(0),
                label="PERCENTAGE",
                start=match.start(),
                end=match.end()
            ))
        
        # Extract time durations
        time_pattern = r'\d+(?:\.\d+)?\s*(?:hours?|hrs?|minutes?|mins?|days?|weeks?|months?)'
        for match in re.finditer(time_pattern, text_lower):
            entities.append(Entity(
                text=match.group(0),
                label="DURATION",
                start=match.start(),
                end=match.end()
            ))
        
        # Extract dates
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{4}-\d{2}-\d{2}',
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}'
        ]
        for pattern in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    text=match.group(0),
                    label="DATE",
                    start=match.start(),
                    end=match.end()
                ))
        
        # Extract supplier references
        supplier_pattern = r'\b(?:supplier|vendor)\s+[A-Z][a-zA-Z\s]+(?:Inc|LLC|Ltd|Co|Corp)?\.?'
        for match in re.finditer(supplier_pattern, text):
            entities.append(Entity(
                text=match.group(0),
                label="SUPPLIER",
                start=match.start(),
                end=match.end()
            ))
        
        return entities
    
    def classify_failure_mode(self, text: str) -> Optional[Tuple[str, float]]:
        """Classify the failure mode from maintenance description."""
        text_lower = text.lower()
        
        for pattern, failure_type in self.failure_patterns.items():
            if re.search(pattern, text_lower):
                return (failure_type, 0.85)
        
        return None
    
    def assess_severity(self, text: str) -> str:
        """Assess severity level from text."""
        text_lower = text.lower()
        
        for severity, keywords in self.severity_patterns.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return severity
        
        return "low"
    
    def extract_numeric_values(self, text: str) -> Dict[str, List[float]]:
        """Extract and categorize numeric values from text."""
        values = defaultdict(list)
        
        # Extract costs
        cost_pattern = r'\$?([\d,]+(?:\.\d{2})?)\s*(?:dollars|USD)?'
        for match in re.finditer(cost_pattern, text):
            try:
                val = float(match.group(1).replace(',', ''))
                if val > 0:
                    values["costs"].append(val)
            except ValueError:
                pass
        
        # Extract hours
        hours_pattern = r'([\d.]+)\s*(?:hours?|hrs?)'
        for match in re.finditer(hours_pattern, text, re.IGNORECASE):
            try:
                values["hours"].append(float(match.group(1)))
            except ValueError:
                pass
        
        # Extract percentages
        pct_pattern = r'([\d.]+)%'
        for match in re.finditer(pct_pattern, text):
            try:
                values["percentages"].append(float(match.group(1)))
            except ValueError:
                pass
        
        # Extract quantities
        qty_pattern = r'(\d+)\s*(?:units?|pieces?|parts?|items?)'
        for match in re.finditer(qty_pattern, text, re.IGNORECASE):
            try:
                values["quantities"].append(int(match.group(1)))
            except ValueError:
                pass
        
        return dict(values)
    
    def generate_summary(self, text: str, max_sentences: int = 3) -> str:
        """Generate extractive summary of text."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= max_sentences:
            return '. '.join(sentences) + '.'
        
        # Score sentences by keyword importance
        scored = []
        all_keywords = self.cost_keywords + self.efficiency_keywords + self.risk_keywords
        
        for sentence in sentences:
            score = 0
            lower = sentence.lower()
            
            # Score based on keywords
            for kw in all_keywords:
                if kw in lower:
                    score += 2
            
            # Score based on numeric content
            if re.search(r'\d+', sentence):
                score += 1
            
            # Score based on entities
            if re.search(r'\$[\d,]+', sentence):
                score += 2
            
            scored.append((sentence, score))
        
        # Return top sentences in original order
        scored.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored[:max_sentences]]
        
        # Reorder by appearance
        result = []
        for sentence in sentences:
            if sentence in top_sentences:
                result.append(sentence)
        
        return '. '.join(result) + '.'


class InsightExtractor:
    """Extracts actionable insights from manufacturing data."""
    
    def __init__(self):
        self.nlp = ManufacturingNLP()
    
    def analyze_maintenance_patterns(self, maintenance_df: pd.DataFrame) -> List[Insight]:
        """Analyze maintenance data for patterns and insights."""
        insights = []
        
        # Analyze by machine
        machine_costs = maintenance_df.groupby("machine_id").agg({
            "cost": "sum",
            "duration_hours": "sum",
            "event_id": "count",
            "downtime_impact_hours": "sum"
        }).rename(columns={"event_id": "event_count"})
        
        # Find machines with highest maintenance costs
        top_cost_machines = machine_costs.nlargest(3, "cost")
        for machine_id, row in top_cost_machines.iterrows():
            machine_events = maintenance_df[maintenance_df["machine_id"] == machine_id]
            unplanned_pct = (machine_events["event_type"] != "planned").mean() * 100
            
            insights.append(Insight(
                category="maintenance_cost",
                description=f"Machine {machine_id} has accumulated ${row['cost']:,.2f} in maintenance costs "
                           f"over {int(row['event_count'])} events ({unplanned_pct:.0f}% unplanned).",
                severity="high" if row["cost"] > machine_costs["cost"].mean() * 2 else "medium",
                confidence=0.9,
                supporting_data={
                    "machine_id": machine_id,
                    "total_cost": row["cost"],
                    "event_count": row["event_count"],
                    "downtime_hours": row["downtime_impact_hours"],
                    "unplanned_pct": unplanned_pct
                },
                recommendation=f"Consider preventive maintenance program for {machine_id} "
                              f"to reduce unplanned downtime."
            ))
        
        # Analyze failure modes
        failure_counts = maintenance_df[
            maintenance_df["event_type"] != "planned"
        ]["description"].apply(
            lambda x: self.nlp.classify_failure_mode(x)
        ).dropna()
        
        if len(failure_counts) > 0:
            failure_types = [f[0] for f in failure_counts]
            most_common = Counter(failure_types).most_common(3)
            
            for failure_type, count in most_common:
                total_cost = maintenance_df[
                    maintenance_df["description"].str.lower().str.contains(
                        failure_type.replace("_", "|"), regex=True, na=False
                    )
                ]["cost"].sum()
                
                insights.append(Insight(
                    category="failure_pattern",
                    description=f"{failure_type.replace('_', ' ').title()} issues occurred {count} times, "
                               f"costing approximately ${total_cost:,.2f}.",
                    severity="high" if count > 10 else "medium",
                    confidence=0.85,
                    supporting_data={
                        "failure_type": failure_type,
                        "occurrence_count": count,
                        "total_cost": total_cost
                    },
                    recommendation=f"Implement targeted inspection protocol for {failure_type.replace('_', ' ')} "
                                  f"to catch issues before failure."
                ))
        
        # Analyze downtime patterns
        if "downtime_impact_hours" in maintenance_df.columns:
            total_downtime = maintenance_df["downtime_impact_hours"].sum()
            avg_hourly_cost = 75  # Estimated opportunity cost per hour
            downtime_cost = total_downtime * avg_hourly_cost
            
            insights.append(Insight(
                category="downtime_analysis",
                description=f"Total unplanned downtime: {total_downtime:.0f} hours, "
                           f"with estimated production loss of ${downtime_cost:,.2f}.",
                severity="high" if total_downtime > 500 else "medium",
                confidence=0.8,
                supporting_data={
                    "total_downtime_hours": total_downtime,
                    "estimated_loss": downtime_cost,
                    "avg_hourly_cost": avg_hourly_cost
                },
                recommendation="Prioritize maintenance for machines with highest downtime impact."
            ))
        
        return insights
    
    def analyze_supplier_performance(self, deliveries_df: pd.DataFrame) -> List[Insight]:
        """Analyze supplier delivery performance."""
        insights = []
        
        # Aggregate by supplier
        supplier_stats = deliveries_df.groupby("supplier_name").agg({
            "delivery_id": "count",
            "days_late": ["sum", "mean"],
            "on_time": "mean",
            "total_cost": "sum",
            "quality_score": "mean"
        })
        supplier_stats.columns = ["delivery_count", "total_late_days", "avg_late_days", 
                                  "on_time_rate", "total_spend", "avg_quality"]
        
        # Find problematic suppliers
        poor_performers = supplier_stats[supplier_stats["on_time_rate"] < 0.8]
        
        for supplier, row in poor_performers.iterrows():
            production_impact = row["total_late_days"] * 8  # Assume 8 hours production per late day
            
            insights.append(Insight(
                category="supplier_reliability",
                description=f"{supplier} has {row['on_time_rate']*100:.0f}% on-time delivery rate "
                           f"with {int(row['total_late_days'])} total days late across "
                           f"{int(row['delivery_count'])} deliveries.",
                severity="high" if row["on_time_rate"] < 0.7 else "medium",
                confidence=0.9,
                supporting_data={
                    "supplier": supplier,
                    "on_time_rate": row["on_time_rate"],
                    "total_late_days": row["total_late_days"],
                    "delivery_count": row["delivery_count"],
                    "total_spend": row["total_spend"],
                    "estimated_production_impact_hours": production_impact
                },
                recommendation=f"Review contract with {supplier}; consider penalty clauses "
                              f"or alternative suppliers for critical materials."
            ))
        
        # Identify high-value, high-risk combinations
        high_spend = supplier_stats[supplier_stats["total_spend"] > supplier_stats["total_spend"].median()]
        for supplier, row in high_spend.iterrows():
            if row["on_time_rate"] < 0.85 or row["avg_quality"] < 4.0:
                insights.append(Insight(
                    category="supplier_risk",
                    description=f"High-spend supplier {supplier} (${row['total_spend']:,.2f}) "
                               f"shows quality ({row['avg_quality']:.1f}/5) or delivery "
                               f"({row['on_time_rate']*100:.0f}%) concerns.",
                    severity="high",
                    confidence=0.85,
                    supporting_data={
                        "supplier": supplier,
                        "total_spend": row["total_spend"],
                        "quality_score": row["avg_quality"],
                        "on_time_rate": row["on_time_rate"]
                    },
                    recommendation="Diversify supply chain; identify backup suppliers "
                                  "for critical materials from this vendor."
                ))
        
        return insights
    
    def analyze_production_efficiency(self, production_df: pd.DataFrame) -> List[Insight]:
        """Analyze production efficiency metrics."""
        insights = []
        
        # Overall efficiency metrics
        avg_efficiency = production_df["efficiency_pct"].mean()
        avg_defect_rate = production_df["defect_rate"].mean()
        total_scrap_cost = production_df["scrap_cost"].sum()
        
        insights.append(Insight(
            category="production_overview",
            description=f"Overall production efficiency: {avg_efficiency:.1f}% with "
                       f"{avg_defect_rate:.2f}% average defect rate. "
                       f"Total scrap cost: ${total_scrap_cost:,.2f}.",
            severity="medium" if avg_defect_rate > 2.0 else "low",
            confidence=0.95,
            supporting_data={
                "avg_efficiency": avg_efficiency,
                "avg_defect_rate": avg_defect_rate,
                "total_scrap_cost": total_scrap_cost,
                "total_units": production_df["units_produced"].sum()
            }
        ))
        
        # Machine-level efficiency
        machine_efficiency = production_df.groupby("machine_id").agg({
            "efficiency_pct": "mean",
            "defect_rate": "mean",
            "scrap_cost": "sum",
            "run_id": "count"
        }).rename(columns={"run_id": "run_count"})
        
        # Find underperforming machines
        low_efficiency = machine_efficiency[
            machine_efficiency["efficiency_pct"] < machine_efficiency["efficiency_pct"].mean() - 5
        ]
        
        for machine_id, row in low_efficiency.iterrows():
            insights.append(Insight(
                category="machine_efficiency",
                description=f"Machine {machine_id} running at {row['efficiency_pct']:.1f}% efficiency "
                           f"({avg_efficiency - row['efficiency_pct']:.1f}% below average) "
                           f"with ${row['scrap_cost']:,.2f} in scrap costs.",
                severity="high" if row["efficiency_pct"] < 80 else "medium",
                confidence=0.9,
                supporting_data={
                    "machine_id": machine_id,
                    "efficiency": row["efficiency_pct"],
                    "defect_rate": row["defect_rate"],
                    "scrap_cost": row["scrap_cost"],
                    "run_count": row["run_count"]
                },
                recommendation=f"Investigate {machine_id} for calibration issues, tooling wear, "
                              f"or operator training needs."
            ))
        
        # Operator analysis
        operator_stats = production_df.groupby("operator_id").agg({
            "efficiency_pct": "mean",
            "defect_rate": "mean",
            "run_id": "count"
        })
        
        top_operators = operator_stats.nlargest(3, "efficiency_pct")
        insights.append(Insight(
            category="operator_performance",
            description=f"Top performing operators: {', '.join(top_operators.index.tolist())} "
                       f"with {top_operators['efficiency_pct'].mean():.1f}% average efficiency.",
            severity="low",
            confidence=0.85,
            supporting_data={
                "top_operators": top_operators.to_dict()
            },
            recommendation="Document best practices from top performers for training program."
        ))
        
        return insights
    
    def analyze_financial_trends(self, financials_df: pd.DataFrame) -> List[Insight]:
        """Analyze financial trends and cost drivers."""
        insights = []
        
        # Recent trend analysis
        recent = financials_df.tail(3)
        older = financials_df.head(len(financials_df) - 3)
        
        for metric in ["maintenance_cost", "scrap_cost", "profit_margin"]:
            recent_avg = recent[metric].mean()
            older_avg = older[metric].mean() if len(older) > 0 else recent_avg
            
            if older_avg > 0:
                change_pct = ((recent_avg - older_avg) / older_avg) * 100
            else:
                change_pct = 0
            
            if abs(change_pct) > 10:
                direction = "increased" if change_pct > 0 else "decreased"
                
                insights.append(Insight(
                    category="financial_trend",
                    description=f"{metric.replace('_', ' ').title()} has {direction} by "
                               f"{abs(change_pct):.1f}% in recent months.",
                    severity="high" if (change_pct > 20 and metric != "profit_margin") or \
                                       (change_pct < -10 and metric == "profit_margin") else "medium",
                    confidence=0.85,
                    supporting_data={
                        "metric": metric,
                        "recent_avg": recent_avg,
                        "historical_avg": older_avg,
                        "change_percent": change_pct
                    },
                    recommendation=f"Investigate root causes for {metric.replace('_', ' ')} changes."
                ))
        
        # Cost breakdown analysis
        total_costs = financials_df[["material_cost", "labor_cost", "maintenance_cost", 
                                      "scrap_cost", "overhead"]].sum()
        total = total_costs.sum()
        
        cost_breakdown = (total_costs / total * 100).to_dict()
        largest_cost = max(cost_breakdown, key=cost_breakdown.get)
        
        insights.append(Insight(
            category="cost_analysis",
            description=f"Primary cost driver: {largest_cost.replace('_', ' ')} "
                       f"at {cost_breakdown[largest_cost]:.1f}% of total costs. "
                       f"Total annual costs: ${total:,.2f}.",
            severity="low",
            confidence=0.95,
            supporting_data={
                "cost_breakdown": cost_breakdown,
                "total_costs": total
            }
        ))
        
        return insights
    
    def generate_recommendations(self, all_insights: List[Insight]) -> List[Dict]:
        """Generate prioritized recommendations from all insights."""
        recommendations = []
        
        # Sort insights by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_insights = sorted(all_insights, 
                                key=lambda x: severity_order.get(x.severity, 4))
        
        # Generate prioritized recommendations
        for i, insight in enumerate(sorted_insights[:10], 1):
            if insight.recommendation:
                # Estimate potential savings
                savings = self._estimate_savings(insight)
                
                recommendations.append({
                    "priority": i,
                    "category": insight.category,
                    "issue": insight.description,
                    "recommendation": insight.recommendation,
                    "severity": insight.severity,
                    "estimated_savings": savings,
                    "confidence": insight.confidence
                })
        
        return recommendations
    
    def _estimate_savings(self, insight: Insight) -> Optional[float]:
        """Estimate potential savings from addressing an insight."""
        data = insight.supporting_data
        
        if insight.category == "maintenance_cost":
            # Assume 30% reduction possible with preventive maintenance
            return data.get("total_cost", 0) * 0.3
        
        elif insight.category == "supplier_reliability":
            # Estimate savings from reduced delays
            hours_impact = data.get("estimated_production_impact_hours", 0)
            return hours_impact * 75  # $75/hour opportunity cost
        
        elif insight.category == "machine_efficiency":
            # Estimate savings from efficiency improvement
            scrap = data.get("scrap_cost", 0)
            return scrap * 0.5  # 50% scrap reduction possible
        
        elif insight.category == "failure_pattern":
            return data.get("total_cost", 0) * 0.4
        
        return None


class NaturalLanguageGenerator:
    """Generates natural language reports from insights."""
    
    def __init__(self):
        self.templates = {
            "executive_summary": (
                "Executive Summary\n\n"
                "This analysis covers manufacturing operations from {start_date} to {end_date}. "
                "Key findings include:\n\n"
                "{key_findings}\n\n"
                "Top Recommendations:\n{recommendations}\n\n"
                "Estimated total savings opportunity: ${total_savings:,.2f}"
            ),
            "maintenance_report": (
                "Maintenance Analysis Report\n\n"
                "During the analysis period, {total_events} maintenance events occurred "
                "with total costs of ${total_cost:,.2f}. "
                "{unplanned_pct:.0f}% of events were unplanned, resulting in "
                "{downtime_hours:.0f} hours of production downtime.\n\n"
                "Key Issues:\n{issues}\n\n"
                "Recommended Actions:\n{actions}"
            ),
            "supplier_report": (
                "Supplier Performance Report\n\n"
                "Analysis of {delivery_count} deliveries across {supplier_count} suppliers:\n\n"
                "Overall on-time delivery rate: {on_time_rate:.1f}%\n"
                "Total late days: {total_late_days}\n"
                "Estimated production impact: {impact_hours:.0f} hours\n\n"
                "Supplier Issues:\n{issues}\n\n"
                "Recommended Actions:\n{actions}"
            )
        }
    
    def generate_executive_summary(self, insights: List[Insight], 
                                    recommendations: List[Dict],
                                    date_range: Tuple[str, str]) -> str:
        """Generate executive summary report."""
        key_findings = []
        for insight in insights[:5]:
            if insight.severity in ["high", "critical"]:
                key_findings.append(f"• {insight.description}")
        
        rec_text = []
        total_savings = 0
        for rec in recommendations[:5]:
            savings = rec.get("estimated_savings")
            if savings:
                rec_text.append(f"{rec['priority']}. {rec['recommendation']} "
                              f"(Est. savings: ${savings:,.2f})")
                total_savings += savings
            else:
                rec_text.append(f"{rec['priority']}. {rec['recommendation']}")
        
        return self.templates["executive_summary"].format(
            start_date=date_range[0],
            end_date=date_range[1],
            key_findings='\n'.join(key_findings) if key_findings else "No critical issues identified.",
            recommendations='\n'.join(rec_text),
            total_savings=total_savings
        )
    
    def generate_maintenance_report(self, maintenance_df: pd.DataFrame,
                                     insights: List[Insight]) -> str:
        """Generate maintenance-focused report."""
        maint_insights = [i for i in insights if i.category in 
                         ["maintenance_cost", "failure_pattern", "downtime_analysis"]]
        
        total_events = len(maintenance_df)
        total_cost = maintenance_df["cost"].sum()
        unplanned = maintenance_df[maintenance_df["event_type"] != "planned"]
        unplanned_pct = len(unplanned) / total_events * 100 if total_events > 0 else 0
        downtime = maintenance_df["downtime_impact_hours"].sum()
        
        issues = [f"• {i.description}" for i in maint_insights]
        actions = [f"• {i.recommendation}" for i in maint_insights if i.recommendation]
        
        return self.templates["maintenance_report"].format(
            total_events=total_events,
            total_cost=total_cost,
            unplanned_pct=unplanned_pct,
            downtime_hours=downtime,
            issues='\n'.join(issues) if issues else "No significant issues identified.",
            actions='\n'.join(actions) if actions else "Continue current maintenance practices."
        )
    
    def generate_chatbot_response(self, query: str, insights: List[Insight],
                                   data: Dict[str, pd.DataFrame]) -> str:
        """Generate conversational response to user query."""
        query_lower = query.lower()
        
        # Determine query intent
        if any(kw in query_lower for kw in ["cost", "expense", "spending", "money"]):
            return self._respond_costs(insights, data)
        elif any(kw in query_lower for kw in ["maintenance", "repair", "downtime", "breakdown"]):
            return self._respond_maintenance(insights, data)
        elif any(kw in query_lower for kw in ["supplier", "delivery", "late", "vendor"]):
            return self._respond_suppliers(insights, data)
        elif any(kw in query_lower for kw in ["efficiency", "production", "output", "performance"]):
            return self._respond_efficiency(insights, data)
        elif any(kw in query_lower for kw in ["recommend", "suggestion", "improve", "save"]):
            return self._respond_recommendations(insights)
        elif any(kw in query_lower for kw in ["summary", "overview", "report"]):
            return self._respond_summary(insights, data)
        else:
            return self._respond_general(insights)
    
    def _respond_costs(self, insights: List[Insight], data: Dict) -> str:
        """Generate response about costs."""
        financials = data.get("financials")
        if financials is None:
            return "I don't have financial data loaded to analyze costs."
        
        recent = financials.tail(3)
        total_cost = recent["total_cost"].mean()
        
        cost_insights = [i for i in insights if "cost" in i.category]
        
        response = f"Over the recent period, average monthly costs are ${total_cost:,.2f}. "
        
        if cost_insights:
            response += "Key cost concerns include: "
            response += "; ".join([i.description for i in cost_insights[:3]])
        
        return response
    
    def _respond_maintenance(self, insights: List[Insight], data: Dict) -> str:
        """Generate response about maintenance."""
        maint_insights = [i for i in insights if i.category in 
                         ["maintenance_cost", "failure_pattern", "downtime_analysis"]]
        
        if not maint_insights:
            return "No significant maintenance issues have been identified."
        
        response = "Here's what I found about maintenance:\n\n"
        for insight in maint_insights[:3]:
            response += f"• {insight.description}\n"
            if insight.recommendation:
                response += f"  → {insight.recommendation}\n"
        
        return response
    
    def _respond_suppliers(self, insights: List[Insight], data: Dict) -> str:
        """Generate response about suppliers."""
        supplier_insights = [i for i in insights if "supplier" in i.category]
        
        if not supplier_insights:
            return "Supplier performance is within acceptable parameters."
        
        response = "Supplier performance analysis:\n\n"
        for insight in supplier_insights[:3]:
            response += f"• {insight.description}\n"
        
        return response
    
    def _respond_efficiency(self, insights: List[Insight], data: Dict) -> str:
        """Generate response about efficiency."""
        eff_insights = [i for i in insights if i.category in 
                       ["production_overview", "machine_efficiency", "operator_performance"]]
        
        if not eff_insights:
            return "Production efficiency metrics are within normal ranges."
        
        response = "Production efficiency analysis:\n\n"
        for insight in eff_insights:
            response += f"• {insight.description}\n"
        
        return response
    
    def _respond_recommendations(self, insights: List[Insight]) -> str:
        """Generate response with recommendations."""
        high_priority = [i for i in insights if i.severity in ["high", "critical"] 
                        and i.recommendation]
        
        if not high_priority:
            return "No urgent recommendations at this time. Continue monitoring key metrics."
        
        response = "Here are my top recommendations:\n\n"
        for i, insight in enumerate(high_priority[:5], 1):
            savings = insight.supporting_data.get("total_cost", 0) * 0.3
            response += f"{i}. {insight.recommendation}"
            if savings > 0:
                response += f" (Potential savings: ${savings:,.2f})"
            response += "\n"
        
        return response
    
    def _respond_summary(self, insights: List[Insight], data: Dict) -> str:
        """Generate summary response."""
        high_severity = [i for i in insights if i.severity in ["high", "critical"]]
        
        response = f"Summary: I've identified {len(insights)} insights across your manufacturing data. "
        response += f"{len(high_severity)} require immediate attention.\n\n"
        
        if high_severity:
            response += "Priority issues:\n"
            for insight in high_severity[:3]:
                response += f"• {insight.description}\n"
        
        return response
    
    def _respond_general(self, insights: List[Insight]) -> str:
        """Generate general response."""
        return ("I can help you understand your manufacturing data. Try asking about:\n"
                "• Costs and expenses\n"
                "• Maintenance and downtime\n"
                "• Supplier performance\n"
                "• Production efficiency\n"
                "• Recommendations for improvement")


if __name__ == "__main__":
    # Test NLP components
    nlp = ManufacturingNLP()
    
    test_text = """
    Machine #7's motor overheating caused 20 hours of unplanned downtime.
    The repair cost was $4,500 including parts and labor.
    Supplier X delivered materials 3 days late, affecting production by 15%.
    """
    
    print("Testing Entity Extraction:")
    entities = nlp.extract_entities(test_text)
    for entity in entities:
        print(f"  {entity.label}: {entity.text}")
    
    print("\nTesting Failure Classification:")
    failure = nlp.classify_failure_mode(test_text)
    print(f"  Failure type: {failure}")
    
    print("\nTesting Numeric Extraction:")
    values = nlp.extract_numeric_values(test_text)
    print(f"  Values: {values}")
    
    print("\nTesting Summary Generation:")
    summary = nlp.generate_summary(test_text)
    print(f"  Summary: {summary}")
