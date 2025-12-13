"""
LeanNLP: Knowledge Graph Module
Builds and queries a knowledge graph linking manufacturing entities.
"""

import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    node_id: str
    node_type: str
    properties: Dict[str, Any]


@dataclass
class GraphEdge:
    """Represents an edge/relationship in the knowledge graph."""
    source_id: str
    target_id: str
    relationship: str
    properties: Dict[str, Any]


class ManufacturingKnowledgeGraph:
    """Knowledge graph for manufacturing data relationships."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_types = defaultdict(list)
        self.relationship_types = defaultdict(list)
        
        # Define valid relationships
        self.valid_relationships = {
            ("machine", "maintenance_event"): "had_maintenance",
            ("maintenance_event", "machine"): "performed_on",
            ("supplier", "delivery"): "made_delivery",
            ("delivery", "supplier"): "supplied_by",
            ("machine", "production_run"): "produced",
            ("production_run", "machine"): "ran_on",
            ("machine", "cost"): "incurred_cost",
            ("supplier", "material"): "supplies",
            ("production_run", "product"): "produced_product",
            ("maintenance_event", "failure_mode"): "caused_by",
            ("machine", "operator"): "operated_by",
        }
    
    def add_node(self, node: GraphNode) -> None:
        """Add a node to the knowledge graph."""
        self.graph.add_node(
            node.node_id,
            node_type=node.node_type,
            **node.properties
        )
        self.node_types[node.node_type].append(node.node_id)
    
    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge/relationship to the knowledge graph."""
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            relationship=edge.relationship,
            **edge.properties
        )
        self.relationship_types[edge.relationship].append((edge.source_id, edge.target_id))
    
    def build_from_dataframes(self, data: Dict[str, pd.DataFrame]) -> None:
        """Build knowledge graph from manufacturing dataframes."""
        print("Building knowledge graph...")
        
        # Add machine nodes
        if "machines" in data:
            self._add_machines(data["machines"])
        
        # Add supplier nodes
        if "suppliers" in data:
            self._add_suppliers(data["suppliers"])
        
        # Add maintenance events and relationships
        if "maintenance" in data:
            self._add_maintenance_events(data["maintenance"])
        
        # Add production runs and relationships
        if "production" in data:
            self._add_production_runs(data["production"])
        
        # Add deliveries and relationships
        if "deliveries" in data:
            self._add_deliveries(data["deliveries"])
        
        print(f"  ✓ Graph built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
    
    def _add_machines(self, machines_df: pd.DataFrame) -> None:
        """Add machine nodes to graph."""
        for _, row in machines_df.iterrows():
            node = GraphNode(
                node_id=row["machine_id"],
                node_type="machine",
                properties={
                    "name": row["name"],
                    "type": row["type"],
                    "hourly_rate": row["hourly_rate"],
                    "status": row["status"],
                    "age_years": row.get("age_years", 0)
                }
            )
            self.add_node(node)
    
    def _add_suppliers(self, suppliers_df: pd.DataFrame) -> None:
        """Add supplier nodes to graph."""
        for _, row in suppliers_df.iterrows():
            node = GraphNode(
                node_id=row["supplier_id"],
                node_type="supplier",
                properties={
                    "name": row["name"],
                    "category": row["category"],
                    "reliability_score": row["reliability_score"],
                    "avg_lead_time_days": row["avg_lead_time_days"],
                    "contract_value": row["contract_value"]
                }
            )
            self.add_node(node)
    
    def _add_maintenance_events(self, maintenance_df: pd.DataFrame) -> None:
        """Add maintenance events and link to machines."""
        failure_modes = set()
        
        for _, row in maintenance_df.iterrows():
            # Add maintenance event node
            event_node = GraphNode(
                node_id=row["event_id"],
                node_type="maintenance_event",
                properties={
                    "event_type": row["event_type"],
                    "description": row["description"],
                    "duration_hours": row["duration_hours"],
                    "cost": row["cost"],
                    "root_cause": row.get("root_cause", "N/A"),
                    "downtime_impact": row.get("downtime_impact_hours", 0)
                }
            )
            self.add_node(event_node)
            
            # Link to machine
            self.add_edge(GraphEdge(
                source_id=row["machine_id"],
                target_id=row["event_id"],
                relationship="had_maintenance",
                properties={"date": str(row["start_time"])}
            ))
            
            # Extract and link failure modes
            if row["event_type"] != "planned" and row.get("root_cause"):
                failure_mode_id = f"FM_{row['root_cause'].replace(' ', '_')}"
                
                if failure_mode_id not in failure_modes:
                    failure_modes.add(failure_mode_id)
                    self.add_node(GraphNode(
                        node_id=failure_mode_id,
                        node_type="failure_mode",
                        properties={"name": row["root_cause"]}
                    ))
                
                self.add_edge(GraphEdge(
                    source_id=row["event_id"],
                    target_id=failure_mode_id,
                    relationship="caused_by",
                    properties={}
                ))
    
    def _add_production_runs(self, production_df: pd.DataFrame) -> None:
        """Add production run nodes and relationships."""
        products = set()
        operators = set()
        
        for _, row in production_df.iterrows():
            # Add production run node
            run_node = GraphNode(
                node_id=row["run_id"],
                node_type="production_run",
                properties={
                    "units_produced": row["units_produced"],
                    "units_defective": row["units_defective"],
                    "defect_rate": row["defect_rate"],
                    "efficiency_pct": row["efficiency_pct"],
                    "duration_hours": row["duration_hours"],
                    "scrap_cost": row["scrap_cost"]
                }
            )
            self.add_node(run_node)
            
            # Link to machine
            self.add_edge(GraphEdge(
                source_id=row["run_id"],
                target_id=row["machine_id"],
                relationship="ran_on",
                properties={}
            ))
            
            # Add and link product
            product_id = row["product_code"]
            if product_id not in products:
                products.add(product_id)
                self.add_node(GraphNode(
                    node_id=product_id,
                    node_type="product",
                    properties={"code": product_id}
                ))
            
            self.add_edge(GraphEdge(
                source_id=row["run_id"],
                target_id=product_id,
                relationship="produced_product",
                properties={"quantity": row["units_produced"]}
            ))
            
            # Add and link operator
            operator_id = row["operator_id"]
            if operator_id not in operators:
                operators.add(operator_id)
                self.add_node(GraphNode(
                    node_id=operator_id,
                    node_type="operator",
                    properties={"id": operator_id}
                ))
            
            self.add_edge(GraphEdge(
                source_id=row["run_id"],
                target_id=operator_id,
                relationship="operated_by",
                properties={}
            ))
    
    def _add_deliveries(self, deliveries_df: pd.DataFrame) -> None:
        """Add delivery nodes and link to suppliers."""
        materials = set()
        
        for _, row in deliveries_df.iterrows():
            # Add delivery node
            delivery_node = GraphNode(
                node_id=row["delivery_id"],
                node_type="delivery",
                properties={
                    "po_number": row["po_number"],
                    "days_late": row["days_late"],
                    "on_time": row["on_time"],
                    "quantity": row["quantity"],
                    "total_cost": row["total_cost"],
                    "quality_score": row["quality_score"]
                }
            )
            self.add_node(delivery_node)
            
            # Link to supplier
            self.add_edge(GraphEdge(
                source_id=row["supplier_id"],
                target_id=row["delivery_id"],
                relationship="made_delivery",
                properties={}
            ))
            
            # Add and link material
            material_id = row["material_code"]
            if material_id not in materials:
                materials.add(material_id)
                self.add_node(GraphNode(
                    node_id=material_id,
                    node_type="material",
                    properties={
                        "code": material_id,
                        "category": row["material_category"]
                    }
                ))
            
            self.add_edge(GraphEdge(
                source_id=row["delivery_id"],
                target_id=material_id,
                relationship="contains_material",
                properties={"quantity": row["quantity"]}
            ))
    
    def query_node(self, node_id: str) -> Optional[Dict]:
        """Get node details by ID."""
        if node_id in self.graph:
            return dict(self.graph.nodes[node_id])
        return None
    
    def query_neighbors(self, node_id: str, relationship: Optional[str] = None) -> List[Dict]:
        """Get all neighbors of a node, optionally filtered by relationship."""
        neighbors = []
        
        # Outgoing edges
        for _, target, data in self.graph.out_edges(node_id, data=True):
            if relationship is None or data.get("relationship") == relationship:
                neighbors.append({
                    "node_id": target,
                    "direction": "outgoing",
                    "relationship": data.get("relationship"),
                    "properties": dict(self.graph.nodes[target])
                })
        
        # Incoming edges
        for source, _, data in self.graph.in_edges(node_id, data=True):
            if relationship is None or data.get("relationship") == relationship:
                neighbors.append({
                    "node_id": source,
                    "direction": "incoming",
                    "relationship": data.get("relationship"),
                    "properties": dict(self.graph.nodes[source])
                })
        
        return neighbors
    
    def query_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """Find shortest path between two nodes."""
        try:
            path = nx.shortest_path(self.graph, source_id, target_id)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def query_by_type(self, node_type: str) -> List[Dict]:
        """Get all nodes of a specific type."""
        results = []
        for node_id in self.node_types.get(node_type, []):
            results.append({
                "node_id": node_id,
                **dict(self.graph.nodes[node_id])
            })
        return results
    
    def aggregate_costs(self, node_id: str) -> Dict[str, float]:
        """Aggregate costs associated with a node (e.g., machine)."""
        costs = defaultdict(float)
        
        node_data = self.query_node(node_id)
        if not node_data:
            return dict(costs)
        
        node_type = node_data.get("node_type")
        
        if node_type == "machine":
            # Get maintenance costs
            for neighbor in self.query_neighbors(node_id, "had_maintenance"):
                if neighbor["direction"] == "outgoing":
                    cost = neighbor["properties"].get("cost", 0)
                    costs["maintenance"] += cost
            
            # Get production scrap costs
            for neighbor in self.query_neighbors(node_id):
                if neighbor["properties"].get("node_type") == "production_run":
                    scrap = neighbor["properties"].get("scrap_cost", 0)
                    costs["scrap"] += scrap
        
        elif node_type == "supplier":
            # Get delivery costs
            for neighbor in self.query_neighbors(node_id, "made_delivery"):
                if neighbor["direction"] == "outgoing":
                    cost = neighbor["properties"].get("total_cost", 0)
                    costs["materials"] += cost
        
        costs["total"] = sum(costs.values())
        return dict(costs)
    
    def find_risk_patterns(self) -> List[Dict]:
        """Identify risk patterns in the knowledge graph."""
        risks = []
        
        # Find machines with high maintenance frequency
        for machine_id in self.node_types.get("machine", []):
            maintenance_events = self.query_neighbors(machine_id, "had_maintenance")
            unplanned_count = sum(
                1 for e in maintenance_events 
                if e["properties"].get("event_type") != "planned"
            )
            
            if unplanned_count > 5:
                risks.append({
                    "type": "high_maintenance_machine",
                    "node_id": machine_id,
                    "severity": "high" if unplanned_count > 10 else "medium",
                    "details": f"{unplanned_count} unplanned maintenance events"
                })
        
        # Find suppliers with delivery issues
        for supplier_id in self.node_types.get("supplier", []):
            deliveries = self.query_neighbors(supplier_id, "made_delivery")
            late_count = sum(
                1 for d in deliveries 
                if d["properties"].get("days_late", 0) > 0
            )
            
            if len(deliveries) > 0 and late_count / len(deliveries) > 0.2:
                risks.append({
                    "type": "unreliable_supplier",
                    "node_id": supplier_id,
                    "severity": "high" if late_count / len(deliveries) > 0.3 else "medium",
                    "details": f"{late_count}/{len(deliveries)} late deliveries"
                })
        
        # Find common failure modes
        failure_mode_counts = defaultdict(int)
        for fm_id in self.node_types.get("failure_mode", []):
            incoming = self.query_neighbors(fm_id)
            failure_mode_counts[fm_id] = len([
                n for n in incoming if n["direction"] == "incoming"
            ])
        
        for fm_id, count in failure_mode_counts.items():
            if count > 5:
                fm_data = self.query_node(fm_id)
                risks.append({
                    "type": "recurring_failure_mode",
                    "node_id": fm_id,
                    "severity": "high" if count > 10 else "medium",
                    "details": f"Occurred {count} times: {fm_data.get('name', fm_id)}"
                })
        
        return risks
    
    def get_graph_statistics(self) -> Dict:
        """Get overall statistics about the knowledge graph."""
        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": {k: len(v) for k, v in self.node_types.items()},
            "relationship_types": {k: len(v) for k, v in self.relationship_types.items()},
            "density": nx.density(self.graph),
            "connected_components": nx.number_weakly_connected_components(self.graph)
        }
        return stats
    
    def export_for_visualization(self) -> Dict:
        """Export graph data for visualization (e.g., with pyvis)."""
        nodes = []
        edges = []
        
        # Color mapping for node types
        colors = {
            "machine": "#4CAF50",
            "supplier": "#2196F3",
            "maintenance_event": "#FF9800",
            "production_run": "#9C27B0",
            "delivery": "#00BCD4",
            "product": "#E91E63",
            "operator": "#795548",
            "failure_mode": "#F44336",
            "material": "#607D8B"
        }
        
        for node_id in self.graph.nodes():
            node_data = dict(self.graph.nodes[node_id])
            node_type = node_data.get("node_type", "unknown")
            
            nodes.append({
                "id": node_id,
                "label": node_data.get("name", node_id),
                "type": node_type,
                "color": colors.get(node_type, "#9E9E9E"),
                "title": json.dumps(node_data, indent=2, default=str)
            })
        
        for source, target, data in self.graph.edges(data=True):
            edges.append({
                "from": source,
                "to": target,
                "label": data.get("relationship", ""),
                "arrows": "to"
            })
        
        return {"nodes": nodes, "edges": edges}
    
    def natural_language_query(self, query: str) -> str:
        """Answer natural language queries about the graph."""
        query_lower = query.lower()
        
        # Machine queries
        if "machine" in query_lower:
            machine_id = None
            # Try to extract machine ID
            import re
            match = re.search(r'm\d{3}', query_lower)
            if match:
                machine_id = match.group(0).upper()
            
            if machine_id and machine_id in self.graph:
                node_data = self.query_node(machine_id)
                costs = self.aggregate_costs(machine_id)
                neighbors = self.query_neighbors(machine_id)
                
                maint_count = len([n for n in neighbors if "maintenance" in n.get("relationship", "")])
                
                return (f"Machine {machine_id} ({node_data.get('name', '')}):\n"
                       f"• Type: {node_data.get('type', 'Unknown')}\n"
                       f"• Status: {node_data.get('status', 'Unknown')}\n"
                       f"• Maintenance events: {maint_count}\n"
                       f"• Total costs: ${costs.get('total', 0):,.2f}")
            
            # List all machines
            machines = self.query_by_type("machine")
            return f"Found {len(machines)} machines in the system: {', '.join([m['node_id'] for m in machines[:10]])}"
        
        # Supplier queries
        if "supplier" in query_lower:
            suppliers = self.query_by_type("supplier")
            response = f"Found {len(suppliers)} suppliers:\n"
            for s in suppliers[:5]:
                response += f"• {s['node_id']}: {s.get('name', 'Unknown')} (Reliability: {s.get('reliability_score', 0):.0%})\n"
            return response
        
        # Risk queries
        if "risk" in query_lower or "problem" in query_lower or "issue" in query_lower:
            risks = self.find_risk_patterns()
            if not risks:
                return "No significant risks identified in the current data."
            
            response = f"Identified {len(risks)} risk patterns:\n"
            for risk in risks[:5]:
                response += f"• [{risk['severity'].upper()}] {risk['type']}: {risk['details']}\n"
            return response
        
        # Statistics queries
        if "stat" in query_lower or "overview" in query_lower or "summary" in query_lower:
            stats = self.get_graph_statistics()
            return (f"Knowledge Graph Overview:\n"
                   f"• Total entities: {stats['total_nodes']}\n"
                   f"• Total relationships: {stats['total_edges']}\n"
                   f"• Entity types: {', '.join(f'{k}({v})' for k, v in stats['node_types'].items())}")
        
        return "I can answer questions about machines, suppliers, risks, and statistics. Try asking something more specific!"


if __name__ == "__main__":
    # Test knowledge graph
    from data.data_generator import SyntheticDataGenerator
    
    print("Testing Knowledge Graph...")
    
    # Generate test data
    generator = SyntheticDataGenerator(seed=42)
    data = generator.generate_all_data()
    
    # Build graph
    kg = ManufacturingKnowledgeGraph()
    kg.build_from_dataframes(data)
    
    # Test queries
    print("\nGraph Statistics:")
    stats = kg.get_graph_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nRisk Patterns:")
    risks = kg.find_risk_patterns()
    for risk in risks[:5]:
        print(f"  [{risk['severity']}] {risk['type']}: {risk['details']}")
    
    print("\nNatural Language Query Test:")
    print(kg.natural_language_query("Tell me about machine M001"))
    print("\n" + kg.natural_language_query("What suppliers do we have?"))
    print("\n" + kg.natural_language_query("What are the main risks?"))
