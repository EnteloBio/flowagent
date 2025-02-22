from typing import Dict, List, Set, Any
from dataclasses import dataclass
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import json
from datetime import datetime

@dataclass
class ToolCall:
    tool_name: str
    inputs: Dict[str, Any]
    outputs: Set[Path]  # Files/directories generated
    dependencies: Set[str]  # Tool call IDs this depends on
    call_id: str
    status: str = "pending"
    start_time: datetime = None
    end_time: datetime = None

    def to_dict(self):
        return {
            'tool_name': self.tool_name,
            'inputs': self.inputs,
            'outputs': [str(p) for p in self.outputs],
            'dependencies': list(self.dependencies),
            'call_id': self.call_id,
            'status': self.status,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None
        }

class ToolTracker:
    def __init__(self):
        self.tool_calls: Dict[str, ToolCall] = {}
        self.current_context: List[str] = []  # Stack of tool call IDs
    
    def start_tool_call(self, tool_name: str, inputs: Dict[str, Any]) -> str:
        """Register the start of a tool call and infer dependencies"""
        call_id = f"{tool_name}_{len(self.tool_calls)}"
        
        # Infer dependencies by checking if any inputs use outputs from previous tools
        dependencies = set()
        for value in inputs.values():
            if isinstance(value, (str, Path)):
                path = Path(value)
                for prev_call in self.tool_calls.values():
                    if any(output_path == path for output_path in prev_call.outputs):
                        dependencies.add(prev_call.call_id)
        
        tool_call = ToolCall(
            tool_name=tool_name,
            inputs=inputs,
            outputs=set(),
            dependencies=dependencies,
            call_id=call_id,
            start_time=datetime.now()
        )
        
        self.tool_calls[call_id] = tool_call
        self.current_context.append(call_id)
        return call_id
    
    def finish_tool_call(self, call_id: str, outputs: Set[Path]):
        """Register tool completion and its outputs"""
        if call_id in self.tool_calls:
            self.tool_calls[call_id].outputs = outputs
            self.tool_calls[call_id].status = "completed"
            self.tool_calls[call_id].end_time = datetime.now()
            if self.current_context and self.current_context[-1] == call_id:
                self.current_context.pop()
    
    def build_dag(self) -> nx.DiGraph:
        """Convert the tracked tool calls into a DAG"""
        dag = nx.DiGraph()
        
        # Add nodes for all tool calls
        for call_id, tool_call in self.tool_calls.items():
            dag.add_node(call_id, 
                        tool_name=tool_call.tool_name,
                        status=tool_call.status,
                        inputs=tool_call.inputs,
                        outputs=tool_call.outputs)
        
        # Add edges for dependencies
        for call_id, tool_call in self.tool_calls.items():
            for dep_id in tool_call.dependencies:
                dag.add_edge(dep_id, call_id)
        
        return dag
    
    def visualize_dag(self, output_path: Path):
        """Generate a visualization of the workflow DAG"""
        dag = self.build_dag()
        
        # Set up the plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(dag)
        
        # Draw nodes with different colors based on status
        colors = {
            'pending': 'lightblue',
            'running': 'yellow',
            'completed': 'lightgreen',
            'failed': 'red'
        }
        
        for status in colors:
            nodes = [n for n, d in dag.nodes(data=True) if d.get('status') == status]
            nx.draw_networkx_nodes(dag, pos, nodelist=nodes, node_color=colors[status])
        
        # Draw edges and labels
        nx.draw_networkx_edges(dag, pos)
        labels = {n: f"{d['tool_name']}\n{n}" for n, d in dag.nodes(data=True)}
        nx.draw_networkx_labels(dag, pos, labels)
        
        plt.title("Workflow DAG")
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()
    
    def save_state(self, checkpoint_dir: Path):
        """Save the current state of tool tracking"""
        state = {
            'tool_calls': {k: v.to_dict() for k, v in self.tool_calls.items()},
            'current_context': self.current_context
        }
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with (checkpoint_dir / 'tool_tracker.json').open('w') as f:
            json.dump(state, f, indent=2)
    
    @classmethod
    def load_state(cls, checkpoint_dir: Path) -> 'ToolTracker':
        """Load tool tracking state from checkpoint"""
        tracker = cls()
        
        try:
            with (checkpoint_dir / 'tool_tracker.json').open('r') as f:
                state = json.load(f)
                
            for call_id, call_data in state['tool_calls'].items():
                tool_call = ToolCall(
                    tool_name=call_data['tool_name'],
                    inputs=call_data['inputs'],
                    outputs={Path(p) for p in call_data['outputs']},
                    dependencies=set(call_data['dependencies']),
                    call_id=call_data['call_id'],
                    status=call_data['status']
                )
                if call_data['start_time']:
                    tool_call.start_time = datetime.fromisoformat(call_data['start_time'])
                if call_data['end_time']:
                    tool_call.end_time = datetime.fromisoformat(call_data['end_time'])
                tracker.tool_calls[call_id] = tool_call
                
            tracker.current_context = state['current_context']
            
        except FileNotFoundError:
            pass
            
        return tracker
