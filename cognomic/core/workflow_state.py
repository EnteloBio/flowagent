import json
from datetime import datetime
from pathlib import Path

class WorkflowState:
    def __init__(self, workflow_name: str):
        self.state = {
            'metadata': {
                'name': workflow_name,
                'created_at': datetime.utcnow().isoformat(),
                'version': '1.0'
            },
            'steps': {},
            'artifacts': []
        }

    def add_step(self, step: dict):
        self.state['steps'][step['name']] = {
            'status': 'pending',
            'inputs': step.get('inputs', {}),
            'outputs': {},
            'errors': [],
            'attempts': 0
        }

    def update_step(self, step_name: str, updates: dict):
        if step_name in self.state['steps']:
            self.state['steps'][step_name].update(updates)

    def archive(self, output_dir: Path):
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        archive_path = output_dir / f"workflow_state_{timestamp}.json"
        with open(archive_path, 'w') as f:
            json.dump(self.state, f, indent=2)
        return archive_path
