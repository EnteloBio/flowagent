# Quick Start

This guide will help you get started with FlowAgent quickly.

## Basic Usage

```python
from flowagent import FlowAgent

# Initialize the agent
agent = FlowAgent()

# Create a workflow
workflow = agent.create_workflow("my_workflow")

# Add tasks to the workflow
workflow.add_task("task1", lambda x: x + 1)
workflow.add_task("task2", lambda x: x * 2)

# Run the workflow
result = workflow.run(input_data=5)
print(result)  # Output: 12
```

## Next Steps

- Learn more about [Workflows](../user-guide/workflows.md)
- Explore different types of [Agents](../user-guide/agents.md)
- Read about [Configuration](configuration.md)
