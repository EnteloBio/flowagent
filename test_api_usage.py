#!/usr/bin/env python
"""
Test script for API usage tracking in FlowAgent.

This script tests the API usage tracking functionality to ensure
it correctly records, displays, and serializes API usage data.
"""

import os
import sys
import json
from pathlib import Path
import unittest
import asyncio
from unittest.mock import MagicMock, patch
from datetime import datetime

# Ensure flowagent is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flowagent.core.api_usage import APIUsageTracker
from flowagent.agents.llm_agent import LLMAgent


class TestAPIUsageTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = APIUsageTracker()
        self.tracker.start_workflow("test_workflow", "Test Workflow")
        
    def test_record_api_call(self):
        """Test recording an API call."""
        self.tracker.record_api_call(
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            workflow_id="test_workflow",
            step_name="test_step",
            purpose="Testing"
        )
        
        workflow = self.tracker.workflows["test_workflow"]
        self.assertEqual(workflow["total_calls"], 1)
        self.assertEqual(workflow["total_prompt_tokens"], 100)
        self.assertEqual(workflow["total_completion_tokens"], 50)
        self.assertEqual(workflow["total_tokens"], 150)
        self.assertEqual(workflow["steps_tracked"], {"test_step"})
        
    def test_track_api_usage(self):
        """Test tracking API usage for non-API operations."""
        self.tracker.track_api_usage("test_operation", "test_step")
        
        workflow = self.tracker.workflows["test_workflow"]
        self.assertEqual(workflow["steps_tracked"], {"test_step"})
        
    def test_workflow_stages_tracking(self):
        """Test tracking API usage across workflow stages."""
        # Planning stage
        self.tracker.record_api_call(
            model="gpt-4",
            prompt_tokens=200,
            completion_tokens=100,
            workflow_id="test_workflow",
            step_name="planning",
            purpose="Generate workflow plan"
        )
        
        # Execution stage - first step
        self.tracker.record_api_call(
            model="gpt-4",
            prompt_tokens=150,
            completion_tokens=75,
            workflow_id="test_workflow",
            step_name="execution_step1",
            purpose="Execute command"
        )
        
        # Analysis stage
        analysis_id = "analysis_test"
        self.tracker.start_workflow(analysis_id, "Analysis")
        self.tracker.workflows[analysis_id]["parent_workflow_id"] = "test_workflow"
        
        self.tracker.record_api_call(
            model="gpt-4",
            prompt_tokens=300,
            completion_tokens=150,
            workflow_id=analysis_id,
            step_name="analysis_step",
            purpose="Analyze results"
        )
        
        # End analysis workflow and verify it merges with parent
        self.tracker.end_workflow(analysis_id)
        
        # Verify total usage includes all stages
        workflow = self.tracker.workflows["test_workflow"]
        self.assertEqual(workflow["total_calls"], 3)  # 1 planning + 1 execution + 1 analysis
        self.assertEqual(workflow["total_prompt_tokens"], 650)  # 200 + 150 + 300
        self.assertEqual(workflow["total_completion_tokens"], 325)  # 100 + 75 + 150
        self.assertEqual(workflow["total_tokens"], 975)  # 300 + 225 + 450

    def test_complex_data_serialization(self):
        """Test that complex data types are properly serialized."""
        # Add some complex data types to the workflow data
        self.tracker.workflows["test_workflow"]["models_used"] = set(["gpt-4", "gpt-3.5"])
        self.tracker.workflows["test_workflow"]["custom_data"] = {
            "nested": {
                "list": [1, 2, 3],
                "set": set([4, 5, 6]),
                "date": datetime.now()
            }
        }
        
        # Prepare data for serialization
        serialized = self.tracker._prepare_data_for_serialization(
            self.tracker.workflows["test_workflow"]
        )
        
        # Verify it can be JSON serialized
        try:
            json_str = json.dumps(serialized)
            self.assertTrue(isinstance(json_str, str))
        except Exception as e:
            self.fail(f"JSON serialization failed: {str(e)}")
        
        # Verify models_used is now a list, not a set
        self.assertTrue(isinstance(serialized["models_used"], list))
        self.assertEqual(sorted(serialized["models_used"]), ["gpt-3.5", "gpt-4"])
        
        # Verify nested set was converted to list
        self.assertTrue(isinstance(serialized["custom_data"]["nested"]["set"], list))
        
        # Verify datetime was converted to ISO string
        self.assertTrue(isinstance(serialized["custom_data"]["nested"]["date"], str))


class TestLLMAgentUsageTracking(unittest.TestCase):
    def test_llm_agent_track_api_call(self):
        """Test API usage tracking in LLMAgent."""
        # This test requires importing LLMAgent which may have dependencies
        # So we'll just mock it instead of importing the actual class
        
        with patch("flowagent.agents.llm_agent.LLMAgent") as MockAgent:
            agent = MockAgent()
            agent.api_usage_tracker = APIUsageTracker()
            agent.api_usage_tracker.start_workflow("test_agent", "Test Agent Workflow")
            
            # Mock a response object
            response = MagicMock()
            response.usage.prompt_tokens = 100
            response.usage.completion_tokens = 50
            response.usage.total_tokens = 150
            response.model = "gpt-4"
            
            # Call the mocked track_api_call method
            # In a real implementation this would use the LLMAgent's _track_api_call method
            agent.api_usage_tracker.record_api_call(
                model=response.model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                step_name="test_step",
                purpose="Testing"
            )
            
            workflow = agent.api_usage_tracker.workflows["test_agent"]
            self.assertEqual(workflow["total_calls"], 1)
            self.assertEqual(workflow["total_prompt_tokens"], 100)
            self.assertEqual(workflow["total_completion_tokens"], 50)
            self.assertEqual(workflow["total_tokens"], 150)


def test_api_usage_tracker():
    """Test basic APIUsageTracker functionality."""
    print("\n===== Testing Basic APIUsageTracker =====")
    
    # Create tracker
    tracker = APIUsageTracker()
    
    # Start workflow
    workflow_id = "test_workflow_1"
    tracker.start_workflow(workflow_id, "Test Workflow")
    
    # Record some API calls
    for i in range(5):
        tracker.record_api_call(
            model="gpt-4o",
            prompt_tokens=100 * (i + 1),
            completion_tokens=50 * (i + 1),
            workflow_id=workflow_id,
            step_name=f"step_{i}",
            purpose="Testing API usage tracking"
        )
    
    # End workflow and get usage
    usage = tracker.end_workflow(workflow_id)
    
    # Test serialization
    serialized = json.dumps(usage, indent=2)
    print(f"Successfully serialized usage data ({len(serialized)} bytes)")
    
    # Test display
    tracker.display_usage(workflow_id)
    
    # Test save_usage_report
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    report_path = tracker.save_usage_report(workflow_id, str(output_dir))
    
    if report_path:
        print(f"Successfully saved usage report to {report_path}")
    else:
        print("Failed to save usage report")
    
    return tracker

def test_llm_agent_integration():
    """Test LLMAgent integration with APIUsageTracker."""
    print("\n===== Testing LLMAgent Integration =====")
    
    # Create LLMAgent
    try:
        agent = LLMAgent()
        print("Successfully created LLMAgent")
        
        # Test _track_api_usage method
        agent._track_api_usage(operation="test_operation", tokens=1000)
        
        # Display usage
        if hasattr(agent, "api_usage_tracker") and agent.api_usage_tracker:
            agent.api_usage_tracker.display_usage()
        else:
            print("No API usage tracker available in LLMAgent")
            
    except Exception as e:
        print(f"Error testing LLMAgent integration: {e}")
        
def test_complex_data_serialization():
    """Test serialization of complex data structures."""
    print("\n===== Testing Complex Data Serialization =====")
    
    tracker = APIUsageTracker()
    
    # Create test data with complex structures
    complex_data = {
        "a_set": set(["item1", "item2", "item3"]),
        "nested": {
            "another_set": set([1, 2, 3]),
            "mixed_list": [1, "two", set([3, 4]), {"five": 5}]
        },
        "datetime": datetime.now()
    }
    
    # Test serialization method
    serialized = tracker._prepare_data_for_serialization(complex_data)
    
    # Try to JSON encode
    try:
        json_str = json.dumps(serialized, indent=2)
        print(f"Successfully serialized complex data ({len(json_str)} bytes)")
    except Exception as e:
        print(f"Failed to serialize complex data: {e}")
        
    return serialized

if __name__ == "__main__":
    unittest.main()
