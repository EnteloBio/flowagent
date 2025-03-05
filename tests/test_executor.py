"""
Test script for the Executor class.
"""

import os
import sys
import json
import asyncio
import logging
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowagent.core.executor import Executor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

class TestExecutor(unittest.TestCase):
    """Test cases for the Executor class."""
    
    def setUp(self):
        """Set up test environment."""
        self.executor = Executor(executor_type="local")
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove test output directory
        import shutil
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
    
    def test_execute_step_success(self):
        """Test executing a step successfully."""
        # Define a simple step
        step = {
            "name": "Test step",
            "description": "A simple test step",
            "command": "echo 'Hello, world!'",
            "tools": []
        }
        
        # Execute the step
        result = asyncio.run(self.executor.execute_step(step, self.output_dir))
        
        # Check the result
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["step"], step)
        self.assertEqual(result["stdout"].strip(), "Hello, world!")
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(os.path.exists(result["log_file"]))
    
    def test_execute_step_error(self):
        """Test executing a step that fails."""
        # Define a step that will fail
        step = {
            "name": "Test error step",
            "description": "A step that will fail",
            "command": "command_that_does_not_exist",
            "tools": []
        }
        
        # Execute the step
        result = asyncio.run(self.executor.execute_step(step, self.output_dir))
        
        # Check the result
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["step"], step)
        self.assertNotEqual(result["exit_code"], 0)
        self.assertTrue(os.path.exists(result["log_file"]))
    
    def test_execute_step_no_command(self):
        """Test executing a step with no command."""
        # Define a step with no command
        step = {
            "name": "Test no command step",
            "description": "A step with no command",
            "tools": []
        }
        
        # Execute the step
        result = asyncio.run(self.executor.execute_step(step, self.output_dir))
        
        # Check the result
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["step"], step)
        self.assertEqual(result["reason"], "No command specified")
    
    def test_execute_step_with_output(self):
        """Test executing a step that produces output files."""
        # Define a step that creates a file
        step = {
            "name": "Test output step",
            "description": "A step that creates a file",
            "command": "echo 'Test content' > test_file.txt",
            "tools": []
        }
        
        # Execute the step
        result = asyncio.run(self.executor.execute_step(step, self.output_dir))
        
        # Check the result
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["step"], step)
        self.assertEqual(result["exit_code"], 0)
        
        # Check that the file was created
        output_file = os.path.join(self.output_dir, "test_file.txt")
        self.assertTrue(os.path.exists(output_file))
        
        # Check the content of the file
        with open(output_file, "r") as f:
            content = f.read().strip()
            self.assertEqual(content, "Test content")

if __name__ == "__main__":
    unittest.main()
