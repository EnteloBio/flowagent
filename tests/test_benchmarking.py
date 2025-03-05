"""Tests for the benchmarking module."""

import os
import unittest
import tempfile
from unittest import mock
from pathlib import Path

from flowagent.benchmarking import BenchmarkRunner, BenchmarkMetrics, BenchmarkVisualizer


class TestBenchmarkMetrics(unittest.TestCase):
    """Test the benchmark metrics component."""

    def setUp(self):
        """Set up test fixtures."""
        self.metrics = BenchmarkMetrics()

    def test_extract_metrics(self):
        """Test that metrics can be properly extracted from workflow results."""
        # Mock result data
        result = {
            "status": "success",
            "steps": [
                {"status": "success"},
                {"status": "success"},
                {"status": "error"}
            ]
        }
        duration = 10.5

        # Extract metrics
        metrics = self.metrics.extract_metrics(result, duration)

        # Check that basic metrics are extracted correctly
        self.assertEqual(metrics["status"], "success")
        self.assertEqual(metrics["duration"], 10.5)
        self.assertEqual(metrics["total_steps"], 3)
        self.assertEqual(metrics["successful_steps"], 2)
        self.assertEqual(metrics["failed_steps"], 1)

    def test_analyze_command_quality(self):
        """Test command quality analysis."""
        # Test commands
        commands = [
            "kallisto index -i transcripts.idx transcripts.fa",
            "fastqc sample1.fastq.gz -o results/fastqc",
            "kallisto quant -i transcripts.idx -o results/kallisto sample1.fastq.gz"
        ]

        # Analyze commands
        metrics = self.metrics.analyze_command_quality(commands)

        # Check command metrics
        self.assertEqual(metrics["command_count"], 3)
        self.assertGreater(metrics["avg_command_length"], 0)
        self.assertIn("kallisto", metrics["tools_used"])
        self.assertEqual(metrics["tools_used"]["kallisto"], 2)
        self.assertIn("fastqc", metrics["tools_used"])
        self.assertEqual(metrics["tools_used"]["fastqc"], 1)
        self.assertIn("-i", metrics["flags_used"])
        self.assertEqual(metrics["flags_used"]["-i"], 2)


class TestBenchmarkVisualizer(unittest.TestCase):
    """Test the benchmark visualizer component."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.visualizer = BenchmarkVisualizer(output_dir=self.temp_dir.name)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    @mock.patch('matplotlib.pyplot.savefig')
    def test_visualize_execution_time(self, mock_savefig):
        """Test execution time visualization."""
        # Mock results data
        model_results = {
            "gpt-3.5-turbo": [
                {"metrics": {"duration": 10.5}, "workflow_type": "rna_seq_kallisto"},
                {"metrics": {"duration": 11.2}, "workflow_type": "rna_seq_kallisto"}
            ],
            "gpt-4": [
                {"metrics": {"duration": 8.3}, "workflow_type": "rna_seq_kallisto"},
                {"metrics": {"duration": 9.1}, "workflow_type": "rna_seq_kallisto"}
            ]
        }

        # Generate visualization
        self.visualizer.visualize_execution_time(model_results)

        # Check if savefig was called (visualization was created)
        mock_savefig.assert_called_once()


class TestBenchmarkRunner(unittest.TestCase):
    """Test the benchmark runner component."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.benchmark_dir = Path(self.temp_dir.name)
        self.runner = BenchmarkRunner(benchmark_dir=self.temp_dir.name)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test proper initialization of the benchmark runner."""
        self.assertEqual(self.runner.benchmark_dir, self.benchmark_dir)
        self.assertIsInstance(self.runner.metrics, BenchmarkMetrics)
        self.assertIsInstance(self.runner.visualizer, BenchmarkVisualizer)

    def test_generate_summary(self):
        """Test summary generation from benchmark results."""
        # Mock results data
        results = {
            "gpt-3.5-turbo": [
                {
                    "repetition": 1,
                    "status": "success",
                    "duration": 10.5,
                    "metrics": {
                        "memory_usage": 150.2,
                        "cpu_usage": 35.6,
                        "completed_steps_detected_ratio": 0.8,
                        "steps_skipped_ratio": 0.6
                    }
                },
                {
                    "repetition": 2,
                    "status": "error",
                    "duration": 5.3,
                    "error": "Workflow failed"
                }
            ],
            "gpt-4": [
                {
                    "repetition": 1,
                    "status": "success",
                    "duration": 9.8,
                    "metrics": {
                        "memory_usage": 180.4,
                        "cpu_usage": 42.1,
                        "completed_steps_detected_ratio": 0.9,
                        "steps_skipped_ratio": 0.7
                    }
                }
            ]
        }

        # Mock metadata
        metadata = {
            "id": "benchmark-123",
            "prompt": "Test prompt",
            "models": ["gpt-3.5-turbo", "gpt-4"],
            "repetitions": 2
        }

        # Generate summary
        summary = self.runner.generate_summary(results, metadata)

        # Check summary contents
        self.assertEqual(summary["benchmark_id"], "benchmark-123")
        self.assertEqual(summary["models"], ["gpt-3.5-turbo", "gpt-4"])
        self.assertEqual(len(summary["model_summaries"]), 2)
        self.assertEqual(summary["model_summaries"]["gpt-3.5-turbo"]["success_rate"], 0.5)
        self.assertEqual(summary["model_summaries"]["gpt-4"]["success_rate"], 1.0)
        self.assertIn("comparison", summary)


if __name__ == "__main__":
    unittest.main()
