import asyncio
import logging
import os
import json
from pathlib import Path
from typing import Dict, Any
from .core.workflow_manager import WorkflowManager
from .core.llm import LLMInterface
from .analysis.report_generator import ReportGenerator
from .agents.agentic.analysis_system import AgenticAnalysisSystem

logger = logging.getLogger(__name__)

def format_agentic_results(results: Dict[str, Any]) -> str:
    """Format agentic analysis results in a human-readable format."""
    output = []
    
    # Header
    output.append(f"Agentic Analysis Report")
    output.append(f"Generated: {results['timestamp']}")
    output.append(f"Analyzing: {results['directory']}")
    output.append("")
    
    # Workflow Info
    output.append("1. Workflow Information")
    output.append(f"Type: {results['workflow_info']['type']}")
    output.append(f"Tools: {', '.join(results['workflow_info']['tools_used'])}")
    output.append("")
    
    # Quality Analysis
    output.append("2. Quality Control Analysis")
    qa = results['quality_analysis']
    output.append(f"FastQC Reports: {'Available' if qa['fastqc_available'] else 'Not Found'}")
    output.append(f"MultiQC Report: {'Available' if qa['multiqc_available'] else 'Not Found'}")
    if qa.get('issues'):
        output.append("\nQuality Issues:")
        for issue in qa['issues']:
            output.append(f"- [{issue['severity'].upper()}] {issue['description']}")
    output.append("")
    
    # Quantification Analysis
    output.append("3. Quantification Analysis")
    quant = results['quantification_analysis']
    
    # Kallisto metrics
    kallisto = quant['tools']['kallisto']
    output.append(f"\nKallisto Analysis ({kallisto['samples']} samples):")
    for sample, data in kallisto['metrics'].items():
        metrics = data['metrics']
        output.append(f"\nSample: {sample}")
        output.append(f"- Total Transcripts: {metrics['total_transcripts']:,}")
        output.append(f"- Expressed Transcripts: {metrics['expressed_transcripts']:,}")
        output.append(f"- Median TPM: {metrics['median_tpm']:.2f}")
        output.append(f"- Mean TPM: {metrics['mean_tpm']:.2f}")
    
    if quant.get('issues'):
        output.append("\nQuantification Issues:")
        for issue in quant['issues']:
            output.append(f"- [{issue['severity'].upper()}] {issue['description']}")
    output.append("")
    
    # Technical Analysis
    output.append("4. Technical Analysis")
    tech = results['technical_analysis']
    if tech['tool_versions']:
        output.append("\nTool Versions:")
        for tool, version in tech['tool_versions'].items():
            output.append(f"- {tool}: {version}")
    output.append(f"\nLog Files Found: {tech['log_files']}")
    
    if tech.get('summary'):
        output.append(f"Resource Efficiency: {tech['summary']['resource_efficiency']}")
        output.append(f"Errors Found: {tech['summary']['error_count']}")
    
    # Combined Recommendations
    output.append("\n5. Recommendations")
    all_recs = []
    if qa.get('recommendations'):
        all_recs.extend(qa['recommendations'])
    if quant.get('recommendations'):
        all_recs.extend(quant['recommendations'])
    if tech.get('recommendations'):
        all_recs.extend(tech['recommendations'])
    if results.get('recommendations'):
        all_recs.extend(results['recommendations'])
    
    for i, rec in enumerate(all_recs, 1):
        output.append(f"{i}. {rec}")
    
    return "\n".join(output)

async def analyze_workflow(analysis_dir: str, save_report: bool = True) -> Dict[str, Any]:
    """
    Analyze workflow results in the specified directory using both standard and agentic analysis.
    
    Args:
        analysis_dir: Directory containing workflow results
        save_report: Whether to save the analysis report to a file
        
    Returns:
        Dict containing analysis results and report
    """
    try:
        logger.info(f"Analyzing workflow results in {analysis_dir}")
        
        # Initialize analysis systems
        report_gen = ReportGenerator()
        agentic_system = AgenticAnalysisSystem()
        
        # Run agentic analysis
        logger.info("Running agentic analysis...")
        agentic_results = await agentic_system.analyze_results(Path(analysis_dir))
        
        # Generate analysis report incorporating agentic results
        report = await report_gen.generate_analysis_report(Path(analysis_dir))
        
        # Save reports if requested
        report_files = {}
        if save_report:
            # Save reports in the analysis directory
            report_file = os.path.join(analysis_dir, "analysis_report.md")
            with open(report_file, "w") as f:
                f.write(report)
            report_files["standard"] = report_file
            
            agentic_file = os.path.join(analysis_dir, "agentic_analysis.md")
            with open(agentic_file, "w") as f:
                f.write(format_agentic_results(agentic_results))
            report_files["agentic"] = agentic_file
            
            logger.info(f"Saved analysis reports to {', '.join(report_files.values())}")
            
        # Print reports to terminal
        print("\nStandard Analysis Report:")
        print("=" * 80)
        print(report)
        print("=" * 80)
        
        print("\nAgentic Analysis Report:")
        print("=" * 80)
        print(format_agentic_results(agentic_results))
        print("=" * 80)
        
        return {
            "status": "success",
            "report": report,
            "agentic_report": format_agentic_results(agentic_results),
            "report_files": report_files,
            "analysis_results": agentic_results
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze workflow: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

async def run_workflow(prompt: str, checkpoint_dir: str = None, resume: bool = False) -> None:
    """Run workflow from prompt."""
    try:
        # Initialize workflow manager
        workflow_manager = WorkflowManager()
        
        # Create checkpoint directory if needed
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        else:
            checkpoint_path = Path("workflow_state")
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Execute workflow
        logger.info(f"Executing workflow with checkpoint dir: {checkpoint_path}")
        result = await workflow_manager.execute_workflow(prompt)
        
        if result["status"] != "success":
            raise Exception(result.get("error", "Unknown error"))
            
        logger.info("Workflow completed successfully!")
        
        # Generate workflow visualization
        if hasattr(workflow_manager, 'dag'):
            output_path = Path(checkpoint_path) / "workflow_dag.png"
            workflow_manager.dag.visualize(output_path)
        
        # Get output directory from workflow results
        output_dir = result.get('output_directory')
        if not output_dir:
            # Look for output directory in workflow results
            for step in result.get('steps', []):
                cmd = step.get('command', '')
                if 'results/' in cmd:
                    output_dir = cmd.split('results/')[1].split()[0].rstrip('/')
                    output_dir = f"results/{output_dir}"
                    break
        
        if not output_dir:
            logger.warning("No output directory found in workflow results")
            output_dir = "results"  # Default to results directory
        
        logger.info(f"Using output directory: {output_dir}")
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            logger.warning(f"Output directory does not exist: {output_dir}")
            return
            
        logger.info(f"Generating analysis report from {output_dir}...")
        await analyze_workflow(output_dir)
        
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        raise
