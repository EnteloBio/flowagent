import re
import os
import json
import signal
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

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
    kallisto = quant['tools'].get('kallisto', {})
    if kallisto and kallisto.get('samples'):
        sample_count = kallisto.get('sample_count', 0)
        output.append(f"\nKallisto Analysis ({sample_count} sample{'s' if sample_count != 1 else ''}):")
        for sample, data in sorted(kallisto['samples'].items()):  # Sort samples for consistent output
            metrics = data.get('metrics', {})
            if metrics:
                output.append(f"\nSample: {sample}")
                output.append(f"- Total Transcripts: {metrics.get('total_transcripts', 0):,}")
                output.append(f"- Expressed Transcripts: {metrics.get('expressed_transcripts', 0):,}")
                output.append(f"- Median TPM: {metrics.get('median_tpm', 0):.2f}")
                output.append(f"- Mean TPM: {metrics.get('mean_tpm', 0):.2f}")
    
    if quant.get('issues'):
        output.append("\nQuantification Issues:")
        seen_issues = set()  # Track unique issues
        for issue in quant['issues']:
            issue_text = f"[{issue['severity'].upper()}] {issue['description']}"
            if issue_text not in seen_issues:  # Only add unique issues
                output.append(f"- {issue_text}")
                seen_issues.add(issue_text)
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
    
    # Deduplicate recommendations
    seen_recs = set()
    unique_recs = []
    for rec in all_recs:
        if rec not in seen_recs:
            unique_recs.append(rec)
            seen_recs.add(rec)
    
    for i, rec in enumerate(unique_recs, 1):
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
        
        # Plan the workflow first
        logger.info(f"Planning workflow with checkpoint dir: {checkpoint_path}")
        try:
            workflow_plan = await workflow_manager.plan_workflow(prompt)
            # Print the workflow steps
            print_workflow_plan(workflow_plan)
        except Exception as e:
            logger.error(f"Error planning workflow: {str(e)}")
            print(f"\nError planning workflow: {str(e)}")
            print("Proceeding with direct execution...")
            workflow_plan = None
        
        # Execute workflow
        logger.info(f"Executing workflow with checkpoint dir: {checkpoint_path}")
        result = await workflow_manager.plan_and_execute_workflow(prompt, checkpoint_dir=checkpoint_path)
        
        if result.get("status") != "success":
            error_msg = result.get("error", "Unknown error")
            logger.error(f"Workflow failed: {error_msg}")
            raise Exception(f"Workflow failed: {error_msg}")
            
        logger.info("Workflow completed successfully!")
        
        # Generate workflow visualization
        if hasattr(workflow_manager, 'dag'):
            output_path = Path(checkpoint_path) / "workflow_dag.png"
            workflow_manager.dag.visualize(output_path)
        
        # Get output directory from workflow results
        output_dir = result.get('output_dir')
        if not output_dir:
            # Look for output directory in workflow results
            for step_name, step_result in result.get('results', {}).items():
                cmd = step_result.get('command', '')
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
        result = await analyze_workflow(output_dir)
        
        # Print reports to terminal
        print("\nStandard Analysis Report:")
        print("=" * 80)
        print(result["report"])
        print("=" * 80)
        
        print("\nAgentic Analysis Report:")
        print("=" * 80)
        print(result["agentic_report"])
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        raise

def print_workflow_plan(workflow_plan: Dict[str, Any], timeout: int = 30) -> None:
    """Print workflow plan with a timeout for user input."""
    if not workflow_plan:
        logger.warning("No workflow plan available to print")
        return
    
    if "steps" not in workflow_plan:
        logger.warning("Workflow plan has no steps to print")
        return
    
    try:
        print("\n" + "=" * 80)
        print(f"WORKFLOW PLAN: {workflow_plan.get('workflow_type', 'Custom Workflow')}")
        print("=" * 80)
        
        for i, step in enumerate(workflow_plan["steps"], 1):
            print(f"\nStep {i}: {step.get('name', 'Unnamed Step')}")
            print(f"  Command: {step.get('command', 'No command specified')}")
            
            if "description" in step:
                print(f"  Description: {step['description']}")
                
            if "dependencies" in step and step["dependencies"]:
                print(f"  Dependencies: {', '.join(step['dependencies'])}")
                
            if "profile_name" in step:
                print(f"  Resource Profile: {step['profile_name']}")
        
        print("\n" + "=" * 80)
        print("To execute this workflow, press Enter. To cancel, press Ctrl+C.")
        
        # Add a timeout to prevent hanging
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Input timed out. Proceeding with workflow execution.")
        
        # Set a 10-second timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            input()
        except (KeyboardInterrupt, TimeoutError) as e:
            if isinstance(e, KeyboardInterrupt):
                print("\nWorkflow cancelled by user.")
                sys.exit(0)
            else:
                print("\nInput timed out. Proceeding with workflow execution.")
        finally:
            # Cancel the alarm
            signal.alarm(0)
            
    except Exception as e:
        logger.error(f"Error printing workflow plan: {str(e)}")
        print("\nError displaying workflow plan. Proceeding with execution.")

def extract_geo_accession(prompt: str) -> List[str]:
    """
    Extract GEO accession numbers from a prompt.
    
    Args:
        prompt: The user prompt
        
    Returns:
        List of GEO accession numbers found in the prompt
    """
    # Pattern to match GSE followed by 4-8 digits
    pattern = r'GSE\d{4,8}'
    matches = re.findall(pattern, prompt)
    return matches

def _add_geo_download_steps(workflow_plan: Dict[str, Any], geo_accessions: List[str], output_dir: str) -> Dict[str, Any]:
    """
    Add steps to download data from GEO to the workflow plan.
    
    Args:
        workflow_plan: The current workflow plan
        geo_accessions: List of GEO accession numbers
        output_dir: The output directory for the workflow
        
    Returns:
        Updated workflow plan with GEO download steps
    """
    if not geo_accessions:
        return workflow_plan
    
    # Create a data directory if not specified in the workflow
    data_dir = os.path.join(output_dir, "data")
    
    # Check if we already have the necessary tools in dependencies
    dependencies = workflow_plan.get("dependencies", {})
    tools = dependencies.get("tools", [])
    
    # Ensure we have the necessary tools for GEO download
    required_tools = ["esearch", "efetch", "prefetch", "fasterq-dump"]
    for tool in required_tools:
        if tool not in tools and {"name": tool} not in tools:
            if isinstance(tools, list):
                tools.append({"name": tool})
    
    # Update dependencies in the workflow plan
    if "dependencies" not in workflow_plan:
        workflow_plan["dependencies"] = {}
    workflow_plan["dependencies"]["tools"] = tools
    
    # Get the current steps
    steps = workflow_plan.get("steps", [])
    
    # Add steps for each GEO accession
    for geo_accession in geo_accessions:
        # Create directory for this accession
        accession_dir = os.path.join(data_dir, geo_accession)
        
        # Add step to create the directory
        steps.insert(0, {
            "name": f"Create directory for {geo_accession}",
            "command": f"mkdir -p {accession_dir}",
            "description": f"Create directory to store data from {geo_accession}"
        })
        
        # Add step to get SRR IDs
        steps.insert(1, {
            "name": f"Get SRR IDs for {geo_accession}",
            "command": f"cd {accession_dir} && esearch -db sra -query '{geo_accession}[Accession]' | efetch -format runinfo > {geo_accession}_runinfo.csv",
            "description": f"Retrieve SRR IDs for {geo_accession} from NCBI SRA database"
        })
        
        # Add step to extract SRR IDs
        steps.insert(2, {
            "name": f"Extract SRR IDs for {geo_accession}",
            "command": f"cd {accession_dir} && tail -n +2 {geo_accession}_runinfo.csv | cut -d',' -f1 > {geo_accession}_srr_ids.txt",
            "description": f"Extract SRR IDs from runinfo CSV file for {geo_accession}"
        })
        
        # Add step to download FASTQ files
        steps.insert(3, {
            "name": f"Download FASTQ files for {geo_accession}",
            "command": f"cd {accession_dir} && cat {geo_accession}_srr_ids.txt | xargs -I{{}} sh -c 'prefetch {{}} && fasterq-dump {{}} && gzip *.fastq'",
            "description": f"Download and compress FASTQ files for {geo_accession}"
        })
    
    # Update steps in the workflow plan
    workflow_plan["steps"] = steps
    
    return workflow_plan

async def generate_workflow(prompt: str, llm_client: Any, output_dir: str = None) -> Dict[str, Any]:
    """Generate workflow from prompt."""
    try:
        # Extract GEO accession numbers from the prompt
        geo_accessions = extract_geo_accession(prompt)
        
        # Generate workflow plan using LLM
        workflow_plan = await llm_client.generate_workflow_plan(prompt)
        
        # Add GEO download steps if needed
        if geo_accessions:
            if output_dir is None:
                output_dir = workflow_plan.get("output_dir", os.getcwd())
            workflow_plan = _add_geo_download_steps(workflow_plan, geo_accessions, output_dir)
        
        return workflow_plan
    except Exception as e:
        logging.error(f"Error generating workflow: {str(e)}")
        return {"status": "error", "message": f"Error generating workflow: {str(e)}"}
