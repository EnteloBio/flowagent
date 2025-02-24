"""Command line interface for Cognomic."""

import asyncio
import logging
import click
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

from .core.workflow_manager import WorkflowManager
from .utils.logging import get_logger
from .analysis.report_generator import ReportGenerator

# Configure logging to show debug messages
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s [%(levelname)s] %(name)s - %(message)s')
logger = get_logger(__name__)

async def run_both_analyses(analysis_dir: Path, prompt: str) -> Tuple[str, dict]:
    """Run both report generator and agent-based analysis in parallel."""
    # Create both analyzers
    report_gen = ReportGenerator()
    workflow_manager = WorkflowManager()
    
    # Run both analyses in parallel
    report_task = asyncio.create_task(
        report_gen.generate_analysis_report(analysis_dir, query=prompt)
    )
    
    # Load workflow results for agent analysis
    workflow_results_path = analysis_dir / "workflow_results.json"
    if workflow_results_path.exists():
        with open(workflow_results_path) as f:
            workflow_results = json.load(f)
        agent_task = asyncio.create_task(
            workflow_manager.analyze_results(workflow_results)
        )
    else:
        agent_task = None
    
    # Wait for both to complete
    report = await report_task
    agent_analysis = await agent_task if agent_task else None
    
    return report, agent_analysis

@click.command()
@click.argument('prompt')
@click.option('--analysis-dir', type=click.Path(exists=True), help='Directory containing workflow results to analyze')
@click.option('--format', type=click.Choice(['text', 'json', 'html']), default='text', 
              help='Output format for analysis')
@click.option('--debug/--no-debug', default=False, help='Enable debug logging')
def run(prompt: str, analysis_dir: Optional[str] = None, format: str = 'text', debug: bool = False):
    """Run Cognomic workflow or analyze results.
    
    There are two main modes:
    
    1. Run a workflow:
       cognomic "run rna-seq analysis"
       
    2. Analyze existing results:
       cognomic "analyze workflow results" --analysis-dir=results
    """
    try:
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            
        # Check if this is an analysis request - either explicit "analyze" or when analysis_dir is provided
        is_analysis = "analyze" in prompt.lower() or analysis_dir is not None
            
        if is_analysis and analysis_dir:
            logger.info(f"Starting analysis of results in directory: {analysis_dir}")
            
            # Run both analyses
            report, agent_analysis = asyncio.run(
                run_both_analyses(Path(analysis_dir), prompt)
            )
            
            if format == 'json':
                # Combine both analyses into one JSON output
                combined_report = {
                    "file_analysis": report,
                    "agent_analysis": agent_analysis["report"] if agent_analysis else None
                }
                click.echo(json.dumps(combined_report, indent=2))
            elif format == 'html':
                # Convert markdown report to HTML
                try:
                    import markdown2
                    html_content = markdown2.markdown(
                        report,
                        extras=['tables', 'fenced-code-blocks']
                    )
                    
                    # Add agent analysis if available
                    if agent_analysis and agent_analysis["status"] == "success":
                        agent_html = """
                        <h2>Agent-Based Analysis</h2>
                        <div class="section">
                            <h3>Overall Assessment</h3>
                            {assessment}
                        </div>
                        <div class="section">
                            <h3>Issues</h3>
                            {issues}
                        </div>
                        <div class="section">
                            <h3>Recommendations</h3>
                            {recommendations}
                        </div>
                        """.format(
                            assessment="".join(
                                f"<p><strong>{k}:</strong> {v}</p>"
                                for k, v in agent_analysis["report"]["overall_assessment"].items()
                            ),
                            issues="".join(
                                f'<div class="issue {issue["severity"]}">'
                                f'<strong>{issue["severity"].upper()}:</strong> {issue["description"]}</div>'
                                for issue in agent_analysis["report"]["issues"]
                            ),
                            recommendations="<ol>" + "".join(
                                f"<li>{rec}</li>"
                                for rec in agent_analysis["report"]["recommendations"]
                            ) + "</ol>"
                        )
                        html_content += agent_html
                    
                    html_report = f"""
                    <html>
                        <head>
                            <title>Cognomic Analysis Report</title>
                            <style>
                                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                                .section {{ margin: 20px 0; }}
                                table {{ border-collapse: collapse; width: 100%; }}
                                th, td {{ border: 1px solid #ddd; padding: 8px; }}
                                th {{ background-color: #f5f5f5; }}
                                code {{ background-color: #f5f5f5; padding: 2px 4px; }}
                                .issue {{ padding: 10px; margin: 5px 0; border-radius: 5px; }}
                                .high {{ background-color: #ffe6e6; }}
                                .medium {{ background-color: #fff3e6; }}
                                .low {{ background-color: #e6ffe6; }}
                            </style>
                        </head>
                        <body>
                            <h1>Cognomic Analysis Report</h1>
                            <h2>File-Based Analysis</h2>
                            {html_content}
                        </body>
                    </html>
                    """
                    output_path = Path(analysis_dir) / "analysis_report.html"
                    with open(output_path, 'w') as f:
                        f.write(html_report)
                    click.echo(f"Analysis report saved to: {output_path}")
                except ImportError:
                    click.echo("Warning: markdown2 not installed. Falling back to text format.")
                    click.echo(report)
                    if agent_analysis and agent_analysis["status"] == "success":
                        click.echo("\nAgent-Based Analysis:")
                        click.echo("====================")
                        
                        click.echo("\nOverall Assessment:")
                        for k, v in agent_analysis["report"]["overall_assessment"].items():
                            click.echo(f"- {k}: {v}")
                        
                        if agent_analysis["report"]["issues"]:
                            click.echo("\nIssues:")
                            for issue in agent_analysis["report"]["issues"]:
                                click.echo(f"[{issue['severity'].upper()}] {issue['description']}")
                        
                        if agent_analysis["report"]["recommendations"]:
                            click.echo("\nRecommendations:")
                            for i, rec in enumerate(agent_analysis["report"]["recommendations"], 1):
                                click.echo(f"{i}. {rec}")
            else:
                # Text format - show both reports
                click.echo("File-Based Analysis:")
                click.echo("===================")
                click.echo(report)
                
                if agent_analysis and agent_analysis["status"] == "success":
                    click.echo("\nAgent-Based Analysis:")
                    click.echo("====================")
                    
                    click.echo("\nOverall Assessment:")
                    for k, v in agent_analysis["report"]["overall_assessment"].items():
                        click.echo(f"- {k}: {v}")
                    
                    if agent_analysis["report"]["issues"]:
                        click.echo("\nIssues:")
                        for issue in agent_analysis["report"]["issues"]:
                            click.echo(f"[{issue['severity'].upper()}] {issue['description']}")
                    
                    if agent_analysis["report"]["recommendations"]:
                        click.echo("\nRecommendations:")
                        for i, rec in enumerate(agent_analysis["report"]["recommendations"], 1):
                            click.echo(f"{i}. {rec}")
        else:
            # Run workflow with analysis
            workflow_manager = WorkflowManager()
            results = asyncio.run(workflow_manager.execute_workflow(prompt))
            
            if results["workflow_results"]["status"] == "success":
                click.echo("✓ Workflow completed successfully!")
                
                # Show analysis
                if "analysis" in results and results["analysis"]["status"] == "success":
                    if format == 'json':
                        click.echo(json.dumps(results["analysis"]["report"], indent=2))
                    elif format == 'html':
                        html_report = _generate_html_report(results["analysis"]["report"])
                        output_path = Path("analysis_report.html")
                        with open(output_path, 'w') as f:
                            f.write(html_report)
                        click.echo(f"Analysis report saved to: {output_path}")
                    else:
                        click.echo("\nAnalysis Report:")
                        click.echo("===============")
                        
                        # Overall Assessment
                        click.echo("\nOverall Assessment:")
                        for component, summary in results["analysis"]["report"]["overall_assessment"].items():
                            click.echo(f"- {component}: {summary}")
                        
                        # Issues
                        if results["analysis"]["report"]["issues"]:
                            click.echo("\nIssues Found:")
                            for issue in results["analysis"]["report"]["issues"]:
                                severity = issue.get("severity", "unknown").upper()
                                click.echo(f"[{severity}] {issue.get('description')}")
                        
                        # Recommendations
                        if results["analysis"]["report"]["recommendations"]:
                            click.echo("\nRecommendations:")
                            for i, rec in enumerate(results["analysis"]["report"]["recommendations"], 1):
                                click.echo(f"{i}. {rec}")
            else:
                click.echo("✗ Workflow failed!")
                if "error" in results["workflow_results"]:
                    click.echo(f"Error: {results['workflow_results']['error']}")
                    
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        sys.exit(1)

def _generate_html_report(report: dict) -> str:
    """Generate HTML report from analysis results."""
    # Basic HTML report template
    return f"""
    <html>
        <head>
            <title>Cognomic Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin: 20px 0; }}
                .issue {{ padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .high {{ background-color: #ffe6e6; }}
                .medium {{ background-color: #fff3e6; }}
                .low {{ background-color: #e6ffe6; }}
            </style>
        </head>
        <body>
            <h1>Cognomic Analysis Report</h1>
            
            <div class="section">
                <h2>Overall Assessment</h2>
                {''.join(f"<p><strong>{k}:</strong> {v}</p>" 
                        for k, v in report['overall_assessment'].items())}
            </div>
            
            <div class="section">
                <h2>Issues</h2>
                {''.join(f"<div class='issue {issue.get('severity', 'low')}'>"
                        f"<strong>{issue.get('severity', '').upper()}:</strong> "
                        f"{issue.get('description')}</div>"
                        for issue in report['issues'])}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ol>
                    {''.join(f"<li>{rec}</li>" 
                            for rec in report['recommendations'])}
                </ol>
            </div>
        </body>
    </html>
    """

if __name__ == '__main__':
    run()
