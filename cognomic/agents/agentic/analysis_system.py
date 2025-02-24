"""Coordinates analysis agents for comprehensive result interpretation."""

import json
import logging
from typing import Dict, Any, List
from pathlib import Path

from .analysis_agents import (
    QualityAnalysisAgent,
    QuantificationAnalysisAgent,
    TechnicalQCAgent
)

logger = logging.getLogger(__name__)

class AgenticAnalysisSystem:
    """Coordinates analysis agents for comprehensive result interpretation."""
    
    def __init__(self):
        """Initialize the analysis system."""
        self.agents = {
            "quality": QualityAnalysisAgent(),
            "quantification": QuantificationAnalysisAgent(),
            "technical": TechnicalQCAgent()
        }
    
    async def analyze_workflow_results(self, workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workflow results using specialized agents."""
        try:
            # Prepare data for analysis
            analysis_data = self._prepare_analysis_data(workflow_results)
            
            # Run analyses in parallel
            analyses = {}
            for agent_name, agent in self.agents.items():
                try:
                    analysis = await agent.analyze(analysis_data)
                    analyses[agent_name] = analysis
                except Exception as e:
                    logger.error(f"Analysis failed for agent {agent_name}: {str(e)}")
                    analyses[agent_name] = {"error": str(e)}
            
            # Generate comprehensive report
            report = await self._generate_comprehensive_report(analyses)
            
            return {
                "status": "success",
                "analyses": analyses,
                "report": report
            }
            
        except Exception as e:
            logger.error(f"Analysis system failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _prepare_analysis_data(self, workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare workflow results data for analysis."""
        try:
            # Extract relevant data for each analysis type
            analysis_data = {
                # Quality data
                "fastqc_reports": self._find_fastqc_reports(workflow_results),
                "multiqc_report": self._find_multiqc_report(workflow_results),
                
                # Quantification data
                "kallisto_output": self._find_kallisto_output(workflow_results),
                
                # Technical data
                "resource_usage": workflow_results.get("resource_usage", {}),
                "tool_versions": workflow_results.get("tool_versions", {}),
                "logs": workflow_results.get("logs", [])
            }
            
            return analysis_data
            
        except Exception as e:
            logger.error(f"Failed to prepare analysis data: {str(e)}")
            raise
    
    def _find_fastqc_reports(self, results: Dict[str, Any]) -> List[str]:
        """Find FastQC report files in workflow results."""
        # Implementation would find and parse FastQC reports
        # For now returning empty list
        return []
    
    def _find_multiqc_report(self, results: Dict[str, Any]) -> str:
        """Find MultiQC report in workflow results."""
        # Implementation would find and parse MultiQC report
        # For now returning None
        return None
    
    def _find_kallisto_output(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Find Kallisto output in workflow results."""
        # Implementation would find and parse Kallisto output
        # For now returning empty dict
        return {}
    
    async def _generate_comprehensive_report(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        try:
            # Extract results from each analysis
            quality_results = analyses.get("quality", {}).get("results", {})
            quant_results = analyses.get("quantification", {}).get("results", {})
            tech_results = analyses.get("technical", {}).get("results", {})
            
            # Combine analyses into comprehensive assessment
            assessment = {
                "quality_summary": quality_results.get("overall_assessment"),
                "quantification_summary": quant_results.get("mapping_statistics", {}).get("assessment"),
                "technical_summary": tech_results.get("execution_quality", {})
            }
            
            # Aggregate all issues and recommendations
            issues = []
            recommendations = []
            
            # Add quality issues and recommendations
            if "issues" in quality_results:
                issues.extend(quality_results["issues"])
            if "recommendations" in quality_results:
                recommendations.extend(quality_results["recommendations"])
            
            # Add quantification issues and recommendations
            if "quality_flags" in quant_results:
                issues.extend(quant_results["quality_flags"])
            if "recommendations" in quant_results:
                recommendations.extend(quant_results["recommendations"])
            
            # Add technical issues and recommendations
            if "execution_quality" in tech_results:
                if "failed_steps" in tech_results["execution_quality"]:
                    issues.extend([
                        {"severity": "high", "description": f"Failed step: {step}"}
                        for step in tech_results["execution_quality"]["failed_steps"]
                    ])
            if "recommendations" in tech_results:
                recommendations.extend(tech_results["recommendations"])
            
            return {
                "overall_assessment": assessment,
                "issues": issues,
                "recommendations": recommendations,
                "detailed_results": {
                    "quality_analysis": quality_results,
                    "quantification_analysis": quant_results,
                    "technical_analysis": tech_results
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {str(e)}")
            return {
                "error": f"Failed to generate comprehensive report: {str(e)}"
            }
