"""Dependency management utilities for FlowAgent."""

import subprocess
import sys
import logging
import shutil
from typing import Dict, List, Optional, Union, Set
import pkg_resources
import importlib
import json
import os
from pathlib import Path

from ..core.llm import LLMInterface

logger = logging.getLogger(__name__)

class DependencyManager:
    """Manages software dependencies for FlowAgent."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.llm = LLMInterface()
        
    async def analyze_workflow_dependencies(self, workflow_plan: Dict) -> Dict[str, Set[str]]:
        """Use LLM to analyze workflow dependencies from the workflow plan."""
        try:
            # Extract workflow information for LLM
            steps = []
            for step in workflow_plan.get("steps", []):
                steps.append({
                    "name": step.get("name", ""),
                    "command": step.get("command", ""),
                    "description": step.get("description", "")
                })
            
            workflow_info = {
                "workflow_type": workflow_plan.get("type", "custom"),
                "description": workflow_plan.get("description", ""),
                "steps": steps
            }
            
            # Create prompt for LLM
            prompt = f"""Analyze this bioinformatics workflow and determine all required software dependencies.
            
Workflow Information:
{json.dumps(workflow_info, indent=2)}

For each command in the workflow steps, identify:
1. Required command-line tools (available via conda-forge or bioconda)
2. Required Python packages (available via conda-forge or pip)
3. Required R packages (from CRAN or Bioconductor)

Special Cases:
- If you see 'kb' commands being used, this requires the 'kb-python' package which must be installed via pip
- Some tools may have Python package equivalents that are preferred (e.g., kb-python instead of kb)

Return a JSON object with this structure:
{{
    "tools": [
        {{
            "name": "tool_name",
            "channel": "conda-forge or bioconda",
            "min_version": "minimum version if critical",
            "reason": "why this tool is needed"
        }}
    ],
    "python_packages": [
        {{
            "name": "package_name",
            "channel": "conda-forge or pip",
            "min_version": "minimum version if critical",
            "reason": "why this package is needed"
        }}
    ],
    "r_packages": [
        {{
            "name": "package_name",
            "repository": "cran or bioconductor",
            "min_version": "minimum version if critical",
            "reason": "why this package is needed"
        }}
    ]
}}

Focus on identifying dependencies that are actually used in the commands or are essential for the workflow type.
For Python packages that must be installed via pip, make sure to specify "channel": "pip".
"""
            
            # Get LLM response
            response = await self.llm._call_openai(
                messages=[
                    {
                        "role": "system", 
                        "content": """You are an expert in bioinformatics workflows and software dependencies.
You must respond with valid JSON only, no other text.
Example format:
{
    "tools": [{"name": "tool_name", "channel": "conda-forge", "min_version": "1.0.0", "reason": "explanation"}],
    "python_packages": [{"name": "package", "channel": "pip", "min_version": "2.0.0", "reason": "explanation"}],
    "r_packages": [{"name": "package", "channel": "cran", "min_version": "0.1.0", "reason": "explanation"}]
}"""
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            
            try:
                # Clean the response to ensure it's valid JSON
                response = response.strip()
                if response.startswith("```json"):
                    response = response[7:]
                if response.startswith("```"):
                    response = response[3:]
                if response.endswith("```"):
                    response = response[:-3]
                response = response.strip()
                
                # Parse dependencies
                dependencies = json.loads(response)
                self.logger.info(f"Identified dependencies: {json.dumps(dependencies, indent=2)}")
                
                return dependencies
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
                self.logger.error(f"Raw response: {response}")
                raise
            
        except Exception as e:
            self.logger.error(f"Failed to analyze workflow dependencies: {str(e)}")
            raise
            
    def check_python_package(self, package_name: str, min_version: Optional[str] = None) -> bool:
        """Check if a Python package is installed with minimum version."""
        try:
            pkg = pkg_resources.get_distribution(package_name)
            if min_version:
                return pkg.version >= min_version
            return True
        except pkg_resources.DistributionNotFound:
            return False
            
    def check_r_package(self, package_name: str) -> bool:
        """Check if an R package is installed."""
        try:
            cmd = f"Rscript -e 'if (!require({package_name})) quit(status=1)'"
            result = subprocess.run(cmd, shell=True, capture_output=True)
            return result.returncode == 0
        except Exception:
            return False
            
    def check_tool(self, tool_name: str) -> bool:
        """Check if a command-line tool is available."""
        return shutil.which(tool_name) is not None
        
    def install_python_package(self, package_info: Dict[str, str]) -> bool:
        """Install a Python package using conda or pip."""
        package_name = package_info["name"]
        channel = package_info.get("channel", "conda-forge")
        
        try:
            if channel == "pip":
                cmd = f"pip install {package_name}"
                if "min_version" in package_info:
                    cmd += f">={package_info['min_version']}"
            else:
                cmd = f"conda install -y -c {channel} {package_name}"
                if "min_version" in package_info:
                    cmd += f"={package_info['min_version']}"
                    
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0 and channel != "pip":
                self.logger.warning(f"Failed to install {package_name} via conda, trying pip...")
                return self.install_python_package({"name": package_name, "channel": "pip"})
                
            return result.returncode == 0
            
        except Exception as e:
            self.logger.error(f"Failed to install {package_name}: {str(e)}")
            return False
            
    def install_r_package(self, package_info: Dict[str, str]) -> bool:
        """Install an R package from CRAN or Bioconductor."""
        package_name = package_info["name"]
        repository = package_info.get("repository", "cran")
        
        try:
            if repository == "bioconductor":
                cmd = f"""Rscript -e 'if (!require("BiocManager")) install.packages("BiocManager"); BiocManager::install("{package_name}")'"""
            else:
                cmd = f"""Rscript -e 'if (!require("{package_name}")) install.packages("{package_name}")'"""
                
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception as e:
            self.logger.error(f"Failed to install R package {package_name}: {str(e)}")
            return False
            
    def install_tool(self, tool_info: Dict[str, str]) -> bool:
        """Install a command-line tool using conda."""
        tool_name = tool_info["name"]
        channel = tool_info.get("channel", "conda-forge")
        
        try:
            cmd = f"conda install -y -c {channel} {tool_name}"
            if "min_version" in tool_info:
                cmd += f"={tool_info['min_version']}"
                
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception as e:
            self.logger.error(f"Failed to install tool {tool_name}: {str(e)}")
            return False
            
    async def ensure_workflow_dependencies(self, workflow_plan: Dict) -> bool:
        """Ensure all required workflow dependencies are installed."""
        try:
            # Get dependencies from LLM analysis
            dependencies = await self.analyze_workflow_dependencies(workflow_plan)
            all_installed = True
            
            # Check for kb-python special case
            if any(tool["name"] == "kb" for tool in dependencies.get("tools", [])):
                self.logger.info("Detected kb tool requirement, installing kb-python via pip")
                result = subprocess.run("pip install kb-python", shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    self.logger.error(f"Failed to install kb-python: {result.stderr}")
                    all_installed = False
                else:
                    self.logger.info("Successfully installed kb-python")
                    # Remove kb from tools list since we've handled it
                    dependencies["tools"] = [t for t in dependencies["tools"] if t["name"] != "kb"]
            
            # Check and install remaining tools
            for tool in dependencies.get("tools", []):
                if not self.check_tool(tool["name"]):
                    self.logger.info(f"Installing tool: {tool['name']} ({tool['reason']})")
                    if not self.install_tool(tool):
                        self.logger.error(f"Failed to install tool: {tool['name']}")
                        all_installed = False
                        
            # Check and install Python packages
            for pkg in dependencies.get("python_packages", []):
                if not self.check_python_package(pkg["name"], pkg.get("min_version")):
                    self.logger.info(f"Installing Python package: {pkg['name']} ({pkg['reason']})")
                    if not self.install_python_package(pkg):
                        self.logger.error(f"Failed to install Python package: {pkg['name']}")
                        all_installed = False
                        
            # Check and install R packages
            for pkg in dependencies.get("r_packages", []):
                if not self.check_r_package(pkg["name"]):
                    self.logger.info(f"Installing R package: {pkg['name']} ({pkg['reason']})")
                    if not self.install_r_package(pkg):
                        self.logger.error(f"Failed to install R package: {pkg['name']}")
                        all_installed = False
                        
            return all_installed
            
        except Exception as e:
            self.logger.error(f"Failed to ensure workflow dependencies: {str(e)}")
            raise
