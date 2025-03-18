"""Dependency management utilities for FlowAgent."""

import subprocess
import sys
import logging
import shutil
from typing import Dict, List, Optional, Union, Set, Tuple, Any
import pkg_resources
import importlib
import json
import os
import glob
from pathlib import Path

from ..core.llm import LLMInterface

logger = logging.getLogger(__name__)

class DependencyManager:
    """Manages software dependencies for FlowAgent."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.llm = LLMInterface()
        
        # Dictionary of tool suites with their components and detection patterns
        self.TOOL_SUITES = {
            "entrez-direct": {
                "components": ["esearch", "efetch"],
                "patterns": [
                    {"type": "conda", "name": "entrez-direct", "channel": "bioconda"},
                    {"type": "path", "paths": ["/usr/bin/", "/usr/local/bin/"]}
                ]
            },
            "sra-toolkit": {
                "components": ["prefetch", "fasterq-dump", "fastq-dump", "sam-dump"],
                "detection_patterns": [
                    "/usr/local/bin/{}",
                    "/usr/bin/{}",
                    "~/miniconda3/bin/{}",
                    "~/anaconda3/bin/{}",
                    "/opt/homebrew/bin/{}",
                    "/opt/conda/bin/{}",
                    "~/.conda/envs/*/bin/{}"
                ]
            },
            "blast": {
                "components": ["blastn", "blastp", "blastx", "tblastn", "tblastx", "makeblastdb"],
                "detection_patterns": [
                    "/usr/local/bin/{}",
                    "/usr/bin/{}",
                    "~/miniconda3/bin/{}",
                    "~/anaconda3/bin/{}",
                    "/opt/homebrew/bin/{}",
                    "/opt/ncbi/blast/bin/{}"
                ]
            },
            "samtools": {
                "components": ["samtools", "bcftools", "tabix", "bgzip"],
                "detection_patterns": [
                    "/usr/local/bin/{}",
                    "/usr/bin/{}",
                    "~/miniconda3/bin/{}",
                    "~/anaconda3/bin/{}",
                    "/opt/homebrew/bin/{}"
                ]
            },
            "alignment": {
                "components": ["bwa", "bowtie", "bowtie2", "STAR", "hisat2"],
                "detection_patterns": [
                    "/usr/local/bin/{}",
                    "/usr/bin/{}",
                    "~/miniconda3/bin/{}",
                    "~/anaconda3/bin/{}",
                    "/opt/homebrew/bin/{}"
                ]
            },
            "rna-seq": {
                "components": ["kallisto", "salmon", "stringtie", "rsem-prepare-reference", "rsem-calculate-expression"],
                "detection_patterns": [
                    "/usr/local/bin/{}",
                    "/usr/bin/{}",
                    "~/miniconda3/bin/{}",
                    "~/anaconda3/bin/{}",
                    "/opt/homebrew/bin/{}"
                ]
            },
            "qc": {
                "components": ["fastqc", "multiqc", "trimmomatic", "cutadapt", "fastp"],
                "detection_patterns": [
                    "/usr/local/bin/{}",
                    "/usr/bin/{}",
                    "~/miniconda3/bin/{}",
                    "~/anaconda3/bin/{}",
                    "/opt/homebrew/bin/{}"
                ]
            },
            "variant": {
                "components": ["gatk", "picard", "vcftools", "bedtools"],
                "detection_patterns": [
                    "/usr/local/bin/{}",
                    "/usr/bin/{}",
                    "~/miniconda3/bin/{}",
                    "~/anaconda3/bin/{}",
                    "/opt/homebrew/bin/{}"
                ]
            },
            "single-cell": {
                "components": ["kb", "cellranger", "velocyto"],
                "detection_patterns": [
                    "/usr/local/bin/{}",
                    "/usr/bin/{}",
                    "~/miniconda3/bin/{}",
                    "~/anaconda3/bin/{}",
                    "/opt/homebrew/bin/{}"
                ]
            }
        }
        
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
                
                # Handle the case where tools can be either strings or dictionaries
                tools_list = []
                for pkg in dependencies.get("tools", []):
                    if isinstance(pkg, str):
                        # Convert 'ncbi-entrez-direct' to 'entrez-direct' if needed
                        pkg_name = 'entrez-direct' if pkg.lower() == 'ncbi-entrez-direct' else pkg
                        tools_list.append({"name": pkg_name, "channel": "bioconda"})
                    elif isinstance(pkg, dict) and "name" in pkg:
                        # If it's already a dictionary with a name, use it directly
                        # But still convert 'ncbi-entrez-direct' to 'entrez-direct' if needed
                        if pkg["name"].lower() == 'ncbi-entrez-direct':
                            pkg["name"] = 'entrez-direct'
                        # Ensure it has a channel if not specified
                        if "channel" not in pkg:
                            pkg["channel"] = "bioconda"
                        tools_list.append(pkg)
                    else:
                        self.logger.warning(f"Skipping invalid tool specification: {pkg}")
                
                return {
                    "tools": tools_list,
                    "python_packages": dependencies.get("python_packages", []),
                    "r_packages": dependencies.get("r_packages", [])
                }
                
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
            
    def check_tool(self, tool_name: str, timeout: int = 5) -> bool:
        """
        Check if a tool is installed and available in the system.
        
        Args:
            tool_name: Name of the tool to check
            timeout: Timeout in seconds for the check
            
        Returns:
            True if the tool is installed, False otherwise
        """
        self.logger.debug(f"Checking if tool '{tool_name}' is installed...")
        
        # Special case for Java-based tools
        if tool_name.lower() in ["picard", "gatk", "trimmomatic"]:
            return self._check_java_tool(tool_name)
        
        # Check if the tool is part of a known suite
        for suite_name, suite_info in self.TOOL_SUITES.items():
            if tool_name in suite_info["components"]:
                return self._check_tool_in_suite(tool_name, suite_info, timeout)
        
        # Default check using 'which' command
        return self._check_tool_with_which(tool_name, timeout)
    
    def _check_tool_in_suite(self, tool_name: str, suite_info: Dict[str, Any], timeout: int = 5) -> bool:
        """
        Check if a tool from a known suite is installed.
        
        Args:
            tool_name: Name of the tool to check
            suite_info: Information about the suite
            timeout: Timeout in seconds for the check
            
        Returns:
            True if the tool is installed, False otherwise
        """
        import os
        import subprocess
        import shlex
        from pathlib import Path
        
        # First try the standard which command
        if self._check_tool_with_which(tool_name, timeout):
            return True
        
        # Try to find the tool using the detection patterns
        for pattern in suite_info["detection_patterns"]:
            try:
                # Replace {} with the tool name
                path_pattern = pattern.format(tool_name)
                
                # Expand user home directory
                path_pattern = os.path.expanduser(path_pattern)
                
                # Handle glob patterns
                if "*" in path_pattern:
                    import glob
                    matching_paths = glob.glob(path_pattern)
                    if matching_paths:
                        self.logger.debug(f"Found {tool_name} at {matching_paths[0]}")
                        return True
                else:
                    # Check if the file exists and is executable
                    if os.path.isfile(path_pattern) and os.access(path_pattern, os.X_OK):
                        self.logger.debug(f"Found {tool_name} at {path_pattern}")
                        return True
            except Exception as e:
                self.logger.debug(f"Error checking pattern {pattern} for {tool_name}: {str(e)}")
        
        # Special handling for SRA Toolkit and Entrez Direct tools
        if tool_name in ["prefetch", "fasterq-dump", "fastq-dump", "sam-dump"]:
            # Check if the SRA Toolkit is installed via conda
            try:
                cmd = f"conda list -f sra-tools"
                result = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=timeout)
                if "sra-tools" in result.stdout:
                    self.logger.debug(f"Found {tool_name} via conda sra-tools package")
                    return True
            except Exception as e:
                self.logger.debug(f"Error checking conda for sra-tools: {str(e)}")
        
        if tool_name in ["esearch", "efetch", "einfo", "elink", "xtract"]:
            # Check if Entrez Direct is installed in the home directory
            edirect_path = os.path.expanduser("~/edirect")
            if os.path.isdir(edirect_path):
                tool_path = os.path.join(edirect_path, tool_name)
                if os.path.isfile(tool_path) and os.access(tool_path, os.X_OK):
                    self.logger.debug(f"Found {tool_name} at {tool_path}")
                    return True
            
            # Check if Entrez Direct is installed via conda
            try:
                cmd = f"conda list -f entrez-direct"
                result = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=timeout)
                if "entrez-direct" in result.stdout:
                    self.logger.debug(f"Found {tool_name} via conda entrez-direct package")
                    return True
            except Exception as e:
                self.logger.debug(f"Error checking conda for entrez-direct: {str(e)}")
        
        return False
    
    def _check_tool_with_which(self, tool_name: str, timeout: int = 5) -> bool:
        """
        Check if a tool is installed using the 'which' command.
        
        Args:
            tool_name: Name of the tool to check
            timeout: Timeout in seconds for the check
            
        Returns:
            True if the tool is installed, False otherwise
        """
        import subprocess
        import shlex
        
        try:
            cmd = f"which {tool_name}"
            result = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=timeout)
            
            if result.returncode == 0 and result.stdout.strip():
                self.logger.debug(f"Found {tool_name} at {result.stdout.strip()}")
                return True
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Timeout expired while checking for {tool_name}")
        except Exception as e:
            self.logger.debug(f"Error checking for {tool_name}: {str(e)}")
        
        return False
    
    def _check_java_tool(self, tool_name: str) -> bool:
        """
        Check if a Java-based tool is installed.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if the tool is installed, False otherwise
        """
        import os
        import subprocess
        import shlex
        from pathlib import Path
        
        # Check if Java is installed
        if not self._check_tool_with_which("java"):
            self.logger.debug(f"Java is not installed, cannot run {tool_name}")
            return False
        
        # Common locations for JAR files
        jar_locations = [
            "/usr/local/share",
            "/usr/share",
            "~/miniconda3/share",
            "~/anaconda3/share",
            "/opt/homebrew/share",
            "~/.conda/envs/*/share"
        ]
        
        # Expand user home directory in all locations
        jar_locations = [os.path.expanduser(loc) for loc in jar_locations]
        
        # Handle glob patterns
        import glob
        expanded_locations = []
        for loc in jar_locations:
            if "*" in loc:
                expanded_locations.extend(glob.glob(loc))
            else:
                expanded_locations.append(loc)
        
        # Look for JAR files
        for loc in expanded_locations:
            if os.path.isdir(loc):
                # Look for tool-specific directories
                tool_dir = os.path.join(loc, tool_name.lower())
                if os.path.isdir(tool_dir):
                    # Look for JAR files
                    jar_files = [f for f in os.listdir(tool_dir) if f.endswith(".jar")]
                    if jar_files:
                        self.logger.debug(f"Found {tool_name} JAR at {os.path.join(tool_dir, jar_files[0])}")
                        return True
        
        # Check if the tool is installed via conda
        try:
            cmd = f"conda list -f {tool_name.lower()}"
            result = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=5)
            if tool_name.lower() in result.stdout.lower():
                self.logger.debug(f"Found {tool_name} via conda package")
                return True
        except Exception as e:
            self.logger.debug(f"Error checking conda for {tool_name}: {str(e)}")
        
        return False
    
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
        tool_name = tool_info.get("name", "")
        channel = tool_info.get("channel", "bioconda")
        min_version = tool_info.get("min_version", "")
        reason = tool_info.get("reason", "")
        
        if not tool_name:
            self.logger.error("No tool name provided for installation")
            return False
            
        self.logger.info(f"Installing tool: {tool_name} ({reason})")
        
        # Check if the tool is already available before attempting installation
        if tool_name in ["esearch", "efetch", "einfo", "elink", "xtract"]:
            if self._check_entrez_direct_tool(tool_name):
                self.logger.info(f"Entrez Direct tool {tool_name} is already available, skipping installation")
                return True
        elif tool_name in ["prefetch", "fasterq-dump", "fastq-dump", "sam-dump"]:
            if self._check_sra_toolkit_tool(tool_name):
                self.logger.info(f"SRA Toolkit tool {tool_name} is already available, skipping installation")
                return True
        elif self.check_tool(tool_name, timeout=1):
            self.logger.info(f"Tool {tool_name} is already available, skipping installation")
            return True
            
        # Try to install using conda
        try:
            version_spec = f"={min_version}" if min_version else ""
            cmd = f"conda install -y -c {channel} {tool_name}{version_spec}"
            
            self.logger.info(f"Running: {cmd}")
            
            # Set a reasonable timeout for conda installation (3 minutes)
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0:
                self.logger.info(f"Successfully installed {tool_name}")
                return True
            else:
                self.logger.error(f"Failed to install {tool_name} using conda: {result.stderr}")
                
                # Check if the tool is available despite installation error
                if tool_name in ["esearch", "efetch", "einfo", "elink", "xtract"]:
                    if self._check_entrez_direct_tool(tool_name):
                        self.logger.warning(f"Entrez Direct tool {tool_name} installation reported an error, but it is available")
                        return True
                elif tool_name in ["prefetch", "fasterq-dump", "fastq-dump", "sam-dump"]:
                    if self._check_sra_toolkit_tool(tool_name):
                        self.logger.warning(f"SRA Toolkit tool {tool_name} installation reported an error, but it is available")
                        return True
                elif self.check_tool(tool_name, timeout=1):
                    self.logger.warning(f"Tool {tool_name} installation reported an error, but it is available")
                    return True
                
                # Provide installation guidance
                self._provide_installation_guidance(tool_name, channel)
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout expired while installing {tool_name}")
            
            # Check if the tool is available despite timeout
            if tool_name in ["esearch", "efetch", "einfo", "elink", "xtract"]:
                if self._check_entrez_direct_tool(tool_name):
                    self.logger.warning(f"Entrez Direct tool {tool_name} installation timed out, but it is available")
                    return True
            elif tool_name in ["prefetch", "fasterq-dump", "fastq-dump", "sam-dump"]:
                if self._check_sra_toolkit_tool(tool_name):
                    self.logger.warning(f"SRA Toolkit tool {tool_name} installation timed out, but it is available")
                    return True
            elif self.check_tool(tool_name, timeout=1):
                self.logger.warning(f"Tool {tool_name} installation timed out, but it is available")
                return True
            
            # Provide installation guidance
            self._provide_installation_guidance(tool_name, channel)
            return False
            
        except Exception as e:
            self.logger.error(f"Error installing {tool_name}: {str(e)}")
            
            # Check if the tool is available despite error
            if tool_name in ["esearch", "efetch", "einfo", "elink", "xtract"]:
                if self._check_entrez_direct_tool(tool_name):
                    self.logger.warning(f"Entrez Direct tool {tool_name} installation error, but it is available")
                    return True
            elif tool_name in ["prefetch", "fasterq-dump", "fastq-dump", "sam-dump"]:
                if self._check_sra_toolkit_tool(tool_name):
                    self.logger.warning(f"SRA Toolkit tool {tool_name} installation error, but it is available")
                    return True
            elif self.check_tool(tool_name, timeout=1):
                self.logger.warning(f"Tool {tool_name} installation error, but it is available")
                return True
            
            # Provide installation guidance
            self._provide_installation_guidance(tool_name, channel)
            return False
            
    def _provide_installation_guidance(self, tool_name: str, channel: str = "bioconda") -> None:
        """Provide guidance for manual installation of a tool.
        
        Args:
            tool_name: Name of the tool
            channel: Conda channel for the tool
        """
        self.logger.info(f"Providing installation guidance for {tool_name}")
        
        # Special case for Entrez Direct tools
        if tool_name in ["esearch", "efetch", "einfo", "elink", "xtract"]:
            self.logger.info("""
To install Entrez Direct tools manually:

Option 1: Using conda (recommended)
    conda install -c bioconda entrez-direct

Option 2: Using the NCBI installer script
    cd ~
    perl -MNet::FTP -e '
        $ftp = new Net::FTP("ftp.ncbi.nlm.nih.gov", Passive => 1);
        $ftp->login; $ftp->binary;
        $ftp->get("/entrez/entrezdirect/edirect.tar.gz");'
    gunzip -c edirect.tar.gz | tar xf -
    rm edirect.tar.gz
    ~/edirect/setup.sh

After installation, add to your PATH:
    export PATH=$PATH:$HOME/edirect
            """)
            
        # Special case for SRA Toolkit tools
        elif tool_name in ["prefetch", "fasterq-dump", "fastq-dump", "sam-dump"]:
            self.logger.info("""
To install SRA Toolkit tools manually:

Option 1: Using conda (recommended)
    conda install -c bioconda sra-tools

Option 2: Download from NCBI
    1. Visit https://github.com/ncbi/sra-tools/wiki/01.-Downloading-SRA-Toolkit
    2. Download the appropriate version for your system
    3. Extract the archive and add the bin directory to your PATH
            """)
            
        # General case for other tools
        else:
            self.logger.info(f"""
To install {tool_name} manually:

Option 1: Using conda (recommended)
    conda install -c {channel} {tool_name}

Option 2: Check the tool's documentation for alternative installation methods
            """)
            
    async def ensure_workflow_dependencies(self, workflow_plan: Dict) -> Tuple[bool, List[str]]:
        """Ensure all required workflow dependencies are installed.
        
        Args:
            workflow_plan: The workflow plan containing steps to analyze
            
        Returns:
            Tuple[bool, List[str]]: A tuple containing:
                - all_installed: True if all dependencies were successfully installed
                - available_but_failed_install: List of tools that failed installation but are available
        """
        try:
            # Get dependencies from LLM analysis
            dependencies = await self.analyze_workflow_dependencies(workflow_plan)
            self.logger.info(f"Identified dependencies: {json.dumps(dependencies, indent=2)}")
            
            all_installed = True
            available_but_failed_install = []
            
            # Check for kb-python special case
            if any(tool["name"] == "kb" for tool in dependencies.get("tools", [])):
                self.logger.info("Detected kb tool requirement, installing kb-python via pip")
                result = subprocess.run("pip install kb-python", shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    self.logger.error(f"Failed to install kb-python: {result.stderr}")
                    all_installed = False
                    # Check if kb is still available despite installation failure
                    if self.check_tool("kb"):
                        self.logger.warning("kb-python installation failed, but kb is available in PATH")
                        available_but_failed_install.append("kb")
                else:
                    self.logger.info("Successfully installed kb-python")
                # Remove kb from tools list since we've handled it
                dependencies["tools"] = [t for t in dependencies["tools"] if t["name"] != "kb"]
            
            # Check and install remaining tools
            for tool in dependencies.get("tools", []):
                tool_name = tool["name"]
                
                # Map package names to their component tools
                component_tools = self._map_package_to_tools(tool_name)
                
                # Check if any of the component tools are already available
                all_components_available = True
                for component in component_tools:
                    if component in ["esearch", "efetch", "einfo", "elink", "xtract"]:
                        if not self._check_entrez_direct_tool(component):
                            all_components_available = False
                            break
                    elif component in ["prefetch", "fasterq-dump", "fastq-dump", "sam-dump"]:
                        if not self._check_sra_toolkit_tool(component):
                            all_components_available = False
                            break
                    elif not self.check_tool(component, timeout=1):
                        all_components_available = False
                        break
                
                if all_components_available:
                    self.logger.info(f"Tool {tool_name} and its components are already available, skipping installation")
                    continue
                
                # Try to install the tool
                self.logger.info(f"Installing tool: {tool_name} ({tool.get('reason', '')})")
                if not self.install_tool(tool):
                    # Check if the tool is available despite installation failure
                    all_components_available_after_failure = True
                    for component in component_tools:
                        if component in ["esearch", "efetch", "einfo", "elink", "xtract"]:
                            if not self._check_entrez_direct_tool(component):
                                all_components_available_after_failure = False
                                break
                        elif component in ["prefetch", "fasterq-dump", "fastq-dump", "sam-dump"]:
                            if not self._check_sra_toolkit_tool(component):
                                all_components_available_after_failure = False
                                break
                        elif not self.check_tool(component, timeout=1):
                            all_components_available_after_failure = False
                            break
                    
                    if all_components_available_after_failure:
                        self.logger.warning(f"Tool {tool_name} installation reported an error, but all components are available")
                        available_but_failed_install.append(tool_name)
                    else:
                        self.logger.error(f"Failed to install tool: {tool_name}")
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
                    
            return all_installed, available_but_failed_install
            
        except Exception as e:
            self.logger.error(f"Failed to ensure workflow dependencies: {str(e)}")
            raise

    def _map_package_to_tools(self, package_name: str) -> List[str]:
        """Map a package name to its component tools.
        
        Args:
            package_name: Name of the package
            
        Returns:
            List of tool names associated with the package
        """
        package_mapping = {
            "entrez-direct": ["esearch", "efetch"],
            "sra-tools": ["prefetch", "fasterq-dump", "fastq-dump", "sam-dump"],
            "blast": ["blastn", "blastp", "blastx", "tblastn", "tblastx", "makeblastdb"],
            "samtools": ["samtools"],
            "bcftools": ["bcftools"],
            "htslib": ["tabix", "bgzip"],
            "bwa": ["bwa"],
            "bowtie": ["bowtie"],
            "bowtie2": ["bowtie2"],
            "star": ["STAR"],
            "hisat2": ["hisat2"],
            "kallisto": ["kallisto"],
            "salmon": ["salmon"],
            "stringtie": ["stringtie"],
            "fastqc": ["fastqc"],
            "multiqc": ["multiqc"],
            "trimmomatic": ["trimmomatic"],
            "cutadapt": ["cutadapt"],
            "fastp": ["fastp"],
            "gatk4": ["gatk"],
            "picard": ["picard"],
            "vcftools": ["vcftools"],
            "bedtools": ["bedtools"],
            "kb-python": ["kb"],
            "cellranger": ["cellranger"],
            "velocyto": ["velocyto"],
            "coreutils": ["mkdir", "tail", "cut", "cat", "echo", "rm", "cp", "mv"],
            "gzip": ["gzip"]
        }
        
        return package_mapping.get(package_name.lower(), [])

    def ensure_workflow_dependencies_sync(self, workflow_plan):
        """Ensure that all dependencies required by a workflow are installed.
        This is a synchronous version of ensure_workflow_dependencies.
        
        Args:
            workflow_plan: Dictionary containing workflow plan with dependencies
            
        Returns:
            tuple: (all_installed, available_but_failed_install)
                all_installed: True if all dependencies were successfully installed
                available_but_failed_install: List of dependencies that failed installation but are available
        """
        try:
            # Extract dependencies from workflow plan
            if "dependencies" not in workflow_plan or not workflow_plan["dependencies"]:
                self.logger.info("No dependencies specified in workflow plan")
                return True, []
            
            dependencies = workflow_plan["dependencies"]
            self.logger.info(f"Identified dependencies: {json.dumps(dependencies, indent=2)}")
            
            # Track which dependencies failed installation but are actually available
            available_but_failed_install = []
            all_installed = True
            
            # Process tool dependencies
            for tool in dependencies.get("tools", []):
                tool_name = tool["name"] if isinstance(tool, dict) else tool
                
                # Map package names to their component tools
                component_tools = self._map_package_to_tools(tool_name)
                
                # Check if any of the component tools are already available
                all_components_available = True
                for component in component_tools:
                    if component in ["esearch", "efetch", "einfo", "elink", "xtract"]:
                        if not self._check_entrez_direct_tool(component):
                            all_components_available = False
                            break
                    elif component in ["prefetch", "fasterq-dump", "fastq-dump", "sam-dump"]:
                        if not self._check_sra_toolkit_tool(component):
                            all_components_available = False
                            break
                    elif not self.check_tool(component, timeout=1):
                        all_components_available = False
                        break
                
                if all_components_available:
                    self.logger.info(f"Tool {tool_name} and its components are already available, skipping installation")
                    continue
                
                # Try to install the tool
                try:
                    self.logger.info(f"Installing tool: {tool_name}")
                    if isinstance(tool, dict):
                        installed = self.install_tool(tool)
                    else:
                        installed = self.install_tool({"name": tool_name, "channel": "bioconda"})
                        
                    if not installed:
                        # Check if the tool is available despite installation failure
                        all_components_available_after_failure = True
                        for component in component_tools:
                            if component in ["esearch", "efetch", "einfo", "elink", "xtract"]:
                                if not self._check_entrez_direct_tool(component):
                                    all_components_available_after_failure = False
                                    break
                            elif component in ["prefetch", "fasterq-dump", "fastq-dump", "sam-dump"]:
                                if not self._check_sra_toolkit_tool(component):
                                    all_components_available_after_failure = False
                                    break
                            elif not self.check_tool(component, timeout=1):
                                all_components_available_after_failure = False
                                break
                        
                        if all_components_available_after_failure:
                            self.logger.warning(f"Tool {tool_name} installation reported an error, but all components are available")
                            available_but_failed_install.append(tool_name)
                        else:
                            self.logger.error(f"Failed to install tool: {tool_name}")
                            all_installed = False
                except Exception as e:
                    self.logger.error(f"Error installing tool {tool_name}: {str(e)}")
                    # Check if the tool is available despite installation error
                    all_components_available_after_failure = True
                    for component in component_tools:
                        if component in ["esearch", "efetch", "einfo", "elink", "xtract"]:
                            if not self._check_entrez_direct_tool(component):
                                all_components_available_after_failure = False
                                break
                        elif component in ["prefetch", "fasterq-dump", "fastq-dump", "sam-dump"]:
                            if not self._check_sra_toolkit_tool(component):
                                all_components_available_after_failure = False
                                break
                        elif not self.check_tool(component, timeout=1):
                            all_components_available_after_failure = False
                            break
                    
                    if all_components_available_after_failure:
                        self.logger.warning(f"Tool {tool_name} installation error, but all components are available")
                        available_but_failed_install.append(tool_name)
                    else:
                        all_installed = False
            
            # Process Python package dependencies
            for pkg in dependencies.get("python_packages", []):
                pkg_name = pkg["name"] if isinstance(pkg, dict) else pkg
                pkg_version = pkg.get("min_version") if isinstance(pkg, dict) else None
                
                if not self.check_python_package(pkg_name, pkg_version):
                    self.logger.info(f"Installing Python package: {pkg_name}")
                    if isinstance(pkg, dict):
                        installed = self.install_python_package(pkg)
                    else:
                        installed = self.install_python_package({"name": pkg_name, "channel": "pip"})
                        
                    if not installed:
                        self.logger.error(f"Failed to install Python package: {pkg_name}")
                        all_installed = False
            
            # Process R package dependencies
            for pkg in dependencies.get("r_packages", []):
                pkg_name = pkg["name"] if isinstance(pkg, dict) else pkg
                
                if not self.check_r_package(pkg_name):
                    self.logger.info(f"Installing R package: {pkg_name}")
                    if isinstance(pkg, dict):
                        installed = self.install_r_package(pkg)
                    else:
                        installed = self.install_r_package({"name": pkg_name, "repository": "cran"})
                        
                    if not installed:
                        self.logger.error(f"Failed to install R package: {pkg_name}")
                        all_installed = False
            
            return all_installed, available_but_failed_install
            
        except Exception as e:
            self.logger.error(f"Failed to ensure workflow dependencies: {str(e)}")
            return False, []

    def check_dependencies(self, workflow_steps: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Check if all dependencies required by workflow steps are available.
        
        Args:
            workflow_steps: List of workflow steps
            
        Returns:
            Dictionary with available and missing dependencies
        """
        # Extract tools from workflow steps
        required_tools = set()
        for step in workflow_steps:
            tools = step.get("tools", [])
            required_tools.update(tools)
        
        self.logger.info(f"Required tools: {', '.join(required_tools)}")
        
        # Check each tool
        available_tools = []
        missing_tools = []
        
        for tool in required_tools:
            # Map package names to their component tools
            component_tools = self._map_package_to_tools(tool)
            
            # Check if all component tools are available
            all_components_available = True
            for component in component_tools:
                # Special handling for known tool suites
                if component in ["esearch", "efetch", "einfo", "elink", "xtract"]:
                    is_available = self._check_entrez_direct_tool(component)
                    if not is_available:
                        all_components_available = False
                        break
                elif component in ["prefetch", "fasterq-dump", "fastq-dump", "sam-dump"]:
                    is_available = self._check_sra_toolkit_tool(component)
                    if not is_available:
                        all_components_available = False
                        break
                else:
                    # Try direct command check first with shorter timeout
                    try:
                        # Use subprocess.run with a timeout to check if the command exists
                        import shlex
                        cmd = f"command -v {component}"
                        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1)
                        
                        if result.returncode == 0 and result.stdout.strip():
                            self.logger.info(f"Tool {component} is available at {result.stdout.strip()}")
                            continue
                    except subprocess.TimeoutExpired:
                        self.logger.debug(f"Command check timed out for {component}")
                    except Exception as e:
                        self.logger.debug(f"Error checking command for {component}: {str(e)}")
                    
                    # Try with which command
                    try:
                        cmd = f"which {component}"
                        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1)
                        
                        if result.returncode == 0 and result.stdout.strip():
                            self.logger.info(f"Tool {component} is available at {result.stdout.strip()}")
                            continue
                    except subprocess.TimeoutExpired:
                        self.logger.debug(f"Which check timed out for {component}")
                    except Exception as e:
                        self.logger.debug(f"Error checking which for {component}: {str(e)}")
                    
                    # Try version command check
                    try:
                        cmd = f"{component} --version"
                        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1)
                        
                        if result.returncode == 0:
                            self.logger.info(f"Tool {component} is available (version check passed)")
                            continue
                    except subprocess.TimeoutExpired:
                        self.logger.debug(f"Version check timed out for {component}")
                    except Exception as e:
                        self.logger.debug(f"Error checking version for {component}: {str(e)}")
                    
                    # Try help command check
                    try:
                        cmd = f"{component} --help"
                        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1)
                        
                        if result.returncode == 0 or result.returncode == 1:  # Some tools return 1 for help
                            self.logger.info(f"Tool {component} is available (help check passed)")
                            continue
                    except subprocess.TimeoutExpired:
                        self.logger.debug(f"Help check timed out for {component}")
                    except Exception as e:
                        self.logger.debug(f"Error checking help for {component}: {str(e)}")
                    
                    # If we get here, the component is not available
                    all_components_available = False
                    break
            
            if all_components_available:
                self.logger.info(f"Tool {tool} and its components are available")
                available_tools.append(tool)
            else:
                self.logger.warning(f"Tool {tool} or some of its components are not available")
                missing_tools.append(tool)
        
        return {
            "available": available_tools,
            "missing": missing_tools
        }
        
    def _check_entrez_direct_tool(self, tool_name: str) -> bool:
        """
        Check if an Entrez Direct tool is installed.
        
        Args:
            tool_name: Name of the Entrez Direct tool to check
            
        Returns:
            True if the tool is installed, False otherwise
        """
        self.logger.info(f"Checking for Entrez Direct tool: {tool_name}")
        
        # Check common locations for Entrez Direct tools
        import os
        import glob
        
        # 1. Check if it's in PATH using which (fast)
        try:
            cmd = f"which {tool_name}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1)
            if result.returncode == 0 and result.stdout.strip():
                self.logger.info(f"Found {tool_name} in PATH: {result.stdout.strip()}")
                return True
        except Exception as e:
            self.logger.debug(f"Error checking PATH for {tool_name}: {str(e)}")
        
        # 2. Check in ~/edirect directory (common installation location)
        edirect_path = os.path.expanduser("~/edirect")
        if os.path.isdir(edirect_path):
            tool_path = os.path.join(edirect_path, tool_name)
            if os.path.isfile(tool_path) and os.access(tool_path, os.X_OK):
                self.logger.info(f"Found {tool_name} in ~/edirect: {tool_path}")
                return True
        
        # 3. Check if installed via conda
        try:
            cmd = "conda list -f entrez-direct"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1)
            if "entrez-direct" in result.stdout:
                # Check in conda environment bin directory
                conda_env = os.environ.get("CONDA_PREFIX")
                if conda_env:
                    tool_path = os.path.join(conda_env, "bin", tool_name)
                    if os.path.isfile(tool_path) and os.access(tool_path, os.X_OK):
                        self.logger.info(f"Found {tool_name} in conda environment: {tool_path}")
                        return True
                
                # It's installed via conda but we couldn't find the exact path
                # Let's assume it's available since conda should add it to PATH
                self.logger.info(f"Entrez Direct is installed via conda, assuming {tool_name} is available")
                return True
        except Exception as e:
            self.logger.debug(f"Error checking conda for entrez-direct: {str(e)}")
        
        # 4. Check common conda installation paths
        conda_paths = [
            os.path.expanduser("~/miniconda3/bin"),
            os.path.expanduser("~/anaconda3/bin"),
            "/opt/conda/bin",
            "/usr/local/bin",
            "/usr/bin"
        ]
        
        for path in conda_paths:
            if os.path.isdir(path):
                tool_path = os.path.join(path, tool_name)
                if os.path.isfile(tool_path) and os.access(tool_path, os.X_OK):
                    self.logger.info(f"Found {tool_name} in {path}: {tool_path}")
                    return True
        
        # 5. Check if the tool is available by running a simple command
        # This is a last resort and might be slow, but we'll use a short timeout
        try:
            # Use a simple command that should work for all Entrez Direct tools
            cmd = f"{tool_name} -help"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1)
            if result.returncode == 0 or result.returncode == 1:  # Some tools return 1 for help
                self.logger.info(f"{tool_name} is available (help command worked)")
                return True
        except Exception as e:
            self.logger.debug(f"Error running help command for {tool_name}: {str(e)}")
        
        self.logger.warning(f"Entrez Direct tool {tool_name} not found")
        return False
        
    def _check_sra_toolkit_tool(self, tool_name: str) -> bool:
        """
        Check if an SRA Toolkit tool is installed.
        
        Args:
            tool_name: Name of the SRA Toolkit tool to check
            
        Returns:
            True if the tool is installed, False otherwise
        """
        self.logger.info(f"Checking for SRA Toolkit tool: {tool_name}")
        
        # Check common locations for SRA Toolkit tools
        import os
        import glob
        
        # 1. Check if it's in PATH using which (fast)
        try:
            cmd = f"which {tool_name}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1)
            if result.returncode == 0 and result.stdout.strip():
                self.logger.info(f"Found {tool_name} in PATH: {result.stdout.strip()}")
                return True
        except Exception as e:
            self.logger.debug(f"Error checking PATH for {tool_name}: {str(e)}")
        
        # 2. Check if installed via conda
        try:
            cmd = "conda list -f sra-tools"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1)
            if "sra-tools" in result.stdout:
                # Check in conda environment bin directory
                conda_env = os.environ.get("CONDA_PREFIX")
                if conda_env:
                    tool_path = os.path.join(conda_env, "bin", tool_name)
                    if os.path.isfile(tool_path) and os.access(tool_path, os.X_OK):
                        self.logger.info(f"Found {tool_name} in conda environment: {tool_path}")
                        return True
                
                # It's installed via conda but we couldn't find the exact path
                # Let's assume it's available since conda should add it to PATH
                self.logger.info(f"SRA Tools is installed via conda, assuming {tool_name} is available")
                return True
        except Exception as e:
            self.logger.debug(f"Error checking conda for sra-tools: {str(e)}")
        
        # 3. Check common conda installation paths
        conda_paths = [
            os.path.expanduser("~/miniconda3/bin"),
            os.path.expanduser("~/anaconda3/bin"),
            "/opt/conda/bin",
            "/usr/local/bin",
            "/usr/bin"
        ]
        
        for path in conda_paths:
            if os.path.isdir(path):
                tool_path = os.path.join(path, tool_name)
                if os.path.isfile(tool_path) and os.access(tool_path, os.X_OK):
                    self.logger.info(f"Found {tool_name} in {path}: {tool_path}")
                    return True
        
        # 4. Check if any SRA Toolkit tool is available, which suggests the suite is installed
        for sra_tool in ["fastq-dump", "prefetch", "fasterq-dump", "sam-dump"]:
            if sra_tool == tool_name:
                continue  # Skip the current tool
                
            try:
                cmd = f"which {sra_tool}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1)
                if result.returncode == 0 and result.stdout.strip():
                    # Found another SRA tool, check if our tool is in the same directory
                    sra_dir = os.path.dirname(result.stdout.strip())
                    tool_path = os.path.join(sra_dir, tool_name)
                    if os.path.isfile(tool_path) and os.access(tool_path, os.X_OK):
                        self.logger.info(f"Found {tool_name} in SRA Toolkit directory: {tool_path}")
                        return True
            except Exception as e:
                self.logger.debug(f"Error checking for other SRA tools: {str(e)}")
        
        # 5. Check if the tool is available by running a simple command
        # This is a last resort and might be slow, but we'll use a short timeout
        try:
            # Use a simple command that should work for all SRA Toolkit tools
            cmd = f"{tool_name} --help"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1)
            if result.returncode == 0 or result.returncode == 1:  # Some tools return 1 for help
                self.logger.info(f"{tool_name} is available (help command worked)")
                return True
        except Exception as e:
            self.logger.debug(f"Error running help command for {tool_name}: {str(e)}")
        
        self.logger.warning(f"SRA Toolkit tool {tool_name} not found")
        return False
