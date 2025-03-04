#!/usr/bin/env python3
"""
Diagnostic script to check for the availability of bioinformatics tools.

This script checks for the availability of various bioinformatics tools
and provides installation guidance for missing tools.
"""

import os
import sys
import subprocess
import platform
import shutil
import time
from typing import Dict, List, Tuple, Optional, Set
import logging
from pathlib import Path

# Add the parent directory to the path so we can import from flowagent
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from flowagent.utils.dependency_manager import DependencyManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("check_bioinformatics_tools")

# Define tool suites with their components
TOOL_SUITES = {
    "SRA Toolkit": {
        "components": ["prefetch", "fasterq-dump", "fastq-dump", "sam-dump"],
        "channel": "bioconda",
        "package": "sra-tools"
    },
    "Entrez Direct": {
        "components": ["esearch", "efetch", "einfo", "elink", "xtract"],
        "channel": "bioconda",
        "package": "entrez-direct"
    },
    "BLAST Suite": {
        "components": ["blastn", "blastp", "blastx", "tblastn", "tblastx", "makeblastdb"],
        "channel": "bioconda",
        "package": "blast"
    },
    "Samtools Suite": {
        "components": ["samtools", "bcftools", "tabix", "bgzip"],
        "channel": "bioconda",
        "package": "samtools bcftools htslib"
    },
    "Alignment Tools": {
        "components": ["bwa", "bowtie", "bowtie2", "STAR", "hisat2", "minimap2"],
        "channel": "bioconda",
        "package": "bwa bowtie bowtie2 star hisat2 minimap2"
    },
    "RNA-Seq Tools": {
        "components": ["kallisto", "salmon", "stringtie", "featureCounts"],
        "channel": "bioconda",
        "package": "kallisto salmon stringtie subread"
    },
    "Quality Control Tools": {
        "components": ["fastqc", "multiqc", "trimmomatic", "cutadapt", "fastp"],
        "channel": "bioconda",
        "package": "fastqc multiqc trimmomatic cutadapt fastp"
    },
    "Variant Analysis Tools": {
        "components": ["gatk", "picard", "vcftools", "bedtools", "vcflib"],
        "channel": "bioconda",
        "package": "gatk4 picard vcftools bedtools vcflib"
    },
    "Single-cell Tools": {
        "components": ["kb", "cellranger", "velocyto"],
        "channel": "bioconda",
        "package": "kb-python"
    }
}

def get_system_info() -> Dict[str, str]:
    """Get system information."""
    info = {
        "Platform": platform.platform(),
        "Python Version": platform.python_version(),
        "User": os.environ.get("USER", "Unknown"),
        "Home Directory": os.environ.get("HOME", "Unknown"),
        "Current Directory": os.getcwd(),
        "PATH": os.environ.get("PATH", ""),
    }
    
    # Check for conda environment
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        info["Conda Environment"] = os.path.basename(conda_prefix)
        info["Conda Prefix"] = conda_prefix
    else:
        info["Conda Environment"] = "Not active"
    
    return info

def get_tool_version(tool: str) -> Optional[str]:
    """Get the version of a tool."""
    version_commands = [
        f"{tool} --version",
        f"{tool} -version",
        f"{tool} -v",
        f"{tool} version",
        f"{tool} -h | head -n 1",
        f"{tool} --help | head -n 1"
    ]
    
    for cmd in version_commands:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split("\n")[0]
        except (subprocess.SubprocessError, subprocess.TimeoutExpired):
            continue
    
    return None

def check_tools() -> Dict[str, Dict[str, Dict[str, str]]]:
    """Check for the availability of bioinformatics tools."""
    dependency_manager = DependencyManager()
    results = {}
    
    for suite_name, suite_info in TOOL_SUITES.items():
        suite_results = {}
        components = suite_info["components"]
        
        for tool in components:
            tool_status = {"status": "Not found", "version": "N/A", "location": "N/A"}
            
            # Check for special tools
            if tool in ["esearch", "efetch", "einfo", "elink", "xtract"]:
                if dependency_manager._check_entrez_direct_tool(tool):
                    tool_status["status"] = "Available"
                    tool_status["location"] = shutil.which(tool) or "Unknown"
                    tool_status["version"] = get_tool_version(tool) or "Unknown"
            elif tool in ["prefetch", "fasterq-dump", "fastq-dump", "sam-dump"]:
                if dependency_manager._check_sra_toolkit_tool(tool):
                    tool_status["status"] = "Available"
                    tool_status["location"] = shutil.which(tool) or "Unknown"
                    tool_status["version"] = get_tool_version(tool) or "Unknown"
            # General tool check
            elif dependency_manager.check_tool(tool, timeout=1):
                tool_status["status"] = "Available"
                tool_status["location"] = shutil.which(tool) or "Unknown"
                tool_status["version"] = get_tool_version(tool) or "Unknown"
            
            suite_results[tool] = tool_status
        
        results[suite_name] = suite_results
    
    return results

def print_results(results: Dict[str, Dict[str, Dict[str, str]]]) -> None:
    """Print the results of the tool check."""
    print("\n" + "=" * 80)
    print("BIOINFORMATICS TOOLS AVAILABILITY REPORT")
    print("=" * 80)
    
    for suite_name, suite_results in results.items():
        print(f"\n{suite_name}:")
        print("-" * 80)
        print(f"{'Tool':<20} {'Status':<15} {'Version':<20} {'Location':<50}")
        print("-" * 80)
        
        for tool, tool_status in suite_results.items():
            status = tool_status["status"]
            version = tool_status["version"]
            location = tool_status["location"]
            
            status_color = "\033[92m" if status == "Available" else "\033[91m"  # Green for available, red for not found
            print(f"{tool:<20} {status_color}{status:<15}\033[0m {version:<20} {location:<50}")
    
    print("\n" + "=" * 80)

def print_system_info(info: Dict[str, str]) -> None:
    """Print system information."""
    print("\n" + "=" * 80)
    print("SYSTEM INFORMATION")
    print("=" * 80)
    
    for key, value in info.items():
        if key == "PATH":
            print(f"\n{key}:")
            for path in value.split(":"):
                print(f"  {path}")
        else:
            print(f"{key}: {value}")
    
    print("\n" + "=" * 80)

def provide_installation_guidance(results: Dict[str, Dict[str, Dict[str, str]]]) -> None:
    """Provide installation guidance for missing tools."""
    missing_suites = {}
    
    for suite_name, suite_results in results.items():
        missing_tools = [tool for tool, status in suite_results.items() if status["status"] == "Not found"]
        if missing_tools:
            missing_suites[suite_name] = missing_tools
    
    if not missing_suites:
        print("\nAll tools are available! No installation needed.")
        return
    
    print("\n" + "=" * 80)
    print("INSTALLATION GUIDANCE")
    print("=" * 80)
    
    for suite_name, missing_tools in missing_suites.items():
        suite_info = TOOL_SUITES[suite_name]
        channel = suite_info["channel"]
        package = suite_info["package"]
        
        print(f"\n{suite_name} - Missing tools: {', '.join(missing_tools)}")
        print(f"  Recommended installation:")
        print(f"    conda install -c {channel} {package}")
        
        # Special cases
        if suite_name == "Entrez Direct":
            print("""
  Alternative installation:
    cd ~
    perl -MNet::FTP -e '
        $ftp = new Net::FTP("ftp.ncbi.nlm.nih.gov", Passive => 1);
        $ftp->login; $ftp->binary;
        $ftp->get("/entrez/entrezdirect/edirect.tar.gz");'
    gunzip -c edirect.tar.gz | tar xf -
    rm edirect.tar.gz
    ~/edirect/setup.sh
    
    # Add to your PATH
    export PATH=$PATH:$HOME/edirect
            """)
        elif suite_name == "SRA Toolkit":
            print("""
  Alternative installation:
    1. Visit https://github.com/ncbi/sra-tools/wiki/01.-Downloading-SRA-Toolkit
    2. Download the appropriate version for your system
    3. Extract the archive and add the bin directory to your PATH
            """)
    
    print("\n" + "=" * 80)

def main():
    """Main function."""
    print("\nChecking for bioinformatics tools...")
    
    # Get system information
    system_info = get_system_info()
    print_system_info(system_info)
    
    # Check for tools
    results = check_tools()
    print_results(results)
    
    # Provide installation guidance
    provide_installation_guidance(results)
    
    print("\nCheck complete!")

if __name__ == "__main__":
    main()
