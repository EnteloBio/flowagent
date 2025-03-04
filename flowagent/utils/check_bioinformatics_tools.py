#!/usr/bin/env python3
"""
Check Bioinformatics Tools Script

This script checks for the presence of common bioinformatics tools used by FlowAgent
and provides information about their installation status and versions.
"""

import os
import sys
import subprocess
import shutil
import glob
import platform
from typing import Dict, List, Tuple, Any, Optional

# Define colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Define tool suites to check
TOOL_SUITES = {
    "SRA Toolkit": ["prefetch", "fasterq-dump", "fastq-dump", "sam-dump"],
    "Entrez Direct": ["esearch", "efetch", "einfo", "elink", "xtract"],
    "BLAST Suite": ["blastn", "blastp", "blastx", "tblastn", "tblastx", "makeblastdb"],
    "Samtools/BCFtools": ["samtools", "bcftools", "tabix", "bgzip"],
    "Alignment Tools": ["bwa", "bowtie", "bowtie2", "STAR", "hisat2"],
    "RNA-Seq Tools": ["kallisto", "salmon", "stringtie", "rsem-prepare-reference", "rsem-calculate-expression"],
    "Quality Control": ["fastqc", "multiqc", "trimmomatic", "cutadapt", "fastp"],
    "Variant Analysis": ["gatk", "picard", "vcftools", "bedtools"],
    "Single-cell Tools": ["kb", "cellranger", "velocyto"]
}

def check_tool(tool_name: str, timeout: int = 5) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if a tool is installed and get its version.
    
    Args:
        tool_name: Name of the tool to check
        timeout: Timeout in seconds for the check
        
    Returns:
        Tuple of (is_installed, path, version)
    """
    # First try to find the tool using 'which'
    try:
        result = subprocess.run(["which", tool_name], capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0 and result.stdout.strip():
            path = result.stdout.strip()
            version = get_tool_version(tool_name, path, timeout)
            return True, path, version
    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        pass
    except Exception:
        pass
    
    # Special case for Java-based tools
    if tool_name.lower() in ["picard", "gatk", "trimmomatic"]:
        # Check if Java is installed
        java_installed = shutil.which("java") is not None
        if not java_installed:
            return False, None, None
        
        # Common locations for JAR files
        jar_locations = [
            "/usr/local/share",
            "/usr/share",
            os.path.expanduser("~/miniconda3/share"),
            os.path.expanduser("~/anaconda3/share"),
            "/opt/homebrew/share"
        ]
        
        # Look for JAR files
        for loc in jar_locations:
            if os.path.isdir(loc):
                # Look for tool-specific directories
                tool_dir = os.path.join(loc, tool_name.lower())
                if os.path.isdir(tool_dir):
                    # Look for JAR files
                    jar_files = [f for f in os.listdir(tool_dir) if f.endswith(".jar")]
                    if jar_files:
                        path = os.path.join(tool_dir, jar_files[0])
                        version = get_tool_version(tool_name, path, timeout)
                        return True, path, version
        
        # Check if the tool is installed via conda
        try:
            result = subprocess.run(["conda", "list", "-f", tool_name.lower()], 
                                   capture_output=True, text=True, timeout=timeout)
            if tool_name.lower() in result.stdout.lower():
                # Extract version from conda list output
                for line in result.stdout.splitlines():
                    if tool_name.lower() in line.lower():
                        parts = line.split()
                        if len(parts) >= 2:
                            version = parts[1]
                            return True, "conda", version
        except Exception:
            pass
    
    # Check common installation directories
    common_dirs = [
        "/usr/local/bin",
        "/usr/bin",
        os.path.expanduser("~/miniconda3/bin"),
        os.path.expanduser("~/anaconda3/bin"),
        "/opt/homebrew/bin"
    ]
    
    for directory in common_dirs:
        path = os.path.join(directory, tool_name)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            version = get_tool_version(tool_name, path, timeout)
            return True, path, version
    
    # Check if the tool is available in a conda environment
    try:
        result = subprocess.run(["conda", "list", "-f", tool_name], 
                               capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0 and tool_name in result.stdout:
            # Extract version from conda list output
            for line in result.stdout.splitlines():
                if tool_name in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        version = parts[1]
                        return True, "conda", version
    except Exception:
        pass
    
    return False, None, None

def get_tool_version(tool_name: str, path: str, timeout: int = 5) -> Optional[str]:
    """
    Get the version of a tool.
    
    Args:
        tool_name: Name of the tool
        path: Path to the tool
        timeout: Timeout in seconds
        
    Returns:
        Version string or None if not available
    """
    version_commands = [
        [tool_name, "--version"],
        [tool_name, "-version"],
        [tool_name, "-v"],
        [tool_name, "--help"]
    ]
    
    # Special cases for certain tools
    if tool_name == "java":
        version_commands = [["java", "-version"]]
    elif tool_name in ["picard", "gatk", "trimmomatic"]:
        if path.endswith(".jar"):
            version_commands = [["java", "-jar", path, "--version"]]
    
    for cmd in version_commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                # Extract version from output
                output = result.stdout if result.stdout else result.stderr
                lines = output.splitlines()
                for line in lines:
                    if "version" in line.lower():
                        # Try to extract version number
                        import re
                        version_match = re.search(r'(\d+\.\d+(\.\d+)?)', line)
                        if version_match:
                            return version_match.group(1)
                        return line.strip()
                # If no version line found, return first line
                if lines:
                    return lines[0].strip()
        except Exception:
            continue
    
    return None

def get_system_info() -> Dict[str, str]:
    """
    Get system information.
    
    Returns:
        Dictionary with system information
    """
    info = {
        "System": platform.system(),
        "Release": platform.release(),
        "Version": platform.version(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "Python": platform.python_version()
    }
    
    # Check for conda
    try:
        result = subprocess.run(["conda", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            info["Conda"] = result.stdout.strip()
    except Exception:
        info["Conda"] = "Not found"
    
    # Check for pip
    try:
        result = subprocess.run(["pip", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            info["Pip"] = result.stdout.strip()
    except Exception:
        info["Pip"] = "Not found"
    
    return info

def print_installation_guide(tool_name: str) -> None:
    """
    Print installation guide for a tool.
    
    Args:
        tool_name: Name of the tool
    """
    print(f"\n{Colors.BLUE}Installation guide for {tool_name}:{Colors.END}")
    
    # Common installation methods
    if tool_name in ["prefetch", "fasterq-dump", "fastq-dump", "sam-dump"]:
        print("  Install SRA Toolkit:")
        print("  - Conda: conda install -c bioconda sra-tools")
        print("  - Homebrew: brew install sratoolkit")
        print("  - Manual: Download from https://github.com/ncbi/sra-tools/wiki/01.-Downloading-SRA-Toolkit")
    
    elif tool_name in ["esearch", "efetch", "einfo", "elink", "xtract"]:
        print("  Install Entrez Direct:")
        print("  - Conda: conda install -c bioconda entrez-direct")
        print("  - Manual: sh -c \"$(curl -fsSL https://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh)\"")
    
    elif tool_name in ["blastn", "blastp", "blastx", "tblastn", "tblastx", "makeblastdb"]:
        print("  Install BLAST:")
        print("  - Conda: conda install -c bioconda blast")
        print("  - Homebrew: brew install blast")
        print("  - Manual: Download from https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=Download")
    
    elif tool_name in ["samtools", "bcftools", "tabix", "bgzip"]:
        print("  Install Samtools/BCFtools:")
        print("  - Conda: conda install -c bioconda samtools bcftools")
        print("  - Homebrew: brew install samtools bcftools")
    
    elif tool_name in ["bwa", "bowtie", "bowtie2", "STAR", "hisat2"]:
        print(f"  Install {tool_name}:")
        print(f"  - Conda: conda install -c bioconda {tool_name.lower()}")
        print(f"  - Homebrew: brew install {tool_name.lower()}")
    
    elif tool_name in ["kallisto", "salmon", "stringtie"]:
        print(f"  Install {tool_name}:")
        print(f"  - Conda: conda install -c bioconda {tool_name}")
        print(f"  - Homebrew: brew install {tool_name}")
    
    elif tool_name in ["fastqc", "multiqc", "trimmomatic", "cutadapt", "fastp"]:
        print(f"  Install {tool_name}:")
        print(f"  - Conda: conda install -c bioconda {tool_name}")
        if tool_name != "trimmomatic":  # trimmomatic not available in Homebrew
            print(f"  - Homebrew: brew install {tool_name}")
    
    elif tool_name in ["gatk", "picard", "vcftools", "bedtools"]:
        print(f"  Install {tool_name}:")
        print(f"  - Conda: conda install -c bioconda {tool_name}")
        print(f"  - Homebrew: brew install {tool_name}")
    
    elif tool_name in ["kb", "cellranger", "velocyto"]:
        if tool_name == "kb":
            print("  Install kb-python:")
            print("  - Pip: pip install kb-python")
            print("  - Conda: conda install -c bioconda kb-python")
        elif tool_name == "cellranger":
            print("  Install Cell Ranger:")
            print("  - Download from https://support.10xgenomics.com/single-cell-gene-expression/software/downloads/latest")
        elif tool_name == "velocyto":
            print("  Install Velocyto:")
            print("  - Pip: pip install velocyto")
            print("  - Conda: conda install -c bioconda velocyto.py")
    
    else:
        print(f"  No specific installation guide available for {tool_name}.")
        print("  - Try: conda install -c bioconda " + tool_name.lower())
        print("  - Or: pip install " + tool_name.lower())

def main() -> None:
    """Main function."""
    print(f"\n{Colors.BOLD}{Colors.UNDERLINE}FlowAgent Bioinformatics Tools Check{Colors.END}\n")
    
    # Print system information
    system_info = get_system_info()
    print(f"{Colors.BOLD}System Information:{Colors.END}")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    print(f"\n{Colors.BOLD}Checking for bioinformatics tools...{Colors.END}")
    
    # Check each tool suite
    all_tools_count = 0
    installed_tools_count = 0
    
    for suite_name, tools in TOOL_SUITES.items():
        print(f"\n{Colors.BOLD}{suite_name}:{Colors.END}")
        suite_installed_count = 0
        
        for tool in tools:
            all_tools_count += 1
            installed, path, version = check_tool(tool)
            
            if installed:
                installed_tools_count += 1
                suite_installed_count += 1
                version_str = f" (v{version})" if version else ""
                path_str = f" [{path}]" if path else ""
                print(f"  {Colors.GREEN}✓ {tool}{version_str}{path_str}{Colors.END}")
            else:
                print(f"  {Colors.RED}✗ {tool} (not found){Colors.END}")
                print_installation_guide(tool)
        
        # Print suite summary
        if suite_installed_count == len(tools):
            print(f"  {Colors.GREEN}All {suite_name} tools installed ({suite_installed_count}/{len(tools)}){Colors.END}")
        elif suite_installed_count > 0:
            print(f"  {Colors.YELLOW}Some {suite_name} tools installed ({suite_installed_count}/{len(tools)}){Colors.END}")
        else:
            print(f"  {Colors.RED}No {suite_name} tools installed (0/{len(tools)}){Colors.END}")
    
    # Print overall summary
    print(f"\n{Colors.BOLD}Summary:{Colors.END}")
    percentage = (installed_tools_count / all_tools_count) * 100 if all_tools_count > 0 else 0
    
    if percentage >= 80:
        print(f"  {Colors.GREEN}Excellent! {installed_tools_count}/{all_tools_count} tools installed ({percentage:.1f}%){Colors.END}")
        print("  FlowAgent should work well with your current setup.")
    elif percentage >= 50:
        print(f"  {Colors.YELLOW}Good. {installed_tools_count}/{all_tools_count} tools installed ({percentage:.1f}%){Colors.END}")
        print("  FlowAgent should work for most workflows, but you might need to install additional tools for specific analyses.")
    else:
        print(f"  {Colors.RED}Limited. {installed_tools_count}/{all_tools_count} tools installed ({percentage:.1f}%){Colors.END}")
        print("  You may need to install more tools for FlowAgent to work effectively.")
    
    print("\nFor more information on installing bioinformatics tools, visit:")
    print("  - Bioconda: https://bioconda.github.io/")
    print("  - Homebrew: https://brew.sh/")
    print("  - NCBI SRA Toolkit: https://github.com/ncbi/sra-tools/wiki/01.-Downloading-SRA-Toolkit")
    print("  - Entrez Direct: https://www.ncbi.nlm.nih.gov/books/NBK179288/")

if __name__ == "__main__":
    main()
