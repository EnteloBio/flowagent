#!/usr/bin/env python3
"""
Bioinformatics Tools Diagnostic Script

This script checks for the availability of common bioinformatics tools
in the system PATH and in custom installation locations.
"""

import os
import sys
import shutil
import glob
import subprocess
import platform

# Add the parent directory to the Python path to import flowagent modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowagent.utils.dependency_manager import DependencyManager

def check_command_version(command, version_flag="-v"):
    """Check the version of a command-line tool."""
    try:
        result = subprocess.run([command, version_flag], 
                               capture_output=True, 
                               text=True, 
                               timeout=5)
        if result.returncode == 0:
            return result.stdout.strip() or result.stderr.strip()
        else:
            # Try with --version if -v fails
            result = subprocess.run([command, "--version"], 
                                   capture_output=True, 
                                   text=True, 
                                   timeout=5)
            if result.returncode == 0:
                return result.stdout.strip() or result.stderr.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return "Version information not available"

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def main():
    """Main function to check for bioinformatics tools."""
    print_section_header("Bioinformatics Tools Diagnostic")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    
    # Initialize dependency manager
    dm = DependencyManager()
    
    # Define tool suites to check
    tool_suites = {
        "SRA Toolkit": {
            "suite_name": "sra-tools",
            "components": ["prefetch", "fasterq-dump", "fastq-dump", "sam-dump"],
            "installation_guide": """
            To install SRA Toolkit:
            1. Download from: https://github.com/ncbi/sra-tools/wiki/01.-Downloading-SRA-Toolkit
            2. Extract the archive and add the bin directory to your PATH
            3. Or use conda: conda install -c bioconda sra-tools
            """
        },
        "Entrez Direct": {
            "suite_name": "entrez-direct",
            "components": ["esearch", "efetch", "einfo", "elink"],
            "installation_guide": """
            To install Entrez Direct:
            1. Run: sh -c "$(curl -fsSL https://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh)"
            2. Add ~/edirect to your PATH
            """
        },
        "BLAST": {
            "suite_name": "blast",
            "components": ["blastn", "blastp", "blastx", "tblastn", "tblastx", "makeblastdb"],
            "installation_guide": """
            To install BLAST:
            1. Download from: https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs&DOC_TYPE=Download
            2. Or use conda: conda install -c bioconda blast
            """
        },
        "Samtools/BCFtools": {
            "suite_name": "samtools",
            "components": ["samtools", "bcftools", "tabix", "bgzip"],
            "installation_guide": """
            To install Samtools:
            1. Download from: http://www.htslib.org/download/
            2. Or use conda: conda install -c bioconda samtools bcftools
            """
        },
        "Alignment Tools": {
            "components": ["bwa", "bowtie", "bowtie2", "STAR", "hisat2"],
            "installation_guide": """
            To install alignment tools:
            1. BWA: conda install -c bioconda bwa
            2. Bowtie2: conda install -c bioconda bowtie2
            3. STAR: conda install -c bioconda star
            4. HISAT2: conda install -c bioconda hisat2
            """
        },
        "RNA-Seq Tools": {
            "components": ["kallisto", "salmon", "stringtie"],
            "installation_guide": """
            To install RNA-Seq tools:
            1. Kallisto: conda install -c bioconda kallisto
            2. Salmon: conda install -c bioconda salmon
            3. StringTie: conda install -c bioconda stringtie
            """
        },
        "Quality Control": {
            "components": ["fastqc", "multiqc", "trimmomatic", "cutadapt"],
            "installation_guide": """
            To install QC tools:
            1. FastQC: conda install -c bioconda fastqc
            2. MultiQC: conda install -c bioconda multiqc
            3. Trimmomatic: conda install -c bioconda trimmomatic
            4. Cutadapt: conda install -c bioconda cutadapt
            """
        },
        "Variant Analysis": {
            "components": ["gatk", "picard", "vcftools", "bedtools"],
            "installation_guide": """
            To install variant analysis tools:
            1. GATK: conda install -c bioconda gatk4
            2. Picard: conda install -c bioconda picard
            3. VCFtools: conda install -c bioconda vcftools
            4. BEDtools: conda install -c bioconda bedtools
            """
        },
        "Single-Cell Tools": {
            "components": ["kb", "cellranger", "velocyto"],
            "installation_guide": """
            To install single-cell tools:
            1. kb-python: pip install kb-python
            2. Cell Ranger: Download from 10x Genomics website
            3. Velocyto: pip install velocyto
            """
        }
    }
    
    # Check each tool suite
    for suite_name, suite_info in tool_suites.items():
        print_section_header(suite_name)
        
        # Check if suite is available as a whole (if suite_name is defined)
        if "suite_name" in suite_info:
            suite_available = dm.check_tool(suite_info["suite_name"])
            print(f"Suite '{suite_info['suite_name']}' available: {suite_available}")
        
        # Check each component
        for component in suite_info["components"]:
            component_available = dm.check_tool(component)
            print(f"  {component}: {'✓' if component_available else '✗'}", end="")
            
            if component_available:
                # Get version information
                version_info = check_command_version(component)
                if version_info:
                    # Truncate version info if it's too long
                    if len(version_info) > 60:
                        version_info = version_info.split("\n")[0][:60] + "..."
                    print(f" - {version_info}")
                else:
                    print()
            else:
                print()
        
        # Print installation guide if any component is missing
        if not all(dm.check_tool(component) for component in suite_info["components"]):
            print("\nInstallation Guide:")
            print(suite_info["installation_guide"])
    
    # Print PATH information
    print_section_header("Environment Information")
    print("PATH:")
    for path in os.environ.get("PATH", "").split(os.pathsep):
        if path:
            print(f"  {path}")
    
    # Check for conda environments
    print("\nConda Information:")
    conda_path = shutil.which("conda")
    if conda_path:
        print(f"  Conda found: {conda_path}")
        try:
            result = subprocess.run(["conda", "info"], capture_output=True, text=True)
            if result.returncode == 0:
                # Extract and print relevant conda info
                for line in result.stdout.splitlines():
                    if any(key in line for key in ["active env", "conda version", "base env"]):
                        print(f"  {line.strip()}")
        except subprocess.SubprocessError:
            print("  Error getting conda information")
    else:
        print("  Conda not found in PATH")
    
    print("\nDiagnostic complete. Please check the output above for missing tools and installation guides.")

if __name__ == "__main__":
    main()
