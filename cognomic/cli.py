import sys
import argparse
import os
import shutil
import glob
import asyncio
from .core.workflow_manager import WorkflowManager
from .utils.logging import get_logger, setup_logging

def check_java() -> bool:
    """Check if Java is installed and accessible."""
    try:
        result = os.system('java -version 2>&1')
        return result == 0
    except Exception:
        return False

def check_dependencies() -> bool:
    """Check if required external tools are available."""
    # First check Java as it's required by FastQC
    if not check_java():
        logger.error("Java Runtime Environment (JRE) not found")
        logger.error("FastQC requires Java to run. Please install Java first:")
        logger.error("On macOS:")
        logger.error("  brew install openjdk")
        logger.error("  # or download from https://www.java.com/")
        return False

    # Check bioinformatics tools
    required_tools = ['fastqc', 'multiqc', 'kallisto']
    missing_tools = []
    
    for tool in required_tools:
        if shutil.which(tool) is None:
            missing_tools.append(tool)
    
    if missing_tools:
        logger.error("Missing required tools: " + ", ".join(missing_tools))
        logger.error("Please install the missing tools using one of these methods:")
        logger.error("")
        logger.error("1. Using conda (recommended):")
        logger.error(f"   conda install -c bioconda {' '.join(missing_tools)}")
        logger.error("")
        logger.error("2. Using homebrew:")
        logger.error(f"   brew install {' '.join(missing_tools)}")
        logger.error("")
        logger.error("3. Manual installation:")
        logger.error("   FastQC: https://www.bioinformatics.babraham.ac.uk/projects/fastqc/")
        logger.error("   MultiQC: https://multiqc.info/")
        logger.error("   Kallisto: https://pachterlab.github.io/kallisto/")
        return False
    
    return True

def run():
    """Entry point for the cognomic-run command."""
    parser = argparse.ArgumentParser(description='Run the Cognomic workflow.')
    parser.add_argument('--workflow', type=str, required=True,
                        help='Specify the workflow to execute (e.g., rnaseq, pseudobulk).')
    parser.add_argument('--input', type=str, required=True,
                        help='Specify the input data directory.')
    parser.add_argument('--output', type=str, default='output',
                        help='Specify the output directory. Default: output')
    parser.add_argument('--reference', type=str,
                        help='Path to reference transcriptome (required for RNA-seq workflows).')
    parser.add_argument('--threads', type=int, default=4,
                        help='Number of threads to use. Default: 4')
    parser.add_argument('--memory', type=str, default='16G',
                        help='Memory limit per process. Default: 16G')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for log files')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    logger = get_logger(__name__)
    
    logger.info(f"Starting Cognomic with arguments: {vars(args)}")
    
    # Check dependencies before proceeding
    if not check_dependencies():
        logger.error("Missing required dependencies. Please install them and try again.")
        sys.exit(1)
    
    # Validate input directory
    if not os.path.exists(args.input):
        logger.error(f"Input directory does not exist: {args.input}")
        sys.exit(1)

    # Find FASTQ files
    input_files = glob.glob(os.path.join(args.input, "*.fastq*"))
    if not input_files:
        logger.error(f"No FASTQ files found in input directory: {args.input}")
        logger.error("Expected files with extensions: .fastq, .fastq.gz")
        logger.error(f"Contents of {args.input}:")
        for f in os.listdir(args.input):
            logger.error(f"  {f}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Validate reference file
    if args.reference and not os.path.exists(args.reference):
        logger.error(f"Reference file does not exist: {args.reference}")
        sys.exit(1)

    # Prepare workflow parameters
    workflow_params = {
        'input_files': input_files,
        'output_dir': args.output,
        'threads': args.threads,
        'memory': args.memory,
    }

    # Add reference transcriptome if provided
    if args.reference:
        workflow_params['reference_transcriptome'] = args.reference

    try:
        # Run the workflow
        WorkflowManager.run_workflow(args.workflow, workflow_params)
    except Exception as e:
        logger.error(f"Error running workflow: {e}")
        sys.exit(1)

    sys.exit(0)
