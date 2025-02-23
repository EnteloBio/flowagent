import os
import logging
from typing import List

# Create a logger
logger = logging.getLogger(__name__)

def find_fastq(directory: str) -> List[str]:
    """Find all FASTQ files in the specified directory."""
    fastq_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.fastq') or file.endswith('.fastq.gz'):
                fastq_files.append(os.path.join(root, file))
    
    # Log the found FASTQ files
    logger.info(f"Found FASTQ files: {fastq_files}")
    return fastq_files
