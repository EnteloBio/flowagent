import pytest
import os
import gzip
from pathlib import Path
from unittest.mock import patch

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'sk-test-key',
        'COGNOMIC_DATA_DIR': '/tmp/cognomic_test'
    }):
        yield

@pytest.fixture
def test_data_dir(tmp_path):
    """Create a test data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir

@pytest.fixture(scope="session")
def mock_fastq_files(test_data_dir):
    """Create mock fastq.gz files for testing."""
    # Create sample1 paired-end files
    sample1_r1 = test_data_dir / "sample1_R1.fastq.gz"
    sample1_r2 = test_data_dir / "sample1_R2.fastq.gz"
    
    # Create minimal valid FASTQ content
    fastq_content = (
        "@read1\n"
        "ACTGACTGACTGACTG\n"
        "+\n"
        "FFFFFFFFFFFFFFFF\n"
    ).encode()
    
    # Write gzipped content
    for file in [sample1_r1, sample1_r2]:
        with gzip.open(file, 'wb') as f:
            f.write(fastq_content)
    
    return test_data_dir

@pytest.fixture(scope="session")
def mock_kallisto_index(test_data_dir):
    """Create a mock kallisto index file."""
    index_file = test_data_dir / "transcripts.idx"
    index_file.touch()
    return index_file

@pytest.fixture(scope="function")
def workflow_env(mock_fastq_files, mock_kallisto_index, monkeypatch):
    """Set up workflow environment variables."""
    env_vars = {
        "COGNOMIC_DATA_DIR": str(mock_fastq_files),
        "COGNOMIC_KALLISTO_INDEX": str(mock_kallisto_index),
        "COGNOMIC_OUTPUT_DIR": str(mock_fastq_files / "output"),
        "COGNOMIC_THREADS": "2"
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    return env_vars

@pytest.fixture(scope="function")
def clean_output_dir(workflow_env):
    """Ensure clean output directory for each test."""
    output_dir = Path(workflow_env["COGNOMIC_OUTPUT_DIR"])
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                for subitem in item.iterdir():
                    subitem.unlink()
                item.rmdir()
        output_dir.rmdir()
    output_dir.mkdir(parents=True)
    return output_dir
