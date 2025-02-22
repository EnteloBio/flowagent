"""LLM Agent for handling tool execution and decision making."""

import os
import json
import logging
from typing import Any, Dict, List, Optional
import openai
from pathlib import Path
from dotenv import load_dotenv
from ..utils.logging import get_logger
from ..config.settings import Settings
import asyncio
import time

# Get settings
settings = Settings()

logger = get_logger(__name__)

class LLMAgent:
    """Agent that uses LLM (ChatGPT) for decision making and tool execution."""
    
    def __init__(self):
        """Initialize the LLM agent."""
        self.settings = settings
        self.logger = get_logger(__name__)
        
        if not self.settings.openai_api_key:
            raise ValueError("OpenAI API key not found")
            
        # Configure OpenAI client
        self.client = openai.AsyncOpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.OPENAI_BASE_URL,
            timeout=self.settings.TIMEOUT,
            max_retries=0  # We'll handle retries ourselves
        )
        
        # Get model to use (with fallback)
        self.model = self.settings.openai_model_to_use
        self.logger.info(f"Using OpenAI model: {self.model}")
        
        # Rate limiting state
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum time between requests in seconds
        
    async def _wait_for_rate_limit(self):
        """Wait if needed to respect rate limits."""
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
        
    async def _get_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Get a chat completion from OpenAI's API with retries and rate limiting.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional arguments for completion
            
        Returns:
            The completion text
            
        Raises:
            Exception: If there is an error getting the response
        """
        max_retries = self.settings.MAX_RETRIES
        base_delay = self.settings.RETRY_DELAY
        
        for attempt in range(max_retries):
            try:
                # Wait for rate limit if needed
                await self._wait_for_rate_limit()
                
                # Make the API call
                completion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs
                )
                return completion.choices[0].message.content
                
            except openai.error.RateLimitError as e:
                if attempt == max_retries - 1:
                    self.logger.error("OpenAI API quota exceeded")
                    raise Exception("OpenAI API quota exceeded. Disabling LLM functionality.")
                    
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) * (0.5 + 0.5 * time.time() % 1)
                self.logger.warning(f"Rate limit hit, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to get ChatGPT response after {max_retries} attempts: {str(e)}")
                    
                delay = base_delay * (2 ** attempt)
                self.logger.warning(f"Error in request, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries}): {str(e)}")
                await asyncio.sleep(delay)
                
    async def analyze_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze input data and make decisions about processing.
        
        Args:
            data: Input data including file paths, parameters, etc.
            
        Returns:
            Dict containing analysis results and recommendations
        """
        prompt = self._construct_analysis_prompt(data)
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        try:
            response = await self._get_chat_completion(messages, temperature=0.2, max_tokens=1000)
            return self._parse_analysis(response)
        except Exception as e:
            self.logger.error(f"Error analyzing data: {str(e)}")
            raise
            
    async def plan_execution(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan execution steps based on analysis.
        
        Args:
            analysis: Analysis results from analyze_data
            
        Returns:
            List of execution steps with tool configurations
        """
        prompt = self._construct_planning_prompt(analysis)
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        try:
            response = await self._get_chat_completion(messages, temperature=0.2, max_tokens=1000)
            return self._parse_execution_plan(response)
        except Exception as e:
            self.logger.error(f"Error planning execution: {str(e)}")
            raise
            
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors during execution.
        
        Args:
            error: The error that occurred
            context: Context about what was happening
            
        Returns:
            Dict containing error analysis and recovery steps
        """
        prompt = self._construct_error_prompt(error, context)
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        try:
            response = await self._get_chat_completion(messages, temperature=0.2, max_tokens=1000)
            return self._parse_error_handling(response)
        except Exception as e:
            self.logger.error(f"Error handling error: {str(e)}")
            raise
        
    async def generate_command(self, context: Dict[str, Any]) -> str:
        """Generate a command based on context using LLM.
        
        Args:
            context: Command context including tool, action, and parameters
            
        Returns:
            Generated command string
        """
        prompt = self._construct_command_prompt(context)
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        try:
            response = await self._get_chat_completion(messages, temperature=0.1, max_tokens=500)
            return self._parse_command(response)
        except Exception as e:
            self.logger.error(f"Error generating command: {str(e)}")
            raise

    def _get_system_prompt(self) -> str:
        """Get the system prompt for ChatGPT."""
        return """You are an expert bioinformatics assistant specializing in RNA-seq analysis.
        Your primary role is to generate precise command-line instructions for RNA-seq analysis tools.
        
        Key points:
        1. Only generate commands for requested tools (FastQC, MultiQC, Kallisto)
        2. Use exact paths and parameters as provided
        3. Follow tool-specific syntax and requirements strictly
        4. Return only the command, no explanations or markdown
        5. For Kallisto:
           - Handle single-end and paired-end data appropriately
           - Include all necessary parameters (e.g., fragment length for single-end)
           - Follow exact parameter ordering
        
        DO NOT:
        1. Add explanations or comments
        2. Suggest alternative tools or parameters
        3. Use placeholder values - use only provided values
        4. Include markdown formatting
        """
        
    def _construct_analysis_prompt(self, data: Dict[str, Any]) -> str:
        """Construct prompt for data analysis."""
        return f"""Analyze this RNA-seq data for processing with our Kallisto-based workflow.
        Input data: {str(data)}
        
        Consider:
        1. FastQC quality control requirements
        2. Kallisto index requirements
        3. Kallisto quantification parameters
        4. MultiQC report generation
        
        Provide specific recommendations for processing this data through our established workflow steps.
        """
        
    def _construct_planning_prompt(self, analysis: Dict[str, Any]) -> str:
        """Construct prompt for execution planning."""
        return f"""Based on this analysis, plan the execution of our Kallisto-based workflow:
        Analysis: {str(analysis)}
        
        The workflow must follow these exact steps:
        1. Quality Control (FastQC)
        2. MultiQC Report Generation
        3. Kallisto Transcriptome Indexing
        4. Kallisto Quantification
        5. Kallisto MultiQC Reporting
        
        Provide specific commands and parameters for each step.
        """
        
    def _construct_error_prompt(self, error: Exception, context: Dict[str, Any]) -> str:
        """Construct prompt for error handling."""
        return f"""An error occurred in our Kallisto-based RNA-seq workflow:
        Error: {str(error)}
        Context: {str(context)}
        
        Analyze the error and provide recommendations for:
        1. Root cause analysis
        2. Potential fixes within our established workflow
        3. Recovery steps using only our supported tools
        
        Remember we are using:
        - FastQC for quality control
        - MultiQC for reporting
        - Kallisto for transcriptome indexing and quantification
        """
        
    def _construct_command_prompt(self, context: Dict[str, Any]) -> str:
        """Construct prompt for command generation."""
        tool = context.get("tool", "")
        action = context.get("action", "")
        input_type = context.get("input_type", "")
        params = context.get("parameters", {})
        
        if tool == "fastqc":
            return f"""Generate a FastQC command with these specifications:
            Input Files: {params.get('input_files', [])}
            Output Directory: {params.get('output_dir', '')}
            Threads: {params.get('threads', 1)}
            Extract: {params.get('extract', True)}
            
            Requirements:
            1. Use exact paths as provided
            2. Include -o for output directory
            3. Include -t for threads if > 1
            4. Include --extract if specified
            5. Format: fastqc [options] input_files
            6. FastQC can handle .fastq, .fastq.gz, .fq, and .fq.gz files directly
            
            Return ONLY the command, no explanations or markdown.
            """
        elif tool == "kallisto":
            if action == "quantification":
                return f"""Generate a Kallisto quantification command with these specifications:
                Input Type: {input_type}
                Input Files: {params.get('input_files', [])}
                Index File: {params.get('index_file', '')}
                Output Directory: {params.get('output_dir', '')}
                Threads: {params.get('threads', 4)}
                
                Additional Parameters for Single-End Data:
                Fragment Length: {params.get('fragment_length', 200)}
                Fragment SD: {params.get('fragment_sd', 20)}
                Bootstrap Samples: {params.get('bootstrap_samples', 100)}
                
                Requirements:
                1. Use exact paths as provided
                2. For single-end data, include --single, -l (fragment length), and -s (sd)
                3. Include --bootstrap-samples parameter
                4. Format: kallisto quant [options] -i index -o output input_files
                
                Return ONLY the command, no explanations or markdown.
                """
            elif action == "index":
                return f"""Generate a Kallisto index command with these specifications:
                Reference File: {params.get('reference_file', '')}
                Index Output: {params.get('index_file', '')}
                
                Requirements:
                1. Use exact paths as provided
                2. Format: kallisto index -i index_file fasta_file
                
                Return ONLY the command, no explanations or markdown.
                """
        
        return f"""Generate a command for {tool} {action} with parameters:
        {json.dumps(params, indent=2)}
        
        Return ONLY the command, no explanations or markdown.
        """

    def _parse_analysis(self, response: str) -> Dict[str, Any]:
        """Parse LLM analysis response."""
        return {
            "analysis": response,
            "workflow": "kallisto_pseudobulk"
        }
        
    def _parse_execution_plan(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM execution plan response."""
        return [
            {"name": "quality_control", "tool": "fastqc", "parameters": {}},
            {"name": "multiqc", "tool": "multiqc", "parameters": {}},
            {"name": "kallisto_index", "tool": "kallisto", "parameters": {}},
            {"name": "kal_quant", "tool": "kallisto", "parameters": {}},
            {"name": "kallisto_multiqc", "tool": "multiqc", "parameters": {}}
        ]
        
    def _parse_error_handling(self, response: str) -> Dict[str, Any]:
        """Parse LLM error handling response."""
        return {
            "recommendation": response,
            "continue": True
        }

    def _parse_command(self, response: str) -> str:
        """Parse LLM command response."""
        # Remove any markdown formatting
        response = response.strip('`').strip()
        if response.startswith('bash\n'):
            response = response[5:]
        return response.strip()
