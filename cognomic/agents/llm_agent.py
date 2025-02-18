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
        
    def _get_system_prompt(self) -> str:
        """Get the system prompt for ChatGPT."""
        return """You are an expert bioinformatics assistant specializing in RNA-seq analysis.
        Your role is to help optimize and execute commands for RNA-seq data processing.
        
        When analyzing commands:
        1. Check for potential issues or optimizations
        2. Suggest parameter improvements
        3. Consider resource constraints
        4. Plan for error recovery
        
        When planning execution:
        1. Break complex tasks into steps
        2. Validate inputs and outputs
        3. Monitor resource usage
        4. Prepare error handling
        
        Always format your responses as JSON with clear structure.
        """
        
    def _construct_analysis_prompt(self, data: Dict[str, Any]) -> str:
        """Construct prompt for data analysis."""
        if 'command' in data:
            return f"""Analyze this command for RNA-seq processing:
            
            Command: {data['command']}
            Parameters: {json.dumps(data['parameters'], indent=2)}
            Context: {data.get('context', 'unknown')}
            
            Please provide:
            1. Command validation
            2. Parameter optimization
            3. Resource requirements
            4. Potential issues
            5. Recommended modifications
            
            Format your response as JSON with these keys:
            {{
                "is_valid": true/false,
                "optimized_parameters": {{}},
                "resource_requirements": {{}},
                "potential_issues": [],
                "recommendations": []
            }}
            """
        else:
            return f"""Analyze this RNA-seq data:
            
            Data: {json.dumps(data, indent=2)}
            
            Please provide:
            1. Data validation
            2. Processing recommendations
            3. Resource requirements
            4. Quality control checks
            
            Format your response as JSON with these keys:
            {{
                "is_valid": true/false,
                "processing_steps": [],
                "resource_requirements": {{}},
                "qc_checks": []
            }}
            """
            
    def _construct_planning_prompt(self, analysis: Dict[str, Any]) -> str:
        """Construct prompt for execution planning."""
        return f"""Based on this analysis, plan the execution steps:
        
        Analysis: {json.dumps(analysis, indent=2)}
        
        Please provide a detailed execution plan with:
        1. Sequential steps
        2. Command for each step
        3. Parameters and resources
        4. Validation checks
        
        Format your response as JSON with these keys:
        {{
            "steps": [
                {{
                    "name": "step_name",
                    "command": "command_to_run",
                    "parameters": {{}},
                    "validation": {{}},
                    "is_final": true/false
                }}
            ]
        }}
        """
        
    def _construct_error_prompt(self, error: Exception, context: Dict[str, Any]) -> str:
        """Construct prompt for error handling."""
        return f"""An error occurred during RNA-seq processing:
        
        Error: {str(error)}
        Context: {json.dumps(context, indent=2)}
        
        Please analyze the error and provide:
        1. Error analysis
        2. Recovery steps
        3. Prevention measures
        
        Format your response as JSON with these keys:
        {{
            "error_type": "error_category",
            "severity": "high/medium/low",
            "recoverable": true/false,
            "recovery_steps": [],
            "prevention": []
        }}
        """
        
    def _parse_analysis(self, response: str) -> Dict[str, Any]:
        """Parse ChatGPT's analysis response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing analysis response: {str(e)}")
            self.logger.debug(f"Raw response: {response}")
            raise ValueError("Invalid analysis response format")
        
    def _parse_execution_plan(self, response: str) -> List[Dict[str, Any]]:
        """Parse ChatGPT's execution plan response."""
        try:
            plan = json.loads(response)
            return plan.get("steps", [])
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing execution plan: {str(e)}")
            self.logger.debug(f"Raw response: {response}")
            raise ValueError("Invalid execution plan format")
        
    def _parse_error_handling(self, response: str) -> Dict[str, Any]:
        """Parse ChatGPT's error handling response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing error handling: {str(e)}")
            self.logger.debug(f"Raw response: {response}")
            raise ValueError("Invalid error handling format")
