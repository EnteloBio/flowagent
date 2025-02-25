"""
API module for Cognomic application.

This module defines the FastAPI application and its endpoints for handling
OpenAI chat completions, running workflows, and retrieving model information.
"""

import io
import json
import logging
import os
import subprocess
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, validator

from .config.settings import Settings
from .core.llm import LLMInterface
from .workflow import analyze_workflow, run_workflow

app = FastAPI()
logger = logging.getLogger(__name__)
settings = Settings()

# Generate a UUID for system_fingerprint
SYSTEM_FINGERPRINT = str(uuid.uuid4())


class StringIOHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_capture_string = io.StringIO()

    def emit(self, record):
        log_entry = self.format(record)
        self.log_capture_string.write(log_entry + "\n")

    def get_log_contents(self):
        return self.log_capture_string.getvalue()

    def clear(self):
        self.log_capture_string.truncate(0)
        self.log_capture_string.seek(0)


class WorkflowRequest(BaseModel):
    """Model for workflow request parameters."""

    prompt: str
    checkpoint_dir: str = None
    analysis_dir: str = None
    resume: bool = False
    save_report: bool = True

    @validator("checkpoint_dir", "analysis_dir")
    def validate_paths(cls, v, values, **kwargs):
        """Validator to ensure provided paths exist."""
        if v and not Path(v).exists():
            raise ValueError(f"Path does not exist: {v}")
        if values.get("resume") and not values.get("checkpoint_dir"):
            raise ValueError(
                "Checkpoint directory must be provided if resume is set to True"
            )
        return v


class OpenAIChatMessage(BaseModel):
    """Model for individual chat messages in OpenAI chat request."""

    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    """Model for OpenAI chat request."""

    model: str
    messages: list[OpenAIChatMessage]
    stream: bool = False


class OpenAIChatResponseChoiceMessage(BaseModel):
    """Model for individual messages in OpenAI chat response choices."""

    role: str
    content: str


class OpenAIChatResponseChoice(BaseModel):
    """Model for choices in OpenAI chat response."""

    index: int
    message: OpenAIChatResponseChoiceMessage
    logprobs: dict = {}
    finish_reason: str


class OpenAIChatResponseUsageDetails(BaseModel):
    """Model for usage details in OpenAI chat response."""

    reasoning_tokens: int
    accepted_prediction_tokens: int
    rejected_prediction_tokens: int


class OpenAIChatResponseUsage(BaseModel):
    """Model for usage in OpenAI chat response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: OpenAIChatResponseUsageDetails


class OpenAIChatResponse(BaseModel):
    """Model for OpenAI chat response."""

    id: str
    object: str
    created: int
    model: str
    system_fingerprint: str
    choices: list[OpenAIChatResponseChoice]
    usage: OpenAIChatResponseUsage

    @classmethod
    def from_basic_message(cls, message: str) -> "OpenAIChatResponse":
        """Create an OpenAIChatResponse from a basic message."""
        return cls(
            id=str(uuid.uuid4()),
            object="chat.completion",
            created=int(time.time()),
            model="cognomic-custom-model",
            system_fingerprint=SYSTEM_FINGERPRINT,
            choices=[
                OpenAIChatResponseChoice(
                    index=0,
                    message=OpenAIChatResponseChoiceMessage(
                        role="assistant", content=message
                    ),
                    logprobs={},
                    finish_reason="stop",
                )
            ],
            usage=OpenAIChatResponseUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                completion_tokens_details=OpenAIChatResponseUsageDetails(
                    reasoning_tokens=0,
                    accepted_prediction_tokens=0,
                    rejected_prediction_tokens=0,
                ),
            ),
        )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {exc}")
    try:
        request_body = await request.body()
        request_json = json.loads(request_body)
        pretty_request_json = json.dumps(request_json, indent=4)
        logger.error(f"Request body:\n{pretty_request_json}")
    except Exception as e:
        logger.error(f"Failed to parse request body as JSON: {e}")
        logger.error(f"Raw request body:\n{request_body.decode('utf-8')}")
    return await request_validation_exception_handler(request, exc)


@app.post("/chat/completions")
async def openai_completions(request: OpenAIChatRequest) -> OpenAIChatResponse:
    """
    Endpoint to handle OpenAI chat completions.

    Args:
        engine_id (str): The engine ID.
        request (OpenAIChatRequest): The chat request.

    Returns:
        OpenAIChatResponse: The chat response.
    """
    logger.info(f"Received request: {request.json()}")
    log_handler = StringIOHandler()
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    try:
        llm = LLMInterface()
        analysis = await llm.analyze_prompt(request.messages[-1].content)

        # Validate and set defaults
        analysis["checkpoint_dir"] = analysis.get("checkpoint_dir", None)
        analysis["analysis_dir"] = analysis.get("analysis_dir", None)
        analysis["resume"] = analysis.get("resume", False)
        analysis["save_report"] = analysis.get("save_report", True)

        if analysis["resume"] and not analysis["checkpoint_dir"]:
            message = "```**Error:** Checkpoint directory must be provided if resume is set to True.```"
            return OpenAIChatResponse.from_basic_message(message)

        match analysis["action"]:
            case "analyze":
                if not analysis["analysis_dir"]:
                    log_contents = log_handler.get_log_contents()
                    root_logger.removeHandler(log_handler)

                    message = (
                        f"## Log Contents:\n\n```{log_contents}```\n\n"
                        "```**Error:** Analysis directory is required for analysis.```"
                    )

                    return OpenAIChatResponse.from_basic_message(message)
                await analyze_workflow(
                    analysis["analysis_dir"], analysis["save_report"]
                )
                log_contents = log_handler.get_log_contents()
                root_logger.removeHandler(log_handler)

                message = f"## Analysis Log:\n\n```{log_contents}```"

                return OpenAIChatResponse.from_basic_message(message)
            case "run":
                await run_workflow(
                    analysis["prompt"], analysis["checkpoint_dir"], analysis["resume"]
                )
                log_contents = log_handler.get_log_contents()
                root_logger.removeHandler(log_handler)

                message = f"## Workflow Log:\n\n```{log_contents}```"

                return OpenAIChatResponse.from_basic_message(message)
            case other:
                message = (
                    f"```**Error:** Invalid action extracted from prompt ({other}).```"
                )
                return OpenAIChatResponse.from_basic_message(message)
    except Exception:
        logger.error(f"Operation failed: \n{traceback.format_exc()}")

        log_contents = log_handler.get_log_contents()
        root_logger.removeHandler(log_handler)

        message = (
            f"## Log Contents:\n\n```{log_contents}```\n\n"
            f"```**Error:** Operation failed: {traceback.format_exc()}```"
        )

        return OpenAIChatResponse.from_basic_message(message)


@app.post("/run")
async def run_workflow_endpoint(request: WorkflowRequest):
    """
    Endpoint to run a workflow.

    Args:
        request (WorkflowRequest): The workflow request.

    Returns:
        dict: Status and message of the operation.
    """
    try:
        llm = LLMInterface()
        analysis = await llm.analyze_prompt(request.prompt)

        # Validate and set defaults
        analysis["checkpoint_dir"] = analysis.get("checkpoint_dir", None)
        analysis["analysis_dir"] = analysis.get("analysis_dir", None)
        analysis["resume"] = analysis.get("resume", False)
        analysis["save_report"] = analysis.get("save_report", True)

        if analysis["resume"] and not analysis["checkpoint_dir"]:
            raise HTTPException(
                status_code=400,
                detail="Checkpoint directory must be provided if resume is set to True",
            )

        if analysis["action"] == "analyze" and analysis["analysis_dir"]:
            await analyze_workflow(analysis["analysis_dir"], analysis["save_report"])
        else:
            await run_workflow(
                analysis["prompt"], analysis["checkpoint_dir"], analysis["resume"]
            )
        return {"status": "success", "message": "Operation completed successfully!"}
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Operation failed: {str(e)}")


@app.get("/models")
async def get_models():
    """
    Endpoint to get the list of models.

    Returns:
        dict: List of models.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "cognomic-custom-model",
                "object": "model",
                "created": SYSTEM_FINGERPRINT,
                "owned_by": "caeruleus-genomics",
            }
        ],
    }


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the FastAPI server.

    Args:
        host (str): The host address.
        port (int): The port number.
    """
    import uvicorn

    try:
        # Ensure that open-webui is installed and available in the system's PATH
        env = os.environ.copy()
        env["WEBUI_AUTH"] = "False"
        env["HF_HUB_OFFLINE"] = "1"
        env["OPENAI_API_BASE_URL"] = f"http://{host}:{port}"
        env["ENABLE_AUTOCOMPLETE_GENERATION"] = "False"
        env["ENABLE_TAGS_GENERATION"] = "False"
        env["TITLE_GENERATION_PROMPT_TEMPLATE"] = "This should fail"
        subprocess.Popen(["open-webui", "serve"], env=env)
    except FileNotFoundError:
        raise RuntimeError(
            "Failed to start open-webui server. Ensure that open-webui is installed and available in the system's PATH."
        )
    # Start the FastAPI server
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
