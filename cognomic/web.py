"""
This module provides the web interface for the cognomic chatbot using Chainlit.
It includes functionality to run and analyze workflows, stream log contents, and handle user commands.
"""

import asyncio
import io
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import chainlit as cl

from cognomic.core.llm import LLMInterface
from cognomic.workflow import analyze_workflow, run_workflow

commands = [
    {"id": "Run", "icon": "chart-no-axes-gantt", "description": "Run a workflow"},
    {"id": "Analyse", "icon": "search", "description": "Analyse a workflow"},
]


class RootLogger:
    """
    A context manager for capturing log output in a StringIO buffer.
    """

    def __init__(self):
        # Initialize the root logger and a StringIOHandler for capturing logs.
        self.root_logger = logging.getLogger()
        self.log_handler = StringIOHandler()

    def __enter__(self):
        # Add the StringIOHandler to the root logger.
        self.root_logger.addHandler(self.log_handler)
        return self

    def __getattr__(self, name):
        # Delegate attribute access to the root logger.
        return getattr(self.root_logger, name)

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        # Remove the StringIOHandler from the root logger.
        self.root_logger.removeHandler(self.log_handler)

    def get_log_contents(self):
        """
        Retrieve the current contents of the log buffer.
        """
        return self.log_handler.get_log_contents()


class StringIOHandler(logging.Handler):
    """
    A logging handler that writes log entries to a StringIO buffer.
    """

    def __init__(self):
        super().__init__()
        self.log_capture_string = io.StringIO()

    def emit(self, record):
        # Format the log record and write it to the StringIO buffer.
        log_entry = self.format(record)
        self.log_capture_string.write(log_entry + "\n")

    def get_log_contents(self):
        """
        Retrieve the current contents of the StringIO buffer.
        """
        return self.log_capture_string.getvalue()

    def clear(self):
        """
        Clear the contents of the StringIO buffer.
        """
        self.log_capture_string.truncate(0)
        self.log_capture_string.seek(0)


async def stream_log_contents(root_logger: RootLogger, message: cl.Message):
    """
    Continuously stream log contents to a Chainlit message.
    """
    while True:
        log_contents = root_logger.get_log_contents()
        if log_contents:
            await message.stream_token(log_contents)
        root_logger.log_handler.clear()
        # Sleep for a short time to avoid idling.
        await asyncio.sleep(0.02)


async def raise_error(msg, root_logger: Optional[RootLogger] = None):
    """
    Send an error message to the user, including log contents if available.
    """
    if root_logger:
        log_contents = root_logger.get_log_contents()

        await cl.Message(content=f"```{log_contents}\n\n🚨 Error: {msg}```").send()
    else:
        await cl.Message(content=f"```🚨 Error: {msg}```").send()


async def start_run_workflow(message: cl.Message, root_logger: RootLogger):
    """
    Start and run a workflow based on the user's message content.
    """
    llm = LLMInterface()
    extracted_params = await llm.analyze_run_prompt(message.content)

    response = cl.Message(content="```")

    # TODO: Set this up as context manager so that we guarantee flush.
    stream_task = asyncio.create_task(stream_log_contents(root_logger, response))

    root_logger.info(f"Extracted parameters: {extracted_params}")

    if not extracted_params["success"]:
        await raise_error("Failed to parse input prompt!", root_logger)
        return

    if (
        extracted_params["checkpoint_dir"]
        and not Path(extracted_params["checkpoint_dir"]).exists()
        and not Path(extracted_params["checkpoint_dir"]).is_dir()
    ):
        await raise_error(
            "Checkpoint folder does not exist or is not a folder!", root_logger
        )
        return

    if extracted_params["resume"] and not extracted_params["checkpoint_dir"]:
        await raise_error(
            "Cannot resume workflow without checkpoint folder!", root_logger
        )
        return

    root_logger.info("Starting workflow...")

    await run_workflow(
        extracted_params["prompt"],
        extracted_params["checkpoint_dir"],
        extracted_params["resume"],
    )

    root_logger.info("Workflow completed!")

    stream_task.cancel()

    try:
        await stream_task
    except asyncio.CancelledError:
        pass

    # Final flush of the buffer
    final_log_contents = root_logger.get_log_contents()
    if final_log_contents:
        await response.stream_token(final_log_contents)
        root_logger.log_handler.clear()

    # Complete the code block.
    await response.stream_token("```")


async def start_analyse_workflow(message: cl.Message, root_logger: RootLogger):
    """
    Start and analyze a workflow based on the user's message content.
    """
    llm = LLMInterface()
    extracted_params = await llm.analyze_analyze_prompt(message.content)

    response = cl.Message(content="```")
    stream_task = asyncio.create_task(stream_log_contents(root_logger, response))

    root_logger.info(f"Extracted parameters: {extracted_params}")

    if not extracted_params["success"]:
        await raise_error("Failed to parse input prompt!", root_logger)
        return

    if not extracted_params["analysis_dir"]:
        await raise_error("No results directory specified!", root_logger)
        return

    if (
        not Path(extracted_params["analysis_dir"]).exists()
        or not Path(extracted_params["analysis_dir"]).is_dir()
    ):
        await raise_error(
            "Analysis folder does not exist or is not a folder!", root_logger
        )
        return

    root_logger.info("Starting analysis...")

    await analyze_workflow(
        extracted_params["analysis_dir"], extracted_params["save_report"]
    )

    root_logger.info("Analysis completed!")

    stream_task.cancel()

    try:
        await stream_task
    except asyncio.CancelledError:
        pass

    # Final flush of the buffer
    final_log_contents = root_logger.get_log_contents()
    if final_log_contents:
        await response.stream_token(final_log_contents)
        root_logger.log_handler.clear()

    # Complete the code block.
    await response.stream_token("```")


@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the chatbot session and send a welcome message to the user.
    """
    try:
        user_execution_dir = os.environ["USER_EXECUTION_DIR"]
        os.chdir(user_execution_dir)
    except KeyError:
        await raise_error(
            "Warning: USER_EXECUTION_DIR environment variable not set! "
            "Are you serving the web app correctly?"
        )

    c_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c_wd = os.getcwd()

    init_msg = f"""
    👋 Welcome to the cognomic chatbot!

    🕒 Current time: {c_time}
    📁 Current working directory: {c_wd}

    Example tasks:
    - 💨 /Run: run a workflow. Tell the chatbot:
        - That you want to run a workflow,
        - What you workflow you want it to do,
        - Optionally, a checkpoint folder if you want to be able to
          resume a workflow later,
        - Optionally, if you want to resume an existing workflow under 
          the specified checkpoint folder.
    - 🔎 /Analyse: analyse workflow results. Tell the chatbot:
        - That you want to analyse a workflow,
        - The folder where the workflow results to analyse 
          are stored,
        - Optionally, if you want to save a report of the analysis.
    """

    await cl.Message(
        content=init_msg,
    ).send()

    await cl.context.emitter.set_commands(commands)


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming messages and execute the appropriate command.
    """

    with RootLogger() as root_logger:
        try:
            match message.command:
                case "Run":
                    await start_run_workflow(message, root_logger)
                case "Analyse":
                    await start_analyse_workflow(message, root_logger)
                case other:
                    await raise_error(f"Unknown command: {other}", root_logger)
                    return
        except Exception as e:
            await raise_error(f"Error while processing request: {e}", root_logger)
            return
