"""
This module provides the web interface for the flowagent chatbot using Chainlit.
It includes functionality to run and analyze workflows, stream log contents,
and handle user commands.
"""

import asyncio
import io
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import chainlit as cl

from flowagent.core.llm import LLMInterface
from flowagent.core.providers import create_provider
from flowagent.core.agent_loop import run_agent_loop
from flowagent.config.settings import Settings
from flowagent.workflow import analyze_workflow, run_workflow

commands = [
    {"id": "Run", "icon": "chart-no-axes-gantt", "description": "Run a workflow"},
    {"id": "Analyse", "icon": "search", "description": "Analyse a workflow"},
    {"id": "Agent", "icon": "bot", "description": "Interactive agent with tool calling"},
]


class OutputMessage:
    """
    A class to manage the output messages and log streaming for the Chainlit chatbot.
    """

    def __init__(self):
        """
        Initialize the OutputMessage instance, set up the root logger,
        and start the async log streaming task.
        """
        self.root_logger = RootLogger()
        self.root_logger.__enter__()

        self.response = cl.Message(content="```\n")

        self.async_task = asyncio.create_task(self._async_stream())

    async def __aenter__(self):
        """
        Enter the async context manager, returning the instance itself.
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the async context manager, cancel the async task, flush the logs,
        and clean up the root logger.
        """
        self.async_task.cancel()

        try:
            await self.async_task
        except asyncio.CancelledError:
            pass

        await self.flush()

        self.root_logger.__exit__(exc_type, exc_val, exc_tb)

    async def flush(self):
        """
        Flush the log contents to the Chainlit message stream.
        """
        log_contents = self.root_logger.get_log_contents()

        if log_contents:
            await self.response.stream_token(log_contents)

        self.root_logger.log_handler.clear()

    async def _async_stream(self):
        """
        Continuously stream log contents to a Chainlit message.
        """
        while True:
            log_contents = self.root_logger.get_log_contents()
            self.root_logger.log_handler.clear()

            if log_contents:
                await self.response.stream_token(log_contents)

            # Sleep for a short time to avoid idling.
            await asyncio.sleep(0.05)

    def __getattr__(self, name):
        """
        Delegate attribute access to the root logger.
        """
        return getattr(self.root_logger, name)


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


async def raise_error(msg, root_logger: Optional[RootLogger] = None):
    """
    Send an error message to the user, including log contents if available.

    Effectively deprecated in favor of `OutputMessage`. If you have an instance
    of `OutputMessage`, use `OutputMessage.root_logger.error()` instead. Messages,
    are streamed to the web interface in real-time. This function is retained for
    if you don't have an instance of `OutputMessage` available.
    """
    if root_logger:
        log_contents = root_logger.get_log_contents()

        await cl.Message(content=f"```{log_contents}\n\n🚨 Error: {msg}```").send()
    else:
        await cl.Message(content=f"```🚨 Error: {msg}```").send()


async def start_run_workflow(message: cl.Message, output: OutputMessage):
    """
    Start and run a workflow based on the user's message content.
    """
    llm = LLMInterface()
    extracted_params = await llm.analyze_run_prompt(message.content)

    output.root_logger.info(f"Extracted parameters: {extracted_params}")

    if not extracted_params["success"]:
        output.root_logger.critical("🚨 Failed to parse input prompt!")
        return

    _cp = Path(extracted_params["checkpoint_dir"]) if extracted_params["checkpoint_dir"] else None
    if (
        _cp is not None
        and (not _cp.exists() or not _cp.is_dir())
    ):
        output.root_logger.critical(
            "🚨 Checkpoint folder does not exist or is not a folder!"
        )
        return

    if extracted_params["resume"] and not extracted_params["checkpoint_dir"]:
        output.root_logger.critical(
            "🚨 Cannot resume workflow without checkpoint folder!"
        )
        return

    output.root_logger.info("Starting workflow...")

    await run_workflow(
        extracted_params["prompt"],
        extracted_params["checkpoint_dir"],
        extracted_params["resume"],
    )

    output.root_logger.info("Workflow completed!")


async def start_analyse_workflow(message: cl.Message, output: OutputMessage):
    """
    Start and analyze a workflow based on the user's message content.
    """
    llm = LLMInterface()
    extracted_params = await llm.analyze_analyze_prompt(message.content)

    output.root_logger.info(f"Extracted parameters: {extracted_params}")

    if not extracted_params["success"]:
        output.root_logger.critical("🚨 Failed to parse input prompt!")
        return

    if not extracted_params["analysis_dir"]:
        output.root_logger.critical("🚨 No results directory specified!")
        return

    if (
        not Path(extracted_params["analysis_dir"]).exists()
        or not Path(extracted_params["analysis_dir"]).is_dir()
    ):
        output.root_logger.critical(
            "🚨 Analysis folder does not exist or is not a folder!"
        )
        return

    output.root_logger.info("Starting analysis...")

    await analyze_workflow(
        extracted_params["analysis_dir"], extracted_params["save_report"]
    )

    output.root_logger.info("Analysis completed!")


async def start_agent_chat(message: cl.Message):
    """Run the interactive agent loop with streaming and tool calling."""
    s = Settings()
    api_key = s.active_api_key or s.OPENAI_API_KEY
    provider = create_provider(
        s.LLM_PROVIDER, model=s.LLM_MODEL,
        api_key=api_key, base_url=s.LLM_BASE_URL,
    )

    # Stream a thinking indicator
    thinking_msg = cl.Message(content="")
    await thinking_msg.send()

    result = await run_agent_loop(provider, message.content)

    # Stream final response
    resp_msg = cl.Message(content="")
    await resp_msg.send()
    response_text = result.get("response", "") or ""
    for chunk in [response_text[i:i+50] for i in range(0, len(response_text), 50)]:
        await resp_msg.stream_token(chunk)
    await resp_msg.update()

    # Show tool call summary
    if result["tool_calls"]:
        summary_lines = [f"**Tools used:** {result['iterations']} iterations\n"]
        for tc in result["tool_calls"]:
            summary_lines.append(f"- `{tc['name']}({json.dumps(tc['arguments'])[:80]})`")
        await cl.Message(content="\n".join(summary_lines)).send()


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
            "🚨 Warning: USER_EXECUTION_DIR environment variable not set! "
            "Are you serving the web app correctly?"
        )
        return

    c_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c_wd = os.getcwd()

    init_msg = f"""
    👋 Welcome to the flowagent chatbot!

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

    async with OutputMessage() as output:
        try:
            match message.command:
                case "Run":
                    await start_run_workflow(message, output)
                case "Analyse":
                    await start_analyse_workflow(message, output)
                case "Agent":
                    await start_agent_chat(message)
                case None:
                    output.root_logger.critical(
                        "🚨 No command provided! Start typing with / to see available commands."
                    )
                    return
                case other:
                    output.root_logger.critical(f"🚨 Unknown command: {other}")
                    return
        except Exception as e:
            output.root_logger.critical(f"🚨 Error while processing request: {e}")
            return
