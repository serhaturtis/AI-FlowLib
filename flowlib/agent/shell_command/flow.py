"""
Shell Command Flow for Agent System

This module provides an agent flow that enables running shell commands
on the local system and capturing their output.
"""

import asyncio
import logging
import os
import shlex
import subprocess
import time
import shutil
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime

from ...flows.base import Flow
from ...flows.decorators import pipeline
from ...flows.results import FlowResult
from ...agent.decorators.base import agent_flow
from ...providers.registry import provider_registry
from ...providers.constants import ProviderType
from ...providers.llm.base import LLMProvider
from ...resources.registry import resource_registry
from ...resources.constants import ResourceType
from ...resources.decorators import prompt

logger = logging.getLogger(__name__)


class ShellCommandInput(BaseModel):
    """Input model for shell command execution."""
    command: str = Field(..., description="The shell command to execute")
    working_dir: Optional[str] = Field(None, description="Working directory for command execution")
    timeout: Optional[int] = Field(60, description="Timeout in seconds for command execution")
    env_vars: Optional[Dict[str, str]] = Field(None, description="Environment variables to set")
    capture_output: bool = Field(True, description="Whether to capture command output")
    shell: bool = Field(False, description="Whether to execute the command through the shell")


class ShellCommandOutput(BaseModel):
    """Output model for shell command results."""
    command: str = Field(..., description="The command that was executed")
    exit_code: int = Field(..., description="Command exit code")
    stdout: str = Field("", description="Standard output from the command")
    stderr: str = Field("", description="Standard error from the command")
    execution_time: float = Field(..., description="Execution time in seconds")
    success: bool = Field(..., description="Whether the command executed successfully")
    working_dir: str = Field(..., description="Directory where the command was executed")


class ShellCommandExecuteInput(BaseModel):
    """Complete input model for the execute method including metadata fields."""
    input_data: Optional[ShellCommandInput] = None
    inputs: Optional[Dict[str, Any]] = None
    command: Optional[str] = None
    working_dir: Optional[str] = None
    timeout: Optional[int] = None
    env_vars: Optional[Dict[str, str]] = None
    capture_output: Optional[bool] = None
    shell: Optional[bool] = None
    rationale: Optional[str] = Field(None, description="Reasoning for executing this command (for logging only)")
    flow_context: Optional[Dict[str, Any]] = Field(None, description="Flow execution context")
    
    class Config:
        extra = "allow"  # Allow extra fields for forward compatibility


class ShellCommandIntentInput(BaseModel):
    """Input model describing the *intent* for a shell command."""
    intent: str = Field(..., description="A clear description of the goal (e.g., Download file, Search web, List files, Extract text).")
    target_resource: Optional[str] = Field(None, description="The primary resource involved (e.g., URL, file path, search query).")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional parameters or options (e.g., output filename, specific flags).")
    output_description: str = Field("Return the standard output.", description="How the output should be handled or what to extract (e.g., Save to file X, Return summary, Return raw text).")
    working_dir: Optional[str] = Field(None, description="Optional working directory for command execution")
    timeout: Optional[int] = Field(60, description="Timeout in seconds for command execution")


class GeneratedCommand(BaseModel):
    command: str = Field(..., description="The generated shell command string ready for execution.")


@prompt("shell_command_generation")
class ShellCommandGenerationPrompt:
    """Prompt to generate a shell command based on intent and available tools."""
    template = """
You are an expert shell command generator.
Your task is to generate a SINGLE, executable shell command to achieve the given intent, using ONLY the provided available commands.

**Intent:** {{intent}}
**Target Resource:** {{target_resource}}
**Parameters:** {{parameters}}
**Desired Output:** {{output_description}}

**Available Commands:**
{{available_commands_list}}

**Constraints:**
- Construct a single command line.
- Use ONLY commands from the 'Available Commands' list.
- If the intent requires a command that is NOT in the 'Available Commands' list, you MUST output the command 'echo "Error: Cannot achieve intent with available tools."' exactly. Do NOT attempt to use commands that are not listed.
- Be mindful of quoting and escaping special characters.
- If 'Target Resource' or 'Parameters' are empty or None, generate the command accordingly (e.g., `df` without a target, or omitting flags).
- Do NOT add explanations, just the command.

Generated Shell Command:
"""


@agent_flow(
    name="ShellCommand",
    description="Execute shell commands on the local system and capture output"
)
class ShellCommandFlow(Flow):
    """Flow for executing shell commands on the local system.
    
    This flow enables agents to execute shell commands on the Linux machine
    they're running on, with control over working directory, environment variables,
    and output capturing.
    """
    
    # List of common commands to check for availability
    _COMMON_COMMANDS = [
        "curl", "wget", "jq", "python", "pip", "grep", 
        "sed", "awk", "ls", "cd", "mkdir", "rm", "cat", "echo", "df",
        "git", "docker", "kubectl", "tar", "unzip", "zip"
        # Add more common/useful commands as needed
    ]

    def __init__(self):
        """Initialize the shell command flow."""
        super().__init__(name_or_instance="ShellCommand")
        self.__flow_metadata__ = {
            "agent_flow": True,
            "is_system": True,
            "handles_shell_commands": True,
            "priority": "medium",
            "requires_caution": True
        }

    async def _is_command_available(self, command_name: str) -> bool:
        """Check if a command is available in the system PATH."""
        return shutil.which(command_name) is not None

    def _parse_primary_command(self, command_string: str) -> Optional[str]:
        """Attempt to parse the primary command from a shell command string."""
        try:
            # Handle simple cases and pipes/redirects reasonably
            # This is a basic heuristic and might not cover all shell syntax
            parts = shlex.split(command_string)
            if not parts:
                return None
            
            # Often the first element is the command
            # We iterate in case of env var assignments like `VAR=val command`
            for part in parts:
                 if '=' not in part:
                      return part # Return the first part without '=' as the likely command
            return parts[0] # Fallback to first part if all contain '=' (unlikely for commands)
        except ValueError:
            # Handle complex shell syntax shlex might fail on
            # Try a simpler split
            simple_parts = command_string.strip().split()
            if simple_parts:
                return simple_parts[0]
            return None
            
    @pipeline(input_model=ShellCommandIntentInput, output_model=ShellCommandOutput)
    async def run_pipeline(self, input_data: ShellCommandIntentInput) -> ShellCommandOutput:
        """Generates and executes a shell command based on intent and available tools."""
        
        # --- Get LLM Provider (Just-in-Time) --- 
        # Define default provider/model names here or get from config if available
        llm_provider_name = "llamacpp" 
        llm_model_name = "default" # Or get from config/context if needed
        llm: Optional[LLMProvider] = None
        try:
            llm = await provider_registry.get(ProviderType.LLM, llm_provider_name)
            if not llm:
                raise RuntimeError(f"Could not get LLM provider {llm_provider_name}")
        except Exception as llm_e:
             logger.error(f"Failed to get LLM provider '{llm_provider_name}': {llm_e}", exc_info=True)
             # Handle failure to get LLM - return an error output
             return ShellCommandOutput(
                 command=f"(LLM Provider Error for: {input_data.intent})",
                 exit_code=-1,
                 stderr=f"Error: Failed to get required LLM provider '{llm_provider_name}'. {llm_e}",
                 execution_time=0.0,
                 success=False,
                 working_dir=os.getcwd()
             )
        # --- End LLM Provider Get --- 
        
        start_time = time.time()
        working_dir = input_data.working_dir or os.getcwd()
        generated_command_str = ""
        exit_code = -1
        success = False
        stdout_str = ""
        stderr_str = ""

        logger.info(f"Received shell command intent: {input_data.intent}")

        # 1. Check available commands
        available_commands = [cmd for cmd in self._COMMON_COMMANDS if await self._is_command_available(cmd)]
        available_commands_text = "\n".join([f"- {cmd}" for cmd in available_commands])
        logger.debug(f"Available commands: {available_commands}")

        # 2. Generate the command string using LLM
        try:
            command_gen_prompt = resource_registry.get("shell_command_generation", ResourceType.PROMPT)
            prompt_vars = {
                "intent": input_data.intent,
                "target_resource": input_data.target_resource,
                "parameters": input_data.parameters,
                "output_description": input_data.output_description,
                "available_commands_list": available_commands_text
            }
            
            logger.debug("Generating shell command string...")
            # Using generate_structured with GeneratedCommand model
            structured_result: GeneratedCommand = await llm.generate_structured(
                prompt=command_gen_prompt,
                prompt_variables=prompt_vars,
                output_type=GeneratedCommand,
                model_name=llm_model_name
            )
            generated_command_str = structured_result.command.strip()
            logger.info(f"Generated command: {generated_command_str}")
            
            # Basic check if LLM failed to generate a valid command
            if not generated_command_str or generated_command_str.startswith('Error:'):
                 raise ValueError(f"LLM failed to generate a valid command: {generated_command_str}")

        except Exception as gen_e:
            logger.error(f"Failed to generate shell command: {gen_e}", exc_info=True)
            stderr_str = f"Error: Failed to generate appropriate shell command for intent '{input_data.intent}'. {gen_e}"
            # Go directly to output creation
            execution_time = time.time() - start_time
            output = ShellCommandOutput(
                command=f"(Failed Generation for: {input_data.intent})", # Indicate failure source
                exit_code=exit_code,
                stdout=stdout_str,
                stderr=stderr_str,
                execution_time=execution_time,
                success=success,
                working_dir=working_dir
            )
            return output

        # 3. Execute the *generated* command
        command_to_run = generated_command_str # Use the command generated by LLM
        # (Optional safety check on generated command can be added here)

        # Prepare working directory
        if not os.path.exists(working_dir):
            logger.warning(f"Working directory {working_dir} does not exist, using current directory")
            working_dir = os.getcwd()
        
        # --- Execute Command --- 
        try:
            # Use shell=True for simplicity now, as LLM might generate complex commands
            # Consider security implications of shell=True carefully in production
            process = await asyncio.create_subprocess_shell(
                command_to_run,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=input_data.timeout
                )
                stdout_str = stdout.decode('utf-8', errors='replace') if stdout else ""
                stderr_str = stderr.decode('utf-8', errors='replace') if stderr else ""
                exit_code = process.returncode
                success = exit_code == 0
                
            except asyncio.TimeoutError:
                logger.warning(f"Command execution timed out after {input_data.timeout} seconds")
                try:
                    process.terminate()
                    await asyncio.sleep(0.5)
                    if process.returncode is None: process.kill()
                except Exception as term_e:
                    logger.error(f"Error terminating process: {term_e}")
                stdout_str = "(Command execution timed out)"
                stderr_str = f"Execution timed out after {input_data.timeout} seconds"
                exit_code = -1
                success = False
        
        except FileNotFoundError as fnf_error:
            # Specific handling for FileNotFoundError which indicates command not found
            logger.error(f"Command not found during execution attempt: {fnf_error}")
            stderr_str = f"Error: Command not found: {fnf_error.filename}"
            exit_code = 127
            success = False
        except Exception as e:
            logger.error(f"Error executing command: {e}", exc_info=True)
            stderr_str = f"Error executing command: {str(e)}"
            exit_code = -1
            success = False
        # --- End Execution --- 
        
        # 4. Return results
        execution_time = time.time() - start_time
        output = ShellCommandOutput(
            command=command_to_run, # Log the executed command
            exit_code=exit_code,
            stdout=stdout_str,
            stderr=stderr_str,
            execution_time=execution_time,
            success=success,
            working_dir=working_dir
        )
        
        logger.info(f"Command '{command_to_run}' executed with exit code {exit_code} in {execution_time:.2f} seconds")
        return output
    
    @classmethod
    def create(cls) -> Flow:
        """Create a new instance of this flow.
        
        Returns:
            A Flow instance
        """
        return cls() 