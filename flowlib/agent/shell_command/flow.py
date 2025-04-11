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
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime

from ...flows.base import Flow
from ...flows.decorators import pipeline
from ...flows.results import FlowResult
from ...agent.decorators.base import agent_flow

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


@agent_flow(
    name="ShellCommand",
    description="Execute shell commands on the local system and capture output"
)
class ShellCommandFlow:
    """Flow for executing shell commands on the local system.
    
    This flow enables agents to execute shell commands on the Linux machine
    they're running on, with control over working directory, environment variables,
    and output capturing.
    """
    
    def __init__(self):
        """Initialize the shell command flow."""
        super().__init__(name_or_instance="ShellCommand")
        
        # Add metadata for agent planner
        self.__flow_metadata__ = {
            "agent_flow": True,
            "is_system": True,
            "handles_shell_commands": True,
            "priority": "medium",
            "requires_caution": True  # Flag to indicate this flow requires careful consideration
        }
    
    @pipeline(input_model=ShellCommandInput, output_model=ShellCommandOutput)
    async def run_pipeline(self, input_data: ShellCommandInput) -> FlowResult:
        """Execute the shell command and return the results.
        
        Args:
            input_data: The shell command input parameters
            
        Returns:
            FlowResult containing command execution results including stdout, stderr, and exit code
        """
        logger.info(f"Executing shell command: {input_data.command}")
        
        # Prepare working directory
        working_dir = input_data.working_dir or os.getcwd()
        if not os.path.exists(working_dir):
            logger.warning(f"Working directory {working_dir} does not exist, using current directory")
            working_dir = os.getcwd()
        
        # Prepare environment variables
        env = os.environ.copy()
        if input_data.env_vars:
            env.update(input_data.env_vars)
        
        start_time = time.time()
        
        try:
            # Determine how to execute the command
            if input_data.shell:
                # Execute through shell (useful for pipes, redirects, etc.)
                process = await asyncio.create_subprocess_shell(
                    input_data.command,
                    stdout=asyncio.subprocess.PIPE if input_data.capture_output else None,
                    stderr=asyncio.subprocess.PIPE if input_data.capture_output else None,
                    cwd=working_dir,
                    env=env
                )
            else:
                # Execute directly (safer for commands without shell features)
                # Split command into arguments
                cmd_args = shlex.split(input_data.command)
                process = await asyncio.create_subprocess_exec(
                    *cmd_args,
                    stdout=asyncio.subprocess.PIPE if input_data.capture_output else None,
                    stderr=asyncio.subprocess.PIPE if input_data.capture_output else None,
                    cwd=working_dir,
                    env=env
                )
            
            # Wait for the process to complete with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=input_data.timeout
                )
                
                # Decode output if captured
                stdout_str = stdout.decode('utf-8', errors='replace') if stdout and input_data.capture_output else ""
                stderr_str = stderr.decode('utf-8', errors='replace') if stderr and input_data.capture_output else ""
                
                exit_code = process.returncode
                success = exit_code == 0
                
            except asyncio.TimeoutError:
                # Handle timeout
                logger.warning(f"Command execution timed out after {input_data.timeout} seconds")
                
                # Try to terminate the process
                try:
                    process.terminate()
                    await asyncio.sleep(0.5)
                    if process.returncode is None:
                        process.kill()
                except Exception as e:
                    logger.error(f"Error terminating process: {str(e)}")
                
                stdout_str = "(Command execution timed out)"
                stderr_str = f"Execution timed out after {input_data.timeout} seconds"
                exit_code = -1
                success = False
        
        except Exception as e:
            # Handle execution errors
            logger.error(f"Error executing command: {str(e)}")
            stdout_str = ""
            stderr_str = f"Error executing command: {str(e)}"
            exit_code = -1
            success = False
        
        execution_time = time.time() - start_time
        
        # Create output model
        output = ShellCommandOutput(
            command=input_data.command,
            exit_code=exit_code,
            stdout=stdout_str,
            stderr=stderr_str,
            execution_time=execution_time,
            success=success,
            working_dir=working_dir
        )
        
        logger.info(f"Command executed with exit code {exit_code} in {execution_time:.2f} seconds")
        
        return output
    
    @classmethod
    def create(cls) -> Flow:
        """Create a new instance of this flow.
        
        Returns:
            A Flow instance
        """
        return cls() 