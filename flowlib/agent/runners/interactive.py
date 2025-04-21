"""
Interactive command-line runner for agents.
"""

import logging
import asyncio
from typing import TYPE_CHECKING, Optional

# Avoid circular import, only type hint AgentCore
if TYPE_CHECKING:
    from ..core.agent import AgentCore
    from ..models.state import AgentState # For type hint in run_autonomous

# Import exceptions
from ..core.errors import NotInitializedError, ExecutionError

logger = logging.getLogger(__name__)

# Sentinel value to signal the worker task to stop
_SENTINEL = object()

async def _agent_worker(
    agent: 'AgentCore',
    input_queue: asyncio.Queue,
    output_queue: asyncio.Queue
):
    """Background task that processes messages from the input queue."""
    logger.info(f"Agent worker started for {agent.name}.")
    while True:
        try:
            # Wait for an item from the input queue
            input_item = await input_queue.get()

            # Check for sentinel to stop the worker
            if input_item is _SENTINEL:
                logger.info("Sentinel received, agent worker stopping.")
                break

            # Process the input using the agent's core logic
            if isinstance(input_item, str):
                 # Assuming the input item is the user message string
                 # Use the refactored internal method if available
                if hasattr(agent, '_handle_single_input'):
                     result = await agent._handle_single_input(input_item)
                elif hasattr(agent, 'process_message'): # Fallback to public method
                     result = await agent.process_message(input_item)
                else:
                     logger.error("Agent has no suitable method (_handle_single_input or process_message) to handle input.")
                     result = None # Or some error indicator
            else:
                 logger.warning(f"Received non-string item in input queue: {type(input_item)}")
                 result = None # Or handle other types if needed

            # Put the result onto the output queue
            await output_queue.put(result)

        except asyncio.CancelledError:
            logger.info("Agent worker task cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in agent worker: {e}", exc_info=True)
            # Put an error indicator onto the output queue
            # TODO: Define a better error reporting mechanism if needed
            await output_queue.put(f"WORKER_ERROR: {e}") 
        finally:
            # Mark the processed task as done (important for queue management)
            input_queue.task_done()

    logger.info("Agent worker finished.")

async def run_interactive_session(agent: 'AgentCore'):
    """Runs a standard interactive command-line session for an agent.

    Uses input/output queues to decouple I/O from agent processing.
    Handles user input, puts it on a queue, gets results from another queue,
    and manages agent saving/shutdown on exit.

    Args:
        agent: An initialized AgentCore instance.
    """
    if not agent or not agent.initialized:
        logger.error("Agent must be initialized before running interactive session.")
        return

    task_id = agent.state.task_id
    print("\n=== Interactive Agent Session ===")
    print(f"Agent: {agent.name}")
    print(f"Persona: {getattr(agent, 'persona', 'N/A')}") # Display persona if available
    print(f"Session ID: {task_id}")
    print("Type 'exit' or 'quit' to end the session.")

    # Display recent history if available (modify as needed)
    if agent.state.system_messages:
        print("\n=== Recent History ===")
        max_history = min(3, len(agent.state.user_messages))
        if max_history > 0:
             for i in range(max_history):
                 user_idx = -(i + 1)
                 sys_idx = -(i + 1)
                 if abs(user_idx) <= len(agent.state.user_messages):
                     print(f"\nYou: {agent.state.user_messages[user_idx]}")
                 if abs(sys_idx) <= len(agent.state.system_messages):
                     print(f"Assistant: {agent.state.system_messages[sys_idx]}")
        print("\n======================")

    # Main conversation loop - Moved from dual_path_main
    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()

    # Start the agent worker task in the background
    worker_task = asyncio.create_task(
        _agent_worker(agent, input_queue, output_queue),
        name=f"agent_worker_{agent.name}"
    )

    while True:
        try:
            user_message = await asyncio.to_thread(input, "\nYou: ")

            if user_message.lower().strip() in ['exit', 'quit', 'bye']:
                logger.info("Exit command received. Shutting down agent.")
                # Signal the worker to stop
                await input_queue.put(_SENTINEL)
                break

            # Put the user message onto the input queue
            await input_queue.put(user_message)

            # Wait for the result from the output queue
            result = await output_queue.get()

            # Display response (prefer result, fallback to state)
            response_displayed = False
            if result and hasattr(result, 'status') and result.status == "SUCCESS" and hasattr(result.data, "response"):
                print(f"\nAssistant: {result.data.response}")
                response_displayed = True
            elif agent.state and agent.state.system_messages:
                # Check if a new system message was added corresponding to this turn
                # This logic might need refinement depending on exact state updates
                print(f"\nAssistant: {agent.state.system_messages[-1]}")
                response_displayed = True
            
            if not response_displayed:
                 logger.warning("No response generated or found in state for the last turn.")

        except EOFError:
            logger.info("EOF received. Shutting down agent.")
            await input_queue.put(_SENTINEL) # Signal worker to stop
            break # Exit loop on EOF (e.g., piped input ends)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Shutting down agent.")
            await input_queue.put(_SENTINEL) # Signal worker to stop
            # Worker cancellation might be handled automatically on main loop exit
            # but sending sentinel is cleaner.
            break # Exit loop on Ctrl+C
        except Exception as e:
            logger.error(f"Error during interactive loop: {e}", exc_info=True)
            print(f"\nAssistant: I encountered an error processing that request: {e}")
            # Decide whether to continue or break on error (currently continues)
            # break

    # Wait for the worker task to finish processing remaining items + sentinel
    await input_queue.join() # Wait until all items are processed
    # worker_task might already be stopped by sentinel, but ensure it's finished
    if not worker_task.done():
         worker_task.cancel()
    await asyncio.wait([worker_task], timeout=5.0) # Wait briefly for cleanup

    # Shutdown the agent gracefully
    try:
        logger.info(f"Saving final state for task {agent.state.task_id}...")
        await agent.save_state()
        logger.info("Shutting down agent...")
        await agent.shutdown()
        logger.info("Agent shutdown complete.")
    except Exception as e:
        logger.error(f"Error during agent save/shutdown: {e}", exc_info=True)

    print("\nSession ended.") 