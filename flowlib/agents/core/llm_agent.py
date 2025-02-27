"""LLM-based agent implementation."""

from typing import Dict, Any, List, Optional
import json
import logging

from pydantic import BaseModel, Field
from flowlib.core.resources import ResourceRegistry

from .base import Agent
from ..models.base import AgentAction
from ..tools.flow_tool import FlowTool

logger = logging.getLogger(__name__)

class PlanningResponse(BaseModel):
    """Response from LLM for planning next action."""
    reasoning: str = Field(..., description="Step-by-step reasoning for the decision")
    decision: str = Field(..., description="Either 'TASK_COMPLETE' or the name of the flow to execute")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the decision")

class ReflectionResponse(BaseModel):
    """Response from LLM for reflecting on results."""
    key_insights: List[str] = Field(..., description="List of key insights from the result")
    progress_assessment: str = Field(..., description="Assessment of progress towards goal")
    memory_updates: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {"working": {}, "short_term": {}},
        description="Updates for agent memory"
    )

class LLMAgent(Agent):
    """LLM-powered agent implementation."""
    
    def __init__(
        self,
        tools: List[FlowTool],
        config: Dict[str, Any]
    ):
        """Initialize LLM agent.
        
        Args:
            tools: List of flow tools available to the agent
            config: Agent configuration including:
                - provider_name: Name of LLM provider
                - planner_model: Name of model to use for planning
                - input_generator_model: Name of model to use for input generation
                - reflection_model: Name of model to use for reflection
                - max_retries: Maximum number of retries for LLM calls
        """
        super().__init__(tools, config)
        self.provider = ResourceRegistry.get_resource("provider", config["provider_name"])
        self.max_retries = config.get("max_retries", 3)
    
    async def __aenter__(self):
        """Initialize agent."""
        return await super().__aenter__()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup agent."""
        await super().__aexit__(exc_type, exc_val, exc_tb)
    
    def _format_state_for_prompt(self) -> str:
        """Format current state for prompt context."""
        return f"""
        Current Task: {self.state.current_task}
        
        Completed Flows: {', '.join(self.state.completed_flows) if self.state.completed_flows else 'None'}
        
        Recent Context:
        {json.dumps(self.state.context, indent=2)}
        
        Working Memory:
        {json.dumps(self.state.memory.working, indent=2)}
        
        Recent Results:
        {json.dumps(self.state.artifacts, indent=2)}
        """
    
    async def _plan_next_action(self) -> Optional[AgentAction]:
        """Plan next action using LLM."""
        # Format available tools
        tools_description = "\n\n".join(
            tool.describe_interface()
            for tool in self.tools.values()
        )
        
        # Create planning prompt
        prompt = f"""
        You are an AI agent tasked with solving:
        {self.state.current_task}
        
        Current State:
        {self._format_state_for_prompt()}
        
        Available Tools (Flows):
        {tools_description}
        
        Based on the current state and available tools, what should be the next action?
        If you believe the task is complete, respond with "TASK_COMPLETE".
        
        You must respond with a complete, valid JSON object using this exact structure:
        {{
            "reasoning": "Your step-by-step reasoning for the decision",
            "decision": "Either 'TASK_COMPLETE' or the name of the flow to execute",
            "confidence": 0.95
        }}

        Ensure your response is a complete JSON object with all required fields.
        The confidence must be a number between 0 and 1.
        """
        
        # Get LLM decision
        response = await self.provider.generate_structured(
            prompt=prompt,
            model_name=self.config["planner_model"],
            response_model=PlanningResponse,
            temperature=0.7,
            max_tokens=1024
        )
        
        if response.decision == "TASK_COMPLETE":
            return None
        
        return AgentAction(
            flow_name=response.decision,
            inputs={},  # Inputs will be generated separately
            reasoning=response.reasoning,
            confidence=response.confidence
        )
    
    async def _generate_flow_inputs(
        self,
        flow_tool: FlowTool
    ) -> Optional[BaseModel]:
        """Generate flow inputs using LLM."""
        if not flow_tool.requires_inputs:
            return None
            
        prompt = f"""
        Generate valid inputs for the following flow:
        {flow_tool.describe_interface()}
        
        Current context:
        {self._format_state_for_prompt()}
        
        Generate ONLY the input parameters that match the schema exactly.
        Response must be a valid, complete JSON object matching the input model schema.
        Ensure all required fields are included and the response is properly terminated.
        Do not include any additional text or explanations.
        """
        
        response = await self.provider.generate_structured(
            prompt=prompt,
            model_name=self.config["input_generator_model"],
            response_model=flow_tool.input_model,
            temperature=0.7,
            max_tokens=512,  # Increased token limit for complete response
            top_p=0.95,  # Added to improve response coherence
            repeat_penalty=1.1  # Added to prevent repetition
        )
        
        return response
    
    async def _reflect_on_result(
        self,
        flow_name: str,
        result: BaseModel
    ) -> None:
        """Update agent's understanding using LLM."""
        prompt = f"""
        Analyze the result of flow {flow_name}:
        {result.model_dump_json(indent=2)}
        
        Current State:
        {self._format_state_for_prompt()}
        
        What key information should be remembered? What progress has been made?
        
        You must respond with a complete, valid JSON object using this exact structure:
        {{
            "key_insights": [
                "First key insight",
                "Second key insight"
            ],
            "progress_assessment": "Detailed assessment of progress towards goal",
            "memory_updates": {{
                "working": {{
                    "key1": "value1",
                    "key2": "value2"
                }},
                "short_term": {{
                    "key1": "value1",
                    "key2": "value2"
                }}
            }}
        }}

        Ensure your response is a complete JSON object with all required fields.
        """
        
        reflection = await self.provider.generate_structured(
            prompt=prompt,
            model_name=self.config["reflection_model"],
            response_model=ReflectionResponse,
            temperature=0.6,
            max_tokens=1024
        )
        
        # Update memory with insights
        self.state.memory.working.update(reflection.memory_updates["working"])
        self.state.memory.short_term.update(reflection.memory_updates["short_term"])
        
        # Update context
        self.state.context["last_progress"] = reflection.progress_assessment 