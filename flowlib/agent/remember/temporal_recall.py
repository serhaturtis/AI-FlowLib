from datetime import datetime
from flowlib.flows.base import Flow
from flowlib.agent.remember.models import RecallRequest, RecallResponse, RecallStrategy, MemoryMatch
from flowlib.flows.decorators import flow, stage, pipeline
from typing import Dict, Any, List
from ...flows.decorators import flow, stage, pipeline
from ...core.errors import ExecutionError
from ...core.context import Context
from .flows import BaseRecallFlow

@flow(
    name="TemporalRecallFlow",
    description="Flow for time-based memory recall that retrieves information based on temporal context",
    is_infrastructure=True
)
class TemporalRecallFlow(BaseRecallFlow):
    """Flow for time-based memory recall"""
    
    def __init__(self, name=None):
        """Initialize the temporal recall flow.
        
        Args:
            name: Optional name for the flow instance
        """
        # Call parent constructor with name or default
        super().__init__(name or "TemporalRecallFlow")
    
    @stage("parse_temporal_query")
    async def parse_temporal_query(self, request: RecallRequest) -> dict:
        """Parse the query to extract temporal information"""
        # Use LLM to extract temporal information from query
        temporal_info = await self.agent.llm.analyze_text(
            f"Query: {request.query}\n"
            "Extract temporal information (time periods, dates, sequences) from this query."
        )
        return {"temporal_info": temporal_info}
    
    @stage("execute_temporal_recall")
    async def execute_temporal_recall(self, request: RecallRequest, temporal_info: dict) -> RecallResponse:
        """Execute temporal recall based on extracted information"""
        # Search memory using temporal criteria
        memories = await self.agent.memory.search_temporal(
            query=request.query,
            temporal_info=temporal_info["temporal_info"],
            limit=request.limit,
            memory_types=request.memory_types
        )
        
        # Convert memories to matches
        matches = [
            MemoryMatch(
                memory_id=m.id,
                content=m.content,
                memory_type=m.type,
                relevance_score=m.relevance,
                metadata={
                    "timestamp": m.timestamp.isoformat(),
                    "temporal_context": m.temporal_context
                }
            )
            for m in memories
        ]
        
        return RecallResponse(
            matches=matches,
            strategy_used=RecallStrategy.TEMPORAL,
            total_matches=len(matches),
            query_analysis={"temporal_info": temporal_info["temporal_info"]}
        )
    
    @pipeline(input_model=RecallRequest, output_model=RecallResponse)
    async def run_pipeline(self, input_data: RecallRequest) -> RecallResponse:
        """Run the temporal recall pipeline.
        
        Args:
            input_data: The recall request
            
        Returns:
            Recall response with time-based matches
        """
        # Create context with the input data
        input_context = Context(data=input_data)
        
        # Get stage instances
        validate_stage = self.get_stage("validate_request")
        parse_stage = self.get_stage("parse_temporal_query")
        execute_stage = self.get_stage("execute_temporal_recall")
        
        # Execute validation
        validated_result = await validate_stage.execute(input_context)
        
        # Execute temporal parsing
        parsed_result = await parse_stage.execute(validated_result)
        
        # Execute temporal recall
        recall_result = await execute_stage.execute(parsed_result)
        
        return recall_result.data 