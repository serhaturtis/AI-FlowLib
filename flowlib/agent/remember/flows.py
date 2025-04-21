from typing import Optional
from ...flows.base import Flow
from ...flows.decorators import flow, stage, pipeline
from ...core.errors import ExecutionError
from ...core.context import Context
from .models import RecallRequest, RecallResponse, RecallStrategy, MemoryMatch

@flow(
    name="BaseRecallFlow",
    description="Base flow for memory recall operations that provides core recall functionality",
    is_infrastructure=True
)
class BaseRecallFlow(Flow):
    """Base flow for memory recall operations"""
    
    def __init__(self, name=None):
        """Initialize the base recall flow.
        
        Args:
            name: Optional name for the flow instance
        """
        # Call parent constructor with name or default
        super().__init__(name or "BaseRecallFlow")
    
    @stage("validate_request")
    async def validate_request(self, request: RecallRequest) -> RecallRequest:
        """Validate the recall request"""
        if request.strategy == RecallStrategy.ENTITY and not request.entity_id:
            raise ExecutionError("Entity ID required for entity-based recall")
        return request
    
    @stage("execute_recall")
    async def execute_recall(self, request: RecallRequest) -> RecallResponse:
        """Execute the recall strategy - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement execute_recall")
    
    @pipeline(input_model=RecallRequest, output_model=RecallResponse)
    async def run_pipeline(self, input_data: RecallRequest) -> RecallResponse:
        """Run the recall pipeline.
        
        Args:
            input_data: The recall request
            
        Returns:
            Recall response
        """
        # Create context with the input data
        input_context = Context(data=input_data)
        
        # Get stage instances
        validate_stage = self.get_stage("validate_request")
        execute_stage = self.get_stage("execute_recall")
        
        # Execute validation
        validated_result = await validate_stage.execute(input_context)
        
        # Execute recall
        recall_result = await execute_stage.execute(validated_result)
        
        return recall_result.data

@flow(
    name="ContextualRecallFlow",
    description="Flow for context-aware memory recall that considers surrounding context",
    is_infrastructure=True
)
class ContextualRecallFlow(BaseRecallFlow):
    """Flow for context-based memory recall"""
    
    def __init__(self, name=None):
        """Initialize the contextual recall flow.
        
        Args:
            name: Optional name for the flow instance
        """
        # Call parent constructor with name or default
        super().__init__(name or "ContextualRecallFlow")
    
    @stage("analyze_context")
    async def analyze_context(self, request: RecallRequest) -> dict:
        """Analyze the context to determine recall strategy"""
        # Use LLM to analyze context and determine best recall approach
        analysis = await self.agent.llm.analyze_text(
            f"Context: {request.context}\nQuery: {request.query}\n"
            "Analyze the context and query to determine the best recall strategy."
        )
        return {"analysis": analysis}
    
    @stage("execute_recall")
    async def execute_recall(self, request: RecallRequest) -> RecallResponse:
        """Execute contextual recall"""
        # Search memory using context-aware approach
        memories = await self.agent.memory.search(
            query=request.query,
            context=request.context,
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
                metadata=m.metadata
            )
            for m in memories
        ]
        
        return RecallResponse(
            matches=matches,
            strategy_used=RecallStrategy.CONTEXTUAL,
            total_matches=len(matches),
            query_analysis={"context_analysis": request.context}
        )
    
    @pipeline(input_model=RecallRequest, output_model=RecallResponse)
    async def run_pipeline(self, input_data: RecallRequest) -> RecallResponse:
        """Run the contextual recall pipeline.
        
        Args:
            input_data: The recall request
            
        Returns:
            Recall response with matches based on context
        """
        # Create context with the input data
        input_context = Context(data=input_data)
        
        # Get stage instances
        validate_stage = self.get_stage("validate_request")
        analyze_stage = self.get_stage("analyze_context")
        execute_stage = self.get_stage("execute_recall")
        
        # Execute validation
        validated_result = await validate_stage.execute(input_context)
        
        # Execute context analysis
        analysis_result = await analyze_stage.execute(validated_result)
        
        # Execute recall
        recall_result = await execute_stage.execute(analysis_result)
        
        return recall_result.data 