from flowlib.flows.base import Flow
from flowlib.agent.remember.models import RecallRequest, RecallResponse, RecallStrategy, MemoryMatch
from flowlib.flows.decorators import flow, stage, pipeline
from .flows import BaseRecallFlow
from ...core.context import Context

@flow(
    name="SemanticRecallFlow",
    description="Flow for semantic-based memory recall that matches based on meaning",
    is_infrastructure=True
)
class SemanticRecallFlow(BaseRecallFlow):
    """Flow for semantic-based memory recall"""
    
    def __init__(self, name=None):
        """Initialize the semantic recall flow.
        
        Args:
            name: Optional name for the flow instance
        """
        # Call parent constructor with name or default
        super().__init__(name or "SemanticRecallFlow")
    
    @stage("analyze_semantic_query")
    async def analyze_semantic_query(self, request: RecallRequest) -> dict:
        """Analyze the query for semantic understanding"""
        # Use LLM to analyze semantic aspects of the query
        semantic_analysis = await self.agent.llm.analyze_text(
            f"Query: {request.query}\n"
            "Analyze this query for semantic understanding, including:\n"
            "1. Key concepts and topics\n"
            "2. Semantic relationships\n"
            "3. Contextual meaning"
        )
        return {"semantic_analysis": semantic_analysis}
    
    @stage("execute_semantic_recall")
    async def execute_semantic_recall(self, request: RecallRequest, semantic_analysis: dict) -> RecallResponse:
        """Execute semantic recall based on analysis"""
        # Search memory using semantic search
        memories = await self.agent.memory.semantic_search(
            query=request.query,
            semantic_context=semantic_analysis["semantic_analysis"],
            limit=request.limit,
            memory_types=request.memory_types
        )
        
        # Convert memories to matches
        matches = [
            MemoryMatch(
                memory_id=m.id,
                content=m.content,
                memory_type=m.type,
                relevance_score=m.semantic_score,
                metadata={
                    "semantic_context": m.semantic_context,
                    "concept_matches": m.matching_concepts
                }
            )
            for m in memories
        ]
        
        return RecallResponse(
            matches=matches,
            strategy_used=RecallStrategy.SEMANTIC,
            total_matches=len(matches),
            query_analysis={"semantic_analysis": semantic_analysis["semantic_analysis"]}
        )
    
    @pipeline(input_model=RecallRequest, output_model=RecallResponse)
    async def run_pipeline(self, input_data: RecallRequest) -> RecallResponse:
        """Run the semantic recall pipeline.
        
        Args:
            input_data: The recall request
            
        Returns:
            Recall response with semantic matches
        """
        # Create context with the input data
        input_context = Context(data=input_data)
        
        # Get stage instances
        validate_stage = self.get_stage("validate_request")
        analyze_stage = self.get_stage("analyze_semantic_query")
        execute_stage = self.get_stage("execute_semantic_recall")
        
        # Execute validation
        validated_result = await validate_stage.execute(input_context)
        
        # Execute semantic analysis
        analysis_result = await analyze_stage.execute(validated_result)
        
        # Execute semantic recall
        recall_result = await execute_stage.execute(analysis_result)
        
        return recall_result.data 