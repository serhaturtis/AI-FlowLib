from typing import Dict, Any
from ...flows.base import Flow
from flowlib.agent.learn.models import (
    LearningRequest,
    LearningResponse,
    LearningStrategy,
    Entity,
    Relationship
)
from ...flows.decorators import flow, stage, pipeline
from ...core.errors import ExecutionError
from ...core.context import Context

@flow(name="BaseLearningFlow", description="Base flow for knowledge acquisition")
class BaseLearningFlow(Flow):
    """Base flow for knowledge acquisition"""
    
    def __init__(self, name=None):
        """Initialize the base learning flow.
        
        Args:
            name: Optional name for the flow instance
        """
        # Call parent constructor with name or default
        super().__init__(name or "BaseLearningFlow")
    
    def get_description(self) -> str:
        """Get the flow description.
        
        Returns:
            Flow description
        """
        return "Base flow for knowledge acquisition and learning"
    
    @stage("validate_request")
    async def validate_request(self, request: LearningRequest) -> LearningRequest:
        """Validate the learning request"""
        if not request.content:
            raise ExecutionError("Content is required for learning")
        if request.confidence_threshold < 0 or request.confidence_threshold > 1:
            raise ExecutionError("Confidence threshold must be between 0 and 1")
        return request
    
    @stage("execute_learning")
    async def execute_learning(self, request: LearningRequest) -> LearningResponse:
        """Execute the learning strategy - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement execute_learning")
    
    @pipeline(input_model=LearningRequest, output_model=LearningResponse)
    async def run_pipeline(self, input_data: LearningRequest) -> LearningResponse:
        """Run the learning pipeline.
        
        Args:
            input_data: The learning request with content to process
            
        Returns:
            Learning response with results
        """
        # Create context with the input data
        input_context = Context(data=input_data)
        
        # Get stage instances
        validate_stage = self.get_stage("validate_request")
        execute_stage = self.get_stage("execute_learning")
        
        # Execute validation
        validated_result = await validate_stage.execute(input_context)
        
        # Execute learning
        learning_result = await execute_stage.execute(validated_result)
        
        return learning_result.data

@flow(name="EntityExtractionFlow", description="Flow for extracting entities from content")
class EntityExtractionFlow(BaseLearningFlow):
    """Flow for extracting entities from content"""
    
    @stage("analyze_content")
    async def analyze_content(self, request: LearningRequest) -> Dict[str, Any]:
        """Analyze content for entity extraction"""
        # Use LLM to analyze content and identify potential entities
        analysis = await self.agent.llm.analyze_text(
            f"Content: {request.content}\n"
            "Analyze this content to identify:\n"
            "1. Entities and their types\n"
            "2. Entity properties\n"
            "3. Confidence levels for each entity"
        )
        return {"analysis": analysis}
    
    @stage("extract_entities")
    async def extract_entities(self, request: LearningRequest, analysis: Dict[str, Any]) -> LearningResponse:
        """Extract entities from analyzed content"""
        # Process the analysis to extract entities
        entities = []
        for entity_info in analysis["analysis"]["entities"]:
            if entity_info["confidence"] >= request.confidence_threshold:
                entity = Entity(
                    id=entity_info["id"],
                    type=entity_info["type"],
                    properties=entity_info["properties"],
                    confidence=entity_info["confidence"],
                    source=request.content[:100]  # Use first 100 chars as source
                )
                entities.append(entity)
        
        # Store entities in memory
        for entity in entities:
            await self.agent.memory.store_entity(entity)
        
        return LearningResponse(
            entities=entities,
            relationships=[],  # No relationships in entity extraction
            strategy_used=LearningStrategy.ENTITY_EXTRACTION,
            analysis={"content_analysis": analysis["analysis"]},
            confidence_scores={e.id: e.confidence for e in entities}
        )
    
    @pipeline(input_model=LearningRequest, output_model=LearningResponse)
    async def run_pipeline(self, input_data: LearningRequest) -> LearningResponse:
        """Run the entity extraction pipeline.
        
        Args:
            input_data: The learning request with content to process
            
        Returns:
            Learning response with extracted entities
        """
        # Create context with the input data
        input_context = Context(data=input_data)
        
        # Get stage instances
        validate_stage = self.get_stage("validate_request")
        analyze_stage = self.get_stage("analyze_content")
        extract_stage = self.get_stage("extract_entities")
        
        # Execute validation
        validated_result = await validate_stage.execute(input_context)
        
        # Execute analysis
        analysis_result = await analyze_stage.execute(validated_result)
        
        # Execute entity extraction
        extraction_result = await extract_stage.execute(analysis_result)
        
        return extraction_result.data 