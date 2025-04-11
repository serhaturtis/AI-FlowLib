"""
Entity extraction flow for the learning module.

This flow extracts entities from content and stores them in memory.
"""

from typing import Dict, Any, List
from ...flows.decorators import flow, stage, pipeline
from ...core.errors import ExecutionError
from ...core.context import Context
from .flows import BaseLearningFlow
from .models import (
    LearningRequest,
    LearningResponse,
    LearningStrategy,
    Entity,
    Relationship
)

@flow(
    name="EntityExtractionFlow", 
    description="Extracts entities from content and stores them in memory for later use"
)
class EntityExtractionFlow(BaseLearningFlow):
    """Flow for extracting entities from content"""
    
    def __init__(self, name=None):
        """Initialize the entity extraction flow.
        
        Args:
            name: Optional name for the flow instance
        """
        # Call parent constructor with name or default
        super().__init__(name or "EntityExtractionFlow")
    
    def get_description(self) -> str:
        """Get the flow description.
        
        Returns:
            Flow description
        """
        return "Extracts entities from content and stores them in memory for later use"
    
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