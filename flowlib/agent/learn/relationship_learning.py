from typing import Dict, Any, List
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
from .flows import BaseLearningFlow
from ...core.context import Context

@flow(
    name="RelationshipLearningFlow",
    description="Flow for learning relationships between entities and their properties"
)
class RelationshipLearningFlow(BaseLearningFlow):
    """Flow for learning relationships between entities"""
    
    def __init__(self, name=None):
        """Initialize the relationship learning flow.
        
        Args:
            name: Optional name for the flow instance
        """
        # Call parent constructor with name or default
        super().__init__(name or "RelationshipLearningFlow")
    
    def get_description(self) -> str:
        """Get a description of this flow's purpose.
        
        Returns:
            A string describing what this flow does
        """
        return "Analyzes content to identify and store relationships between existing entities, with confidence scoring and property extraction."
    
    @stage("validate_entities")
    async def validate_entities(self, request: LearningRequest) -> LearningRequest:
        """Validate that required entities exist"""
        if not request.existing_entities:
            raise ExecutionError("Existing entities list is required for relationship learning")
            
        # Verify all entities exist in memory
        for entity_id in request.existing_entities:
            entity = await self.agent.memory.get_entity(entity_id)
            if not entity:
                raise ExecutionError(f"Entity {entity_id} not found in memory")
                
        return request
    
    @stage("analyze_relationships")
    async def analyze_relationships(self, request: LearningRequest) -> Dict[str, Any]:
        """Analyze content for relationships between entities"""
        # Get entity details for context
        entities = []
        for entity_id in request.existing_entities:
            entity = await self.agent.memory.get_entity(entity_id)
            entities.append(entity)
            
        # Use LLM to analyze relationships
        analysis = await self.agent.llm.analyze_text(
            f"Content: {request.content}\n"
            f"Entities: {[e.type for e in entities]}\n"
            "Analyze this content to identify:\n"
            "1. Relationships between entities\n"
            "2. Relationship types and properties\n"
            "3. Confidence levels for each relationship"
        )
        return {"analysis": analysis, "entities": entities}
    
    @stage("extract_relationships")
    async def extract_relationships(self, request: LearningRequest, analysis: Dict[str, Any]) -> LearningResponse:
        """Extract relationships from analyzed content"""
        entities = analysis["entities"]
        relationships = []
        
        # Process the analysis to extract relationships
        for rel_info in analysis["analysis"]["relationships"]:
            if rel_info["confidence"] >= request.confidence_threshold:
                # Find source and target entities
                source_entity = next(
                    (e for e in entities if e.type == rel_info["source_type"]),
                    None
                )
                target_entity = next(
                    (e for e in entities if e.type == rel_info["target_type"]),
                    None
                )
                
                if source_entity and target_entity:
                    relationship = Relationship(
                        id=rel_info["id"],
                        type=rel_info["type"],
                        source_entity_id=source_entity.id,
                        target_entity_id=target_entity.id,
                        properties=rel_info["properties"],
                        confidence=rel_info["confidence"],
                        source=request.content[:100]
                    )
                    relationships.append(relationship)
        
        # Store relationships in memory
        for relationship in relationships:
            await self.agent.memory.store_relationship(relationship)
        
        return LearningResponse(
            entities=[],  # No new entities in relationship learning
            relationships=relationships,
            strategy_used=LearningStrategy.RELATIONSHIP_LEARNING,
            analysis={"relationship_analysis": analysis["analysis"]},
            confidence_scores={r.id: r.confidence for r in relationships}
        )
    
    @pipeline(input_model=LearningRequest, output_model=LearningResponse)
    async def run_pipeline(self, input_data: LearningRequest) -> LearningResponse:
        """Run the relationship learning pipeline.
        
        Args:
            input_data: The learning request with content to process
            
        Returns:
            Learning response with extracted relationships
        """
        # Create context with the input data
        input_context = Context(data=input_data)
        
        # Get stage instances
        validate_entities_stage = self.get_stage("validate_entities")
        analyze_stage = self.get_stage("analyze_relationships")
        extract_stage = self.get_stage("extract_relationships")
        
        # Execute validation
        validated_result = await validate_entities_stage.execute(input_context)
        
        # Execute analysis
        analysis_result = await analyze_stage.execute(validated_result)
        
        # Execute relationship extraction
        extraction_result = await extract_stage.execute(analysis_result)
        
        return extraction_result.data 