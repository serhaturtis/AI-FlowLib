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
    name="ConceptFormationFlow",
    description="Analyzes content to identify patterns and form new concepts, with confidence scoring and property extraction."
)
class ConceptFormationFlow(BaseLearningFlow):
    """Flow for forming new concepts from content"""
    
    def __init__(self, name=None):
        """Initialize the concept formation flow.
        
        Args:
            name: Optional name for the flow instance
        """
        # Call parent constructor with name or default
        super().__init__(name or "ConceptFormationFlow")
    
    def get_description(self) -> str:
        """Get a description of this flow's purpose.
        
        Returns:
            A string describing what this flow does
        """
        return "Analyzes content to identify patterns and form new concepts, with confidence scoring and property extraction."
    
    @stage("analyze_patterns")
    async def analyze_patterns(self, request: LearningRequest) -> Dict[str, Any]:
        """Analyze content for patterns and potential concepts"""
        # Use LLM to analyze patterns and form concepts
        analysis = await self.agent.llm.analyze_text(
            f"Content: {request.content}\n"
            "Analyze this content to identify:\n"
            "1. Recurring patterns and themes\n"
            "2. Abstract concepts and categories\n"
            "3. Hierarchical relationships between concepts\n"
            "4. Defining characteristics of each concept"
        )
        return {"analysis": analysis}
    
    @stage("form_concepts")
    async def form_concepts(self, request: LearningRequest, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Form new concepts from identified patterns"""
        concepts = []
        concept_relationships = []
        
        # Process concept formations
        for concept_info in analysis["analysis"]["concepts"]:
            if concept_info["confidence"] >= request.confidence_threshold:
                # Create concept entity
                concept = Entity(
                    id=concept_info["id"],
                    type="concept",
                    properties={
                        "definition": concept_info["definition"],
                        "characteristics": concept_info["characteristics"],
                        "examples": concept_info["examples"]
                    },
                    confidence=concept_info["confidence"],
                    source=request.content[:100]
                )
                concepts.append(concept)
                
                # Create hierarchical relationships
                for parent_id in concept_info.get("parent_concepts", []):
                    relationship = Relationship(
                        id=f"{concept.id}_parent_{parent_id}",
                        type="is_a",
                        source_entity_id=concept.id,
                        target_entity_id=parent_id,
                        properties={"hierarchy_level": "direct"},
                        confidence=concept_info["confidence"],
                        source=request.content[:100]
                    )
                    concept_relationships.append(relationship)
        
        return {
            "analysis": analysis["analysis"],
            "concepts": concepts,
            "relationships": concept_relationships
        }
    
    @stage("store_concepts")
    async def store_concepts(self, request: LearningRequest, formed: Dict[str, Any]) -> LearningResponse:
        """Store formed concepts and their relationships"""
        # Store concepts
        for concept in formed["concepts"]:
            await self.agent.memory.store_entity(concept)
            
        # Store relationships
        for relationship in formed["relationships"]:
            await self.agent.memory.store_relationship(relationship)
            
        return LearningResponse(
            entities=formed["concepts"],
            relationships=formed["relationships"],
            strategy_used=LearningStrategy.CONCEPT_FORMATION,
            analysis={"concept_analysis": formed["analysis"]},
            confidence_scores={
                **{e.id: e.confidence for e in formed["concepts"]},
                **{r.id: r.confidence for r in formed["relationships"]}
            }
        )
    
    @pipeline(input_model=LearningRequest, output_model=LearningResponse)
    async def run_pipeline(self, input_data: LearningRequest) -> LearningResponse:
        """Run the concept formation pipeline.
        
        Args:
            input_data: The learning request with content to process
            
        Returns:
            Learning response with formed concepts
        """
        # Create context with the input data
        input_context = Context(data=input_data)
        
        # Get stage instances
        validate_stage = self.get_stage("validate_request")
        analyze_stage = self.get_stage("analyze_patterns")
        form_stage = self.get_stage("form_concepts")
        store_stage = self.get_stage("store_concepts")
        
        # Execute validation
        validated_result = await validate_stage.execute(input_context)
        
        # Execute pattern analysis
        analysis_result = await analyze_stage.execute(validated_result)
        
        # Execute concept formation
        formed_result = await form_stage.execute(analysis_result)
        
        # Execute concept storage
        storage_result = await store_stage.execute(formed_result)
        
        return storage_result.data 