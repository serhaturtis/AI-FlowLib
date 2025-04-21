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
    name="KnowledgeIntegrationFlow",
    description="Flow for integrating new knowledge with existing knowledge, resolving conflicts and updating metadata",
    is_infrastructure=True
)
class KnowledgeIntegrationFlow(BaseLearningFlow):
    """Flow for integrating new knowledge with existing knowledge"""
    
    def __init__(self, name=None):
        """Initialize the knowledge integration flow.
        
        Args:
            name: Optional name for the flow instance
        """
        # Call parent constructor with name or default
        super().__init__(name or "KnowledgeIntegrationFlow")
    
    def get_description(self) -> str:
        """Get a description of this flow's purpose.
        
        Returns:
            A string describing what this flow does
        """
        return "Integrates newly learned concepts and relationships with existing knowledge, resolving conflicts and updating metadata."
    
    @stage("analyze_existing_knowledge")
    async def analyze_existing_knowledge(self, request: LearningRequest) -> Dict[str, Any]:
        """Analyze existing knowledge for integration context"""
        # Get relevant existing knowledge
        existing_entities = []
        if request.existing_entities:
            for entity_id in request.existing_entities:
                entity = await self.agent.memory.get_entity(entity_id)
                if entity:
                    existing_entities.append(entity)
                    
        # Get related relationships
        relationships = []
        for entity in existing_entities:
            entity_relationships = await self.agent.memory.get_entity_relationships(entity.id)
            relationships.extend(entity_relationships)
            
        return {
            "existing_entities": existing_entities,
            "existing_relationships": relationships
        }
    
    @stage("analyze_new_knowledge")
    async def analyze_new_knowledge(self, request: LearningRequest, existing: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze new knowledge for integration"""
        # Use LLM to analyze how new knowledge relates to existing
        analysis = await self.agent.llm.analyze_text(
            f"New Content: {request.content}\n"
            f"Existing Entities: {[e.type for e in existing['existing_entities']]}\n"
            f"Existing Relationships: {[r.type for r in existing['existing_relationships']]}\n"
            "Analyze how this new knowledge:\n"
            "1. Relates to existing entities\n"
            "2. Extends or modifies existing relationships\n"
            "3. Introduces new concepts\n"
            "4. Resolves conflicts with existing knowledge"
        )
        return {**existing, "analysis": analysis}
    
    @stage("integrate_knowledge")
    async def integrate_knowledge(self, request: LearningRequest, analysis: Dict[str, Any]) -> LearningResponse:
        """Integrate new knowledge with existing knowledge"""
        entities = []
        relationships = []
        
        # Process entity integrations
        for entity_info in analysis["analysis"]["entity_integrations"]:
            if entity_info["confidence"] >= request.confidence_threshold:
                # Check if entity exists
                existing_entity = next(
                    (e for e in analysis["existing_entities"] if e.id == entity_info["entity_id"]),
                    None
                )
                
                if existing_entity:
                    # Update existing entity
                    updated_entity = Entity(
                        id=existing_entity.id,
                        type=existing_entity.type,
                        properties={**existing_entity.properties, **entity_info["new_properties"]},
                        confidence=entity_info["confidence"],
                        source=request.content
                    )
                    await self.agent.memory.update_entity(updated_entity)
                    entities.append(updated_entity)
                else:
                    # Create new entity
                    new_entity = Entity(
                        id=entity_info["entity_id"],
                        type=entity_info["type"],
                        properties=entity_info["properties"],
                        confidence=entity_info["confidence"],
                        source=request.content
                    )
                    await self.agent.memory.store_entity(new_entity)
                    entities.append(new_entity)
        
        # Process relationship integrations
        for rel_info in analysis["analysis"]["relationship_integrations"]:
            if rel_info["confidence"] >= request.confidence_threshold:
                # Check if relationship exists
                existing_rel = next(
                    (r for r in analysis["existing_relationships"] 
                     if r.source_entity_id == rel_info["source_id"] 
                     and r.target_entity_id == rel_info["target_id"]
                     and r.type == rel_info["type"]),
                    None
                )
                
                if existing_rel:
                    # Update existing relationship
                    updated_rel = Relationship(
                        id=existing_rel.id,
                        type=existing_rel.type,
                        source_entity_id=existing_rel.source_entity_id,
                        target_entity_id=existing_rel.target_entity_id,
                        properties={**existing_rel.properties, **rel_info["new_properties"]},
                        confidence=rel_info["confidence"],
                        source=request.content
                    )
                    await self.agent.memory.update_relationship(updated_rel)
                    relationships.append(updated_rel)
                else:
                    # Create new relationship
                    new_rel = Relationship(
                        id=rel_info["id"],
                        type=rel_info["type"],
                        source_entity_id=rel_info["source_id"],
                        target_entity_id=rel_info["target_id"],
                        properties=rel_info["properties"],
                        confidence=rel_info["confidence"],
                        source=request.content
                    )
                    await self.agent.memory.store_relationship(new_rel)
                    relationships.append(new_rel)
        
        return LearningResponse(
            entities=entities,
            relationships=relationships,
            strategy_used=LearningStrategy.KNOWLEDGE_INTEGRATION,
            analysis={"integration_analysis": analysis["analysis"]},
            confidence_scores={
                **{e.id: e.confidence for e in entities},
                **{r.id: r.confidence for r in relationships}
            }
        )
    
    @pipeline(input_model=LearningRequest, output_model=LearningResponse)
    async def run_pipeline(self, input_data: LearningRequest) -> LearningResponse:
        """Run the knowledge integration pipeline.
        
        Args:
            input_data: The learning request with content to process
            
        Returns:
            Learning response with integrated knowledge
        """
        # Create context with the input data
        input_context = Context(data=input_data)
        
        # Get stage instances
        validate_stage = self.get_stage("validate_request")
        analyze_existing_stage = self.get_stage("analyze_existing_knowledge")
        analyze_new_stage = self.get_stage("analyze_new_knowledge")
        integrate_stage = self.get_stage("integrate_knowledge")
        
        # Execute validation
        validated_result = await validate_stage.execute(input_context)
        
        # Execute existing knowledge analysis
        existing_analysis_result = await analyze_existing_stage.execute(validated_result)
        
        # Execute new knowledge analysis
        new_analysis_result = await analyze_new_stage.execute(existing_analysis_result)
        
        # Execute knowledge integration
        integration_result = await integrate_stage.execute(new_analysis_result)
        
        return integration_result.data 