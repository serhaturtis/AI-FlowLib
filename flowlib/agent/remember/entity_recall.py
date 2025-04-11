from flowlib.flows.base import Flow
from flowlib.agent.remember.models import RecallRequest, RecallResponse, RecallStrategy, MemoryMatch
from flowlib.flows.decorators import flow, stage, pipeline
from .flows import BaseRecallFlow
from ...core.context import Context
from ...core.errors import ExecutionError

@flow(
    name="EntityRecallFlow",
    description="Flow for entity-based memory recall that retrieves information about specific entities"
)
class EntityRecallFlow(BaseRecallFlow):
    """Flow for entity-based memory recall"""
    
    def __init__(self, name=None):
        """Initialize the entity recall flow.
        
        Args:
            name: Optional name for the flow instance
        """
        # Call parent constructor with name or default
        super().__init__(name or "EntityRecallFlow")
    
    @stage("validate_entity")
    async def validate_entity(self, request: RecallRequest) -> RecallRequest:
        """Validate entity ID and existence"""
        if not request.entity_id:
            raise ExecutionError("Entity ID is required for entity recall")
            
        # Check if entity exists in knowledge memory
        entity = await self.agent.memory.get_entity(request.entity_id)
        if not entity:
            raise ExecutionError(f"Entity {request.entity_id} not found in knowledge memory")
            
        return request
    
    @stage("recall_entity_knowledge")
    async def recall_entity_knowledge(self, request: RecallRequest) -> RecallResponse:
        """Recall knowledge about the entity"""
        # Get entity knowledge from memory
        entity = await self.agent.memory.get_entity(request.entity_id)
        relationships = await self.agent.memory.get_entity_relationships(request.entity_id)
        
        # Create knowledge matches
        matches = []
        
        # Add entity properties as matches
        for prop, value in entity.properties.items():
            matches.append(
                MemoryMatch(
                    memory_id=f"{request.entity_id}_prop_{prop}",
                    content=f"{prop}: {value}",
                    memory_type="entity_property",
                    relevance_score=1.0,
                    metadata={"property": prop, "entity_id": request.entity_id}
                )
            )
            
        # Add relationships as matches
        for rel in relationships:
            matches.append(
                MemoryMatch(
                    memory_id=f"{request.entity_id}_rel_{rel.id}",
                    content=f"Relationship: {rel.type} with {rel.target_id}",
                    memory_type="entity_relationship",
                    relevance_score=1.0,
                    metadata={
                        "relationship_type": rel.type,
                        "target_entity": rel.target_id,
                        "entity_id": request.entity_id
                    }
                )
            )
            
        return RecallResponse(
            matches=matches,
            strategy_used=RecallStrategy.ENTITY,
            total_matches=len(matches),
            query_analysis={"entity_id": request.entity_id}
        )
    
    @pipeline(input_model=RecallRequest, output_model=RecallResponse)
    async def run_pipeline(self, input_data: RecallRequest) -> RecallResponse:
        """Run the entity recall pipeline.
        
        Args:
            input_data: The recall request
            
        Returns:
            Recall response with entity knowledge
        """
        # Create context with the input data
        input_context = Context(data=input_data)
        
        # Get stage instances
        validate_stage = self.get_stage("validate_request")
        validate_entity_stage = self.get_stage("validate_entity")
        recall_stage = self.get_stage("recall_entity_knowledge")
        
        # Execute validation
        validated_result = await validate_stage.execute(input_context)
        
        # Execute entity validation
        entity_validated_result = await validate_entity_stage.execute(validated_result)
        
        # Execute entity recall
        recall_result = await recall_stage.execute(entity_validated_result)
        
        return recall_result.data 