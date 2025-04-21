"""
Comprehensive Memory System for the Agent.

Orchestrates interactions between different specialized memory types 
(vector, knowledge graph, working memory) to provide a unified memory interface.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Union

from ..core.errors import MemoryError
from ..core.base import BaseComponent
from .base import BaseMemory
from .interfaces import MemoryInterface
from .models import (
    MemoryItem,
    MemoryStoreRequest,
    MemoryRetrieveRequest,
    MemorySearchRequest,
    MemorySearchResult,
    MemoryContext,
    Entity
)
# Import specialized memory types
from .vector import VectorMemory
from .knowledge import KnowledgeBaseMemory
from .working import WorkingMemory # Assuming this exists

# Import provider/registry if needed for LLM Fusion
from ...providers.registry import provider_registry
from ...providers.constants import ProviderType
from ...providers.llm.base import LLMProvider

# Import registry/constants for prompts
from ...resources.registry import resource_registry
from ...resources.constants import ResourceType

# Import prompts and related models from the new prompts module
from .prompts import (
    MemoryFusionPrompt, 
    FusedMemoryResult, 
    KGQueryExtractionPrompt, 
    ExtractedKGQueryTerms
)

logger = logging.getLogger(__name__)


# --- Prompt and Model definitions moved to prompts.py ---


class ComprehensiveMemory(BaseMemory):
    """Fused memory component orchestrating Vector, Knowledge, and Working memory."""

    def __init__(
        self,
        vector_memory: VectorMemory,
        knowledge_memory: KnowledgeBaseMemory,
        working_memory: WorkingMemory,
        # Add LLM provider/model name for fusion
        fusion_provider_name: str = "llamacpp", # Or get from config
        fusion_model_name: str = "default",
        name: str = "comprehensive_memory"
    ):
        """Initialize comprehensive memory with specialized memory components."""
        super().__init__(name)
        
        if not vector_memory or not knowledge_memory or not working_memory:
            raise ValueError("VectorMemory, KnowledgeBaseMemory, and WorkingMemory instances are required.")
            
        self._vector_memory = vector_memory
        self._knowledge_memory = knowledge_memory
        self._working_memory = working_memory
        self._fusion_provider_name = fusion_provider_name
        self._fusion_model_name = fusion_model_name
        self._fusion_llm: Optional[LLMProvider] = None
        
        logger.info(f"Initialized {self.name} with Vector, Knowledge, and Working memory.")

    async def _initialize_impl(self) -> None:
        """Initialize composed memory components and fusion LLM."""
        try:
            logger.debug(f"Initializing composed memories for {self.name}...")
            await asyncio.gather(
                self._vector_memory.initialize(),
                self._knowledge_memory.initialize(),
                self._working_memory.initialize()
            )
            
            # Initialize LLM provider for fusion
            self._fusion_llm = await provider_registry.get(
                ProviderType.LLM,
                self._fusion_provider_name
            )
            if not self._fusion_llm:
                 raise MemoryError(f"Fusion LLM provider '{self._fusion_provider_name}' not found.")
            

            logger.debug(f"{self.name} initialization complete.")
        except Exception as e:
            logger.error(f"Error initializing {self.name}: {e}", exc_info=True)
            raise MemoryError(f"Failed to initialize {self.name}: {e}") from e

    async def _shutdown_impl(self) -> None:
        """Shutdown composed memory components."""
        logger.debug(f"Shutting down composed memories for {self.name}...")
        await asyncio.gather(
            self._vector_memory.shutdown(),
            self._knowledge_memory.shutdown(),
            self._working_memory.shutdown()
        )
        self._fusion_llm = None # Release provider instance
        logger.debug(f"{self.name} shutdown complete.")

    # ----------------------------------------------------------------------
    # Implementation of MemoryInterface Methods via Orchestration
    # ----------------------------------------------------------------------

    async def _store_impl(
        self, 
        key: str, 
        value: Any, 
        context: str, 
        metadata: Dict[str, Any] = None,
        importance: float = 0.5,
        ttl_seconds: Optional[int] = None,
        **kwargs
    ) -> None:
        """Route storage requests to appropriate specialized memory."""
        metadata = metadata or {}
        is_entity = isinstance(value, Entity) or (isinstance(value, dict) and 'type' in value and 'id' in value)

        storage_tasks = []

        # 1. Always store in Working Memory (for quick access, potential TTL)
        logger.debug(f"Routing storage for key '{key}' to WorkingMemory.")
        storage_tasks.append(
            self._working_memory._store_impl(
                key=key, value=value, context=context, metadata=metadata.copy(),
                importance=importance, ttl_seconds=ttl_seconds, **kwargs
            )
        )
        
        # 2. Store entities in Knowledge Base
        if is_entity:
            logger.debug(f"Routing storage for key '{key}' (Entity) to KnowledgeBaseMemory.")
            storage_tasks.append(
                self._knowledge_memory._store_impl(
                    key=key, value=value, context=context, metadata=metadata.copy(),
                    importance=importance, **kwargs
                )
            )

        # 3. Store text representation in Vector Memory for semantic search
        # (Avoid storing raw entities here if they went to KB, store text form?)
        # Decision: Store *all* items (or their text representation) in Vector Memory 
        # to ensure they are searchable semantically.
        logger.debug(f"Routing storage for key '{key}' (Value/Text Rep) to VectorMemory.")
        storage_tasks.append(
            self._vector_memory._store_impl(
                key=key, value=value, context=context, metadata=metadata.copy(),
                importance=importance, **kwargs
            )
        )
            
        # Execute storage operations concurrently
        results = await asyncio.gather(*storage_tasks, return_exceptions=True)
        
        # Check for errors - adhering to 'no fallback' means we raise if any store fails
        errors = [res for res in results if isinstance(res, Exception)]
        if errors:
            logger.error(f"Errors occurred during concurrent storage for key '{key}': {errors}")
            # Combine error messages or raise the first one? Raise first for simplicity.
            raise MemoryError(f"Failed to store item in one or more memories: {errors[0]}") from errors[0]

    async def _retrieve_impl(
        self, 
        key: str, 
        context: str,
        retrieve_entity: bool = False, # Hint remains useful
        **kwargs
    ) -> Optional[MemoryItem]: # Return MemoryItem consistent with BaseMemory
        """Retrieve by key, checking Working Memory first, then Knowledge Base."""
        
        # 1. Check Working Memory first (fastest access)
        logger.debug(f"Attempting to retrieve key '{key}' from WorkingMemory.")
        try:
            item = await self._working_memory._retrieve_impl(key=key, context=context, **kwargs)
            if item is not None: 
                logger.debug(f"Retrieved key '{key}' from WorkingMemory.")
                return item
        except Exception as e:
            # Log warning, but proceed, as failure here doesn't block checking others
            logger.warning(f"Error retrieving from WorkingMemory (key: {key}): {e}")
            # If we strictly adhere to NO fallbacks, should we raise here? 
            # Let's reconsider: If the *primary* target (Working) fails, maybe we should raise.
            # Decision: Raise if WorkingMemory retrieval fails, as it's the first intended target.
            raise MemoryError(f"Failed retrieving from WorkingMemory: {e}") from e

        # 2. Check Knowledge Base (especially if entity is expected)
        # We only check KB if Working Memory didn't find it.
        logger.debug(f"Attempting to retrieve key '{key}' from KnowledgeBaseMemory.")
        try:
            # KnowledgeBaseMemory._retrieve_impl likely returns an Entity model or similar
            # We need to wrap it in a MemoryItem if found
            kb_result = await self._knowledge_memory._retrieve_impl(key=key, context=context, **kwargs)
            if kb_result:
                # Assume kb_result needs wrapping. Adjust if KB returns MemoryItem directly.
                # This depends on KnowledgeBaseMemory implementation.
                # Let's assume it returns the raw entity/value for now.
                logger.debug(f"Retrieved key '{key}' from KnowledgeBaseMemory.")
                # We need metadata. Get it from KB if possible, else minimal.
                kb_metadata = getattr(kb_result, 'metadata', {}) if hasattr(kb_result, 'metadata') else {}
                return MemoryItem(key=key, value=kb_result, context=context, metadata=kb_metadata)
        except Exception as e:
            logger.error(f"Error retrieving from KnowledgeBaseMemory: {e}", exc_info=True)
            # Raise error, no fallback beyond Working Memory check
            raise MemoryError(f"Failed retrieving from KnowledgeBaseMemory after checking WorkingMemory: {e}") from e
        
        # 3. Vector Memory (Not suitable for direct key retrieval)
        logger.debug(f"Key '{key}' not found in Working or Knowledge memory. Vector memory not used for key retrieval.")
        return None 

    async def _search_impl(
        self, 
        query: str, 
        context: str,
        limit: int = 10,
        search_type: str = "fused",
        threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None
    ) -> FusedMemoryResult:
        """Perform search across memory types, including KG term extraction, and synthesize results."""
        if not self._fusion_llm:
            raise MemoryError("Fusion LLM provider not initialized.")

        if not query.strip():
            logger.warning("Empty query passed to _search_impl, returning empty result")
            return FusedMemoryResult(relevant_items=[], summary="No query provided.")

        logger.debug(f"Performing '{search_type}' search for query: '{query}' in context: '{context}'")
        
        search_tasks = []
        common_search_args = {
            "context": context,
            "limit": limit,
            "threshold": threshold,
            "metadata_filter": metadata_filter,
            "sort_by": sort_by
        }

        # 1. Vector Search (Semantic) - Uses original query
        logger.debug("Scheduling vector search task.")
        search_tasks.append(self._vector_memory.search_with_model(
            MemorySearchRequest(**common_search_args, query=query, search_type="semantic")
        ))

        # 2. Knowledge Graph Search - Pre-process query first
        kg_search_term = None
        try:
            # Get extraction prompt resource
            extraction_prompt = resource_registry.get(
                name="kg_query_extraction", 
                resource_type=ResourceType.PROMPT
            )
            
            logger.debug(f"Attempting KG keyword extraction for query: '{query}'")
            extracted_terms_result: ExtractedKGQueryTerms = await self._fusion_llm.generate_structured(
                prompt=extraction_prompt,
                prompt_variables={"query": query, "context": context},
                output_type=ExtractedKGQueryTerms,
                model_name=self._fusion_model_name # Use same fusion model for consistency
            )
            
            if extracted_terms_result and extracted_terms_result.terms:
                kg_search_term = extracted_terms_result.terms[0] # Use the first term for now
                logger.info(f"Extracted '{kg_search_term}' for Knowledge Graph search.")
            else:
                logger.info("No specific keywords extracted for Knowledge Graph search.")
                
        except Exception as e:
            logger.warning(f"Knowledge Graph keyword extraction failed: {e}", exc_info=True)
            # Proceed without KG search if extraction fails

        # Schedule KG search only if a term was extracted
        if kg_search_term:
            logger.debug(f"Scheduling knowledge graph search task for term: '{kg_search_term}'")
            search_tasks.append(self._knowledge_memory.search_with_model(
                MemorySearchRequest(**common_search_args, query=kg_search_term, search_type="graph")
            ))
        else:
            # Define a dummy coroutine to return empty results immediately
            async def empty_kg_result():
                logger.debug("Skipping knowledge graph search task (no term extracted).")
                return MemorySearchResult(query=query, items=[], context=context)
            search_tasks.append(empty_kg_result())

        # 3. Working Memory Search (Keyword/Recent) - Uses original query
        logger.debug("Scheduling working memory search task.")
        search_tasks.append(self._working_memory.search_with_model(
             MemorySearchRequest(**common_search_args, query=query, search_type="keyword")
        ))

        # Execute searches concurrently
        logger.debug(f"Executing {len(search_tasks)} search tasks concurrently...")
        try:
             results = await asyncio.gather(*search_tasks, return_exceptions=True)
             logger.debug(f"Search tasks completed. Results (incl. errors): {results}")
        except Exception as e:
             # This gather itself shouldn't typically raise unless cancelled
             logger.error(f"Error gathering search results: {e}", exc_info=True)
             raise MemoryError(f"Failed during concurrent memory search: {e}") from e

        # Process and Format Results for Fusion LLM
        vector_results: List[MemoryItem] = []
        knowledge_results: List[MemoryItem] = []
        working_results: List[MemoryItem] = []

        # Safely extract results or log warnings
        if len(results) > 0 and isinstance(results[0], MemorySearchResult):
             vector_results = results[0].items
        elif len(results) > 0 and isinstance(results[0], Exception):
             logger.warning(f"Vector memory search failed: {results[0]}")

        if len(results) > 1 and isinstance(results[1], MemorySearchResult):
            knowledge_results = results[1].items
        elif len(results) > 1 and isinstance(results[1], Exception):
            # Log the specific exception from the KG search task
            logger.warning(f"Knowledge memory search failed: {results[1]}")

        if len(results) > 2 and isinstance(results[2], MemorySearchResult):
            working_results = results[2].items
        elif len(results) > 2 and isinstance(results[2], Exception):
            logger.warning(f"Working memory search failed: {results[2]}")

        # Format for LLM prompt (Handle potentially empty results gracefully)
        vector_text = "\n".join([f"- [Semantic Match Score: {item.metadata.get('score', 'N/A'):.2f}] {getattr(item, 'value', 'N/A')[:200]}..." for item in vector_results]) or "No relevant semantic matches found."
        knowledge_text = "\n".join([f"- [Knowledge Graph Entity/Relation] Key: {getattr(item, 'key', 'N/A')}, Type: {item.metadata.get('entity_type', 'Unknown')}, Value: {str(getattr(item, 'value', 'N/A'))[:200]}..." for item in knowledge_results]) or "No relevant knowledge graph items found."
        working_text = "\n".join([f"- [Working Memory Item] Key: {getattr(item, 'key', 'N/A')}, Value: {str(getattr(item, 'value', 'N/A'))[:150]}... (Stored: {item.metadata.get('stored_at', 'N/A')})" for item in working_results]) or "No relevant items found in short-term working memory."

        # LLM Fusion Step
        try:
            logger.debug("Attempting LLM Memory Fusion.")
            fusion_prompt = resource_registry.get(
                        name="memory_fusion",
                        resource_type=ResourceType.PROMPT
                    )
            
            prompt_vars = {
                "query": query, # Original user query for context
                "vector_results": vector_text,
                "knowledge_results": knowledge_text,
                "working_results": working_text
            }

            # Use the imported FusedMemoryResult model here
            fused_result_structured: FusedMemoryResult = await self._fusion_llm.generate_structured(
                prompt=fusion_prompt,
                prompt_variables=prompt_vars,
                output_type=FusedMemoryResult, 
                model_name=self._fusion_model_name
            )
            logger.debug("LLM Memory Fusion successful.")
            return fused_result_structured

        except Exception as e:
            logger.error(f"LLM Memory Fusion failed: {e}", exc_info=True)
            raise MemoryError(f"Failed to fuse memory results: {e}") from e

    async def search_with_model(
        self, 
        request: MemorySearchRequest
    ) -> MemorySearchResult:
        """Perform a fused search and return results conforming to MemorySearchResult."""
        try:
            # Call the internal fused search implementation, passing specific fields
            context = request.context or self._default_context
            fused_result: FusedMemoryResult = await self._search_impl(
                query=request.query,
                context=context,
                limit=request.limit,
                search_type=request.search_type or "fused",
                threshold=request.threshold,
                metadata_filter=request.metadata_filter,
                sort_by=request.sort_by
            )

            # Convert the fused list of strings back into MemoryItems 
            # This keeps the interface consistent for consumers expecting MemoryItem lists.
            fused_items = []
            for i, text_value in enumerate(fused_result.relevant_items):
                 # Create MemoryItems with metadata indicating fusion source
                 fused_items.append(MemoryItem(
                      key=f"fused_result_{i}", # Generate a placeholder key
                      value=text_value, # The synthesized string is the value
                      context=context,
                      metadata={
                          'source': 'fused_llm',
                          'fusion_summary': fused_result.summary,
                          'original_query': request.query
                      }
                 ))

            # Return within the expected structure
            return MemorySearchResult(query=request.query, items=fused_items)

        except Exception as e:
            logger.error(f"Error in comprehensive search_with_model: {e}", exc_info=True)
            # Adhering to principles: Don't return partial results or fallback. Raise.
            raise MemoryError(f"Comprehensive search failed: {e}") from e

    async def retrieve_relevant(self, query: str, context: str = None, limit: int = 5) -> List[str]:
        """Retrieve relevant memories using the fused search, returning plain strings."""
        context = context or self._default_context
        try:
            # Use the public search_with_model which now handles fusion
            search_request = MemorySearchRequest(query=query, context=context, limit=limit)
            search_result = await self.search_with_model(search_request)
            
            # Extract the string values from the MemoryItems returned by search_with_model
            relevant_strings = [item.value for item in search_result.items]
            logger.debug(f"Retrieved {len(relevant_strings)} relevant items via fused search for query: '{query}'")
            return relevant_strings
        except Exception as e:
            logger.error(f"Error in retrieve_relevant using fused search: {e}", exc_info=True)
            raise MemoryError(f"Failed during relevant memory retrieval: {e}") from e

    # Other MemoryInterface methods (create_context, get_context_model) might need routing
    # or could be considered not applicable/implemented for the comprehensive view.
    # For simplicity now, let's delegate or raise NotImplementedError.

    async def create_context(
        self, context_name: str, parent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None, **kwargs
    ) -> str:
        """Create a context in all composed memory components."""
        logger.debug(f"Creating memory context '{context_name}' in all memory components")
        
        # Note: This method itself is async because it *could* contain awaitable
        # operations, even if the current underlying calls are synchronous.
        # The caller (DualPathAgent) correctly awaits this async method.
        
        created_path = None
        try:
            # Ensure context_name is a string
            if not isinstance(context_name, str):
                context_name = str(context_name)
                logger.warning(f"Non-string context_name converted to string: {context_name}")
            
            # Call underlying context creation methods (assuming they are synchronous)
            # We store the *returned path* which should be consistent
            vector_path = None
            knowledge_path = None
            
            if self._vector_memory:
                # Call synchronously, as BaseMemory.create_context is not async
                vector_path = self._vector_memory.create_context(
                    context_name, parent=parent, metadata=metadata, **kwargs
                )
                # Use the returned path as the definitive one (they should match)
                created_path = vector_path 
            
            if self._knowledge_memory:
                 # Call synchronously, assuming KnowledgeBaseMemory also uses BaseMemory's sync method
                 # If KnowledgeBaseMemory *does* override with an async method, this needs await
                 # Let's assume sync for now based on VectorMemory's pattern.
                knowledge_path = self._knowledge_memory.create_context(
                    context_name, parent=parent, metadata=metadata, **kwargs
                )
                if created_path and knowledge_path != created_path:
                     logger.warning(f"Mismatch in context paths created: Vector='{created_path}', Knowledge='{knowledge_path}'. Using '{created_path}'.")
                elif not created_path:
                     created_path = knowledge_path
                     
            # Working memory doesn't have create_context
            # ...
            
            # If neither vector nor knowledge memory created a path, raise error
            if created_path is None:
                raise MemoryError("Failed to create context in any underlying memory component.")
            
            # Return the consistent context path name that was created/confirmed
            return created_path
            
        except Exception as e:
            logger.error(f"Failed to create context '{context_name}' across all memory components: {str(e)}")
            # Ensure the original exception type isn't lost if it's already a MemoryError
            if isinstance(e, MemoryError):
                raise e 
            else:
                raise MemoryError(f"Failed to create context: {str(e)}") from e

    def get_context_model(self, context_path: str) -> Optional[MemoryContext]:
        """Get a memory context model by path.
        
        Returns a context model from either vector or knowledge memory components,
        as working memory doesn't support context models.
        
        Args:
            context_path: The path of the context to retrieve
            
        Returns:
            Memory context model or None if not found
            
        Raises:
            MemoryError: If context retrieval fails
        """
        # Try vector memory first, then knowledge memory, since WorkingMemory doesn't support it
        try:
            if self._vector_memory:
                try:
                    return self._vector_memory.get_context_model(context_path)
                except NotImplementedError:
                    logger.debug(f"VectorMemory doesn't support get_context_model, trying KnowledgeBaseMemory.")
                except Exception as e:
                    logger.warning(f"Error getting context model from VectorMemory: {e}")
                    
            if self._knowledge_memory:
                try:
                    return self._knowledge_memory.get_context_model(context_path)
                except NotImplementedError:
                    logger.debug(f"KnowledgeBaseMemory doesn't support get_context_model.")
                except Exception as e:
                    logger.warning(f"Error getting context model from KnowledgeBaseMemory: {e}")
                    
            logger.warning(f"No memory component supports context models for path '{context_path}'")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get context model for path '{context_path}': {e}", exc_info=True)
            raise MemoryError(f"Failed to retrieve context model: {e}") from e

    # --- Wipe Implementation --- 

    async def _wipe_context_impl(self, context: str) -> None:
        """Wipe a specific context across all managed memories."""
        logger.warning(f"Wiping memory context '{context}' across all providers...")
        tasks = [
            self._vector_memory.wipe(context=context),
            self._knowledge_memory.wipe(context=context), # This might raise NotImplementedError
            self._working_memory.wipe(context=context)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [res for res in results if isinstance(res, Exception)]
        
        if errors:
            # Log all errors, re-raise the first one encountered (or a combined one)
            for i, err in enumerate(errors):
                logger.error(f"Error wiping memory provider index {i} for context '{context}': {err}")
            # Re-raise the first error to signal failure clearly
            raise MemoryError(f"Failed to wipe context '{context}' in one or more memories: {errors[0]}") from errors[0]
            
        logger.info(f"Memory wipe completed for context '{context}'.")

    async def _wipe_all_impl(self) -> None:
        """Wipe all data across all managed memories."""
        logger.warning(f"Wiping ALL memory across all providers...")
        tasks = [
            self._vector_memory.wipe(context=None),
            self._knowledge_memory.wipe(context=None),
            self._working_memory.wipe(context=None)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [res for res in results if isinstance(res, Exception)]

        if errors:
            for i, err in enumerate(errors):
                logger.error(f"Error wiping memory provider index {i} (ALL): {err}")
            raise MemoryError(f"Failed to wipe ALL data in one or more memories: {errors[0]}") from errors[0]
            
        logger.info(f"Memory wipe completed for ALL contexts.")

    async def store_with_model(
        self, 
        request: MemoryStoreRequest
    ) -> None:
        """Store data in all appropriate memory components."""
        try:
            context = request.context or self._default_context
            await self._store_impl(
                key=request.key,
                value=request.value,
                context=context,
                metadata=request.metadata,
                importance=request.importance,
                ttl_seconds=request.ttl_seconds
            )
        except Exception as e:
            logger.error(f"Error in comprehensive store_with_model: {e}", exc_info=True)
            raise MemoryError(f"Comprehensive memory storage failed: {e}") from e 