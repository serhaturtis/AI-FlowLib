"""Composite flow example showing reusable standalone stages.

This example demonstrates using flowlib's standalone stages and composite flows to
create modular, reusable components for text processing. It shows:

1. Standalone stage creation using the @standalone decorator
2. Composing stages into a pipeline using CompositeFlow
3. Adding stages to a flow dynamically
4. Reusing standalone stages across different flows
5. Handling intermediate results in a composite flow
"""

import asyncio
import logging
from typing import Dict, List, Any
from pydantic import BaseModel, Field

from flowlib.resources.constants import ResourceType
from flowlib.resources.registry import resource_registry
from flowlib.resources.decorators import model, prompt
from flowlib.providers.constants import ProviderType
from flowlib.providers.registry import provider_registry
from flowlib.flows.registry import stage_registry
from flowlib.flows.decorators import flow, stage, pipeline, standalone
from flowlib.core.context import Context

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===================== Define Models =====================

class TextInput(BaseModel):
    """Input text for processing."""
    text: str = Field(..., description="Raw text to process")
    language: str = Field("english", description="Language of the text")

class SplitResult(BaseModel):
    """Result of text splitting."""
    sentences: List[str] = Field(..., description="List of sentences extracted from text")
    sentence_count: int = Field(..., description="Total number of sentences")
    word_count: int = Field(..., description="Total number of words in all sentences")

class ClassificationResult(BaseModel):
    """Result of text classification."""
    category: str = Field(..., description="Primary topic category")
    confidence: float = Field(..., description="Confidence score (0-1)")
    subcategories: List[str] = Field(default_factory=list, description="Subtopics identified")

class EnhancedTextResult(BaseModel):
    """Enhanced text with metadata."""
    enhanced_text: str = Field(..., description="Text with enhancements")
    changes_made: List[str] = Field(..., description="List of enhancements made")

class CompleteTextAnalysis(BaseModel):
    """Complete text analysis combining all results."""
    sentences: List[str] = Field(..., description="Extracted sentences")
    classification: ClassificationResult = Field(..., description="Classification results")
    enhanced_text: str = Field(..., description="Enhanced text version")
    enhancement_count: int = Field(..., description="Number of enhancements made")
    metrics: Dict[str, Any] = Field(..., description="Analysis metrics")

# Models for the Simple Enhancement Flow
class EnhanceStageInput(BaseModel):
    """Input model for the enhance_text stage."""
    classification: ClassificationResult
    original_text: str

# ===================== Define Prompt Resources =====================

@prompt("text-classification")
class ClassificationPrompt:
    """Prompt for text classification."""
    template = """
    Classify the following text into a primary category and optional subcategories.
    
    TEXT:
    {{text}}
    """
    
    config = {
        "max_tokens": 1024,
        "temperature": 0.3,
        "top_p": 0.95
    }

@prompt("text-enhancement")
class EnhancementPrompt:
    """Prompt for text enhancement."""
    template = """
    Enhance the following text by improving clarity, fixing grammatical errors, 
    and improving sentence structure while preserving the original meaning.
    
    ORIGINAL TEXT:
    {{text}}

    ENHANCED TEXT:
    """
    
    config = {
        "max_tokens": 2048,
        "temperature": 0.5,
        "top_p": 0.95
    }

# ===================== Define Standalone Stages =====================

@standalone(
    name="text-splitter",
    input_model=TextInput,
    output_model=SplitResult
)
async def split_text(context: Context) -> SplitResult:
    """Split input text into sentences and count words."""
    logger.info("Splitting text into sentences")
    
    input_data:TextInput = context.data
    text = input_data.text
    
    # Simple sentence splitting
    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    
    # Count words
    word_count = sum(len(s.split()) for s in sentences)
    
    result = SplitResult(
        sentences=sentences,
        sentence_count=len(sentences),
        word_count=word_count
    )
    
    logger.info(f"Split text into {len(sentences)} sentences with {word_count} words")
    return result

@standalone(
    name="text-classifier",
    input_model=TextInput,
    output_model=ClassificationResult
)
async def classify_text(context: Context) -> ClassificationResult:
    """Classify text using LLM."""
    logger.info("Classifying text")
    
    # Handle different types of input data
    input_data:TextInput = context.data
    text = input_data.text
    
    # Get prompt template - Fix: remove await from synchronous method
    classification_prompt = resource_registry.get(
        "text-classification", 
        resource_type=ResourceType.PROMPT
    )
    
    # Get LLM provider - use the pre-registered llamacpp provider
    llm = await provider_registry.get(ProviderType.LLM, "llamacpp")
    
    # Call LLM to classify text - use the text-analysis-model from the multi-stage flow example
    result = await llm.generate_structured(
        classification_prompt,
        ClassificationResult,
        "text-analysis-model",
        {'text': text}
    )
    
    logger.info(f"Text classified as '{result.category}' with confidence {result.confidence:.2f}")
    return result

@standalone(
    name="text-enhancer",
    input_model=TextInput,
    output_model=EnhancedTextResult
)
async def enhance_text(context: Context) -> EnhancedTextResult:
    """Enhance text using LLM."""
    logger.info("Enhancing text")
    
    # Handle different types of input data
    input_data:TextInput = context.data
    text = input_data.text
    
    # Get prompt template - Fix: remove await
    enhancement_prompt = resource_registry.get(
        "text-enhancement", 
        resource_type=ResourceType.PROMPT
    )
    
    llm = await provider_registry.get(ProviderType.LLM, "llamacpp")
    
    try:
        result = await llm.generate_structured(
            enhancement_prompt,
            EnhancedTextResult,
            "text-analysis-model",
            {'text': text}
        )
            
        logger.info(f"Text enhanced with {len(result.changes_made)} improvements")
        return result
    except Exception as e:
        logger.warning(f"Error in structured generation: {str(e)}.")
        return None

# ===================== Define Model Config =====================

@model("text-analysis-model")
class LLMModelConfig:
    """Configuration for the text analysis LLM."""
    path = "/path/to/your/model.gguf"
    model_type = "llama"
    n_ctx = 4096
    n_threads = 4
    n_batch = 512
    use_gpu = True
    n_gpu_layers = -1
    temperature = 0.2

# ===================== Define Provider =====================

# We use the built-in LlamaCppProvider that comes pre-registered with flowlib.
# When we call provider_registry.get(ProviderType.LLM, "llamacpp"),
# it will use the text-analysis-model configuration already in the system.

# ===================== Create and Run Flow =====================

async def run_composite_flow():
    """Create and run a composite flow with standalone stages."""
    print("\n===== Composite Flow Example =====")
    print("Running standalone stages directly...")
    
    # Create sample input
    sample_text = """
    Artificial intelligence has transformed many industries in recent years. 
    Companies are investing heavily in AI technologies. Machine learning 
    algorithms can now recognize patterns in data that humans might miss. 
    However, there are concerns about privacy and bias in AI systems.
    Researchers work to address these issues while advancing the technology.
    """
    
    input_data = TextInput(text=sample_text)
    context = Context(data=input_data)
    
    # Display available standalone stages
    standalone_stage_names = stage_registry.get_stages()
    print(f"Available standalone stages: {', '.join(standalone_stage_names)}")
    
    # Execute standalone stages directly
    print("\nExecuting standalone stages directly...")
    
    # Split text
    split_result = await split_text(context)
    print(f"Split text into {split_result.sentence_count} sentences with {split_result.word_count} words")
    
    # Classify text
    classify_result = await classify_text(context)
    print(f"Text classified as '{classify_result.category}' with confidence {classify_result.confidence:.2f}")
    
    # Enhance text
    enhance_result:EnhancedTextResult = await enhance_text(context)
    print(f"Text enhanced with {len(enhance_result.changes_made)} improvements")
    
    return enhance_result

# ===================== Demonstrate Stage Reuse =====================

@flow(
    name="simple-enhancement-flow",
    description="Flow for enhancing text by improving clarity and structure while preserving meaning"
)
class SimpleEnhancementFlow:
    """Simple flow that reuses the enhance_text standalone stage."""
    
    @stage(input_model=TextInput, output_model=ClassificationResult)
    async def classify_text(self, context: Context) -> ClassificationResult:
        """Classify the text to identify its category and subcategories."""
        logger.info("Classifying text in simple flow")
        
        input_data:TextInput = context.data
        
        # Fix: remove await from synchronous method
        classification_prompt = resource_registry.get(
            "text-classification", 
            resource_type=ResourceType.PROMPT
        )
        
        llm = await provider_registry.get(ProviderType.LLM, "llamacpp")
        
        try:
            result:ClassificationResult = await llm.generate_structured(
                classification_prompt,
                ClassificationResult,
                "text-analysis-model",
                {'text': input_data.text}
            )
            
            logger.info(f"Text classified as '{result.category}' with confidence {result.confidence:.2f}")
            return result
        except Exception as e:
            logger.warning(f"Error using LLM for classification: {str(e)}")
            return None
    
    @stage(input_model=EnhanceStageInput, output_model=EnhancedTextResult)
    async def enhance_text(self, context: Context) -> EnhancedTextResult:
        """Enhance the text based on its classification."""
        logger.info("Enhancing text in simple flow")
        
        # Get the classification and original text directly from the input model
        input_data:EnhanceStageInput = context.data
        
        # Get prompt template for enhancement - Fix: remove await
        enhancement_prompt = resource_registry.get(
            "text-enhancement", 
            resource_type=ResourceType.PROMPT
        )
        
        # Get LLM provider
        llm = await provider_registry.get(ProviderType.LLM, "llamacpp")
        
        try:
            # Use structured generation to get properly formatted results
            result:EnhancedTextResult = await llm.generate_structured(
                enhancement_prompt,
                EnhancedTextResult,
                "text-analysis-model",
                {'text': input_data.original_text}
            )

            # Return the enhanced result
            logger.info(f"Text enhanced with {len(result.changes_made)} improvements")
            return result
            
        except Exception as e:
            logger.warning(f"Error using LLM for enhancement: {str(e)}.")
            return None
            
    
    @pipeline(input_model=TextInput, output_model=EnhancedTextResult)
    async def run_pipeline(self, input_data: TextInput) -> EnhancedTextResult:
        """Execute the text enhancement pipeline.
        
        This pipeline coordinates the execution of all stages:
        1. Classify the text
        2. Enhance the text based on classification
        
        Args:
            input_data: Input text to process
            
        Returns:
            Enhanced text with classification and improvements
        """        
        # Debug: Print the input text
        print(f"\nDEBUG: Original input text: '{input_data.text}'")
        
        # Get stage instances using the get_stage method
        classify_stage = self.get_stage("classify_text")
        enhance_stage = self.get_stage("enhance_text")
        
        classify_context = Context(data=input_data)
        classification_result = await classify_stage.execute(classify_context)
        
        # Extract classification data
        classification_data = classification_result.data
        print(f"DEBUG: Classification result: {classification_data}")
        
        # Create the enhance stage input with proper model
        enhance_input = EnhanceStageInput(
            classification=classification_data,
            original_text=input_data.text
        )
        
        print(f"DEBUG: Enhance input: classification={enhance_input.classification.category}, text='{enhance_input.original_text[:30]}...'")
        
        enhance_context = Context(data=enhance_input)
        enhancement_result = await enhance_stage.execute(enhance_context)
        
        # Return the enhanced text result
        return enhancement_result.data

async def run_simple_flow():
    """Run the simple enhancement flow.
    
    This demonstrates how to create and execute a flow that combines
    text classification and enhancement.
    """
    # Create the flow instance
    flow = SimpleEnhancementFlow()
    
    # Create input data
    input_data = TextInput(text="I had a wonderful experience with this product. It exceeded my expectations!")
    
    # Run the pipeline directly - simpler approach that avoids issues
    print("\n===== Simple Enhancement Flow =====")
    print("Running flow with direct pipeline call...")
    context = Context(data=input_data)
    flow_result = await flow.execute(context)
    
    result:EnhancedTextResult = flow_result.data
    # Show the results - EnhancedTextResult contains the original_text, enhanced_text, and changes_made
    print("\nEnhanced Text:")
    print(result.enhanced_text)
    
    print("\nChanges Made:")
    for change in result.changes_made:
        print(f" - {change}")
    
    print("="*40)
    
    return result

# ===================== Run Both Flows =====================

async def run_demo():
    """Run both flow examples to demonstrate different ways to use flows."""
    # Run the composite flow example
    await run_composite_flow()
    
    # Run the simple flow demonstrating standalone stage reuse
    await run_simple_flow()

# ===================== Main Entry Point =====================

if __name__ == "__main__":
    asyncio.run(run_demo()) 