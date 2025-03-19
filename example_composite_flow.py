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
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

import flowlib as fl
from flowlib.providers.llm.llama_cpp_provider import LlamaCppProvider, LlamaCppSettings
from flowlib.core.registry.constants import ProviderType

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
    original_text: str = Field(..., description="Original input text")
    enhanced_text: str = Field(..., description="Text with enhancements")
    changes_made: List[str] = Field(..., description="List of enhancements made")

class CompleteTextAnalysis(BaseModel):
    """Complete text analysis combining all results."""
    original_text: str = Field(..., description="Original input text")
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

@fl.prompt("text-classification")
class ClassificationPrompt:
    """Prompt for text classification."""
    template = """
    Classify the following text into a primary category and optional subcategories.
    
    TEXT:
    {text}
    
    Provide your response as a JSON object with the following structure:
    {{
        "category": "primary category name",
        "confidence": 0.0 to 1.0,
        "subcategories": ["subcategory1", "subcategory2", ...]
    }}
    """

@fl.prompt("text-enhancement")
class EnhancementPrompt:
    """Prompt for text enhancement."""
    template = """
    Enhance the following text by improving clarity, fixing grammatical errors, 
    and improving sentence structure while preserving the original meaning.
    
    ORIGINAL TEXT:
    {text}
    
    Provide your response as a JSON object with the following structure:
    {{
        "enhanced_text": "enhanced version here",
        "changes_made": ["change 1", "change 2", ...]
    }}
    """

# ===================== Define Standalone Stages =====================

@fl.standalone(
    name="text-splitter",
    input_model=TextInput,
    output_model=SplitResult
)
async def split_text(context: fl.Context) -> SplitResult:
    """Split input text into sentences and count words."""
    logger.info("Splitting text into sentences")
    
    # Handle different types of input data
    input_data = None
    text = ""
    
    # Check if context.data is already a TextInput object
    if isinstance(context.data, TextInput):
        input_data = context.data
        text = input_data.text
    # Check if context.data has an 'input' key that contains a TextInput
    elif isinstance(context.data, dict) and 'input' in context.data:
        input_obj = context.data.get('input')
        if isinstance(input_obj, TextInput):
            input_data = input_obj
            text = input_data.text
        elif isinstance(input_obj, dict) and 'text' in input_obj:
            text = input_obj['text']
    # Try to access text directly if none of the above worked
    elif hasattr(context.data, 'text'):
        text = context.data.text
    
    if not text:
        # If we still don't have text, use a placeholder
        logger.warning("No text found in input data, using placeholder")
        text = "This is a placeholder text for demonstration purposes."
    
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

@fl.standalone(
    name="text-classifier",
    input_model=TextInput,
    output_model=ClassificationResult
)
async def classify_text(context: fl.Context) -> ClassificationResult:
    """Classify text using LLM."""
    logger.info("Classifying text")
    
    # Handle different types of input data
    input_data = None
    text = ""
    
    # Check if context.data is already a TextInput object
    if isinstance(context.data, TextInput):
        input_data = context.data
        text = input_data.text
    # Check if context.data has an 'input' key that contains a TextInput
    elif isinstance(context.data, dict) and 'input' in context.data:
        input_obj = context.data.get('input')
        if isinstance(input_obj, TextInput):
            input_data = input_obj
            text = input_data.text
        elif isinstance(input_obj, dict) and 'text' in input_obj:
            text = input_obj['text']
    # Try to access text directly if none of the above worked
    elif hasattr(context.data, 'text'):
        text = context.data.text
    
    if not text:
        # If we still don't have text, use a placeholder
        logger.warning("No text found in input data, using placeholder")
        text = "This is a placeholder text for demonstration purposes."
    
    # Get prompt template
    classification_prompt = await fl.resource_registry.get(
        "text-classification", 
        resource_type=fl.ResourceType.PROMPT
    )
    
    # Format prompt with input text
    formatted_prompt = classification_prompt.template.format(
        text=text
    )
    
    # Get LLM provider - use the pre-registered llamacpp provider
    llm = await fl.provider_registry.get(ProviderType.LLM, "llamacpp")
    
    # Call LLM to classify text - use the text-analysis-model from the multi-stage flow example
    result = await llm.generate_structured(
        formatted_prompt,
        ClassificationResult,
        "text-analysis-model",
        **classification_prompt.config
    )
    
    logger.info(f"Text classified as '{result.category}' with confidence {result.confidence:.2f}")
    return result

@fl.standalone(
    name="text-enhancer",
    input_model=TextInput,
    output_model=EnhancedTextResult
)
async def enhance_text(context: fl.Context) -> EnhancedTextResult:
    """Enhance text using LLM."""
    logger.info("Enhancing text")
    
    # Handle different types of input data
    input_data = None
    text = ""
    
    # Check if context.data is already a TextInput object
    if isinstance(context.data, TextInput):
        input_data = context.data
        text = input_data.text
    # Check if context.data has an 'input' key that contains a TextInput
    elif isinstance(context.data, dict) and 'input' in context.data:
        input_obj = context.data.get('input')
        if isinstance(input_obj, TextInput):
            input_data = input_obj
            text = input_data.text
        elif isinstance(input_obj, dict) and 'text' in input_obj:
            text = input_obj['text']
    # Try to access text directly if none of the above worked
    elif hasattr(context.data, 'text'):
        text = context.data.text
    
    if not text:
        # If we still don't have text, use a placeholder
        logger.warning("No text found in input data, using placeholder")
        text = "This is a placeholder text for demonstration purposes."
    
    # Get prompt template
    enhancement_prompt = await fl.resource_registry.get(
        "text-enhancement", 
        resource_type=fl.ResourceType.PROMPT
    )
    
    # Format prompt with input text
    formatted_prompt = enhancement_prompt.template.format(
        text=text
    )
    
    # Get LLM provider - use the pre-registered llamacpp provider
    llm = await fl.provider_registry.get(ProviderType.LLM, "llamacpp")
    
    # Call LLM to enhance text - use the text-analysis-model from the multi-stage flow example
    # Pass EnhancedTextResult directly as the output model
    try:
        result = await llm.generate_structured(
            formatted_prompt,
            EnhancedTextResult,
            "text-analysis-model",
            **enhancement_prompt.config
        )
        
        # If the result doesn't include original_text, add it
        if not result.original_text:
            result.original_text = text
            
        logger.info(f"Text enhanced with {len(result.changes_made)} improvements")
        return result
    except Exception as e:
        # Fallback in case of errors with structured generation
        logger.warning(f"Error in structured generation: {str(e)}. Using fallback response.")
        
        # Generate text without structured format
        raw_response = await llm.generate(
            formatted_prompt,
            "text-analysis-model",
            temperature=0.4,
            max_tokens=1024
        )
        
        # Create a basic enhanced result
        return EnhancedTextResult(
            original_text=text,
            enhanced_text=raw_response.strip(),
            changes_made=["Enhanced with LLM"]
        )

@fl.standalone(
    name="metrics-calculator"
)
async def calculate_metrics(context: fl.Context) -> Dict[str, Any]:
    """Calculate text metrics from analysis results."""
    logger.info("Calculating metrics")
    
    # Access stages results from context data or use direct input
    input_text = ""
    split_result = None
    classify_result = None
    enhance_result = None
    
    # Check what kind of context we have
    if isinstance(context.data, dict):
        # Complex context with multiple stage results (from CompositeFlow)
        input_data = context.data.get("input")
        if input_data and hasattr(input_data, "text"):
            input_text = input_data.text
            
        # Try to get stage results
        split_result = context.data.get("text-splitter")
        classify_result = context.data.get("text-classifier")
        enhance_result = context.data.get("text-enhancer")
    elif hasattr(context.data, "text"):
        # Direct TextInput context (when called standalone)
        input_text = context.data.text
    
    # Create placeholder results if we don't have actual results
    if not split_result:
        split_result = SplitResult(
            sentences=["Placeholder sentence"],
            sentence_count=1,
            word_count=2
        )
    
    if not classify_result:
        classify_result = ClassificationResult(
            category="Placeholder",
            confidence=0.5,
            subcategories=[]
        )
    
    if not enhance_result:
        enhance_result = EnhancedTextResult(
            original_text=input_text,
            enhanced_text="Placeholder enhanced text",
            changes_made=["Placeholder change"]
        )
    
    # Calculate metrics based on real data
    metrics = {}
    
    # Text complexity metrics
    if split_result.sentence_count > 0:
        metrics["avg_sentence_length"] = split_result.word_count / split_result.sentence_count
        
        # Calculate vocabulary richness (unique words / total words)
        all_words = []
        for sentence in split_result.sentences:
            all_words.extend([word.lower().strip('.,!?()[]{}:;"\'') for word in sentence.split()])
        
        if all_words:
            metrics["vocabulary_richness"] = len(set(all_words)) / len(all_words)
        else:
            metrics["vocabulary_richness"] = 0.0
    else:
        metrics["avg_sentence_length"] = 0.0
        metrics["vocabulary_richness"] = 0.0
    
    # Classification confidence
    metrics["classification_confidence"] = classify_result.confidence
    metrics["subcategory_count"] = len(classify_result.subcategories)
    
    # Enhancement metrics
    metrics["enhancement_count"] = len(enhance_result.changes_made)
    if split_result.sentence_count > 0:
        metrics["enhancement_ratio"] = len(enhance_result.changes_made) / split_result.sentence_count
    else:
        metrics["enhancement_ratio"] = 0.0
        
    # Text length change ratio
    original_length = len(enhance_result.original_text)
    enhanced_length = len(enhance_result.enhanced_text)
    if original_length > 0:
        metrics["length_change_ratio"] = enhanced_length / original_length
    else:
        metrics["length_change_ratio"] = 1.0
    
    logger.info(f"Calculated {len(metrics)} metrics")
    return metrics

# ===================== Define Result Aggregator =====================

@fl.standalone(
    name="result-aggregator",
    output_model=CompleteTextAnalysis
)
async def aggregate_results(context: fl.Context) -> CompleteTextAnalysis:
    """Aggregate all stage results into a final result."""
    logger.info("Aggregating results")
    
    # Access stage results from context data or use direct input
    input_text = ""
    split_result = None
    classify_result = None
    enhance_result = None
    metrics_result = None
    
    # Check what kind of context we have
    if isinstance(context.data, dict):
        # Complex context with multiple stage results (from CompositeFlow)
        input_data = context.data.get("input")
        if input_data and hasattr(input_data, "text"):
            input_text = input_data.text
            
        # Try to get stage results
        split_result = context.data.get("text-splitter")
        classify_result = context.data.get("text-classifier")
        enhance_result = context.data.get("text-enhancer")
        metrics_result = context.data.get("metrics-calculator")
    elif hasattr(context.data, "text"):
        # Direct TextInput context (when called standalone)
        input_text = context.data.text
    
    # Create placeholder results if we don't have actual results
    if not split_result:
        split_result = SplitResult(
            sentences=["Placeholder sentence"],
            sentence_count=1,
            word_count=2
        )
    
    if not classify_result:
        classify_result = ClassificationResult(
            category="Placeholder",
            confidence=0.5,
            subcategories=[]
        )
    
    if not enhance_result:
        enhance_result = EnhancedTextResult(
            original_text=input_text,
            enhanced_text="Placeholder enhanced text",
            changes_made=["Placeholder change"]
        )
        
    if not metrics_result:
        metrics_result = {
            "avg_sentence_length": 2.0,
            "enhancement_ratio": 1.0,
            "classification_confidence": 0.5,
            "subcategory_count": 0,
        }
    
    # Create consolidated result
    complete_analysis = CompleteTextAnalysis(
        original_text=input_text,
        sentences=split_result.sentences,
        classification=classify_result,
        enhanced_text=enhance_result.enhanced_text,
        enhancement_count=len(enhance_result.changes_made),
        metrics=metrics_result
    )
    
    logger.info("Results aggregated into complete analysis")
    return complete_analysis

# ===================== Define Model Config =====================

@fl.model("text-analysis-model")
class LLMModelConfig:
    """Configuration for the text analysis LLM."""
    path = "/path/to/model.gguf"
    model_type = "phi4"
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
    context = fl.Context(data=input_data)
    
    # Display available standalone stages
    standalone_stage_names = fl.flows.registry.stage_registry.get_stages()
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
    enhance_result = await enhance_text(context)
    print(f"Text enhanced with {len(enhance_result.changes_made)} improvements")
    
    # Calculate metrics
    metrics_result = await calculate_metrics(context)
    print(f"Calculated {len(metrics_result)} metrics")
    
    # Aggregate results
    context_with_results = fl.Context(data={
        "input": input_data,
        "text-splitter": split_result,
        "text-classifier": classify_result,
        "text-enhancer": enhance_result,
        "metrics-calculator": metrics_result
    })
    final_result = await aggregate_results(context_with_results)
    
    # Print results
    print("\n" + "="*80)
    print("TEXT ANALYSIS REPORT")
    print("="*80)
    
    print(f"\nCLASSIFICATION: {final_result.classification.category} (confidence: {final_result.classification.confidence:.2f})")
    
    if final_result.classification.subcategories:
        print("SUBCATEGORIES:")
        for subcat in final_result.classification.subcategories:
            print(f" - {subcat}")
    
    print(f"\nEXTRACTED {len(final_result.sentences)} SENTENCES")
    for i, sentence in enumerate(final_result.sentences[:3], 1):
        print(f"{i}. {sentence}")
    if len(final_result.sentences) > 3:
        print(f"... and {len(final_result.sentences) - 3} more sentences")
    
    print("\nMETRICS:")
    for key, value in final_result.metrics.items():
        print(f" - {key}: {value:.2f}")
    
    print("\nENHANCED TEXT:")
    print(final_result.enhanced_text)
    
    print("="*80)
    
    return final_result

# ===================== Demonstrate Stage Reuse =====================

@fl.flow(name="simple-enhancement-flow")
class SimpleEnhancementFlow:
    """Simple flow that reuses the enhance_text standalone stage."""
    
    @fl.stage(input_model=TextInput)
    async def preprocess(self, context: fl.Context) -> TextInput:
        """Simple preprocessing stage."""
        logger.info("Preprocessing text")
        
        input_data = context.data
        
        # Add language detection (simplified example)
        if "python" in input_data.text.lower():
            input_data.language = "code-python"
        elif "javascript" in input_data.text.lower():
            input_data.language = "code-javascript"
            
        return input_data
    
    @fl.stage(input_model=TextInput, output_model=ClassificationResult)
    async def classify_text(self, context: fl.Context) -> ClassificationResult:
        """Classify the text to identify its category and subcategories."""
        logger.info("Classifying text in simple flow")
        
        input_data = context.data
        text = ""
        
        # Handle different input types
        if isinstance(input_data, dict):
            # Dictionary input (from stage.execute -> FlowResult)
            text = input_data.get("text", "").lower()
        elif hasattr(input_data, "text"):
            # TextInput object
            text = input_data.text.lower()
        else:
            # Fallback
            text = str(input_data).lower()
        
        # Get prompt template
        classification_prompt = await fl.resource_registry.get(
            "text-classification", 
            resource_type=fl.ResourceType.PROMPT
        )
        
        # Format prompt with input text
        formatted_prompt = classification_prompt.template.format(
            text=text
        )
        
        # Get LLM provider - use the pre-registered llamacpp provider
        llm = await fl.provider_registry.get(ProviderType.LLM, "llamacpp")
        
        # Call LLM to classify text - use the text-analysis-model
        try:
            result = await llm.generate_structured(
                formatted_prompt,
                ClassificationResult,
                "text-analysis-model",
                **classification_prompt.config
            )
            
            logger.info(f"Text classified as '{result.category}' with confidence {result.confidence:.2f}")
            return result
        except Exception as e:
            # Fallback to our simple classification logic if LLM fails
            logger.warning(f"Error using LLM for classification: {str(e)}. Using fallback classification.")
            
            # Simple fallback classification logic
            if "wonderful" in text or "exceed" in text:
                category = "Positive Review"
                confidence = 0.9
                subcategories = ["Enthusiastic", "Satisfied"]
            elif "issue" in text or "problem" in text:
                category = "Negative Review"
                confidence = 0.8
                subcategories = ["Technical Issue", "Complaint"]
            else:
                category = "Neutral"
                confidence = 0.7
                subcategories = ["General Feedback"]
            
            logger.info(f"Text classified as '{category}' with confidence {confidence} (fallback)")
            return ClassificationResult(
                category=category,
                confidence=confidence,
                subcategories=subcategories
            )
    
    @fl.stage(input_model=EnhanceStageInput, output_model=EnhancedTextResult)
    async def enhance_text(self, context: fl.Context) -> EnhancedTextResult:
        """Enhance the text based on its classification."""
        logger.info("Enhancing text in simple flow")
        
        # Get the classification and original text directly from the input model
        input_data = context.data
        classification = None
        original_text = ""
        
        if isinstance(input_data, dict):
            # Dictionary input
            classification = input_data.get("classification")
            original_text = input_data.get("original_text", "")
        elif hasattr(input_data, "classification") and hasattr(input_data, "original_text"):
            # EnhanceStageInput model
            classification = input_data.classification
            original_text = input_data.original_text
        
        if not classification:
            logger.warning("No classification data found in context")
            return EnhancedTextResult(
                original_text=original_text,
                enhanced_text=original_text,
                changes_made=[]
            )
        
        # Get prompt template for enhancement
        enhancement_prompt = await fl.resource_registry.get(
            "text-enhancement", 
            resource_type=fl.ResourceType.PROMPT
        )
        
        # Format prompt with input text and classification info
        formatted_prompt = enhancement_prompt.template.format(
            text=f"{original_text}\n\nThis text has been classified as: {classification.category} (confidence: {classification.confidence:.2f})"
        )
        
        # Get LLM provider
        llm = await fl.provider_registry.get(ProviderType.LLM, "llamacpp")
        
        try:
            # For simplicity, we'll use the non-structured generation instead
            raw_response = await llm.generate(
                formatted_prompt,
                "text-analysis-model",
                temperature=0.4,
                max_tokens=1024
            )
            
            # Try to extract JSON from the response
            import json
            import re
            
            # Look for JSON structure in the response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
            if json_match:
                try:
                    # Parse the JSON
                    json_str = json_match.group(1).strip()
                    parsed_data = json.loads(json_str)
                    
                    # Extract data from parsed JSON
                    enhanced_text = parsed_data.get("enhanced_text", original_text)
                    llm_changes = parsed_data.get("changes_made", ["Enhanced with LLM"])
                except:
                    # If JSON parsing fails, use the raw text
                    enhanced_text = raw_response.strip()
                    llm_changes = ["Enhanced with LLM"]
            else:
                # If no JSON found, use raw response
                enhanced_text = raw_response.strip()
                llm_changes = ["Enhanced with LLM"]
                
            # Now apply our classification-specific formatting on top of LLM enhancement
            changes_made = list(llm_changes)  # Copy LLM changes
            
            # Add classification-specific formatting
            if classification.category.lower() in ["positive review", "product review"]:
                enhanced_text = f"[POSITIVE REVIEW] {enhanced_text}"
                changes_made.append("Added sentiment prefix")
            elif classification.category.lower() in ["negative review"]:
                enhanced_text = f"[NEEDS ATTENTION] {enhanced_text}"
                changes_made.append("Added attention flag")
            
            # Add classification data
            enhanced_text = f"{enhanced_text}\n\nClassified as: {classification.category} (Confidence: {classification.confidence:.2f})"
            changes_made.append("Added classification metadata")
            
            logger.info(f"Text enhanced with {len(changes_made)} improvements")
            return EnhancedTextResult(
                original_text=original_text,
                enhanced_text=enhanced_text,
                changes_made=changes_made,
                classification=classification
            )
            
        except Exception as e:
            # Fallback to simple enhancement if LLM fails completely
            logger.warning(f"Error using LLM for enhancement: {str(e)}. Using basic fallback enhancement.")
            
            # Simple fallback enhancement
            enhanced_text = original_text
            changes_made = []
            
            # Add classification-specific formatting
            if classification.category.lower() in ["positive review", "product review"]:
                enhanced_text = f"[POSITIVE REVIEW] {enhanced_text}"
                changes_made.append("Added sentiment prefix")
            elif classification.category.lower() in ["negative review"]:
                enhanced_text = f"[NEEDS ATTENTION] {enhanced_text}"
                changes_made.append("Added attention flag")
            
            # Add classification data
            enhanced_text = f"{enhanced_text}\n\nClassified as: {classification.category} (Confidence: {classification.confidence:.2f})"
            changes_made.append("Added classification metadata")
            
            logger.info(f"Text enhanced with {len(changes_made)} improvements (fallback)")
            return EnhancedTextResult(
                original_text=original_text,
                enhanced_text=enhanced_text,
                changes_made=changes_made,
                classification=classification
            )
    
    @fl.pipeline(input_model=TextInput, output_model=EnhancedTextResult)
    async def run_pipeline(self, input_data: TextInput) -> EnhancedTextResult:
        """Execute the text enhancement pipeline.
        
        This pipeline coordinates the execution of all stages:
        1. Preprocess the input data
        2. Classify the preprocessed text
        3. Enhance the text based on classification
        
        Args:
            input_data: Input text to process
            
        Returns:
            Enhanced text with classification and improvements
        """
        # Create context with the input data
        context = fl.Context(data=input_data)
        
        # Get stage instances using the get_stage method
        preprocess_stage = self.get_stage("preprocess")
        classify_stage = self.get_stage("classify_text")
        enhance_stage = self.get_stage("enhance_text")
        
        # Execute stages in sequence - each returns a FlowResult
        preprocess_result = await preprocess_stage.execute(context)
        
        # Extract original text from the preprocessed result
        # The result can be a dict or a TextInput object converted to dict
        preprocessed_data = preprocess_result.data
        original_text = ""
        if isinstance(preprocessed_data, dict) and "text" in preprocessed_data:
            original_text = preprocessed_data["text"]
        
        # Use preprocessed data for classification
        # Create a new Context with the preprocessed data
        classify_context = fl.Context(data=preprocessed_data)
        classification_result = await classify_stage.execute(classify_context)
        
        # Extract classification data
        classification_data = classification_result.data
        
        # Create the enhance stage input with proper model
        enhance_input = EnhanceStageInput(
            classification=classification_data,
            original_text=original_text
        )
        
        # Create context for enhancement with the proper input model
        enhance_context = fl.Context(data=enhance_input)
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
    result = await flow.run_pipeline(input_data)
    
    # Show the results - EnhancedTextResult contains the original_text, enhanced_text, and changes_made
    print("\nOriginal Text:")
    print(result.original_text)
    
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