"""Multi-stage flow example with LLM-powered text processing.

This example demonstrates using flowlib to create a multi-stage flow that processes
text through a series of LLM-powered stages:
1. Information extraction - Extract key entities and facts from input text
2. Sentiment analysis - Analyze sentiment of the extracted information
3. Summarization - Generate a concise summary
4. Report generation - Create a final structured report

The example shows:
- Flow and stage definition with decorators
- Input/output validation with Pydantic models
- LLM provider configuration and initialization
- Context passing between stages
- Structured data extraction with LLMs
"""

import asyncio
import logging
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import inspect

from flowlib.core.errors import ValidationError, ExecutionError
from flowlib.resources.constants import ResourceType
from flowlib.resources.registry import resource_registry
from flowlib.resources.decorators import model, prompt
from flowlib.providers.constants import ProviderType
from flowlib.providers.registry import provider_registry
from flowlib.flows.decorators import flow, stage, pipeline
from flowlib.core.context import Context

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===================== Define Models for Each Stage =====================

class InputText(BaseModel):
    """Input model for the text processing flow."""
    text: str = Field(..., description="Raw text to process")
    max_entities: int = Field(5, description="Maximum number of entities to extract")

class ExtractedInfo(BaseModel):
    """Output model from information extraction stage."""
    entities: List[Dict[str, str]] = Field(..., description="List of entities with type and value")
    key_facts: List[str] = Field(..., description="Key facts extracted from the text")
    topics: List[str] = Field(..., description="Main topics identified in the text")

class SentimentResult(BaseModel):
    """Output model from sentiment analysis stage."""
    sentiment: str = Field(..., description="Overall sentiment (positive, negative, neutral)")
    score: float = Field(..., description="Sentiment score from -1 (negative) to 1 (positive)")
    entity_sentiments: Optional[Dict[str, float]] = Field(None, description="Sentiment for each entity")

class SummaryInput(BaseModel):
    extract_information: ExtractedInfo = Field(..., description="Extracted information")
    analyze_sentiment: SentimentResult = Field(..., description="Sentiment analysis results")

class TextSummary(BaseModel):
    """Output model from summarization stage."""
    summary: str = Field(..., description="Concise summary of the text")
    key_points: List[str] = Field(..., description="Key points from the text")

class ReportInput(BaseModel):
    extract_information: ExtractedInfo = Field(..., description="Extracted information")
    analyze_sentiment: SentimentResult = Field(..., description="Sentiment analysis results")
    generate_summary: TextSummary = Field(..., description="Summary of the text")

class FinalReport(BaseModel):
    """Output model for the final report."""
    title: str = Field(..., description="Report title")
    summary: str = Field(..., description="Executive summary")
    extracted_info: ExtractedInfo = Field(..., description="Extracted information")
    sentiment: SentimentResult = Field(..., description="Sentiment analysis results")
    recommendations: List[str] = Field(..., description="Generated recommendations")

class ReportResponse(BaseModel):
    """Model for parsing the LLM-generated report response."""
    title: str = Field(..., description="Report title")
    summary: str = Field(..., description="Executive summary")
    recommendations: List[str] = Field(..., description="Generated recommendations")

# ===================== Define Prompts =====================

@prompt("information-extraction")
class InformationExtractionPrompt:
    """Prompt for extracting information from text."""
    template = """
    Extract key information from the following text:
    
    TEXT:
    {{text}}
    
    Please identify:
    1. Entities (people, organizations, locations, products) - up to {{max_entities}}
    2. Key facts stated in the text
    3. Main topics discussed
    """

@prompt("sentiment-analysis")
class SentimentAnalysisPrompt:
    """Prompt for analyzing sentiment of text."""
    template = """
    Analyze the sentiment of the following extracted information:
    
    ENTITIES:
    {{entities}}
    
    KEY FACTS:
    {{key_facts}}
    
    TOPICS:
    {{topics}}
    
    Provide sentiment analysis with:
    1. Overall sentiment (positive, negative, or neutral)
    2. Sentiment score (-1.0 to 1.0, where -1 is very negative, 0 is neutral, 1 is very positive)
    3. Sentiment for each entity when possible
    """
    
    config = {
        "max_tokens": 1000,
        "temperature": 0.2
    }

@prompt("summarization")
class SummarizationPrompt:
    """Prompt for summarizing text."""
    template = """
    Create a concise summary of the following information:
    
    ENTITIES:
    {{entities}}
    
    KEY FACTS:
    {{key_facts}}
    
    TOPICS:
    {{topics}}
    
    SENTIMENT:
    Overall sentiment: {{sentiment}}
    
    Please provide:
    1. A concise summary paragraph
    2. 3-5 key points from the information
    """
    
    config = {
        "max_tokens": 1000,
        "temperature": 0.3
    }

@prompt("report-generation")
class ReportGenerationPrompt:
    """Prompt for generating the final report."""
    template = """
    Generate a comprehensive report based on the following analysis:
    
    EXTRACTED INFORMATION:
    - Entities: {{entities}}
    - Key Facts: {{key_facts}}
    - Topics: {{topics}}
    
    SENTIMENT ANALYSIS:
    - Overall Sentiment: {{sentiment}}
    - Sentiment Score: {{score}}
    
    SUMMARY:
    {{summary}}
    
    Generate a report with:
    1. An appropriate title
    2. Executive summary
    3. 3-5 recommendations based on the analysis
    """
    
    config = {
        "max_tokens": 1500,
        "temperature": 0.4
    }

# ===================== Define LLM Model Configuration =====================

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
    max_tokens = 1000

# Debugging model registration
print("Checking if 'text-analysis-model' is registered:")
# Use contains method which is synchronous
if resource_registry.contains("text-analysis-model", resource_type="model"):
    print("Model is registered in the registry")
else:
    print("Model is NOT registered in the registry")

# We can't await here directly since this is not an async context
# We'll log what's available instead
print(f"Available models in registry: {resource_registry.list('model')}")

# ===================== Define Flow =====================

@flow(
    name="text-analysis-flow",
    description="Flow for multi-stage text analysis including extraction, sentiment, summarization and reporting"
)
class TextAnalysisFlow:
    """Multi-stage flow for analyzing text with LLM.
    
    This flow demonstrates a multi-stage processing pipeline for text analysis:
    1. Extract key information from input text
    2. Analyze sentiment of extracted information
    3. Generate a summary based on the analysis
    4. Create a final formatted report
    """
    
    @stage(input_model=InputText, output_model=ExtractedInfo)
    async def extract_information(self, context: Context) -> ExtractedInfo:
        """Extract key information from the input text."""
        logger.info("Extracting information from text")
        
        # Get input data
        input_data:InputText = context.data
        
        # Get prompt from resource registry - Fix: remove await from synchronous method
        extraction_prompt = resource_registry.get(
            "information-extraction", 
            resource_type=ResourceType.PROMPT
        )
        
        # Get LLM provider - using our registered provider
        llm = await provider_registry.get(ProviderType.LLM, "llamacpp")
        
        # Create prompt variables
        prompt_vars = {
            "text": input_data.text,
            "max_entities": input_data.max_entities
        }
        
        # Ensure output_type is a class, not an instance
        print(f"DEBUG: ExtractedInfo type: {type(ExtractedInfo)}")
        print(f"DEBUG: ExtractedInfo is class: {inspect.isclass(ExtractedInfo)}")
        
        # Generate structured output using the LLM with Flowlib's template handling
        result = await llm.generate_structured(
            prompt=extraction_prompt,
            output_type=ExtractedInfo,  # This is a class, not an instance
            model_name="text-analysis-model",
            prompt_variables=prompt_vars
        )
        
        # Print debug information about the result
        print(f"DEBUG: Result type from llm.generate_structured: {type(result)}")
        print(f"DEBUG: Result data: {result}")
        
        # Validate and clean entities - ensure they have required keys
        validated_entities = []
        for entity in result.entities:
            # Only include entities that have both type and value
            if isinstance(entity, dict) and 'type' in entity and 'value' in entity:
                validated_entities.append(entity)
            else:
                logger.warning(f"Skipping invalid entity: {entity}")
        
        # Update entities with validated list
        result.entities = validated_entities
        
        logger.info(f"Extracted {len(result.entities)} entities and {len(result.key_facts)} facts")
        return result
    
    @stage(input_model=ExtractedInfo, output_model=SentimentResult)
    async def analyze_sentiment(self, context: Context) -> SentimentResult:
        """Analyze sentiment of extracted information."""
        logger.info("Analyzing sentiment")
        
        # Get extracted information from previous stage
        extracted_info = context.data
        
        # Get prompt from resource registry
        sentiment_prompt = resource_registry.get(
            "sentiment-analysis", 
            resource_type=ResourceType.PROMPT
        )
        
        # Get LLM provider - using our registered provider
        llm = await provider_registry.get(ProviderType.LLM, "llamacpp")
        
        # Format entities for prompt - validate that entities have required keys
        entities_text_items = []
        for entity in extracted_info.entities:
            if 'type' in entity and 'value' in entity:
                entities_text_items.append(f"- {entity['type']}: {entity['value']}")
            elif entity:  # If entity has some data but not the expected structure
                # Use whatever keys are available
                entity_str = ", ".join(f"{k}: {v}" for k, v in entity.items())
                if entity_str:
                    entities_text_items.append(f"- {entity_str}")
        
        entities_text = "\n".join(entities_text_items) if entities_text_items else "No entities found."
        key_facts_text = "\n".join([f"- {fact}" for fact in extracted_info.key_facts]) if extracted_info.key_facts else "No key facts found."
        topics_text = "\n".join([f"- {topic}" for topic in extracted_info.topics]) if extracted_info.topics else "No topics identified."
        
        # Prepare prompt variables
        prompt_vars = {
            "entities": entities_text,
            "key_facts": key_facts_text,
            "topics": topics_text
        }
        
        # Generate structured output
        result = await llm.generate_structured(
            prompt=sentiment_prompt,
            output_type=SentimentResult,
            model_name="text-analysis-model",
            prompt_variables=prompt_vars
        )
        
        logger.info(f"Sentiment analysis complete: {result.sentiment} (score: {result.score})")
        return result
    
    @stage(output_model=TextSummary)
    async def generate_summary(self, context: Context) -> TextSummary:
        """Generate a summary based on extracted information and sentiment."""
        logger.info("Generating summary")
        
        # Get data from previous stages - now we have a SummaryInput model
        combined_data:SummaryInput = context.data
        extracted_info:ExtractedInfo = combined_data.extract_information
        sentiment_result:SentimentResult = combined_data.analyze_sentiment
        
        # Get prompt
        summarization_prompt = resource_registry.get(
            "summarization", 
            resource_type=ResourceType.PROMPT
        )
        
        # Get LLM provider - using our registered provider
        llm = await provider_registry.get(ProviderType.LLM, "llamacpp")
        
        # Format entities safely for prompt
        entities_text_items = []
        for entity in extracted_info.entities:
            if 'type' in entity and 'value' in entity:
                entities_text_items.append(f"- {entity['type']}: {entity['value']}")
        
        entities_text = "\n".join(entities_text_items) if entities_text_items else "No entities found."
        key_facts_text = "\n".join([f"- {fact}" for fact in extracted_info.key_facts]) if extracted_info.key_facts else "No key facts found."
        topics_text = "\n".join([f"- {topic}" for topic in extracted_info.topics]) if extracted_info.topics else "No topics identified."
            
        # Get sentiment value directly from the model
        sentiment_value = sentiment_result.sentiment
        
        # Prepare prompt variables
        prompt_vars = {
            "entities": entities_text,
            "key_facts": key_facts_text,
            "topics": topics_text,
            "sentiment": sentiment_value
        }
        
        # Generate structured output
        result = await llm.generate_structured(
            prompt=summarization_prompt,
            output_type=TextSummary,
            model_name="text-analysis-model",
            prompt_variables=prompt_vars
        )
        
        logger.info(f"Summary generated with {len(result.key_points)} key points")
        return result
    
    @stage(output_model=FinalReport)
    async def generate_report(self, context: Context) -> FinalReport:
        """Generate final report combining all previous results."""
        logger.info("Generating final report")
        
        # Get data from all previous stages - now we have a ReportInput model
        combined_data = context.data
        extracted_info = combined_data.extract_information
        sentiment_result = combined_data.analyze_sentiment
        summary_result = combined_data.generate_summary
        
        # Get prompt
        report_prompt = resource_registry.get(
            "report-generation", 
            resource_type=ResourceType.PROMPT
        )
        
        # Get LLM provider - using our registered provider
        llm = await provider_registry.get(ProviderType.LLM, "llamacpp")
        
        # Format entities safely for prompt
        entities_text_items = []
        for entity in extracted_info.entities:
            if 'type' in entity and 'value' in entity:
                entities_text_items.append(f"- {entity['type']}: {entity['value']}")
        
        entities_text = "\n".join(entities_text_items) if entities_text_items else "No entities found."
        key_facts_text = "\n".join([f"- {fact}" for fact in extracted_info.key_facts]) if extracted_info.key_facts else "No key facts found."
        topics_text = "\n".join([f"- {topic}" for topic in extracted_info.topics]) if extracted_info.topics else "No topics identified."
            
        # Get sentiment and score values directly from the model
        sentiment_value = sentiment_result.sentiment
        score_value = sentiment_result.score
            
        # Get summary value directly from the model
        summary_value = summary_result.summary
        
        # Prepare prompt variables
        prompt_vars = {
            "entities": entities_text,
            "key_facts": key_facts_text,
            "topics": topics_text,
            "sentiment": sentiment_value,
            "score": score_value,
            "summary": summary_value
        }
        
        # Generate structured response
        result = await llm.generate_structured(
            prompt=report_prompt,
            output_type=ReportResponse,
            model_name="text-analysis-model",
            prompt_variables=prompt_vars
        )
        
        # Create the final report using the models we already have
        final_report = FinalReport(
            title=result.title,
            summary=result.summary,
            extracted_info=extracted_info,
            sentiment=sentiment_result,
            recommendations=result.recommendations
        )
        
        logger.info("Final report generated")
        return final_report

    @pipeline(input_model=InputText, output_model=FinalReport)
    async def run_pipeline(self, input_data: InputText) -> FinalReport:
        """Execute the text analysis pipeline.
        
        This pipeline method coordinates the execution of all stages:
        1. Create a context with the input data
        2. Execute each stage in sequence
        3. Return the final report 
        
        Args:
            input_data: Input data to process
            
        Returns:
            Final report with analysis results
        """
        # Create context with the input data
        # Input data is already a Pydantic model
        input_context = Context(data=input_data)
        
        # Get stage instances using the get_stage method
        extract_stage = self.get_stage("extract_information")
        sentiment_stage = self.get_stage("analyze_sentiment")
        summary_stage = self.get_stage("generate_summary")
        report_stage = self.get_stage("generate_report")
        
        # Execute extraction stage
        extract_result = await extract_stage.execute(input_context)
        # The extracted_info is now directly available as a model in result.data
        extracted_info = extract_result.data
        
        # Create a context with the extracted info as input (must be a model, not a dict)
        sentiment_context = Context(data=extracted_info)
        sentiment_result = await sentiment_stage.execute(sentiment_context)
        # Get sentiment info directly from result
        sentiment_info = sentiment_result.data
        
        summary_input = SummaryInput(
            extract_information=extracted_info,
            analyze_sentiment=sentiment_info
        )
        
        summary_context = Context(data=summary_input)
        summary_result = await summary_stage.execute(summary_context)
        # Get summary info directly from result
        summary_info = summary_result.data

        report_input = ReportInput(
            extract_information=extracted_info,
            analyze_sentiment=sentiment_info,
            generate_summary=summary_info
        )
        
        report_context = Context(data=report_input)
        final_result = await report_stage.execute(report_context)
        
        return final_result.data

# ===================== Run the Flow =====================

async def run_flow():
    """Run the text analysis flow."""
    try:
        # Create sample input
        sample_text = """
        Apple Inc. reported record quarterly earnings today, with revenue reaching $89.5 billion. 
        CEO Tim Cook announced a new line of MacBook Pro laptops featuring the M3 chip. 
        The company's stock price increased by 5% following the announcement. 
        Analysts from Goldman Sachs predict continued growth for Apple's services division.
        In contrast, some market watchers expressed concerns about supply chain constraints in China 
        affecting future iPhone production. The upcoming iPhone models are expected to feature 
        significant camera improvements according to Bloomberg reports.
        """
        
        input_data = InputText(text=sample_text, max_entities=8)
        
        # Create flow instance
        flow = TextAnalysisFlow()
        
        # Display available stages
        available_stages = flow.get_stages()
        logger.info(f"Available stages: {', '.join(available_stages)}")
        
        # Execute flow using the execute method instead of directly calling the pipeline method
        logger.info("Starting text analysis flow")
        result = await flow.execute(Context(data=input_data))
        
        # Print results
        print("\n" + "="*80)

        report:FinalReport = result.data
        
        # Access model attributes directly 
        print(f"REPORT: {report.title}")
        print("="*80)
        print(f"\nEXECUTIVE SUMMARY:\n{report.summary}")
        
        print("\nEXTRACTED ENTITIES:")
        for entity in report.extracted_info.entities:
            print(f"- {entity['type']}: {entity['value']}")
            
        print("\nKEY FACTS:")
        for fact in report.extracted_info.key_facts:
            print(f"- {fact}")
            
        print(f"\nSENTIMENT: {report.sentiment.sentiment} (score: {report.sentiment.score})")
        
        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")
            
        print("="*80)
    
    except ValidationError as e:
        logger.error(f"Validation Error: {e}")
        if hasattr(e, 'validation_errors') and e.validation_errors:
            logger.error("Validation Errors Details:")
            for err in e.validation_errors:
                location = err.get('location', 'unknown')
                message = err.get('message', 'unknown error')
                error_type = err.get('type', 'unknown type')
                logger.error(f"  - Location: {location}")
                logger.error(f"    Message: {message}")
                logger.error(f"    Type: {error_type}")
        if hasattr(e, 'context') and e.context:
            logger.error("Error Context:")
            for key, value in e.context.__dict__.items():
                if key != 'parent' and value:  # Skip parent and empty values
                    logger.error(f"  - {key}: {value}")
                    
    except ExecutionError as e:
        logger.error(f"Execution Error: {e}")
        cause = getattr(e, 'cause', None)
        if cause:
            logger.error(f"Caused by: {type(cause).__name__}: {str(cause)}")
            
        if hasattr(e, 'context') and e.context:
            logger.error("Error Context:")
            for key, value in e.context.__dict__.items():
                if key != 'parent' and value:  # Skip parent and empty values
                    logger.error(f"  - {key}: {value}")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

# ===================== Main Entry Point =====================

if __name__ == "__main__":
    asyncio.run(run_flow()) 