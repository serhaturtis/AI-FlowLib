from typing import Dict, Any, Type
from pathlib import Path
from new_framework.flows.stage import Stage
from new_framework.flows.composite import CompositeFlow
from new_framework.providers.llm import LLMProvider

from ..models.analysis import TopicAnalysis, SentimentAnalysis, Summary, AnalysisResult

def create_llm_processor(
    provider: LLMProvider,
    prompt_file: str,
    model: Type[AnalysisResult],
    flow_config: Dict[str, Any]
) -> Stage:
    """Create an LLM processor stage.
    
    Args:
        provider: LLM provider instance
        prompt_file: Name of prompt template file
        model: Output model type
        flow_config: Flow configuration
        
    Returns:
        Stage configured for LLM processing
    """
    async def processor(data: Dict[str, Any]) -> Dict[str, Any]:
        result = await provider.generate_structured(
            prompt=_load_prompt(prompt_file).format(
                text=data["text"],
                schema=model.model_json_schema()
            ),
            model_name=flow_config["model_name"],
            response_model=model,
            **(flow_config.get("generation_params", {}))
        )
        return model.model_validate(result).model_dump()
    
    return processor

def _load_prompt(filename: str) -> str:
    """Load prompt template from file."""
    prompt_path = Path("./prompts") / filename
    with open(prompt_path) as f:
        return f.read()

def create_analysis_pipeline(
    provider: LLMProvider,
    flow_config: Dict[str, Any]
) -> CompositeFlow:
    """Create the main analysis pipeline.
    
    Args:
        provider: LLM provider instance
        flow_config: Flow configuration
        
    Returns:
        Composite flow for document analysis
    """
    # Create topic analysis stage
    topic_stage = Stage(
        name="topic_analyzer",
        process_fn=create_llm_processor(
            provider,
            "topic_analysis.txt",
            TopicAnalysis,
            flow_config
        ),
        output_schema=TopicAnalysis
    )
    
    # Create sentiment analysis stage
    sentiment_stage = Stage(
        name="sentiment_analyzer",
        process_fn=create_llm_processor(
            provider,
            "sentiment_analysis.txt",
            SentimentAnalysis,
            flow_config
        ),
        output_schema=SentimentAnalysis
    )
    
    # Create summary stage
    summary_stage = Stage(
        name="summarizer",
        process_fn=create_llm_processor(
            provider,
            "summary.txt",
            Summary,
            flow_config
        ),
        output_schema=Summary
    )
    
    # Create and return composite flow
    return CompositeFlow(
        name="analysis_pipeline",
        flows=[topic_stage, sentiment_stage, summary_stage],
        output_schema=AnalysisResult
    ) 