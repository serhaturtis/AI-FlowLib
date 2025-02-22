"""Document analyzer implementation."""

from pathlib import Path
from typing import Dict, Any
from new_framework import flow, stage, pipeline, managed
from new_framework.providers.llm import ModelConfig

from ..models.analysis import (
    Document, AnalysisResult, TopicAnalysis,
    SentimentAnalysis, Summary
)
from ..config.analyzer_config import AnalyzerConfig

@flow("document_analyzer")
@managed
class DocumentAnalyzer:
    """Document analyzer using flow framework."""
    
    def __init__(self):
        """Initialize analyzer."""
        # Load configuration
        self.config = AnalyzerConfig.load()
        
        # Create model config
        model_configs = {
            "analysis_model": ModelConfig(
                path=self.config.Provider.Models.ANALYSIS_MODEL,
                n_ctx=self.config.Provider.Models.N_CTX,
                n_threads=self.config.Provider.Models.N_THREADS,
                n_batch=self.config.Provider.Models.N_BATCH,
                use_gpu=self.config.Provider.Models.USE_GPU
            )
        }
        
        # Setup provider
        self.provider = managed.factory.llm(
            name=self.config.Provider.NAME,
            model_configs=model_configs,
            max_models=self.config.Provider.MAX_MODELS
        )
    
    async def __aenter__(self) -> 'DocumentAnalyzer':
        """Async context manager entry."""
        await self.provider.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.provider.cleanup()
    
    def _load_prompt(self, filename: str) -> str:
        """Load prompt template from file."""
        prompt_path = Path("./prompts") / filename
        with open(prompt_path) as f:
            return f.read()
    
    @stage(output_model=TopicAnalysis)
    async def analyze_topics(self, text: str) -> TopicAnalysis:
        """Analyze document topics."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("topic_analysis.txt").format(
                text=text,
                schema=TopicAnalysis.model_json_schema()
            ),
            model_name=self.config.Flow.MODEL_NAME,
            response_model=TopicAnalysis,
            max_tokens=self.config.Flow.Generation.MAX_TOKENS,
            temperature=self.config.Flow.Generation.TEMPERATURE,
            top_p=self.config.Flow.Generation.TOP_P,
            top_k=self.config.Flow.Generation.TOP_K,
            repeat_penalty=self.config.Flow.Generation.REPEAT_PENALTY
        )
        return TopicAnalysis.model_validate(result)
    
    @stage(output_model=SentimentAnalysis)
    async def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """Analyze document sentiment."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("sentiment_analysis.txt").format(
                text=text,
                schema=SentimentAnalysis.model_json_schema()
            ),
            model_name=self.config.Flow.MODEL_NAME,
            response_model=SentimentAnalysis,
            max_tokens=self.config.Flow.Generation.MAX_TOKENS,
            temperature=self.config.Flow.Generation.TEMPERATURE,
            top_p=self.config.Flow.Generation.TOP_P,
            top_k=self.config.Flow.Generation.TOP_K,
            repeat_penalty=self.config.Flow.Generation.REPEAT_PENALTY
        )
        return SentimentAnalysis.model_validate(result)
    
    @stage(output_model=Summary)
    async def create_summary(self, text: str) -> Summary:
        """Create document summary."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("summary.txt").format(
                text=text,
                schema=Summary.model_json_schema()
            ),
            model_name=self.config.Flow.MODEL_NAME,
            response_model=Summary,
            max_tokens=self.config.Flow.Generation.MAX_TOKENS,
            temperature=self.config.Flow.Generation.TEMPERATURE,
            top_p=self.config.Flow.Generation.TOP_P,
            top_k=self.config.Flow.Generation.TOP_K,
            repeat_penalty=self.config.Flow.Generation.REPEAT_PENALTY
        )
        return Summary.model_validate(result)
    
    @pipeline(output_model=AnalysisResult)
    async def analyze(self, text: str) -> AnalysisResult:
        """Analyze document text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Analysis result
        """
        # Run analysis stages
        topics = await self.analyze_topics(text)
        sentiment = await self.analyze_sentiment(text)
        summary = await self.create_summary(text)
        
        # Create final result
        return AnalysisResult(
            topics=topics,
            sentiment=sentiment,
            summary=summary,
            requires_review=topics.topic_confidence < 0.7 or sentiment.confidence < 0.7
        ) 