"""Framework for building flow-based applications."""

from .flows.decorators import flow, stage, pipeline
from .core.application.config import config
from .core.resources.factory import managed
from .core.resources.managed_resource import ManagedResource
from .flows.builder import FlowBuilder
from .flows.base import Flow
from .core.models.context import Context
from .core.models.results import FlowResult, FlowStatus
from .core.errors.base import ValidationError, ErrorContext

__version__ = "0.1.0"
__all__ = [
    # Decorators for flow creation
    "flow",
    "stage",
    "pipeline",
    
    # Configuration management
    "config",
    
    # Resource management
    "managed",
    "ManagedResource",
    
    # Flow building
    "FlowBuilder",
    "Flow",
    
    # Core models
    "Context",
    "FlowResult",
    "FlowStatus",
    
    # Error handling
    "ValidationError",
    "ErrorContext"
]

# Example usage:
"""
from new_framework import flow, stage, pipeline, config, managed

@config
class AppConfig:
    MODEL_PATH: str = "models/default.gguf"
    USE_GPU: bool = True
    
    class Provider:
        NAME: str = "default"
        MAX_MODELS: int = 2

@flow("document_analysis")
@managed
class DocumentAnalyzer:
    def __init__(self):
        self.config = AppConfig.load()
        self.provider = managed.llm(
            self.config.Provider.NAME,
            model_path=self.config.MODEL_PATH,
            use_gpu=self.config.USE_GPU
        )
    
    @stage(output_model=TopicAnalysis)
    async def analyze_topics(self, text: str) -> TopicAnalysis:
        # Implementation
        pass
    
    @stage(output_model=SentimentAnalysis)
    async def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        # Implementation
        pass
    
    @pipeline(output_model=AnalysisResult)
    async def process(self, text: str) -> AnalysisResult:
        topics = await self.analyze_topics(text)
        sentiment = await self.analyze_sentiment(text)
        return AnalysisResult(topics=topics, sentiment=sentiment)

# Usage:
async with DocumentAnalyzer() as analyzer:
    result = await analyzer.process("Sample text to analyze")
""" 