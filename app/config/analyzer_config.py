"""Configuration for document analyzer."""

from typing import Dict, Any
from new_framework import config

@config
class AnalyzerConfig:
    """Configuration for document analyzer."""
    
    class Provider:
        """LLM provider configuration."""
        NAME: str = "document_analyzer"
        MAX_MODELS: int = 2
        
        class Models:
            """Model configurations."""
            ANALYSIS_MODEL: str = "/home/swr/tools/models/my_models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
            N_CTX: int = 2048
            N_THREADS: int = 4
            N_BATCH: int = 512
            USE_GPU: bool = True
    
    class Flow:
        """Flow configuration."""
        NAME: str = "document_analyzer"
        MODEL_NAME: str = "analysis_model"
        PROMPT_TEMPLATE: str = "./prompts/analysis.txt"
        
        class Generation:
            """Generation parameters."""
            MAX_TOKENS: int = 500
            TEMPERATURE: float = 0.7
            TOP_P: float = 0.9
            TOP_K: int = 40
            REPEAT_PENALTY: float = 1.1 