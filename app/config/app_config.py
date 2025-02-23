"""Configuration for document analyzer."""

from flowlib import config

@config
class AppConfig:
    """Configuration for document analyzer."""
    
    class Provider:
        """LLM provider configuration."""
        NAME: str = "document_analyzer"
        MAX_MODELS: int = 2
        
        class Models:
            """Model configurations."""
            ANALYSIS_MODEL: str = "/path/to/your/analysis_model.gguf"
            MODEL_TYPE: str = "llama3"  # Model type for prompt formatting (chatml, llama2, phi2, etc.)
            N_CTX: int = 4096
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
            MAX_TOKENS: int = 2048
            TEMPERATURE: float = 0.7
            TOP_P: float = 0.9
            TOP_K: int = 40
            REPEAT_PENALTY: float = 1.1 