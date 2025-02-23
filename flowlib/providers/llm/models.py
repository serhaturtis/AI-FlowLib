"""Model configuration and generation parameters."""

from typing import Optional
from pydantic import BaseModel, Field

class GenerationParams(BaseModel):
    """Generation parameters validation."""
    max_tokens: int = Field(default=100, gt=0)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0)
    repeat_penalty: float = Field(default=1.1, ge=1.0)

class ModelConfig(BaseModel):
    """Model configuration."""
    path: str = Field(..., description="Path to model file")
    model_type: str = Field(default="default", description="Type of model for prompt formatting (e.g., llama2, chatml, phi2)")
    n_ctx: int = Field(default=2048, gt=0)
    n_threads: int = Field(default=4, gt=0)
    n_batch: int = Field(default=512, gt=0)
    use_gpu: bool = Field(default=True, description="Whether to use GPU if available") 