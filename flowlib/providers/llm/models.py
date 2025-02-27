"""Model configuration and generation parameters."""

from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    """Model configuration."""
    path: str = Field(..., description="Path to model file")
    model_type: str = Field(default="default", description="Type of model for prompt formatting (e.g., llama2, chatml, phi2)")
    n_ctx: int = Field(default=2048, gt=0)
    n_threads: int = Field(default=4, gt=0)
    n_batch: int = Field(default=512, gt=0)
    use_gpu: bool = Field(default=True, description="Whether to use GPU if available")
    n_gpu_layers: int = Field(default=-1, description="Number of layers to offload to GPU. -1 means use all layers, 0 means CPU only") 