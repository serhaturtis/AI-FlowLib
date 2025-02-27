"""Model configurations."""

from flowlib.providers.llm import ModelConfig
from flowlib.core.resources import model

@model("video_content")
class VideoContentModel(ModelConfig):
    """Configuration for video content generation model."""
    path: str = ""
    model_type: str = "phi4"
    n_ctx: int = 8192
    n_threads: int = 12
    n_batch: int = 512
    use_gpu: bool = True
    n_gpu_layers: int = 48

@model("document_analysis")
class DocumentAnalysisModel(ModelConfig):
    """Configuration for document analysis model."""
    path: str = ""
    model_type: str = "phi4"
    n_ctx: int = 8192
    n_threads: int = 12
    n_batch: int = 512
    use_gpu: bool = True
    n_gpu_layers: int = 48

@model("flow_generator")
class FlowGeneratorModel(ModelConfig):
    """Configuration for flow generation model."""
    path: str = ""
    model_type: str = "phi4"
    n_ctx: int = 8192
    n_threads: int = 12
    n_batch: int = 512
    use_gpu: bool = True
    n_gpu_layers: int = 48 