"""LLM provider utilities."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import subprocess
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

@dataclass
class GPUInfo:
    """Information about detected GPU."""
    name: str
    vram_total: int  # in MB
    cuda_version: str
    compute_capability: Tuple[int, int]
    architecture: str

class GPUConfigManager:
    """Manages automatic GPU configuration based on detected hardware."""
    
    # Known GPU architectures and their characteristics
    GPU_ARCHITECTURES = {
        "Ada Lovelace": {
            "series": ["RTX 40"],
            "optimal_batch": 1024,
            "supports_f16_kv": True,
            "min_layers": 28,
            "system_reserve_mb": 2000
        },
        "Ampere": {
            "series": ["RTX 30"],
            "optimal_batch": 1024,
            "supports_f16_kv": True,
            "min_layers": 24,
            "system_reserve_mb": 2000
        },
        "Turing": {
            "series": ["RTX 20", "GTX 16"],
            "optimal_batch": 512,
            "supports_f16_kv": True,
            "min_layers": 20,
            "system_reserve_mb": 1500
        }
    }

    @staticmethod
    def _try_nvidia_smi() -> Optional[Dict[str, Any]]:
        """Try to get GPU info using nvidia-smi."""
        try:
            nvidia_smi = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=gpu_name,memory.total,cuda_version", 
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.PIPE
            ).decode()
            
            gpu_name, vram_total, cuda_version = nvidia_smi.strip().split(", ")
            return {
                "name": gpu_name,
                "vram_total": int(float(vram_total)),
                "cuda_version": cuda_version
            }
        except:
            return None

    @staticmethod
    def _try_cuda_detect() -> Optional[Dict[str, Any]]:
        """Try to detect CUDA capabilities directly."""
        try:
            # Try to import torch for CUDA detection
            import torch
            if not torch.cuda.is_available():
                return None
                
            return {
                "name": torch.cuda.get_device_name(0),
                "vram_total": torch.cuda.get_device_properties(0).total_memory // (1024*1024),  # Convert to MB
                "cuda_version": torch.version.cuda
            }
        except:
            return None

    @staticmethod
    def _try_env_vars() -> Optional[Dict[str, Any]]:
        """Try to get GPU info from environment variables."""
        try:
            # Check for CUDA environment variables
            cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible_devices is None or cuda_visible_devices == "-1":
                return None
                
            # If we have CUDA devices but no detailed info, return minimal info
            return {
                "name": "CUDA GPU",
                "vram_total": 8000,  # Assume 8GB as safe default
                "cuda_version": os.environ.get("CUDA_VERSION", "11.0")
            }
        except:
            return None

    @staticmethod
    def get_gpu_info() -> Optional[GPUInfo]:
        """Detect GPU information using multiple methods."""
        # Try different detection methods in order
        gpu_data = None
        for method in [GPUConfigManager._try_nvidia_smi, 
                      GPUConfigManager._try_cuda_detect,
                      GPUConfigManager._try_env_vars]:
            gpu_data = method()
            if gpu_data:
                break
        
        if not gpu_data:
            logger.warning("No GPU detected through any method")
            return None
            
        # Determine architecture from name
        architecture = "Unknown"
        gpu_name_lower = gpu_data["name"].lower()
        
        if any(x in gpu_name_lower for x in ["4060", "4070", "4080", "4090", "ada"]):
            architecture = "Ada Lovelace"
            compute_capability = (8, 9)
        elif any(x in gpu_name_lower for x in ["3060", "3070", "3080", "3090", "ampere"]):
            architecture = "Ampere"
            compute_capability = (8, 6)
        elif any(x in gpu_name_lower for x in ["2060", "2070", "2080", "turing"]):
            architecture = "Turing"
            compute_capability = (7, 5)
        else:
            # Default to Ampere settings for unknown GPUs as a safe choice
            architecture = "Ampere"
            compute_capability = (8, 6)
        
        return GPUInfo(
            name=gpu_data["name"],
            vram_total=gpu_data["vram_total"],
            cuda_version=gpu_data["cuda_version"],
            compute_capability=compute_capability,
            architecture=architecture
        )

    def get_optimal_config(self, model_path: Path) -> Dict[str, Any]:
        """Get optimal configuration based on detected GPU and model size."""
        # Get model size in GB
        model_size_gb = model_path.stat().st_size / (1024 * 1024 * 1024)
        
        gpu_info = self.get_gpu_info()
        if not gpu_info:
            logger.warning("No GPU detected, using CPU configuration")
            return self._get_cpu_config()
            
        logger.info(f"Detected GPU: {gpu_info.name}")
        logger.info(f"VRAM: {gpu_info.vram_total}MB")
        logger.info(f"CUDA Version: {gpu_info.cuda_version}")
        logger.info(f"Architecture: {gpu_info.architecture}")
        
        # Get architecture-specific settings
        arch_settings = self.GPU_ARCHITECTURES.get(
            gpu_info.architecture, 
            {
                "optimal_batch": 512,
                "supports_f16_kv": True,
                "min_layers": 20,
                "system_reserve_mb": 2000
            }
        )
        
        # Calculate available VRAM with architecture-specific system reserve
        available_vram = gpu_info.vram_total - arch_settings["system_reserve_mb"]
        model_vram_mb = int(model_size_gb * 1024)
        
        # Adjust batch size based on model size
        batch_size = min(
            arch_settings["optimal_batch"],
            max(128, int(arch_settings["optimal_batch"] * (8 / model_size_gb)))
        )
        
        # Calculate GPU layers based on model size and available VRAM
        if gpu_info.vram_total >= 24000:  # For 24GB+ cards
            n_gpu_layers = 100  # Load all layers
        elif gpu_info.vram_total >= 16000:  # For 16GB cards
            n_gpu_layers = min(80, int((available_vram / model_vram_mb) * 100))
        elif gpu_info.vram_total >= 12000:  # For 12GB cards
            n_gpu_layers = min(60, int((available_vram / model_vram_mb) * 100))
        elif gpu_info.vram_total >= 8000:   # For 8GB cards
            n_gpu_layers = min(40, int((available_vram / model_vram_mb) * 100))
        else:  # For smaller cards
            n_gpu_layers = min(32, int((available_vram / model_vram_mb) * 100))
        
        # Ensure minimum layers per architecture
        n_gpu_layers = max(n_gpu_layers, arch_settings["min_layers"])
        
        # Adjust context size based on available VRAM
        n_ctx = min(
            4096,  # Maximum context size
            max(512, int((available_vram / 2) / (model_size_gb * 128)))  # Scale with model size
        )
        
        # Determine optimal configuration
        config = {
            "n_gpu_layers": n_gpu_layers,
            "n_batch": batch_size,
            "n_ctx": n_ctx,
            "f16_kv": arch_settings["supports_f16_kv"],
            "offload_kqv": True,
            "vocab_only": False,
            "use_mmap": True,
            "use_mlock": False
        }
        
        logger.info(f"GPU Configuration:")
        logger.info(f"- GPU Layers: {config['n_gpu_layers']}")
        logger.info(f"- Batch Size: {config['n_batch']}")
        logger.info(f"- Context Size: {config['n_ctx']}")
        
        return config
    
    def _get_cpu_config(self) -> Dict[str, Any]:
        """Get CPU-only configuration."""
        return {
            "n_gpu_layers": 0,
            "n_batch": 512,
            "n_ctx": 1024,
            "f16_kv": False,
            "offload_kqv": False,
            "vocab_only": False,
            "use_mmap": True,
            "use_mlock": False
        } 