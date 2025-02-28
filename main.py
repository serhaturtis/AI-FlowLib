"""Main script to run the flow generator."""

import os
from pathlib import Path

# Add the workspace root to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Get the absolute path to the flowgen package
FLOWGEN_ROOT = Path(__file__).parent / "flowgen"

# Set up environment variables
os.environ["ROOT_FOLDER"] = str(FLOWGEN_ROOT)
os.environ["PROMPTS_FOLDER"] = str(FLOWGEN_ROOT / "prompts")

from flowgen.generate_simple_flow import main as generate_flow

if __name__ == "__main__":
    generate_flow()

