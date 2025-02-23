"""Models for docstring generation."""

from typing import List, Optional
from pydantic import BaseModel, Field

class FunctionInfo(BaseModel):
    """Information about a Python function."""
    name: str = Field(description="Name of the function")
    args: List[str] = Field(default_factory=list, description="List of argument names")
    returns: Optional[str] = Field(None, description="Return type if specified")
    code: str = Field(description="Function code")
    line_number: int = Field(description="Line number where function starts")
    
class DocstringResult(BaseModel):
    """Generated docstring for a function."""
    docstring: str = Field(
        description="The complete docstring text following Google style guide format. "
        "Must include a one-line summary, detailed description if needed, Args section, "
        "Returns section, and Raises section if applicable."
    )
    confidence: float = Field(
        description="Confidence score for the generated docstring (0.0 to 1.0). "
        "Should reflect understanding of the function's purpose and completeness of documentation.",
        ge=0.0,
        le=1.0
    )
    
class FileResult(BaseModel):
    """Results for a single file."""
    filepath: str = Field(description="Path to the processed file")
    functions: List[DocstringResult] = Field(default_factory=list, description="Docstrings for each function")
    requires_review: bool = Field(description="Whether manual review is recommended")
    review_comments: Optional[str] = Field(None, description="Comments for review if needed") 