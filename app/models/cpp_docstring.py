"""Models for C++ docstring generation."""

from typing import List, Optional
from pydantic import BaseModel, Field

class CppFunctionInfo(BaseModel):
    """Information about a C++ function."""
    name: str = Field(description="Name of the function")
    args: List[str] = Field(default_factory=list, description="List of argument names with types")
    returns: Optional[str] = Field(None, description="Return type if specified")
    code: str = Field(description="Function code")
    line_number: int = Field(description="Line number where function starts")
    is_method: bool = Field(default=False, description="Whether this is a class method")
    class_name: Optional[str] = Field(None, description="Name of the class if this is a method")
    
class CppDocstringResult(BaseModel):
    """Generated docstring for a C++ function."""
    docstring: str = Field(
        description="The complete docstring text following Doxygen format. "
        "Must include a brief description, detailed description if needed, "
        "@param tags for parameters, @return tag if applicable, and "
        "@throws tag if exceptions are thrown."
    )
    confidence: float = Field(
        description="Confidence score for the generated docstring (0.0 to 1.0). "
        "Should reflect understanding of the function's purpose and completeness of documentation.",
        ge=0.0,
        le=1.0
    )
    
class CppFileResult(BaseModel):
    """Results for a single C++ file."""
    filepath: str = Field(description="Path to the processed file")
    functions: List[CppDocstringResult] = Field(default_factory=list, description="Docstrings for each function")
    requires_review: bool = Field(description="Whether manual review is recommended")
    review_comments: Optional[str] = Field(None, description="Comments for review if needed") 