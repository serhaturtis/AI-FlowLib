from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class Document(BaseModel):
    """Document model."""
    text: str = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TopicAnalysis(BaseModel):
    """Topic analysis result."""
    topics: List[str] = Field(..., min_items=1)
    main_topic: str = Field(...)
    topic_confidence: float = Field(..., ge=0.0, le=1.0)

class SentimentAnalysis(BaseModel):
    """Sentiment analysis result."""
    sentiment: str = Field(..., pattern="^(positive|negative|neutral)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    key_phrases: List[str] = Field(..., min_items=1)

class Summary(BaseModel):
    """Document summary."""
    brief: str = Field(..., min_length=50, max_length=200)
    detailed: str = Field(..., min_length=200)

class AnalysisResult(BaseModel):
    """Complete analysis result."""
    topics: TopicAnalysis
    sentiment: SentimentAnalysis
    summary: Summary
    requires_review: bool = Field(default=False)
    review_comments: Optional[str] = None 