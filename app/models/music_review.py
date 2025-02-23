"""Models for music album review generation."""

from typing import List, Optional
from pydantic import BaseModel, Field

class TextResponse(BaseModel):
    """Simple text response model."""
    text: str = Field(description="Generated text content")

class AlbumInfo(BaseModel):
    """Input information about an album."""
    title: str = Field(description="Album title")
    artist: str = Field(description="Artist/band name")
    genre: str = Field(description="Primary music genre")
    year: int = Field(description="Release year")
    mood: Optional[str] = Field(None, description="Overall mood or vibe of the album")
    themes: Optional[List[str]] = Field(default_factory=list, description="Main themes or topics")
    additional_info: Optional[str] = Field(None, description="Any additional context or information")

class Song(BaseModel):
    """Information about a single song."""
    title: str = Field(description="Song title")
    duration: str = Field(description="Song duration in MM:SS format")
    track_number: int = Field(description="Track number on the album")
    description: Optional[str] = Field(None, description="Brief description or highlights")

class TrackList(BaseModel):
    """List of songs in an album."""
    tracks: List[Song] = Field(description="List of songs in the album")

class SongReview(BaseModel):
    """Detailed review of a single song."""
    song: Song = Field(description="Song information")
    review: str = Field(description="Detailed song review")
    highlights: List[str] = Field(default_factory=list, description="Key musical highlights or moments")
    rating: float = Field(ge=0.0, le=5.0, description="Rating out of 5.0")

class AlbumReview(BaseModel):
    """Complete album review."""
    album_info: AlbumInfo = Field(description="Basic album information")
    overview: str = Field(description="General album overview and description")
    themes_analysis: str = Field(description="Analysis of album themes and artistic direction")
    songs: List[Song] = Field(description="List of songs on the album")
    song_reviews: List[SongReview] = Field(description="Individual song reviews")
    overall_rating: float = Field(ge=0.0, le=5.0, description="Overall album rating out of 5.0")
    highlights: List[str] = Field(default_factory=list, description="Key album highlights")
    conclusion: str = Field(description="Final thoughts and recommendation") 