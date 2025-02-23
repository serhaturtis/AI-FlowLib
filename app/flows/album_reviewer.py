"""Album review generator implementation."""

import json
import logging
from pathlib import Path
from typing import List

from flowlib import flow, stage, pipeline, managed
from flowlib.providers.llm import ModelConfig

from ..models.music_review import (
    AlbumInfo,
    Song,
    SongReview,
    AlbumReview,
    TextResponse,
    TrackList
)
from ..config.app_config import AppConfig

logger = logging.getLogger(__name__)

@flow("album_reviewer")
@managed
class AlbumReviewer:
    """Album review generator using flow framework."""
    
    def __init__(self):
        """Initialize generator."""
        # Load configuration
        self.config = AppConfig.load()
        
        # Create model config
        model_configs = {
            "analysis_model": ModelConfig(
                path=self.config.Provider.Models.ANALYSIS_MODEL,
                model_type=self.config.Provider.Models.MODEL_TYPE,
                n_ctx=self.config.Provider.Models.N_CTX,
                n_threads=self.config.Provider.Models.N_THREADS,
                n_batch=self.config.Provider.Models.N_BATCH,
                use_gpu=self.config.Provider.Models.USE_GPU
            )
        }
        
        # Setup provider
        self.provider = managed.factory.llm(
            name=self.config.Provider.NAME,
            model_configs=model_configs,
            max_models=self.config.Provider.MAX_MODELS
        )
    
    async def __aenter__(self) -> 'AlbumReviewer':
        """Async context manager entry."""
        await self.provider.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.provider.cleanup()
    
    def _load_prompt(self, filename: str) -> str:
        """Load prompt template from file."""
        prompt_path = Path("./prompts") / filename
        with open(prompt_path) as f:
            return f.read()
    
    @stage(output_model=str)
    async def generate_album_overview(self, album_info: AlbumInfo) -> str:
        """Generate general album overview and description."""
        # Get and format the schema
        schema = TextResponse.model_json_schema()
        
        # Format the prompt
        formatted_prompt = self._load_prompt("album_overview.txt").format(
            album_info=album_info.model_dump_json(indent=2),
            schema=json.dumps(schema, indent=2)
        )
        
        # Generate overview
        result = await self.provider.generate_structured(
            prompt=formatted_prompt,
            model_name=self.config.Flow.MODEL_NAME,
            response_model=TextResponse,
            max_tokens=self.config.Flow.Generation.MAX_TOKENS,
            temperature=0.7  # Slightly higher for creative description
        )
        
        return result.text.strip()
    
    @stage(output_model=str)
    async def analyze_themes(self, album_info: AlbumInfo, overview: str) -> str:
        """Analyze album themes and artistic direction."""
        # Get and format the schema
        schema = TextResponse.model_json_schema()
        
        # Format the prompt
        formatted_prompt = self._load_prompt("theme_analysis.txt").format(
            album_info=album_info.model_dump_json(indent=2),
            overview=overview,
            schema=json.dumps(schema, indent=2)
        )
        
        # Generate analysis
        result = await self.provider.generate_structured(
            prompt=formatted_prompt,
            model_name=self.config.Flow.MODEL_NAME,
            response_model=TextResponse,
            max_tokens=self.config.Flow.Generation.MAX_TOKENS,
            temperature=0.6  # Balanced between creativity and analysis
        )
        
        return result.text.strip()
    
    @stage(output_model=List[Song])
    async def generate_tracklist(self, album_info: AlbumInfo) -> List[Song]:
        """Generate a plausible track listing."""
        # Get and format the schema
        schema = TrackList.model_json_schema()
        
        # Format the prompt
        formatted_prompt = self._load_prompt("tracklist_generation.txt").format(
            album_info=album_info.model_dump_json(indent=2),
            schema=json.dumps(schema, indent=2)
        )
        
        # Generate track listing
        result = await self.provider.generate_structured(
            prompt=formatted_prompt,
            model_name=self.config.Flow.MODEL_NAME,
            response_model=TrackList,
            max_tokens=self.config.Flow.Generation.MAX_TOKENS,
            temperature=0.6
        )
        
        return result.tracks
    
    @stage(output_model=SongReview)
    async def review_song(self, song: Song, album_info: AlbumInfo, themes_analysis: str) -> SongReview:
        """Generate a review for a single song."""
        # Get and format the schema
        schema = SongReview.model_json_schema()
        
        # Format the prompt
        formatted_prompt = self._load_prompt("song_review.txt").format(
            song=song.model_dump_json(indent=2),
            album_info=album_info.model_dump_json(indent=2),
            themes_analysis=themes_analysis,
            schema=json.dumps(schema, indent=2)
        )
        
        # Generate song review
        result = await self.provider.generate_structured(
            prompt=formatted_prompt,
            model_name=self.config.Flow.MODEL_NAME,
            response_model=SongReview,
            max_tokens=self.config.Flow.Generation.MAX_TOKENS,
            temperature=0.7  # Higher for creative reviews
        )
        
        return result
    
    @stage(output_model=str)
    async def generate_conclusion(
        self,
        album_info: AlbumInfo,
        overview: str,
        themes_analysis: str,
        song_reviews: List[SongReview]
    ) -> str:
        """Generate final thoughts and conclusion."""
        # Calculate overall rating
        ratings = [review.rating for review in song_reviews]
        avg_rating = sum(ratings) / len(ratings)
        
        # Get and format the schema
        schema = TextResponse.model_json_schema()
        
        # Format the prompt
        formatted_prompt = self._load_prompt("conclusion.txt").format(
            album_info=album_info.model_dump_json(indent=2),
            overview=overview,
            themes_analysis=themes_analysis,
            avg_rating=avg_rating,
            num_songs=len(song_reviews),
            schema=json.dumps(schema, indent=2)
        )
        
        # Generate conclusion
        result = await self.provider.generate_structured(
            prompt=formatted_prompt,
            model_name=self.config.Flow.MODEL_NAME,
            response_model=TextResponse,
            max_tokens=self.config.Flow.Generation.MAX_TOKENS,
            temperature=0.6
        )
        
        return result.text.strip()
    
    @pipeline(output_model=AlbumReview)
    async def generate_review(self, album_info: AlbumInfo) -> AlbumReview:
        """Generate a complete album review."""
        # Generate album overview
        overview = await self.generate_album_overview(album_info)
        logger.info("Generated album overview")
        
        # Analyze themes
        themes_analysis = await self.analyze_themes(album_info, overview)
        logger.info("Analyzed themes")
        
        # Generate track listing
        songs = await self.generate_tracklist(album_info)
        logger.info(f"Generated track listing with {len(songs)} songs")
        
        # Review each song
        song_reviews = []
        for song in songs:
            review = await self.review_song(song, album_info, themes_analysis)
            song_reviews.append(review)
            logger.info(f"Generated review for song: {song.title}")
        
        # Generate conclusion
        conclusion = await self.generate_conclusion(
            album_info,
            overview,
            themes_analysis,
            song_reviews
        )
        logger.info("Generated conclusion")
        
        # Calculate overall rating and collect highlights
        ratings = [review.rating for review in song_reviews]
        overall_rating = sum(ratings) / len(ratings)
        
        # Collect unique highlights from song reviews
        highlights = []
        for review in song_reviews:
            highlights.extend(review.highlights)
        highlights = list(set(highlights))  # Remove duplicates
        
        # Create final review
        return AlbumReview(
            album_info=album_info,
            overview=overview,
            themes_analysis=themes_analysis,
            songs=songs,
            song_reviews=song_reviews,
            overall_rating=overall_rating,
            highlights=highlights,
            conclusion=conclusion
        ) 