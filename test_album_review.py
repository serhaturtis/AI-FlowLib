"""Test script for album review generator."""

import asyncio
import json
import logging
from pathlib import Path

from app.flows.album_reviewer import AlbumReviewer
from app.models.music_review import AlbumInfo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main_async():
    """Run the album review generator."""
    # Example album info
    album_info = AlbumInfo(
        title="The Quantum Garden",
        artist="Stellar Drift",
        genre="Electronic/Ambient",
        year=2024,
        mood="Contemplative, ethereal",
        themes=[
            "Space exploration",
            "Scientific discovery",
            "Human consciousness",
            "Nature vs technology"
        ],
        additional_info="A concept album exploring the intersection of quantum physics and human experience through electronic and ambient soundscapes."
    )
    
    logger.info("Generating review for: %s by %s", album_info.title, album_info.artist)
    
    # Create and use reviewer
    async with AlbumReviewer() as reviewer:
        try:
            # Generate review
            result = await reviewer.generate_review(album_info)
            
            # Save result
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            output_file = output_dir / f"{album_info.artist.replace(' ', '_')}_{album_info.title.replace(' ', '_')}.json"
            with open(output_file, 'w') as f:
                json.dump(result.model_dump(), f, indent=2)
            
            # Print summary
            logger.info("\nReview generated successfully!")
            logger.info("Album: %s by %s", result.album_info.title, result.album_info.artist)
            logger.info("Overall rating: %.1f/5.0", result.overall_rating)
            logger.info("Number of tracks: %d", len(result.songs))
            logger.info("\nHighlights:")
            for highlight in result.highlights[:5]:  # Show top 5 highlights
                logger.info("- %s", highlight)
            logger.info("\nSaved full review to: %s", output_file)
            
        except Exception as e:
            logger.error("Error generating review: %s", str(e))
            raise

def main():
    """Main entry point."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 