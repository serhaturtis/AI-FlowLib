"""Test script for document analyzer."""

import asyncio
import logging
import argparse
import sys

from app.flows.document_analyzer import DocumentAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Document analysis tool")
    parser.add_argument(
        "--text",
        default="Your sample text here...",
        help="Text to analyze"
    )
    return parser.parse_args()

async def main_async() -> None:
    """Async main execution."""
    args = parse_args()
    
    try:
        # Create and use analyzer with context manager
        async with DocumentAnalyzer() as analyzer:
            # Analyze document
            result = await analyzer.analyze(args.text)
            
            # Print results
            logger.info("\nAnalysis Results:")
            logger.info(f"Topics: {result.topics.topics}")
            logger.info(f"Main Topic: {result.topics.main_topic} (Confidence: {result.topics.topic_confidence:.2f})")
            logger.info(f"Sentiment: {result.sentiment.sentiment} (Confidence: {result.sentiment.confidence:.2f})")
            logger.info(f"Key Phrases: {', '.join(result.sentiment.key_phrases)}")
            logger.info("\nBrief Summary:")
            logger.info(result.summary.brief)
            logger.info("\nDetailed Summary:")
            logger.info(result.summary.detailed)
            
            if result.requires_review:
                logger.warning("\nThis analysis requires review!")
                logger.warning(f"Review Comments: {result.review_comments}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

def main() -> None:
    """Main entry point."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 