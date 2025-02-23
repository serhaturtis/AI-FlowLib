"""Test script for C++ docstring generator."""

import asyncio
import logging
import argparse
import sys
import traceback
import shutil
from pathlib import Path
from typing import List

from app.flows.cpp_docstring_generator import CppDocstringGenerator

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="C++ Docstring generation tool")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to C++ file or directory to process"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update the source files with generated docstrings"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup files before updating (only used with --update)"
    )
    return parser.parse_args()

def get_cpp_files(path: str) -> List[str]:
    """Get list of C++ files to process.
    
    Args:
        path: Path to file or directory
        
    Returns:
        List of file paths to process
    """
    path_obj = Path(path)
    if path_obj.is_file():
        if path_obj.suffix in ['.cpp', '.hpp', '.h', '.cc']:
            return [str(path_obj)]
        return []
        
    cpp_files = []
    for ext in ['.cpp', '.hpp', '.h', '.cc']:
        cpp_files.extend(str(p) for p in path_obj.rglob(f'*{ext}'))
    return sorted(cpp_files)

def backup_file(file_path: str) -> None:
    """Create backup of source file.
    
    Args:
        file_path: Path to file to backup
    """
    backup_path = file_path + '.bkp'
    shutil.copy2(file_path, backup_path)
    logger.info(f"Created backup: {backup_path}")

async def main_async() -> None:
    """Async main execution."""
    args = parse_args()
    
    try:
        # Get files to process
        files = get_cpp_files(args.path)
        if not files:
            logger.error("No C++ files found to process")
            sys.exit(1)
            
        logger.info(f"Found {len(files)} C++ files to process")
        
        # Create backups if requested
        if args.update and args.backup:
            logger.info("Creating backup files...")
            for file_path in files:
                backup_file(file_path)
        
        # Create and use generator with context manager
        async with CppDocstringGenerator() as generator:
            for file_path in files:
                logger.info(f"\nProcessing {file_path}")
                
                try:
                    # Process file
                    result = await generator.process_file(file_path, update_file=args.update)
                    
                    # Print results
                    logger.info(f"\nResults for {result.filepath}:")
                    for i, func_result in enumerate(result.functions, 1):
                        logger.info(f"\nFunction {i}:")
                        logger.info(f"Generated Docstring:\n{func_result.docstring}")
                        logger.info(f"Confidence: {func_result.confidence:.2f}")
                    
                    if result.requires_review:
                        logger.warning("\nThis file requires review!")
                        logger.warning(f"Review Comments: {result.review_comments}")
                    
                    if args.update:
                        logger.info(f"Updated {file_path} with new docstrings")
                        
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

def main() -> None:
    """Main entry point."""
    asyncio.run(main_async())

if __name__ == '__main__':
    main() 