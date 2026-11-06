# Command-line interface for NAM tone matching

import argparse
import sys
from typing import List


def create_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for the CLI.
    
    Returns:
        ArgumentParser object
    """
    parser = argparse.ArgumentParser(description='NAM Tone Matching CLI')
    
    parser.add_argument('input_file', help='Input audio file')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--reference', '-r', help='Reference audio file')
    parser.add_argument('--isolate', action='store_true', help='Isolate guitar from input')
    parser.add_argument('--match', action='store_true', help='Match tone with reference')
    parser.add_argument('--effects', action='store_true', help='Apply effects')
    
    return parser


def main(args: List[str] = None) -> int:
    """
    Main CLI entry point.
    
    Args:
        args: Command line arguments (default: sys.argv[1:])
        
    Returns:
        Exit code
    """
    if args is None:
        args = sys.argv[1:]
    
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Placeholder for actual implementation
    print(f"CLI called with arguments: {parsed_args}")
    
    return 0