#!/usr/bin/env python3
"""
Create Worklist CSV for Radiology Feature Extraction

Scans a directory for CT/MRI files and creates a worklist.csv with their paths.
"""

import argparse
import csv
from pathlib import Path


def create_worklist(input_dir, output_csv, extensions=None, keywords=None, keywords_avoid=None, recursive=True):
    """
    Create worklist CSV from slide files in input directory
    
    Args:
        input_dir: Path to directory containing slide files
        output_csv: Path where worklist.csv will be saved
        extensions: List of file extensions to include (default: common slide formats).
                   If empty list provided, searches all files.
        keywords: List of keywords to filter filenames (optional). Files must contain at least one keyword.
        keywords_avoid: List of keywords to filter out filenames (optional). Files which contain any of the keywords are not included
        recursive: If True, search subdirectories; if False, only search top-level directory
    """
    # Default to common slide image formats if None, but allow empty list for all files
    if extensions is None:
        extensions = ['.nii', '.nii.gz', '.nrrd', '.mhd', '.dcm', '.mha']
    
    input_path = Path(input_dir)
    output_path = Path(output_csv)
    
    # Check input directory exists
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_path}")
        return
    
    if not input_path.is_dir():
        print(f"Error: Input path is not a directory: {input_path}")
        return
    
    # Find all slide files
    search_mode = "recursively" if recursive else "(top-level only)"
    print(f"Scanning directory {search_mode}: {input_path}")
    slide_files = []
    
    # If extensions is empty list, search for all files
    if extensions == []:
        print("Searching for all file types")
        if recursive:
            found = [f for f in input_path.rglob("*") if f.is_file()]
        else:
            found = [f for f in input_path.glob("*") if f.is_file()]
        # Filter out macOS metadata files
        found = [f for f in found if not f.name.startswith('._')]
        slide_files.extend(found)
    else:
        # Search for specific extensions
        for ext in extensions:
            # Use rglob for recursive search, glob for top-level only
            if recursive:
                found = list(input_path.rglob(f"*{ext}"))
            else:
                found = list(input_path.glob(f"*{ext}"))
            # Filter out macOS metadata files
            found = [f for f in found if not f.name.startswith('._')]
            slide_files.extend(found)
    
    # Apply keyword filtering if specified
    if keywords:
        print(f"Filtering for keywords: {', '.join(keywords)}")
        filtered_files = []
        for slide_file in slide_files:
            # Check if any keyword is in the filename (case-insensitive)
            if any(keyword.lower() in slide_file.name.lower() for keyword in keywords):
                filtered_files.append(slide_file)
        slide_files = filtered_files
        print(f"After keyword filtering: {len(slide_files)} files")

    # Enables filtering out of keywords
    if keywords_avoid:
        print(f"Filtering out keywords: {', '.join(keywords_avoid)}")
        filtered_files = []
        for slide_file in slide_files:
            # Check if any keyword is in the file path (case-insensitive)
            if not any(keyword.lower() in str(slide_file.absolute()).lower() for keyword in keywords_avoid):
                filtered_files.append(slide_file)
        slide_files = filtered_files
        print(f"After keyword filtering: {len(slide_files)} files")
    
    # Sort for consistent ordering
    slide_files = sorted(slide_files)
    
    if not slide_files:
        print(f"Warning: No files found in {input_path}")
        if extensions:
            print(f"Looking for extensions: {', '.join(extensions)}")
        if keywords:
            print(f"With keywords: {', '.join(keywords)}")
        return
    
    print(f"Found {len(slide_files)} files")
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write worklist CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file_path'])  # Header
        
        for slide_file in slide_files:
            writer.writerow([str(slide_file.absolute())])
    
    print(f"✅ Created worklist: {output_path}")
    print(f"   {len(slide_files)} files listed")
    
    # Show first few entries
    print(f"\nFirst 5 entries:")
    for slide_file in slide_files[:5]:
        print(f"  {slide_file.name}")
    
    if len(slide_files) > 5:
        print(f"  ... and {len(slide_files) - 5} more")


def main():
    parser = argparse.ArgumentParser(
        description="Create worklist CSV for whole slide image feature extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create worklist from all image volumes in a directory (default extensions)
  python create_worklist.py /data/slides/ worklist.csv
  
  # Specify custom extensions (e.g., for rare cases)
  python create_worklist.py /data/radiology/ worklist.csv --ext .v .mnc .hdr
  
  # Search all file types
  python create_worklist.py /data/files/ worklist.csv --ext
  
  # Filter by keywords
  python create_worklist.py /data/slides/ worklist.csv --keywords tumor patient1
  
  # Combine extensions and keywords
  python create_worklist.py /data/radiology/ worklist.csv --ext .mha --keywords CT MRI
        """
    )
    
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing files (searches recursively by default)"
    )
    
    parser.add_argument(
        "output_csv",
        type=str,
        help="Output path for worklist.csv"
    )
    
    parser.add_argument(
        "--ext",
        nargs="*",
        help="File extensions to include (default: common slide formats). Use --ext with no arguments to search all files.",
        default=None
    )
    
    parser.add_argument(
        "--keywords",
        nargs="+",
        help="Keywords to filter filenames (files must contain at least one keyword)",
        default=None
    )

    parser.add_argument(
            "--keywords_avoid",
            nargs="+",
            help="Keywords to filter out filenames (files containing any keyword are not included)",
            default=None
        )
    
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only search top-level directory, don't search subdirectories"
    )
    
    args = parser.parse_args()
    
    create_worklist(args.input_dir, args.output_csv, args.ext, args.keywords, args.keywords_avoid, recursive=not args.no_recursive)


if __name__ == "__main__":
    main()
