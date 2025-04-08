import os
import glob
import numpy as np
import cv2
from pathlib import Path
import sys

# Add project root to sys.path to import config if needed later, although not strictly necessary for this script
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# sys.path.append(str(PROJECT_ROOT))
# from src.pipelineA.config import BASE_DATA_DIR # Using hardcoded path for simplicity here

# Define the base directory where MIT, Harvard, UCL datasets reside
# Adjust this path if your structure is different
BASE_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "CW2-Dataset" / "data"

# Number of sample depth files to analyze per sequence/dataset
NUM_SAMPLES_TO_CHECK = 3

def analyze_intrinsics(intrinsics_path):
    """Analyzes the format of an intrinsics file."""
    analysis = {"exists": False, "lines": 0, "format_ok": False, "error": None}
    if not intrinsics_path.exists():
        analysis["error"] = "File not found"
        return analysis

    analysis["exists"] = True
    try:
        with open(intrinsics_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()] # Read non-empty lines
        analysis["lines"] = len(lines)

        # Basic check for 2 lines with 3 numbers each (expected format)
        if len(lines) == 2:
            line1_parts = lines[0].split()
            line2_parts = lines[1].split()
            if len(line1_parts) == 3 and len(line2_parts) == 3:
                # Try converting to float to ensure they are numbers
                [float(p) for p in line1_parts]
                [float(p) for p in line2_parts]
                analysis["format_ok"] = True
            else:
                 analysis["error"] = "Expected 3 numbers per line"
        else:
            analysis["error"] = f"Expected 2 lines, found {len(lines)}"

    except Exception as e:
        analysis["error"] = f"Error reading/parsing: {e}"

    return analysis

def analyze_depth_map(depth_path):
    """Analyzes a single depth map file."""
    analysis = {"path": str(depth_path.name), "dtype": None, "shape": None, "min_val": None, "max_val": None, "mean_nz": None, "scale_guess": "Unknown", "error": None}
    try:
        # Load depth map using similar logic as dataset.py's load_depth_map
        ext = depth_path.suffix.lower()
        depth_map = None
        is_raw_candidate = "harvard_tea_2" in str(depth_path) or "ucl" in str(depth_path) # Heuristic

        if ext in ['.png', '.jpg', '.jpeg']:
             # Try loading as image (could be raw uint16 or processed float saved as image)
             img = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR) # Read any depth/color info
             if img is None:
                 raise ValueError("cv2.imread failed")
             depth_map = img.astype(np.float32) # Convert to float for analysis
             analysis["dtype"] = str(img.dtype) # Report original dtype

        elif ext == '.npy':
             depth_map = np.load(str(depth_path))
             analysis["dtype"] = str(depth_map.dtype)
             depth_map = depth_map.astype(np.float32) # Convert to float for analysis
        else:
             raise ValueError(f"Unsupported extension: {ext}")

        analysis["shape"] = depth_map.shape

        # Analyze non-zero values
        non_zero_depths = depth_map[depth_map > 1e-6] # Avoid pure zeros
        if non_zero_depths.size > 0:
            analysis["min_val"] = np.min(non_zero_depths)
            analysis["max_val"] = np.max(non_zero_depths)
            analysis["mean_nz"] = np.mean(non_zero_depths)

            # Guess scale based on max value (heuristic)
            # If max value is large (e.g., > 50), likely millimeters before scaling
            # If max value is small (e.g., < 20), likely meters (already scaled or TSDF)
            if analysis["max_val"] > 50:
                 analysis["scale_guess"] = "Millimeters (Raw?)"
            elif analysis["max_val"] > 0:
                 analysis["scale_guess"] = "Meters (Scaled?)"
            else:
                 analysis["scale_guess"] = "Zero values only?"
        else:
            analysis["scale_guess"] = "All zero/near-zero values"

    except Exception as e:
        analysis["error"] = f"Error loading/analyzing: {e}"

    return analysis

def analyze_sequence(seq_path):
    """Analyzes a single sequence directory (MIT or Harvard sub-sequence)."""
    print(f"\n--- Analyzing Sequence: {seq_path.relative_to(BASE_DATA_DIR)} ---")

    # 1. Analyze Intrinsics
    intrinsics_path = seq_path / "intrinsics.txt"
    intrinsics_analysis = analyze_intrinsics(intrinsics_path)
    print(f"  Intrinsics ({intrinsics_path.name}):")
    print(f"    Exists: {intrinsics_analysis['exists']}")
    if intrinsics_analysis["exists"]:
        print(f"    Lines: {intrinsics_analysis['lines']}")
        print(f"    Format OK (2 lines, 3 numbers/line): {intrinsics_analysis['format_ok']}")
    if intrinsics_analysis["error"]:
        print(f"    Error: {intrinsics_analysis['error']}")

    # 2. Find and Analyze Depth Directory/Files
    depth_dir_tsdf = seq_path / "depthTSDF"
    depth_dir_raw = seq_path / "depth"
    depth_dir = None
    depth_dir_name = None

    if depth_dir_tsdf.exists():
        depth_dir = depth_dir_tsdf
        depth_dir_name = "depthTSDF"
        print(f"  Depth Directory: Found '{depth_dir_name}'")
    elif depth_dir_raw.exists():
        depth_dir = depth_dir_raw
        depth_dir_name = "depth"
        print(f"  Depth Directory: Found '{depth_dir_name}'")
    else:
        print("  Depth Directory: Not found ('depthTSDF' or 'depth')")
        return # Cannot proceed without depth

    depth_files = sorted(list(depth_dir.glob("*.png")) + list(depth_dir.glob("*.npy")))
    print(f"  Depth Files: Found {len(depth_files)} (.png or .npy)")

    if not depth_files:
        print("    No depth files to analyze.")
        return

    # Analyze sample depth maps
    print(f"  Analyzing first {min(NUM_SAMPLES_TO_CHECK, len(depth_files))} depth maps:")
    for i, depth_file in enumerate(depth_files):
        if i >= NUM_SAMPLES_TO_CHECK:
            break
        depth_analysis = analyze_depth_map(depth_file)
        print(f"    - Sample {i+1} ({depth_analysis['path']}):")
        if depth_analysis["error"]:
            print(f"      Error: {depth_analysis['error']}")
        else:
            print(f"      Dtype: {depth_analysis['dtype']}")
            print(f"      Shape: {depth_analysis['shape']}")
            print(f"      Min (Non-Zero): {depth_analysis['min_val']:.4f}" if depth_analysis['min_val'] is not None else "N/A")
            print(f"      Max (Non-Zero): {depth_analysis['max_val']:.4f}" if depth_analysis['max_val'] is not None else "N/A")
            print(f"      Mean (Non-Zero): {depth_analysis['mean_nz']:.4f}" if depth_analysis['mean_nz'] is not None else "N/A")
            print(f"      Scale Guess: {depth_analysis['scale_guess']}")


if __name__ == "__main__":
    print(f"Starting Dataset Format Analysis...")
    print(f"Base Data Directory: {BASE_DATA_DIR}")

    if not BASE_DATA_DIR.exists():
        print(f"\nERROR: Base data directory not found at {BASE_DATA_DIR}")
        print("Please ensure the 'CW2-Dataset/data' directory exists relative to the project root.")
        sys.exit(1)

    # Analyze MIT sequences
    print("\n=======================================")
    print("           MIT Dataset")
    print("=======================================")
    mit_sequences_found = False
    for item in sorted(BASE_DATA_DIR.iterdir()):
        if item.is_dir() and item.name.startswith("mit_"):
            mit_sequences_found = True
            # MIT sequences have one sub-sequence directory inside
            sub_seq_dirs = [d for d in item.iterdir() if d.is_dir()]
            if len(sub_seq_dirs) == 1:
                 analyze_sequence(sub_seq_dirs[0])
            else:
                 print(f"\n--- Skipping MIT Sequence: {item.name} (Expected 1 sub-dir, found {len(sub_seq_dirs)}) ---")
    if not mit_sequences_found:
        print("No MIT sequences found directly under base data directory.")


    # Analyze Harvard sequences
    print("\n=======================================")
    print("         Harvard Dataset")
    print("=======================================")
    harvard_sequences_found = False
    for item in sorted(BASE_DATA_DIR.iterdir()):
         if item.is_dir() and item.name.startswith("harvard_"):
             harvard_sequences_found = True
             # Harvard sequences have one sub-sequence directory inside
             sub_seq_dirs = [d for d in item.iterdir() if d.is_dir()]
             if len(sub_seq_dirs) == 1:
                  analyze_sequence(sub_seq_dirs[0])
             else:
                  print(f"\n--- Skipping Harvard Sequence: {item.name} (Expected 1 sub-dir, found {len(sub_seq_dirs)}) ---")
    if not harvard_sequences_found:
        print("No Harvard sequences found directly under base data directory.")


    # Analyze UCL dataset
    ucl_dir = BASE_DATA_DIR / "ucl"
    if ucl_dir.exists():
         print("\n=======================================")
         print("           UCL Dataset")
         print("=======================================")
         # UCL dataset doesn't have the same sub-sequence structure
         print(f"\n--- Analyzing Dataset: {ucl_dir.relative_to(BASE_DATA_DIR)} ---")
         # 1. Analyze Intrinsics
         intrinsics_path = ucl_dir / "intrinsics.txt"
         intrinsics_analysis = analyze_intrinsics(intrinsics_path)
         print(f"  Intrinsics ({intrinsics_path.name}):")
         print(f"    Exists: {intrinsics_analysis['exists']}")
         if intrinsics_analysis["exists"]:
             print(f"    Lines: {intrinsics_analysis['lines']}")
             print(f"    Format OK (2 lines, 3 numbers/line): {intrinsics_analysis['format_ok']}")
         if intrinsics_analysis["error"]:
             print(f"    Error: {intrinsics_analysis['error']}")

         # 2. Find and Analyze Depth Directory/Files
         depth_dir = ucl_dir / "depth" # Expecting 'depth' for UCL raw data
         if depth_dir.exists():
             print(f"  Depth Directory: Found 'depth'")
             depth_files = sorted(list(depth_dir.glob("*.png")) + list(depth_dir.glob("*.npy"))) # Check both just in case
             print(f"  Depth Files: Found {len(depth_files)} (.png or .npy)")

             if not depth_files:
                 print("    No depth files to analyze.")
             else:
                 # Analyze sample depth maps
                 print(f"  Analyzing first {min(NUM_SAMPLES_TO_CHECK, len(depth_files))} depth maps:")
                 for i, depth_file in enumerate(depth_files):
                     if i >= NUM_SAMPLES_TO_CHECK:
                         break
                     depth_analysis = analyze_depth_map(depth_file)
                     print(f"    - Sample {i+1} ({depth_analysis['path']}):")
                     if depth_analysis["error"]:
                         print(f"      Error: {depth_analysis['error']}")
                     else:
                         print(f"      Dtype: {depth_analysis['dtype']}")
                         print(f"      Shape: {depth_analysis['shape']}")
                         print(f"      Min (Non-Zero): {depth_analysis['min_val']:.4f}" if depth_analysis['min_val'] is not None else "N/A")
                         print(f"      Max (Non-Zero): {depth_analysis['max_val']:.4f}" if depth_analysis['max_val'] is not None else "N/A")
                         print(f"      Mean (Non-Zero): {depth_analysis['mean_nz']:.4f}" if depth_analysis['mean_nz'] is not None else "N/A")
                         print(f"      Scale Guess: {depth_analysis['scale_guess']}")
         else:
             print("  Depth Directory: Not found ('depth')")
    else:
        print("\nWarning: UCL dataset directory not found.")

    print("\nAnalysis Complete.")
