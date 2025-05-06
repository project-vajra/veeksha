import os
import shutil
from pathlib import Path

# --- Configuration ---
SOURCE_DIR = Path("visualizations")
TARGET_DIR = SOURCE_DIR / "aggregate_plots"
FILENAME_TO_FIND = "aggregate_hit_rates.png"
# --- End Configuration ---

def aggregate_plots():
    """
    Finds specific PNG files, renames them based on their parent directory,
    and copies them to a target aggregate directory.
    """
    print(f"Source directory: {SOURCE_DIR.resolve()}")
    print(f"Target directory: {TARGET_DIR.resolve()}")

    # Create the target directory if it doesn't exist
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ensured target directory exists: {TARGET_DIR}")

    copied_count = 0
    skipped_count = 0

    # Recursively find all files matching the filename within the source directory
    # Use rglob for recursive search
    for filepath in SOURCE_DIR.rglob(FILENAME_TO_FIND):
        # Skip files that are already in the target directory
        if filepath.parent == TARGET_DIR:
            # print(f"Skipping file already in target directory: {filepath}")
            continue # Use continue instead of return inside loop

        # Get the parent directory name
        parent_dir_name = filepath.parent.name

        # Construct the new filename
        new_filename = f"{filepath.stem}_{parent_dir_name}{filepath.suffix}"

        # Construct the full destination path
        destination_path = TARGET_DIR / new_filename

        # Copy the file, preserving metadata (overwrites if exists)
        print(f"Copying '{filepath}' to '{destination_path}'")
        try:
            shutil.copy2(filepath, destination_path)
            copied_count += 1
        except Exception as e:
            print(f"Error copying '{filepath}': {e}")
            skipped_count += 1 # Count errors as skipped

    print("\n--- Summary ---")
    print(f"Copying process completed.")
    print(f"Files copied: {copied_count}")
    print(f"Files skipped (due to error): {skipped_count}")

if __name__ == "__main__":
    aggregate_plots()
