import os
import shutil
import sys

def delete_non_txt_files(directory):
    """Recursively delete all non-txt files in the given directory"""
    deleted_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.lower().endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
    
    return deleted_count

def merge_folders(source_dir, target_dir):
    """Recursively merge files from source directory to target directory"""
    merged_count = 0
    
    # Walk through all files and subdirectories in source_dir
    for root, dirs, files in os.walk(source_dir):
        # Calculate the corresponding path in target_dir
        rel_path = os.path.relpath(root, source_dir)
        target_path = os.path.join(target_dir, rel_path)
        
        # Create the target directory if it doesn't exist
        os.makedirs(target_path, exist_ok=True)
        
        # Copy each file from source to target
        for file in files:
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_path, file)
            
            # Check if target file already exists - skip if it does
            if os.path.exists(target_file):
                print(f"Skipping (already exists): {target_file}")
                continue
            
            try:
                shutil.copy2(source_file, target_file)
                merged_count += 1
                print(f"Merged: {target_file}")
            except Exception as e:
                print(f"Error copying {source_file} to {target_file}: {e}")
    
    return merged_count

def main():
    # Define paths to the exact locations
    nppad_dir = "/Users/hariomnabira/Desktop/sem_project_nppad/NuclearPowerPlantAccidentData/NPPAD"
    operation_csv_dir = "/Users/hariomnabira/Desktop/sem_project_nppad/NuclearPowerPlantAccidentData/Operation_csv_data"
    
    # Check if directories exist
    if not os.path.exists(nppad_dir):
        print(f"Error: NPPAD directory not found at {nppad_dir}")
        return
    
    if not os.path.exists(operation_csv_dir):
        print(f"Error: Operation_csv_data directory not found at {operation_csv_dir}")
        return
    
    # Step 1: Delete non-txt files in NPPAD folder
    print("Step 1: Deleting non-txt files in NPPAD folder...")
    deleted_count = delete_non_txt_files(nppad_dir)
    print(f"Deleted {deleted_count} non-txt files from {nppad_dir}")
    
    # Step 2: Merge Operation_csv_data into NPPAD
    print("\nStep 2: Merging Operation_csv_data files into NPPAD...")
    merged_count = merge_folders(operation_csv_dir, nppad_dir)
    print(f"Merged {merged_count} files from {operation_csv_dir} to {nppad_dir}")
    
    print("\nOperation completed successfully!")

if __name__ == "__main__":
    main() 