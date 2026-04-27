import os
import zipfile
from tqdm import tqdm

def unzip_all_recursive(source_root, extract_to_root):
    """
    Recursively finds all zip files in source_root and extracts them 
    into a mirrored structure in extract_to_root.
    """
    # 1. Collect all zip files first for the progress bar
    all_zips = []
    for root, dirs, files in os.walk(source_root):
        for file in files:
            if file.endswith('.zip'):
                all_zips.append(os.path.join(root, file))
    
    print(f"Found {len(all_zips)} zip files across all scene folders.")

    # 2. Extract each file
    for zip_path in tqdm(all_zips, desc="Extracting Scenes"):
        try:
            # Determine the relative path to maintain folder structure if desired
            # Or just extract everything to the target_folder
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # We extract directly to target_folder
                zip_ref.extractall(extract_to_root)
        except zipfile.BadZipFile:
            print(f"Skipping {zip_path}: Not a valid zip file.")
        except Exception as e:
            print(f"Error extracting {zip_path}: {e}")

if __name__ == "__main__":
    # Updated paths based on your current DolphinFS SSD location
    # source_folder: where the Hugging Face script saves the files
    source_folder = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangquan/code/lhxk/workspace/streamIGGT/data/processed_infinigen"
    
    # target_folder: where you want the actual dataset files to live
    target_folder = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangquan/code/lhxk/workspace/streamIGGT/data/infinigen_extracted"
    
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Created directory: {target_folder}")

    unzip_all_recursive(source_folder, target_folder)
    print(f"\nExtraction complete!")
    print(f"Total available space on this drive: [run 'df -h {target_folder}' to check]")