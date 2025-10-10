"""
ASL Dataset Landmark Extractor
Processes ASL alphabet images and extracts hand landmarks using MediaPipe
Creates a CSV file with: letter, x1, y1, z1, ..., x21, y21, z21 (63 features)
"""

import cv2
import mediapipe as mp
import os
import csv
from pathlib import Path
import sys

print("=" * 70)
print("  ASL DATASET LANDMARK EXTRACTOR")
print("=" * 70)
print()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  # For images, not video
    max_num_hands=1,
    min_detection_confidence=0.5,  # Lower for varied images
    model_complexity=1  # Use fuller model for better accuracy on images
)

def extract_landmarks(image_path):
    """
    Extract hand landmarks from an image
    Returns: list of 63 floats [x1,y1,z1, x2,y2,z2, ..., x21,y21,z21] or None
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    # Convert to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = hands.process(rgb_image)
    
    # Check if hand detected
    if not results.multi_hand_landmarks:
        return None
    
    # Extract landmarks (use first hand if multiple detected)
    landmarks = results.multi_hand_landmarks[0]
    
    # Flatten to list of 63 values
    landmark_list = []
    for landmark in landmarks.landmark:
        landmark_list.extend([landmark.x, landmark.y, landmark.z])
    
    return landmark_list

def process_dataset(dataset_path, output_csv):
    """
    Process all images in dataset and create CSV
    
    Expected structure:
    dataset_path/
        A/
            image1.jpg
            image2.jpg
        B/
            image1.jpg
        ...
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ ERROR: Dataset path does not exist: {dataset_path}")
        print()
        print("Please ensure you have the ASL dataset downloaded.")
        print("Expected structure:")
        print("  data/asl_images/")
        print("    A/")
        print("    B/")
        print("    ...")
        return False
    
    # Find all letter folders
    letter_folders = [f for f in dataset_path.iterdir() if f.is_dir()]
    
    if not letter_folders:
        print(f"❌ ERROR: No letter folders found in {dataset_path}")
        return False
    
    # Filter out J and Z (require motion)
    valid_letters = set('ABCDEFGHIKLMNOPQRSTUVWXY')  # Exclude J and Z
    letter_folders = [f for f in letter_folders if f.name in valid_letters]
    letter_folders.sort()
    
    print(f"Found {len(letter_folders)} letter folders (excluding J and Z)")
    print(f"Letters: {', '.join([f.name for f in letter_folders])}")
    print()
    
    # Prepare CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    total_images = 0
    successful_extractions = 0
    failed_extractions = 0
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = ['letter']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(header)
        
        # Process each letter
        for letter_folder in letter_folders:
            letter = letter_folder.name
            print(f"Processing letter '{letter}'...", end=' ', flush=True)
            
            # Find all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(letter_folder.glob(ext))
            
            letter_success = 0
            letter_fail = 0
            
            for image_file in image_files:
                total_images += 1
                
                # Extract landmarks
                landmarks = extract_landmarks(image_file)
                
                if landmarks:
                    # Write to CSV
                    row = [letter] + landmarks
                    writer.writerow(row)
                    successful_extractions += 1
                    letter_success += 1
                else:
                    failed_extractions += 1
                    letter_fail += 1
            
            print(f"✓ {letter_success} succeeded, {letter_fail} failed")
    
    # Summary
    print()
    print("=" * 70)
    print("  EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"  Total images processed:    {total_images}")
    print(f"  Successful extractions:    {successful_extractions}")
    print(f"  Failed extractions:        {failed_extractions}")
    print(f"  Success rate:              {(successful_extractions/total_images*100):.1f}%")
    print()
    print(f"  Output saved to: {output_csv}")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    # Configuration
    DATASET_PATH = "data/asl_alphabet_train/asl_alphabet_train"  # Kaggle dataset location
    OUTPUT_CSV = "data/asl_landmarks.csv"
    
    print("Configuration:")
    print(f"  Dataset path: {DATASET_PATH}")
    print(f"  Output CSV:   {OUTPUT_CSV}")
    print()
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print("⚠️  Dataset not found!")
        print()
        print("To use this script:")
        print("1. Download ASL Alphabet dataset from Kaggle:")
        print("   https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        print()
        print("2. Extract to: data/asl_images/")
        print("   Structure should be:")
        print("     data/asl_images/A/*.jpg")
        print("     data/asl_images/B/*.jpg")
        print("     etc.")
        print()
        print("3. Run this script again")
        print()
        sys.exit(1)
    
    print("Starting landmark extraction...")
    print("This may take several minutes for large datasets.")
    print()
    
    success = process_dataset(DATASET_PATH, OUTPUT_CSV)
    
    if success:
        print()
        print("✅ Dataset preparation complete!")
        print()
        print("Next steps:")
        print("  1. Run data_preprocessing.py to split the data")
        print("  2. Run model_training.py to train the neural network")
    else:
        print()
        print("❌ Dataset preparation failed!")
        sys.exit(1)
    
    hands.close()
