# ASL Dataset Setup Instructions

## Option 1: Download from Kaggle (Recommended)

### Step 1: Install Kaggle CLI
```bash
pip install kaggle
```

### Step 2: Setup Kaggle API Key
1. Go to https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. This downloads `kaggle.json`
5. Place it in: `C:\Users\<YourUsername>\.kaggle\kaggle.json`

### Step 3: Download Dataset
```bash
# From project root directory
kaggle datasets download -d grassknoted/asl-alphabet
```

### Step 4: Extract Dataset
```bash
# Extract to data/asl_images/
unzip asl-alphabet.zip -d data/asl_images/
```

### Step 5: Verify Structure
Your directory should look like:
```
data/
  asl_images/
    asl_alphabet_train/
      A/
        *.jpg (3000 images)
      B/
        *.jpg (3000 images)
      ...
```

---

## Option 2: Manual Download

1. Visit: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
2. Click "Download" button (requires Kaggle account)
3. Extract the ZIP file
4. Move contents to `data/asl_images/`

---

## Option 3: Use Alternative Dataset

If you can't access Kaggle, we can:
1. Collect our own small dataset using the visualizer
2. Use a different ASL dataset
3. Use Sign Language MNIST (simpler but less images)

---

## Dataset Info

**ASL Alphabet Dataset:**
- **Size:** ~87,000 images
- **Classes:** 29 (A-Z + space + del + nothing)
- **We'll use:** 24 classes (A-Z excluding J, Z)
- **Image size:** 200x200 pixels
- **Format:** JPG
- **Split:** ~3000 images per letter

**Why exclude J and Z?**
- J and Z are motion-based letters in ASL
- Requires video/sequence data
- Static images don't capture the motion

---

## Quick Start Commands

### If you have the dataset ready:
```bash
# From project root
python python/dataset_preparation.py
```

### If dataset is in different location:
Edit `dataset_preparation.py` line 169:
```python
DATASET_PATH = "path/to/your/dataset"  # Change this
```

---

## What Happens Next?

1. **Landmark Extraction** (~10-30 min for full dataset)
   - Processes all images
   - Extracts 21 hand landmarks (63 features)
   - Saves to `data/asl_landmarks.csv`

2. **Data Preprocessing**
   - Splits into train/val/test
   - Prepares for neural network

3. **Model Training**
   - Builds neural network
   - Trains on landmarks
   - Saves trained model

---

## Troubleshooting

### "Dataset path does not exist"
- Check the path in `dataset_preparation.py`
- Ensure you extracted the ZIP correctly

### "No letter folders found"
- Dataset might be nested deeper (e.g., `asl_alphabet_train/`)
- Update DATASET_PATH to point to folder containing A/, B/, etc.

### Kaggle API not working
- Verify `kaggle.json` is in correct location
- Check file permissions
- Try manual download instead

---

## Ready to Start?

Once you have the dataset downloaded and extracted:
```bash
python python/dataset_preparation.py
```

This will create `data/asl_landmarks.csv` with all the hand landmarks! 🚀
