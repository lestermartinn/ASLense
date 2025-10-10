# Phase 2 Quick Reference

## One-Line Summary
Train a neural network to recognize 24 ASL letters (A-Z excluding J,Z) from hand landmark data.

---

## Prerequisites
- ✅ Phase 1 complete (hand tracking working)
- ⏳ Download ASL dataset (~3.5 GB, 87,000 images)
- ⏳ Install ML libraries

---

## Installation (5 minutes)

```powershell
# Activate virtual environment
.\.venv\Scripts\activate

# Install ML packages
pip install scikit-learn matplotlib seaborn pandas

# Install Kaggle CLI (for dataset download)
pip install kaggle
```

---

## Dataset Download (10 minutes)

**Option 1: Kaggle CLI (Recommended)**
```powershell
# Setup API key first
# 1. Go to https://www.kaggle.com/settings/account
# 2. Click "Create New Token" (downloads kaggle.json)
# 3. Place in C:\Users\<YourUsername>\.kaggle\kaggle.json

# Download dataset
kaggle datasets download -d grassknoted/asl-alphabet

# Extract
Expand-Archive asl-alphabet.zip -DestinationPath data\
```

**Option 2: Manual Download**
1. Visit: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
2. Click "Download" button
3. Extract to `data/asl_images/`

---

## Execution Pipeline (2-3 hours)

### Step 1: Extract Landmarks (10-30 min)
```powershell
python python/dataset_preparation.py
```
**Output**: `data/asl_landmarks.csv`

---

### Step 2: Preprocess Data (1-2 min)
```powershell
python python/data_preprocessing.py
```
**Outputs**: 
- `data/train_data.npz`, `val_data.npz`, `test_data.npz`
- `models/label_encoder.pkl`

---

### Step 3: Train Model (1-2 hours)
```powershell
python python/model_training.py
```
**Outputs**: 
- `models/asl_model.keras`
- `models/best_model.keras`

**Target**: >90% accuracy

---

### Step 4: Evaluate Model (2-3 min)
```powershell
python python/model_evaluation.py
```
**Outputs**: 
- Confusion matrix
- Per-class accuracy
- Performance report

---

### Step 5: Export to TFLite (3-5 min)
```powershell
python python/model_export.py
```
**Outputs**: 
- `models/asl_model.tflite`
- `models/asl_model_quantized.tflite`

---

## Success Criteria

| Metric | Target | What it means |
|--------|--------|---------------|
| Test Accuracy | >90% | Overall correct predictions |
| Per-class F1 | >85% | Good performance on each letter |
| Inference Time | <100ms | Fast enough for real-time |
| Model Size | <100 KB | Small enough for deployment |

---

## Troubleshooting

**Problem**: Low accuracy (<85%)
- **Solution**: Check dataset quality, increase epochs, adjust architecture

**Problem**: Training too slow
- **Solution**: Reduce batch size, use fewer epochs, try on GPU

**Problem**: Model overfits (train >> validation accuracy)
- **Solution**: Increase dropout, add regularization, get more data

**Problem**: Import errors
- **Solution**: Ensure virtual environment is activated

---

## Expected Timeline

| Step | Time | Can Skip? |
|------|------|-----------|
| Install packages | 5 min | No |
| Download dataset | 10 min | No |
| Extract landmarks | 10-30 min | No |
| Preprocess data | 1-2 min | No |
| Train model | 1-2 hours | No |
| Evaluate model | 2-3 min | Yes (but recommended) |
| Export TFLite | 3-5 min | No |
| **TOTAL** | **2-3 hours** | |

---

## Files Created

### Scripts (Already Created)
- `python/dataset_preparation.py`
- `python/data_preprocessing.py`
- `python/model_training.py`
- `python/model_evaluation.py`
- `python/model_export.py`

### Data (You Will Generate)
- `data/asl_landmarks.csv` (~50 MB)
- `data/*.npz` (train/val/test splits)
- `models/*.keras` (trained models)
- `models/*.tflite` (deployment models)
- `models/*.png` (visualizations)

---

## Next Phase

**Phase 3: C++ TensorFlow Lite Integration**
- Load `.tflite` model in C++
- Real-time prediction from camera
- Display recognized letters

**Phase 4: Interactive Learning Mode**
- Practice specific letters
- Get feedback (correct/incorrect)
- Track progress

---

## Key Concepts Learned

1. **Feature Extraction**: MediaPipe landmarks as ML features
2. **Data Pipeline**: Raw images → landmarks → train/val/test splits
3. **Neural Networks**: Feedforward architecture with dropout
4. **Training**: Batch processing, early stopping, checkpointing
5. **Evaluation**: Confusion matrix, per-class metrics
6. **Deployment**: TensorFlow Lite conversion and quantization

---

## Quick Commands (Copy-Paste)

```powershell
# Full pipeline (run after dataset download)
.\.venv\Scripts\activate
python python/dataset_preparation.py
python python/data_preprocessing.py
python python/model_training.py
python python/model_evaluation.py
python python/model_export.py
```

---

**Status**: Ready to execute! Download the dataset and run the scripts. 🚀
