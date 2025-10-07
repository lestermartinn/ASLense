# Phase 2: Machine Learning Training - COMPLETE ✅

## Overview
Phase 2 focuses on training a neural network to recognize ASL letters from hand landmark data. All infrastructure and scripts are now ready for execution.

---

## What's Been Created

### 📄 Documentation
1. **PHASE2_PLAN.md** - Complete roadmap with:
   - Neural network architecture
   - Training pipeline steps
   - Expected timeline (5-7 hours)
   - Technologies to learn

2. **DATASET_SETUP.md** - Instructions for:
   - Downloading Kaggle ASL Alphabet dataset
   - Three download methods (CLI, manual, alternative)
   - Dataset structure and format

3. **This file** - Status and execution guide

### 🐍 Python Scripts (Ready to Run)

#### 1. `dataset_preparation.py`
**Purpose**: Extract hand landmarks from images  
**Input**: `data/asl_images/asl_alphabet_train/{A,B,C,...}/*.jpg`  
**Output**: `data/asl_landmarks.csv`  
**What it does**:
- Uses MediaPipe to detect hands in static images
- Extracts 21 landmarks × 3 coordinates = 63 features per image
- Creates CSV with format: `[letter, x0, y0, z0, ..., x20, y20, z20]`
- Reports success/failure statistics per letter

#### 2. `data_preprocessing.py`
**Purpose**: Prepare data for training  
**Input**: `data/asl_landmarks.csv`  
**Output**: 
- `data/train_data.npz` (70%)
- `data/val_data.npz` (15%)
- `data/test_data.npz` (15%)
- `models/label_encoder.pkl`
- `data/class_distribution.png`

**What it does**:
- Loads landmark CSV
- Analyzes class distribution (ensures balance)
- Encodes labels (A→0, B→1, etc.)
- Splits data with stratification
- Visualizes class distribution
- Saves processed data for training

#### 3. `model_training.py`
**Purpose**: Build and train neural network  
**Input**: `data/train_data.npz`, `data/val_data.npz`, `data/test_data.npz`  
**Output**: 
- `models/asl_model.keras`
- `models/best_model.keras` (best validation accuracy)
- `models/training_history.png`

**Architecture**:
```
Input(63) 
    ↓
Dense(128, ReLU) → Dropout(0.3)
    ↓
Dense(64, ReLU) → Dropout(0.3)
    ↓
Output(24, Softmax)
```

**Training Details**:
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Batch size: 32
- Max epochs: 100
- Callbacks: Early stopping, model checkpoint, learning rate reduction
- Target: >90% accuracy

#### 4. `model_evaluation.py`
**Purpose**: Comprehensive model evaluation  
**Input**: `models/best_model.keras`, `data/test_data.npz`  
**Output**:
- `models/confusion_matrix.png`
- `models/per_class_accuracy.png`
- `models/confidence_distribution.png`
- Detailed performance report

**What it analyzes**:
- Overall test accuracy
- Per-class precision, recall, F1-score
- Confusion matrix (identifies similar letters)
- Best/worst performing letters
- Prediction confidence distribution
- Inference speed (target: <100ms)

#### 5. `model_export.py`
**Purpose**: Convert to TensorFlow Lite for deployment  
**Input**: `models/best_model.keras`, `data/test_data.npz`  
**Output**:
- `models/asl_model.tflite` (float32)
- `models/asl_model_quantized.tflite` (int8)
- Comparison report

**What it does**:
- Converts Keras model to TFLite
- Creates quantized version (smaller, faster)
- Validates accuracy preservation
- Compares inference speeds
- Recommends best model for deployment

---

## Execution Workflow

### Step 1: Install ML Dependencies ⏳
```bash
# Activate virtual environment
.\.venv\Scripts\activate

# Install new packages
pip install scikit-learn matplotlib seaborn pandas
```

**Expected time**: 2-3 minutes

---

### Step 2: Download Dataset ⏳
```bash
# Option A: Kaggle CLI (recommended)
pip install kaggle
# Set up API key in C:\Users\<Username>\.kaggle\kaggle.json
kaggle datasets download -d grassknoted/asl-alphabet
unzip asl-alphabet.zip -d data/

# Option B: Manual download from Kaggle website
# See DATASET_SETUP.md for details
```

**Expected time**: 5-10 minutes (depends on internet speed)  
**Dataset size**: ~3.5 GB (87,000 images)

**Verify structure**:
```
data/asl_images/asl_alphabet_train/
    A/
        A1.jpg, A2.jpg, ...
    B/
        B1.jpg, B2.jpg, ...
    ...
    Z/
```

---

### Step 3: Extract Landmarks ⏳
```bash
python python/dataset_preparation.py
```

**Expected time**: 10-30 minutes  
**What to watch for**:
- Processing progress per letter
- Success rate per letter (should be >80%)
- Total landmarks extracted

**Output**: `data/asl_landmarks.csv` (~87,000 rows × 64 columns)

---

### Step 4: Preprocess Data ⏳
```bash
python python/data_preprocessing.py
```

**Expected time**: 1-2 minutes  
**What to watch for**:
- Class distribution balance
- Train/val/test split sizes (70%/15%/15%)
- All 24 classes present in each split

**Outputs**:
- `data/train_data.npz`
- `data/val_data.npz`
- `data/test_data.npz`
- `models/label_encoder.pkl`
- `data/class_distribution.png`

---

### Step 5: Train Model ⏳
```bash
python python/model_training.py
```

**Expected time**: 1-2 hours (GPU recommended but not required)  
**What to watch for**:
- Training accuracy increasing
- Validation accuracy increasing and stabilizing
- Early stopping if validation stops improving
- Best epoch saved

**Outputs**:
- `models/asl_model.keras`
- `models/best_model.keras`
- `models/training_history.png`

**Success criteria**: Test accuracy >90%

---

### Step 6: Evaluate Model ⏳
```bash
python python/model_evaluation.py
```

**Expected time**: 2-3 minutes  
**What to watch for**:
- Overall test accuracy
- Per-class F1-scores
- Common misclassifications (similar letters like M/N/S)
- Inference speed (<100ms target)

**Outputs**:
- `models/confusion_matrix.png`
- `models/per_class_accuracy.png`
- `models/confidence_distribution.png`
- Detailed performance report in terminal

---

### Step 7: Export to TFLite ⏳
```bash
python python/model_export.py
```

**Expected time**: 3-5 minutes  
**What to watch for**:
- Accuracy preservation (should be within 1-2%)
- Inference speedup from quantization
- Model size reduction

**Outputs**:
- `models/asl_model.tflite` (float32, ~50 KB)
- `models/asl_model_quantized.tflite` (int8, ~15 KB)
- Comparison report with recommendation

---

## Expected Results

### Model Performance Targets
- ✅ **Test Accuracy**: >90%
- ✅ **Per-class F1**: >85% for most letters
- ✅ **Inference Time**: <100ms (ideally <50ms)
- ✅ **Model Size**: <100 KB

### Known Challenges
1. **Similar Letters**: M, N, S may confuse (similar hand shapes)
2. **Dataset Quality**: Some images may have poor hand visibility
3. **Class Imbalance**: Should be minimal with this dataset
4. **Overfitting**: Dropout layers help prevent this

### Troubleshooting

**If accuracy is low (<85%)**:
- Check dataset quality (are hands visible?)
- Verify landmark extraction success rate (should be >80%)
- Consider data augmentation
- Try increasing model capacity (more neurons/layers)
- Train for more epochs

**If training is too slow**:
- Reduce batch size (try 16 instead of 32)
- Use fewer epochs (try 50 instead of 100)
- Reduce dataset size for testing (use subset)

**If model overfits** (train acc >> val acc):
- Increase dropout (try 0.4 or 0.5)
- Add more data augmentation
- Reduce model capacity
- Add L2 regularization

---

## Files Generated by Phase 2

### Data Files
```
data/
    asl_images/                     # Downloaded Kaggle dataset
    asl_landmarks.csv               # Extracted landmarks
    train_data.npz                  # Training set
    val_data.npz                    # Validation set
    test_data.npz                   # Test set
    class_distribution.png          # Visualization
```

### Model Files
```
models/
    label_encoder.pkl               # Letter encoding (A→0, B→1, ...)
    asl_model.keras                 # Final Keras model
    best_model.keras                # Best validation model
    asl_model.tflite               # TFLite model (float32)
    asl_model_quantized.tflite     # TFLite model (quantized)
    training_history.png            # Training curves
    confusion_matrix.png            # Classification errors
    per_class_accuracy.png          # Per-letter performance
    confidence_distribution.png     # Prediction confidence
```

---

## Next Steps: Phase 3

Once Phase 2 is complete with >90% accuracy:

### Phase 3: C++ TensorFlow Lite Integration
1. **Install TFLite C++**: Download prebuilt library or build from source
2. **Create `inference_engine.cpp`**: C++ class to load .tflite model
3. **Integrate with `LandmarkService`**: Feed landmarks to model
4. **Real-time prediction**: Display predicted letter on screen
5. **Performance optimization**: Ensure <100ms end-to-end latency

### Phase 4: Interactive Learning Mode
1. **UI Development**: Show target letter to practice
2. **Feedback System**: Green/red/yellow based on prediction
3. **Statistics Tracking**: Accuracy, attempts, progress
4. **Reference Images**: Show correct hand positions

---

## Dependencies

### Already Installed (Phase 1)
- opencv-python 4.11.0
- mediapipe 0.10.21
- numpy

### Need to Install (Phase 2)
```bash
pip install scikit-learn matplotlib seaborn pandas
```

**Versions**:
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- pandas >= 2.0.0

---

## Current Status

✅ **Phase 0**: Environment setup complete  
✅ **Phase 1**: Hand tracking working (29 FPS, 64% detection rate)  
✅ **Phase 2 Infrastructure**: All scripts ready  
⏳ **Phase 2 Execution**: Waiting to download dataset  

---

## Quick Start Commands

```bash
# 1. Activate environment
.\.venv\Scripts\activate

# 2. Install ML packages
pip install scikit-learn matplotlib seaborn pandas

# 3. Download dataset (choose one method from DATASET_SETUP.md)
kaggle datasets download -d grassknoted/asl-alphabet

# 4. Run Phase 2 pipeline
python python/dataset_preparation.py      # 10-30 min
python python/data_preprocessing.py       # 1-2 min
python python/model_training.py           # 1-2 hours
python python/model_evaluation.py         # 2-3 min
python python/model_export.py             # 3-5 min
```

**Total estimated time**: 2-3 hours (mostly training)

---

## Learning Outcomes

By completing Phase 2, you will learn:

1. **Dataset Preparation**: Extract features from images using MediaPipe
2. **Data Preprocessing**: Train/val/test splits, encoding, stratification
3. **Neural Networks**: Build feedforward network with TensorFlow/Keras
4. **Training**: Callbacks, early stopping, learning rate scheduling
5. **Evaluation**: Confusion matrix, per-class metrics, confidence analysis
6. **Model Export**: TensorFlow Lite conversion, quantization
7. **Performance**: Balance accuracy, speed, and model size

---

## Support

If you encounter issues:
1. Check error messages carefully
2. Verify file paths exist
3. Ensure virtual environment is activated
4. Check dataset structure matches expected format
5. Review PHASE2_PLAN.md for detailed explanations

---

**Ready to start Phase 2! 🚀**

Download the dataset and run the scripts in order. Each script provides clear output showing progress and success criteria.
