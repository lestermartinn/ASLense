# Phase 2: Machine Learning Model Training
**Goal:** Train a neural network to recognize ASL alphabet letters from hand landmarks

---

## Overview

We'll train a **feedforward neural network** that takes the 63-element landmark vector and predicts which ASL letter (A-Z, excluding J and Z which require motion).

### Architecture
```
Input Layer (63 features)
    ↓
Dense Layer (128 neurons, ReLU)
    ↓
Dropout (30%)
    ↓
Dense Layer (64 neurons, ReLU)
    ↓
Dropout (30%)
    ↓
Output Layer (24 classes, Softmax)
```

---

## Step-by-Step Plan

### Step 1: Dataset Preparation
**Goal:** Get ASL alphabet images and extract landmarks

**Tasks:**
1. Download ASL alphabet dataset from Kaggle
2. Create Python script to batch-process images
3. Extract 63-element landmark vectors for each image
4. Save as CSV: `[letter, x1, y1, z1, ..., x21, y21, z21]`

**Dataset Options:**
- Kaggle ASL Alphabet Dataset (~87,000 images)
- Or collect our own using the visualizer

**Output:** `data/asl_landmarks.csv`

---

### Step 2: Data Preprocessing
**Goal:** Prepare data for training

**Tasks:**
1. Load CSV file
2. Split into training (70%), validation (15%), test (15%)
3. Normalize features (0-1 range, already normalized from MediaPipe)
4. One-hot encode labels (24 classes)
5. Check for class imbalance

**Output:** 
- `X_train, y_train` - Training data
- `X_val, y_val` - Validation data  
- `X_test, y_test` - Test data

---

### Step 3: Model Architecture
**Goal:** Build neural network in TensorFlow/Keras

**Code Structure:**
```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(24, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

### Step 4: Training
**Goal:** Train the model

**Hyperparameters:**
- Batch size: 32
- Epochs: 50-100 (with early stopping)
- Learning rate: 0.001 (default Adam)
- Loss: Categorical crossentropy
- Metrics: Accuracy

**Techniques:**
- Early stopping (patience=10)
- Model checkpointing (save best model)
- Learning rate reduction on plateau

**Target Performance:**
- Training accuracy: >95%
- Validation accuracy: >90%
- Test accuracy: >90%

**Output:** `models/asl_model.h5` or `models/asl_model.keras`

---

### Step 5: Evaluation
**Goal:** Test model performance

**Tasks:**
1. Evaluate on test set
2. Generate confusion matrix
3. Calculate per-class accuracy
4. Identify problematic letter pairs (e.g., M vs N)
5. Plot training curves (loss, accuracy)

**Metrics:**
- Overall accuracy
- Per-class precision/recall/F1
- Confusion matrix

---

### Step 6: Export to TensorFlow Lite
**Goal:** Prepare model for C++ integration

**Tasks:**
1. Convert Keras model to TFLite format
2. Quantize (optional - float16 or int8)
3. Test TFLite model accuracy vs original
4. Measure inference time

**Output:** `models/asl_model.tflite`

---

## File Structure (Phase 2)

```
python/
  ├── dataset_preparation.py      # Step 1: Extract landmarks from images
  ├── data_preprocessing.py       # Step 2: Load and split data
  ├── model_training.py           # Step 3-4: Build and train model
  ├── model_evaluation.py         # Step 5: Evaluate performance
  └── model_export.py             # Step 6: Convert to TFLite

data/
  ├── asl_images/                 # Raw dataset images
  │   ├── A/
  │   ├── B/
  │   └── ...
  ├── asl_landmarks.csv           # Extracted landmark data
  ├── train.csv                   # Training split
  ├── val.csv                     # Validation split
  └── test.csv                    # Test split

models/
  ├── asl_model.h5               # Trained Keras model
  ├── asl_model.tflite           # TFLite model for C++
  └── training_history.png       # Training curves

notebooks/                        # Optional: Jupyter notebooks for exploration
  └── model_exploration.ipynb
```

---

## Technologies You'll Learn

### TensorFlow/Keras
- ✅ Sequential API for building neural networks
- ✅ Dense (fully connected) layers
- ✅ Activation functions (ReLU, Softmax)
- ✅ Dropout for regularization
- ✅ Loss functions (categorical crossentropy)
- ✅ Optimizers (Adam)
- ✅ Callbacks (EarlyStopping, ModelCheckpoint)

### Data Science
- ✅ Train/validation/test splits
- ✅ One-hot encoding
- ✅ Normalization
- ✅ Confusion matrices
- ✅ Evaluation metrics (precision, recall, F1)

### Model Deployment
- ✅ Model serialization (H5, SavedModel)
- ✅ TensorFlow Lite conversion
- ✅ Model quantization
- ✅ Inference optimization

---

## Expected Challenges & Solutions

### Challenge 1: Dataset Quality
**Problem:** Some images might not have visible hands
**Solution:** Filter out images where MediaPipe can't detect hands

### Challenge 2: Similar Letters
**Problem:** M, N, S might look similar
**Solution:** 
- Augment training data
- Add more features (hand orientation, finger angles)
- Use ensemble methods

### Challenge 3: Overfitting
**Problem:** Model memorizes training data
**Solution:**
- Use dropout layers (30%)
- Data augmentation (slight rotations, scaling)
- Early stopping

### Challenge 4: Class Imbalance
**Problem:** Some letters have more examples than others
**Solution:**
- Balance dataset (oversample or undersample)
- Use class weights during training

---

## Success Criteria

### Minimum Viable Product (MVP)
- ✅ Model trains successfully
- ✅ >80% accuracy on test set
- ✅ Successfully converts to TFLite
- ✅ Inference time <100ms

### Stretch Goals
- ✅ >95% accuracy on test set
- ✅ Real-time inference (<50ms)
- ✅ Works well with live camera feed
- ✅ Handles slight hand rotations

---

## Timeline Estimate

| Step | Estimated Time |
|------|----------------|
| Dataset download & setup | 30 min |
| Landmark extraction | 1-2 hours |
| Data preprocessing | 30 min |
| Model building | 30 min |
| Training & tuning | 1-2 hours |
| Evaluation & analysis | 30 min |
| TFLite conversion | 30 min |
| **Total** | **5-7 hours** |

---

## Next Immediate Steps

1. **Download ASL dataset** from Kaggle
2. **Create landmark extraction script**
3. **Process all images** → CSV file
4. **Build & train model**

---

## Resources

### Datasets
- Kaggle ASL Alphabet: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- Alternative: Sign Language MNIST

### TensorFlow Resources
- Keras Sequential API: https://keras.io/guides/sequential_model/
- TFLite Converter: https://www.tensorflow.org/lite/convert

### MediaPipe
- Hand Landmark Model: https://google.github.io/mediapipe/solutions/hands.html

---

**Ready to start with Step 1: Dataset Preparation?** 🚀

We'll begin by either:
- A) Downloading an existing ASL dataset from Kaggle
- B) Creating our own small dataset using the visualizer you just tested

Which approach would you prefer? (A is faster, B is more custom)
