"""
Model Training for ASL Recognition
Builds and trains neural network on landmark data
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pickle
import time

print("=" * 70)
print("  ASL MODEL TRAINING")
print("=" * 70)
print()

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load data
print("[1/6] Loading preprocessed data...")
train_data = np.load('data/train_data.npz')
val_data = np.load('data/val_data.npz')
test_data = np.load('data/test_data.npz')

X_train, y_train = train_data['X'], train_data['y']
X_val, y_val = val_data['X'], val_data['y']
X_test, y_test = test_data['X'], test_data['y']

num_classes = len(np.unique(y_train))

print(f"  Training:   {X_train.shape[0]} samples")
print(f"  Validation: {X_val.shape[0]} samples")
print(f"  Test:       {X_test.shape[0]} samples")
print(f"  Features:   {X_train.shape[1]}")
print(f"  Classes:    {num_classes}")
print()

# Convert labels to categorical (one-hot encoding)
print("[2/6] Converting labels to one-hot encoding...")
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_val_cat = keras.utils.to_categorical(y_val, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)
print(f"  Train labels shape: {y_train_cat.shape}")
print()

# Build model
print("[3/6] Building neural network...")
model = keras.Sequential([
    # Input layer
    layers.Input(shape=(63,)),
    
    # First hidden layer
    layers.Dense(128, activation='relu', name='dense_1'),
    layers.Dropout(0.3, name='dropout_1'),
    
    # Second hidden layer
    layers.Dense(64, activation='relu', name='dense_2'),
    layers.Dropout(0.3, name='dropout_2'),
    
    # Output layer
    layers.Dense(num_classes, activation='softmax', name='output')
], name='ASL_Recognition_Model')

model.summary()
print()

# Compile model
print("[4/6] Compiling model...")
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("  ✓ Optimizer: Adam")
print("  ✓ Loss: Categorical Crossentropy")
print("  ✓ Metrics: Accuracy")
print()

# Setup callbacks
print("[5/6] Setting up training callbacks...")
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'models/best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]
print("  ✓ Early stopping (patience=10)")
print("  ✓ Model checkpoint (best_model.keras)")
print("  ✓ Learning rate reduction (factor=0.5, patience=5)")
print()

# Train model
print("[6/6] Training model...")
print("=" * 70)
start_time = time.time()

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time
print("=" * 70)
print(f"  Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
print()

# Evaluate on test set
print("Evaluating on test set...")
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"  Test Loss:     {test_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy*100:.2f}%")
print()

# Save final model
print("Saving model...")
model.save('models/asl_model.keras')
print("  ✓ Saved asl_model.keras")
print()

# Plot training history
print("Plotting training history...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss plot
ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/training_history.png', dpi=150)
print("  ✓ Saved training_history.png")
print()

# Summary
print("=" * 70)
print("  TRAINING COMPLETE")
print("=" * 70)
print(f"  Final validation accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
print(f"  Best validation accuracy:  {max(history.history['val_accuracy'])*100:.2f}%")
print(f"  Test accuracy:             {test_accuracy*100:.2f}%")
print(f"  Training time:             {training_time/60:.1f} minutes")
print(f"  Epochs trained:            {len(history.history['loss'])}")
print()

if test_accuracy >= 0.90:
    print("  ✅ EXCELLENT! Model achieved target accuracy (>90%)")
elif test_accuracy >= 0.85:
    print("  ✓ GOOD! Model accuracy is acceptable (>85%)")
else:
    print("  ⚠️  Model accuracy is below target (<85%)")
    print("     Consider:")
    print("       - Collecting more/better data")
    print("       - Adjusting model architecture")
    print("       - Tuning hyperparameters")

print()
print("=" * 70)
print()

print("Next step:")
print("  Run: python python/model_evaluation.py")
print()
