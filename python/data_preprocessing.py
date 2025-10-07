"""
Data Preprocessing for ASL Recognition
Loads landmark CSV and prepares train/validation/test splits
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("  ASL DATA PREPROCESSING")
print("=" * 70)
print()

# Load data
print("[1/5] Loading landmark data...")
df = pd.read_csv('data/asl_landmarks.csv')
print(f"✓ Loaded {len(df)} samples")
print(f"  Columns: {len(df.columns)} (1 label + 63 features)")
print()

# Check class distribution
print("[2/5] Analyzing class distribution...")
class_counts = df['letter'].value_counts().sort_index()
print(f"  Classes: {len(class_counts)}")
print(f"  Min samples: {class_counts.min()} ({class_counts.idxmin()})")
print(f"  Max samples: {class_counts.max()} ({class_counts.idxmax()})")
print(f"  Mean samples: {class_counts.mean():.0f}")
print()

# Visualize distribution
plt.figure(figsize=(12, 4))
class_counts.plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Letter')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('data/class_distribution.png')
print("  ✓ Saved class distribution plot to data/class_distribution.png")
print()

# Separate features and labels
print("[3/5] Separating features and labels...")
X = df.iloc[:, 1:].values  # All columns except first (letter)
y = df.iloc[:, 0].values   # First column (letter)

print(f"  Features shape: {X.shape}")
print(f"  Labels shape: {y.shape}")
print(f"  Feature range: [{X.min():.3f}, {X.max():.3f}]")
print()

# Encode labels
print("[4/5] Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"  Classes: {label_encoder.classes_}")
print(f"  Encoded range: {y_encoded.min()} to {y_encoded.max()}")
print()

# Save label encoder mapping
label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
print("  Label mapping:")
for idx, letter in label_mapping.items():
    print(f"    {idx:2d} -> {letter}")
print()

# Save label encoder
import pickle
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("  ✓ Saved label encoder to models/label_encoder.pkl")
print()

# Split data
print("[5/5] Splitting data...")
# First split: 85% train+val, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
)

# Second split: 70% train, 15% val (from the 85%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
)

print(f"  Training set:   {len(X_train):5d} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Validation set: {len(X_val):5d} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Test set:       {len(X_test):5d} samples ({len(X_test)/len(X)*100:.1f}%)")
print()

# Verify stratification
print("  Verifying class balance in splits:")
print(f"    Train classes:      {np.unique(y_train).shape[0]}")
print(f"    Validation classes: {np.unique(y_val).shape[0]}")
print(f"    Test classes:       {np.unique(y_test).shape[0]}")
print()

# Save splits
print("Saving processed data...")
np.savez('data/train_data.npz', X=X_train, y=y_train)
np.savez('data/val_data.npz', X=X_val, y=y_val)
np.savez('data/test_data.npz', X=X_test, y=y_test)
print("  ✓ Saved train_data.npz")
print("  ✓ Saved val_data.npz")
print("  ✓ Saved test_data.npz")
print()

# Summary statistics
print("=" * 70)
print("  PREPROCESSING COMPLETE")
print("=" * 70)
print(f"  Total samples:      {len(X)}")
print(f"  Features per sample: 63 (21 landmarks × 3 coordinates)")
print(f"  Classes:            {len(np.unique(y_encoded))}")
print()
print("  Data is ready for training!")
print("=" * 70)
print()

print("Next step:")
print("  Run: python python/model_training.py")
print()
