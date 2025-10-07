"""
Model Evaluation for ASL Recognition
Comprehensive evaluation of trained model with visualizations
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pickle

print("=" * 70)
print("  ASL MODEL EVALUATION")
print("=" * 70)
print()

# Load model
print("[1/5] Loading trained model...")
model = keras.models.load_model('models/best_model.keras')
print("  ✓ Loaded best_model.keras")
print()

# Load data
print("[2/5] Loading test data...")
test_data = np.load('data/test_data.npz')
X_test, y_test = test_data['X'], test_data['y']

# Load label encoder
with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

class_names = label_encoder.classes_
num_classes = len(class_names)

print(f"  Test samples: {len(X_test)}")
print(f"  Classes: {num_classes}")
print()

# Make predictions
print("[3/5] Making predictions...")
y_test_cat = keras.utils.to_categorical(y_test, num_classes)
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Calculate overall metrics
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"  Test Loss:     {test_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy*100:.2f}%")
print()

# Confusion matrix
print("[4/5] Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150)
print("  ✓ Saved confusion_matrix.png")
print()

# Per-class metrics
print("[5/5] Calculating per-class metrics...")
print()
print("=" * 70)
print("  PER-CLASS PERFORMANCE")
print("=" * 70)

report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

# Display as table
print()
print("Letter | Precision | Recall | F1-Score | Support")
print("-------+-----------+--------+----------+--------")
for letter in class_names:
    metrics = report[letter]
    print(f"   {letter}   |   {metrics['precision']:.3f}   | {metrics['recall']:.3f} |  {metrics['f1-score']:.3f}   |  {int(metrics['support']):4d}")

print("-------+-----------+--------+----------+--------")
print(f" Macro |   {report['macro avg']['precision']:.3f}   | {report['macro avg']['recall']:.3f} |  {report['macro avg']['f1-score']:.3f}   |  {int(report['macro avg']['support']):4d}")
print()

# Find best and worst performing classes
f1_scores = {letter: report[letter]['f1-score'] for letter in class_names}
best_classes = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)[:5]
worst_classes = sorted(f1_scores.items(), key=lambda x: x[1])[:5]

print("Best performing letters:")
for letter, score in best_classes:
    print(f"  {letter}: {score*100:.1f}%")
print()

print("Worst performing letters:")
for letter, score in worst_classes:
    print(f"  {letter}: {score*100:.1f}%")
print()

# Analyze common misclassifications
print("Most common misclassifications:")
misclassified = []
for i in range(num_classes):
    for j in range(num_classes):
        if i != j and cm[i][j] > 0:
            misclassified.append((class_names[i], class_names[j], cm[i][j]))

misclassified.sort(key=lambda x: x[2], reverse=True)
for true_label, pred_label, count in misclassified[:10]:
    print(f"  {true_label} → {pred_label}: {count} times")
print()

# Per-class accuracy visualization
print("Creating per-class accuracy visualization...")
class_accuracies = []
for i, letter in enumerate(class_names):
    correct = cm[i][i]
    total = cm[i].sum()
    accuracy = (correct / total * 100) if total > 0 else 0
    class_accuracies.append(accuracy)

plt.figure(figsize=(14, 5))
bars = plt.bar(class_names, class_accuracies)

# Color bars based on accuracy
for i, bar in enumerate(bars):
    if class_accuracies[i] >= 95:
        bar.set_color('green')
    elif class_accuracies[i] >= 85:
        bar.set_color('orange')
    else:
        bar.set_color('red')

plt.axhline(y=90, color='r', linestyle='--', label='Target (90%)')
plt.title('Per-Class Accuracy')
plt.xlabel('Letter')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 105)
plt.legend()
plt.tight_layout()
plt.savefig('models/per_class_accuracy.png', dpi=150)
print("  ✓ Saved per_class_accuracy.png")
print()

# Prediction confidence analysis
print("Analyzing prediction confidence...")
confidences = np.max(y_pred_probs, axis=1)
correct_mask = (y_pred == y_test)

correct_confidences = confidences[correct_mask]
incorrect_confidences = confidences[~correct_mask]

print(f"  Correct predictions:   {len(correct_confidences)} (avg confidence: {correct_confidences.mean()*100:.1f}%)")
print(f"  Incorrect predictions: {len(incorrect_confidences)} (avg confidence: {incorrect_confidences.mean()*100:.1f}%)")
print()

plt.figure(figsize=(10, 5))
plt.hist([correct_confidences, incorrect_confidences], bins=20, label=['Correct', 'Incorrect'])
plt.xlabel('Prediction Confidence')
plt.ylabel('Count')
plt.title('Prediction Confidence Distribution')
plt.legend()
plt.tight_layout()
plt.savefig('models/confidence_distribution.png', dpi=150)
print("  ✓ Saved confidence_distribution.png")
print()

# Model summary
print("=" * 70)
print("  EVALUATION SUMMARY")
print("=" * 70)
print(f"  Overall Accuracy:           {test_accuracy*100:.2f}%")
print(f"  Macro-Average Precision:    {report['macro avg']['precision']*100:.2f}%")
print(f"  Macro-Average Recall:       {report['macro avg']['recall']*100:.2f}%")
print(f"  Macro-Average F1-Score:     {report['macro avg']['f1-score']*100:.2f}%")
print()
print(f"  Best performing:  {best_classes[0][0]} ({best_classes[0][1]*100:.1f}%)")
print(f"  Worst performing: {worst_classes[0][0]} ({worst_classes[0][1]*100:.1f}%)")
print()
print(f"  Correct predictions:   {len(correct_confidences)} / {len(y_test)}")
print(f"  Incorrect predictions: {len(incorrect_confidences)} / {len(y_test)}")
print()

# Calculate inference time
print("Measuring inference speed...")
import time
single_sample = X_test[0:1]
warmup_pred = model.predict(single_sample, verbose=0)  # Warmup

times = []
for _ in range(100):
    start = time.time()
    _ = model.predict(single_sample, verbose=0)
    times.append((time.time() - start) * 1000)  # Convert to ms

avg_time = np.mean(times)
std_time = np.std(times)
print(f"  Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
print()

if avg_time < 50:
    print("  ✅ EXCELLENT! Inference time well below 100ms target")
elif avg_time < 100:
    print("  ✓ GOOD! Inference time meets target (<100ms)")
else:
    print("  ⚠️  Inference time exceeds target (>100ms)")

print()
print("=" * 70)
print()

print("Saved files:")
print("  - models/confusion_matrix.png")
print("  - models/per_class_accuracy.png")
print("  - models/confidence_distribution.png")
print()

print("Next step:")
print("  Run: python python/model_export.py")
print()
