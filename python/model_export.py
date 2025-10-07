"""
Model Export for ASL Recognition
Converts Keras model to TensorFlow Lite for deployment
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import time

print("=" * 70)
print("  ASL MODEL EXPORT TO TENSORFLOW LITE")
print("=" * 70)
print()

# Load model
print("[1/6] Loading trained model...")
model = keras.models.load_model('models/best_model.keras')
print("  ✓ Loaded best_model.keras")
print()

# Load test data for validation
print("[2/6] Loading test data...")
test_data = np.load('data/test_data.npz')
X_test, y_test = test_data['X'], test_data['y']

# Load label encoder
with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

num_classes = len(label_encoder.classes_)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

print(f"  Test samples: {len(X_test)}")
print()

# Get baseline Keras model accuracy
print("[3/6] Evaluating original Keras model...")
keras_loss, keras_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"  Keras accuracy: {keras_accuracy*100:.2f}%")

# Measure Keras inference time
print("  Measuring Keras inference speed...")
single_sample = X_test[0:1]
_ = model.predict(single_sample, verbose=0)  # Warmup

keras_times = []
for _ in range(100):
    start = time.time()
    _ = model.predict(single_sample, verbose=0)
    keras_times.append((time.time() - start) * 1000)

keras_avg_time = np.mean(keras_times)
print(f"  Keras inference: {keras_avg_time:.2f} ms/sample")
print()

# Convert to TensorFlow Lite (without quantization)
print("[4/6] Converting to TensorFlow Lite (float32)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
tflite_path = 'models/asl_model.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

tflite_size = len(tflite_model) / 1024
print(f"  ✓ Saved asl_model.tflite ({tflite_size:.1f} KB)")
print()

# Test TFLite model
print("[5/6] Testing TensorFlow Lite model...")
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("  Input details:")
print(f"    Shape: {input_details[0]['shape']}")
print(f"    Type:  {input_details[0]['dtype']}")
print()
print("  Output details:")
print(f"    Shape: {output_details[0]['shape']}")
print(f"    Type:  {output_details[0]['dtype']}")
print()

# Evaluate TFLite model accuracy
print("  Evaluating TFLite accuracy...")
tflite_predictions = []
for sample in X_test:
    interpreter.set_tensor(input_details[0]['index'], sample.reshape(1, -1).astype(np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    tflite_predictions.append(np.argmax(output))

tflite_predictions = np.array(tflite_predictions)
tflite_accuracy = np.mean(tflite_predictions == y_test)
print(f"  TFLite accuracy: {tflite_accuracy*100:.2f}%")

# Measure TFLite inference time
print("  Measuring TFLite inference speed...")
sample = X_test[0].reshape(1, -1).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], sample)
interpreter.invoke()  # Warmup

tflite_times = []
for _ in range(100):
    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    tflite_times.append((time.time() - start) * 1000)

tflite_avg_time = np.mean(tflite_times)
print(f"  TFLite inference: {tflite_avg_time:.2f} ms/sample")
print()

# Quantized model (optional)
print("[6/6] Converting to TensorFlow Lite (quantized)...")
converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]

# Provide representative dataset for quantization
def representative_dataset():
    for sample in X_test[:100]:
        yield [sample.reshape(1, -1).astype(np.float32)]

converter_quant.representative_dataset = representative_dataset
tflite_quant_model = converter_quant.convert()

# Save quantized model
tflite_quant_path = 'models/asl_model_quantized.tflite'
with open(tflite_quant_path, 'wb') as f:
    f.write(tflite_quant_model)

tflite_quant_size = len(tflite_quant_model) / 1024
print(f"  ✓ Saved asl_model_quantized.tflite ({tflite_quant_size:.1f} KB)")
print()

# Test quantized model
print("  Testing quantized model...")
interpreter_quant = tf.lite.Interpreter(model_path=tflite_quant_path)
interpreter_quant.allocate_tensors()

input_details_quant = interpreter_quant.get_input_details()
output_details_quant = interpreter_quant.get_output_details()

# Evaluate quantized model accuracy
quant_predictions = []
for sample in X_test:
    interpreter_quant.set_tensor(input_details_quant[0]['index'], sample.reshape(1, -1).astype(np.float32))
    interpreter_quant.invoke()
    output = interpreter_quant.get_tensor(output_details_quant[0]['index'])
    quant_predictions.append(np.argmax(output))

quant_predictions = np.array(quant_predictions)
quant_accuracy = np.mean(quant_predictions == y_test)
print(f"  Quantized accuracy: {quant_accuracy*100:.2f}%")

# Measure quantized inference time
interpreter_quant.set_tensor(input_details_quant[0]['index'], sample)
interpreter_quant.invoke()  # Warmup

quant_times = []
for _ in range(100):
    start = time.time()
    interpreter_quant.set_tensor(input_details_quant[0]['index'], sample)
    interpreter_quant.invoke()
    quant_times.append((time.time() - start) * 1000)

quant_avg_time = np.mean(quant_times)
print(f"  Quantized inference: {quant_avg_time:.2f} ms/sample")
print()

# Comparison table
print("=" * 70)
print("  MODEL COMPARISON")
print("=" * 70)
print()
print("Model            | Accuracy | Inference Time | Size")
print("-----------------+----------+----------------+--------")
print(f"Keras (original) | {keras_accuracy*100:6.2f}%  | {keras_avg_time:10.2f} ms | N/A")
print(f"TFLite (float32) | {tflite_accuracy*100:6.2f}%  | {tflite_avg_time:10.2f} ms | {tflite_size:5.1f} KB")
print(f"TFLite (quant)   | {quant_accuracy*100:6.2f}%  | {quant_avg_time:10.2f} ms | {tflite_quant_size:5.1f} KB")
print()

# Accuracy comparison
accuracy_diff = abs(keras_accuracy - tflite_accuracy) * 100
quant_accuracy_diff = abs(keras_accuracy - quant_accuracy) * 100

print("Accuracy comparison:")
print(f"  Float32 vs Keras: {accuracy_diff:.2f}% difference")
print(f"  Quantized vs Keras: {quant_accuracy_diff:.2f}% difference")

if accuracy_diff < 1.0:
    print("  ✅ Float32 model maintains accuracy")
else:
    print("  ⚠️  Float32 model has noticeable accuracy loss")

if quant_accuracy_diff < 2.0:
    print("  ✅ Quantized model maintains reasonable accuracy")
else:
    print("  ⚠️  Quantized model has significant accuracy loss")

print()

# Speed comparison
speedup_float = keras_avg_time / tflite_avg_time
speedup_quant = keras_avg_time / quant_avg_time

print("Speed comparison:")
print(f"  Float32 speedup: {speedup_float:.2f}x")
print(f"  Quantized speedup: {speedup_quant:.2f}x")
print()

# Size comparison
print("Size comparison:")
print(f"  Size reduction (quantized): {(1 - tflite_quant_size/tflite_size)*100:.1f}%")
print()

# Recommendation
print("=" * 70)
print("  RECOMMENDATION")
print("=" * 70)
if quant_accuracy_diff < 2.0 and quant_avg_time < 100:
    print("  ✅ Use QUANTIZED model:")
    print(f"     - Smaller size ({tflite_quant_size:.1f} KB)")
    print(f"     - Faster inference ({quant_avg_time:.2f} ms)")
    print(f"     - Minimal accuracy loss ({quant_accuracy_diff:.2f}%)")
else:
    print("  ✅ Use FLOAT32 model:")
    print(f"     - Better accuracy (only {accuracy_diff:.2f}% loss)")
    print(f"     - Fast enough ({tflite_avg_time:.2f} ms < 100ms)")
    print(f"     - Size acceptable ({tflite_size:.1f} KB)")

print()
print("=" * 70)
print()

print("Exported models:")
print("  - models/asl_model.tflite (float32)")
print("  - models/asl_model_quantized.tflite (int8)")
print()

print("Next step:")
print("  Phase 3: C++ TensorFlow Lite integration")
print("  This will require:")
print("    1. Install TensorFlow Lite C++ library")
print("    2. Create inference_engine.cpp")
print("    3. Integrate with LandmarkService")
print()
