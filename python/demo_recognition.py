"""
ASL Recognition Demo - Complete End-to-End System
Shows real-time ASL letter recognition using trained model
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import sys

print("=" * 70)
print("  ASL REAL-TIME RECOGNITION DEMO")
print("=" * 70)
print()

# Load model
print("Loading trained model...")
try:
    interpreter = tf.lite.Interpreter(model_path='models/asl_model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("  ✓ Model loaded successfully")
except Exception as e:
    print(f"  ❌ Error loading model: {e}")
    sys.exit(1)

# Load label encoder
print("Loading label encoder...")
try:
    with open('models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    class_names = label_encoder.classes_
    print(f"  ✓ Loaded {len(class_names)} classes: {', '.join(class_names)}")
except Exception as e:
    print(f"  ❌ Error loading label encoder: {e}")
    sys.exit(1)

# Initialize MediaPipe
print("Initializing MediaPipe...")
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0
)
print("  ✓ MediaPipe initialized")

# Open camera
print("Opening camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("  ❌ Could not open camera")
    sys.exit(1)

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
print("  ✓ Camera opened")
print()

print("=" * 70)
print("  READY! Starting recognition...")
print("  Press 'Q' or 'ESC' to quit")
print("=" * 70)
print()

# Statistics
frame_count = 0
detection_count = 0
prediction_history = []

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        frame_count += 1
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_frame)
        
        # Default display
        predicted_letter = "No hand detected"
        confidence = 0.0
        color = (0, 0, 255)  # Red
        
        if results.multi_hand_landmarks:
            detection_count += 1
            
            # Draw hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract landmarks as features (63 values)
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                landmarks_array = np.array(landmarks, dtype=np.float32).reshape(1, -1)
                
                # Run inference
                interpreter.set_tensor(input_details[0]['index'], landmarks_array)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                
                # Get prediction
                predicted_idx = np.argmax(output[0])
                confidence = output[0][predicted_idx]
                predicted_letter = class_names[predicted_idx]
                
                # Store in history
                prediction_history.append(predicted_letter)
                if len(prediction_history) > 10:
                    prediction_history.pop(0)
                
                # Color based on confidence
                if confidence > 0.9:
                    color = (0, 255, 0)  # Green - high confidence
                elif confidence > 0.7:
                    color = (0, 255, 255)  # Yellow - medium confidence
                else:
                    color = (0, 165, 255)  # Orange - low confidence
        
        # Draw HUD (Heads-Up Display)
        height, width = frame.shape[:2]
        
        # Semi-transparent overlay for text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Display predicted letter (large)
        cv2.putText(frame, f"Letter: {predicted_letter}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Display confidence
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Display statistics
        cv2.putText(frame, f"Frames: {frame_count} | Detections: {detection_count}", (20, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Display recent predictions
        if prediction_history:
            recent = ' '.join(prediction_history[-5:])
            cv2.putText(frame, f"Recent: {recent}", (20, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press Q or ESC to quit", (width - 250, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show frame
        cv2.imshow('ASL Recognition Demo', frame)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # Q or ESC
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    # Print statistics
    print()
    print("=" * 70)
    print("  SESSION SUMMARY")
    print("=" * 70)
    print(f"  Total frames:       {frame_count}")
    print(f"  Hand detections:    {detection_count}")
    if frame_count > 0:
        print(f"  Detection rate:     {detection_count/frame_count*100:.1f}%")
    print(f"  Predictions made:   {len(prediction_history)}")
    
    if prediction_history:
        from collections import Counter
        most_common = Counter(prediction_history).most_common(5)
        print()
        print("  Most predicted letters:")
        for letter, count in most_common:
            print(f"    {letter}: {count} times")
    
    print()
    print("=" * 70)
    print("  Demo complete! Your model works great!")
    print("=" * 70)
