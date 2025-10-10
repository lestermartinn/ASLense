"""
ASL Hand Recognition - Enhanced Landmark & Prediction Service
This script runs as a subprocess, continuously reading frames and outputting:
1. Hand landmarks (63 features)
2. ASL letter prediction
3. Confidence score

Communication via JSON over stdout/stdin.
"""

# Suppress stderr FIRST - before any imports
import sys
import os

# Completely suppress stderr to prevent MediaPipe/TensorFlow warnings
sys.stderr = open(os.devnull, 'w')

# Set environment variables to minimize logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import pickle
import json
import time

# Disable MediaPipe logging
import logging
logging.getLogger('mediapipe').setLevel(logging.CRITICAL)
logging.getLogger('absl').setLevel(logging.CRITICAL)

class ASLRecognitionService:
    """Service that extracts hand landmarks and predicts ASL letters."""
    
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=0  # Lite model for speed
        )
        
        # Load TFLite model
        try:
            self.interpreter = tf.lite.Interpreter(model_path='models/asl_model.tflite')
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        except Exception as e:
            sys.stderr = sys.__stderr__  # Restore stderr for error
            print(f"ERROR: Failed to load model: {e}", file=sys.stderr)
            sys.exit(1)
        
        # Load label encoder
        try:
            with open('models/label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)
            self.class_names = label_encoder.classes_
        except Exception as e:
            sys.stderr = sys.__stderr__
            print(f"ERROR: Failed to load label encoder: {e}", file=sys.stderr)
            sys.exit(1)
        
        # Open camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            sys.stderr = sys.__stderr__
            print("ERROR: Could not open camera", file=sys.stderr)
            sys.exit(1)
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.frame_count = 0
    
    def extract_landmarks(self, hand_landmarks):
        """Extract 63 landmark values from MediaPipe hand landmarks."""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    
    def predict_letter(self, landmarks):
        """
        Predict ASL letter from landmarks.
        Returns: (predicted_letter, confidence)
        """
        # Prepare input
        landmarks_array = np.array(landmarks, dtype=np.float32).reshape(1, -1)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], landmarks_array)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Get prediction
        predicted_idx = np.argmax(output[0])
        confidence = float(output[0][predicted_idx])
        predicted_letter = self.class_names[predicted_idx]
        
        return predicted_letter, confidence
    
    def send_message(self, msg_type, data):
        """Send JSON message to stdout."""
        message = {
            "type": msg_type,
            "timestamp": time.time(),
            "frame": self.frame_count,
            "data": data
        }
        print(json.dumps(message), flush=True)
    
    def process_frame(self):
        """Read frame, detect hand, extract landmarks, predict letter."""
        ret, frame = self.cap.read()
        if not ret:
            return False
        
        self.frame_count += 1
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Get first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract landmarks (63 features)
            landmarks = self.extract_landmarks(hand_landmarks)
            
            # Predict letter
            predicted_letter, confidence = self.predict_letter(landmarks)
            
            # Send prediction message
            self.send_message("prediction", {
                "landmarks": landmarks,
                "letter": predicted_letter,
                "confidence": confidence
            })
        else:
            # No hand detected
            self.send_message("no_hand", {})
        
        return True
    
    def run(self):
        """Main service loop."""
        # Send ready message
        self.send_message("ready", {"model_loaded": True, "classes": len(self.class_names)})
        
        # Main loop
        while True:
            if not self.process_frame():
                break
    
    def cleanup(self):
        """Release resources."""
        self.cap.release()
        self.hands.close()

if __name__ == "__main__":
    service = ASLRecognitionService()
    try:
        service.run()
    except KeyboardInterrupt:
        pass
    finally:
        service.cleanup()
