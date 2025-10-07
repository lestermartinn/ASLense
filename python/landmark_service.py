"""
ASL Hand Recognition - Landmark Extractor Service
This script runs as a subprocess, continuously reading frames and outputting hand landmarks.
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
import json
import time

# Disable MediaPipe logging
import logging
logging.getLogger('mediapipe').setLevel(logging.CRITICAL)
logging.getLogger('absl').setLevel(logging.CRITICAL)

class LandmarkExtractorService:
    """Service that extracts hand landmarks and outputs them as JSON."""
    
    def __init__(self):
        # Initialize MediaPipe Hands with lightweight model
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=0  # 0=Lite (faster)
        )
        
        # Open camera with error handling
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow backend on Windows
        
        if not self.cap.isOpened():
            # Try without backend specification
            self.cap = cv2.VideoCapture(0)
            
        if not self.cap.isOpened():
            # Restore stderr temporarily to report critical error
            sys.stderr = sys.__stderr__
            self.send_error("Failed to open camera")
            sys.exit(1)
        
        # Camera settings for performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
        self.frame_count = 0
        self.start_time = time.time()
    
    def send_message(self, msg_type, data):
        """Send a JSON message to stdout."""
        message = {
            "type": msg_type,
            "timestamp": time.time(),
            "frame": self.frame_count,
            "data": data
        }
        print(json.dumps(message), flush=True)
    
    def send_error(self, error_msg):
        """Send an error message."""
        self.send_message("error", {"message": error_msg})
    
    def send_landmarks(self, landmarks):
        """Send hand landmarks as 63-element feature vector."""
        if landmarks and len(landmarks) == 21:
            # Flatten to [x1,y1,z1, x2,y2,z2, ..., x21,y21,z21]
            features = []
            for lm in landmarks:
                features.extend([lm.x, lm.y, lm.z])
            
            self.send_message("landmarks", {
                "features": features,
                "count": len(landmarks)
            })
        else:
            self.send_message("no_hand", {})
    
    def send_stats(self):
        """Send performance statistics."""
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        self.send_message("stats", {
            "fps": round(fps, 2),
            "frames_processed": self.frame_count,
            "elapsed_time": round(elapsed, 2)
        })
    
    def process_frame(self):
        """Capture and process a single frame."""
        ret, frame = self.cap.read()
        if not ret:
            self.send_error("Failed to capture frame")
            return False
        
        self.frame_count += 1
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Extract and send landmarks
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.send_landmarks(hand_landmarks.landmark)
        else:
            self.send_message("no_hand", {})
        
        return True
    
    def run(self):
        """Main processing loop."""
        self.send_message("ready", {
            "camera_width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "camera_height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": int(self.cap.get(cv2.CAP_PROP_FPS))
        })
        
        try:
            while True:
                # Check for commands from stdin (non-blocking)
                # For now, just process continuously
                
                if not self.process_frame():
                    break
                
                # Send stats every 30 frames
                if self.frame_count % 30 == 0:
                    self.send_stats()
                
        except KeyboardInterrupt:
            self.send_message("shutdown", {"reason": "interrupted"})
        except Exception as e:
            self.send_error(f"Unexpected error: {str(e)}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        self.send_stats()
        self.cap.release()
        self.hands.close()
        self.send_message("shutdown", {"reason": "normal"})


def main():
    # Run the service
    service = LandmarkExtractorService()
    service.run()


if __name__ == "__main__":
    main()
