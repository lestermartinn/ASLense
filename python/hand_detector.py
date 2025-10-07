# ASL Hand Recognition - MediaPipe Hand Landmark Detection
# Phase 1: Hand Tracking Setup

import cv2
import mediapipe as mp
import numpy as np
import sys

class HandLandmarkDetector:
    """
    Wrapper for MediaPipe Hands to extract 21 3D landmarks.
    Each landmark has (x, y, z) coordinates.
    """
    def __init__(self, 
                 static_image_mode=False,
                 max_num_hands=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def process_frame(self, frame):
        """
        Process a single frame and extract hand landmarks.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            landmarks: List of 21 (x, y, z) tuples, or None if no hand detected
            annotated_frame: Frame with landmarks drawn
        """
        # Convert BGR to RGB (MediaPipe uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Make a copy for annotation
        annotated_frame = frame.copy()
        
        landmarks = None
        
        if results.multi_hand_landmarks:
            # Get first hand's landmarks
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract normalized coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y, landmark.z))
            
            # Draw landmarks on frame
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        return landmarks, annotated_frame
    
    def landmarks_to_feature_vector(self, landmarks):
        """
        Convert 21 landmarks to 63-element feature vector.
        Format: [x1, y1, z1, x2, y2, z2, ..., x21, y21, z21]
        
        Args:
            landmarks: List of 21 (x, y, z) tuples
            
        Returns:
            numpy array of shape (63,)
        """
        if landmarks is None or len(landmarks) != 21:
            return None
        
        feature_vector = []
        for x, y, z in landmarks:
            feature_vector.extend([x, y, z])
        
        return np.array(feature_vector, dtype=np.float32)
    
    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()


def main():
    """
    Test the hand landmark detector with webcam feed.
    """
    print("=" * 50)
    print("ASL Hand Recognition - MediaPipe Test")
    print("=" * 50)
    print("\nInitializing camera and MediaPipe...")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    print("✓ Camera opened successfully")
    
    # Initialize hand detector
    detector = HandLandmarkDetector(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    print("✓ MediaPipe Hands initialized")
    
    print("\n📋 Instructions:")
    print("   - Show your hand to the camera")
    print("   - Press 'q' or ESC to quit")
    print("   - Press 's' to save current landmarks\n")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            frame_count += 1
            
            # Process frame
            landmarks, annotated_frame = detector.process_frame(frame)
            
            # Add info overlay
            height, width = frame.shape[:2]
            
            if landmarks:
                # Show landmark count
                cv2.putText(annotated_frame, 
                           f"Landmarks: {len(landmarks)}", 
                           (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                
                # Convert to feature vector
                features = detector.landmarks_to_feature_vector(landmarks)
                
                # Show first few features
                feature_text = f"Features: [{features[0]:.3f}, {features[1]:.3f}, {features[2]:.3f}, ...]"
                cv2.putText(annotated_frame, 
                           feature_text, 
                           (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 1)
            else:
                cv2.putText(annotated_frame, 
                           "No hand detected", 
                           (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 255), 2)
            
            # Show frame count
            cv2.putText(annotated_frame, 
                       f"Frame: {frame_count}", 
                       (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            
            # Display
            cv2.imshow('ASL Hand Detection - MediaPipe', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\n✓ Exiting...")
                break
            elif key == ord('s') and landmarks:  # 's' to save
                features = detector.landmarks_to_feature_vector(landmarks)
                print(f"\n📊 Saved landmarks:")
                print(f"   Feature vector shape: {features.shape}")
                print(f"   First 9 features: {features[:9]}")
    
    except KeyboardInterrupt:
        print("\n✓ Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("✓ Resources released")
        print(f"✓ Processed {frame_count} frames")


if __name__ == "__main__":
    # Check if required packages are available
    try:
        import cv2
        import mediapipe
        print("✓ All required packages found")
        print(f"✓ OpenCV version: {cv2.__version__}")
        print(f"✓ MediaPipe version: {mediapipe.__version__}\n")
        main()
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("\n📦 Install required packages:")
        print("   pip install opencv-python mediapipe")
        sys.exit(1)
