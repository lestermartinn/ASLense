"""
ASL Interactive Learning Mode
Practice ASL letters with real-time feedback!

Features:
- Sequential practice mode (A→Z)
- Real-time recognition feedback
- Progress tracking
- Visual feedback with colors
- Statistics and scoring
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import sys
import time
import random
from collections import Counter

class ASLInteractiveLearning:
    """Interactive ASL learning application."""
    
    def __init__(self):
        print("=" * 70)
        print("  ASL INTERACTIVE LEARNING MODE")
        print("=" * 70)
        print()
        
        # Load model
        print("Loading trained model...")
        try:
            self.interpreter = tf.lite.Interpreter(model_path='models/asl_model.tflite')
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print("  ✓ Model loaded")
        except Exception as e:
            print(f"  ❌ Error loading model: {e}")
            sys.exit(1)
        
        # Load label encoder
        print("Loading label encoder...")
        try:
            with open('models/label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)
            self.class_names = list(label_encoder.classes_)
            print(f"  ✓ Loaded {len(self.class_names)} classes")
        except Exception as e:
            print(f"  ❌ Error loading label encoder: {e}")
            sys.exit(1)
        
        # Initialize MediaPipe
        print("Initializing MediaPipe...")
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
        print("  ✓ MediaPipe initialized")
        
        # Open camera
        print("Opening camera...")
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("  ❌ Could not open camera")
            sys.exit(1)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("  ✓ Camera opened")
        print()
        
        # Learning state
        self.mode = "sequential"  # sequential, random, or custom
        self.target_letter = None
        self.target_index = 0
        self.letters_completed = []
        self.current_attempt_start = None
        self.correct_frame_count = 0
        self.required_correct_frames = 10  # Need 10 consecutive correct predictions
        self.confidence_threshold = 0.80
        
        # Statistics
        self.total_attempts = 0
        self.successful_attempts = 0
        self.session_start = time.time()
        self.letter_times = {}
        
        # UI state
        self.message = ""
        self.message_color = (255, 255, 255)
        self.message_time = 0
        
    def set_next_target(self):
        """Set the next target letter."""
        if self.mode == "sequential":
            if self.target_index >= len(self.class_names):
                # Completed all letters!
                self.target_letter = None
                return False
            self.target_letter = self.class_names[self.target_index]
            self.target_index += 1
        elif self.mode == "random":
            self.target_letter = random.choice(self.class_names)
        
        self.current_attempt_start = time.time()
        self.correct_frame_count = 0
        self.total_attempts += 1
        return True
    
    def extract_landmarks(self, hand_landmarks):
        """Extract 63 landmark values."""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    
    def predict_letter(self, landmarks):
        """Predict ASL letter from landmarks."""
        landmarks_array = np.array(landmarks, dtype=np.float32).reshape(1, -1)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], landmarks_array)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        predicted_idx = np.argmax(output[0])
        confidence = float(output[0][predicted_idx])
        predicted_letter = self.class_names[predicted_idx]
        
        return predicted_letter, confidence
    
    def check_success(self, predicted_letter, confidence):
        """Check if user successfully made the target sign."""
        if predicted_letter == self.target_letter and confidence >= self.confidence_threshold:
            self.correct_frame_count += 1
            
            if self.correct_frame_count >= self.required_correct_frames:
                # SUCCESS!
                elapsed = time.time() - self.current_attempt_start
                self.letters_completed.append(self.target_letter)
                self.letter_times[self.target_letter] = elapsed
                self.successful_attempts += 1
                
                self.show_message(f"✓ CORRECT! {self.target_letter} in {elapsed:.1f}s", (0, 255, 0))
                return True
        else:
            # Reset counter if prediction doesn't match
            self.correct_frame_count = 0
        
        return False
    
    def show_message(self, text, color):
        """Show a temporary message."""
        self.message = text
        self.message_color = color
        self.message_time = time.time()
    
    def draw_ui(self, frame, predicted_letter, confidence, hand_detected):
        """Draw the interactive UI on the frame."""
        height, width = frame.shape[:2]
        
        # Create right panel for target and info
        panel_width = 400
        panel = np.zeros((height, panel_width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Dark gray background
        
        # Draw target letter
        if self.target_letter:
            cv2.putText(panel, "TARGET:", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            
            # Huge letter
            cv2.putText(panel, self.target_letter, (panel_width//2 - 60, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 255, 255), 10)
            
            # Instructions
            cv2.putText(panel, "Make this sign!", (20, 280),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
            
            # Current prediction
            y_offset = 350
            cv2.putText(panel, "Your sign:", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            if hand_detected:
                # Color based on correctness
                if predicted_letter == self.target_letter:
                    pred_color = (0, 255, 0) if confidence >= self.confidence_threshold else (0, 255, 255)
                else:
                    pred_color = (0, 0, 255)
                
                cv2.putText(panel, f"{predicted_letter}", (150, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, pred_color, 3)
                cv2.putText(panel, f"({confidence*100:.0f}%)", (200, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, pred_color, 1)
                
                # Progress bar for consecutive correct frames
                if predicted_letter == self.target_letter and confidence >= self.confidence_threshold:
                    bar_width = 350
                    bar_height = 30
                    bar_x = 20
                    bar_y = y_offset + 30
                    
                    # Background
                    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                                (100, 100, 100), -1)
                    
                    # Progress
                    progress = min(1.0, self.correct_frame_count / self.required_correct_frames)
                    progress_width = int(bar_width * progress)
                    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height),
                                (0, 255, 0), -1)
                    
                    # Text
                    cv2.putText(panel, f"{self.correct_frame_count}/{self.required_correct_frames} frames",
                               (bar_x + 10, bar_y + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(panel, "No hand", (150, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 1)
        
        # Progress
        y_offset = 480
        completed = len(self.letters_completed)
        total = len(self.class_names)
        cv2.putText(panel, f"Progress: {completed}/{total}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Progress bar
        bar_width = 350
        bar_x = 20
        bar_y = y_offset + 10
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20),
                     (100, 100, 100), -1)
        if total > 0:
            progress_width = int(bar_width * (completed / total))
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x + progress_width, bar_y + 20),
                         (0, 255, 255), -1)
        
        # Statistics
        y_offset = 550
        accuracy = (self.successful_attempts / self.total_attempts * 100) if self.total_attempts > 0 else 0
        cv2.putText(panel, f"Accuracy: {accuracy:.0f}%", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        elapsed = time.time() - self.session_start
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        cv2.putText(panel, f"Time: {minutes}m {seconds}s", (20, y_offset + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        # Controls
        y_offset = height - 60
        cv2.putText(panel, "Controls:", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.putText(panel, "[SPACE] Skip  [R] Reset", (20, y_offset + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(panel, "[ESC] Quit", (20, y_offset + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Combine frame and panel
        combined = np.hstack([frame, panel])
        
        # Show message overlay if recent
        if time.time() - self.message_time < 2.0:
            overlay = combined.copy()
            msg_height = 80
            msg_y = height // 2 - msg_height // 2
            cv2.rectangle(overlay, (0, msg_y), (width + panel_width, msg_y + msg_height),
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, combined, 0.3, 0, combined)
            
            cv2.putText(combined, self.message,
                       (width // 2 - 200, height // 2 + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.message_color, 4)
        
        return combined
    
    def run(self):
        """Main learning loop."""
        print("=" * 70)
        print("  STARTING INTERACTIVE LEARNING MODE")
        print("=" * 70)
        print()
        print("Instructions:")
        print("  - Follow the TARGET letter shown on the right")
        print("  - Make the sign and hold it steady")
        print("  - Green progress bar will fill up when correct")
        print("  - Letter advances automatically on success")
        print()
        print("Controls:")
        print("  SPACE - Skip current letter")
        print("  R     - Reset progress")
        print("  ESC   - Quit")
        print()
        print("Press any key to start...")
        
        # Wait for user
        cv2.namedWindow('ASL Interactive Learning', cv2.WINDOW_NORMAL)
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(blank, "Press any key to start...", (400, 360),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.imshow('ASL Interactive Learning', blank)
        cv2.waitKey(0)
        
        # Set first target
        self.set_next_target()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process with MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                predicted_letter = None
                confidence = 0.0
                hand_detected = False
                
                if results.multi_hand_landmarks:
                    hand_detected = True
                    
                    # Draw hand landmarks
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # Predict
                        landmarks = self.extract_landmarks(hand_landmarks)
                        predicted_letter, confidence = self.predict_letter(landmarks)
                        
                        # Check for success
                        if self.target_letter and self.check_success(predicted_letter, confidence):
                            # Move to next letter after brief pause
                            time.sleep(0.5)
                            if not self.set_next_target():
                                # Completed all letters!
                                self.show_message("🎉 COMPLETED ALL LETTERS!", (0, 255, 255))
                                break
                
                # Draw UI
                display = self.draw_ui(frame, predicted_letter, confidence, hand_detected)
                
                # Show
                cv2.imshow('ASL Interactive Learning', display)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord(' '):  # SPACE - skip
                    if self.target_letter:
                        self.show_message(f"Skipped {self.target_letter}", (255, 128, 0))
                        time.sleep(0.3)
                        if not self.set_next_target():
                            break
                elif key == ord('r') or key == ord('R'):  # Reset
                    self.target_index = 0
                    self.letters_completed = []
                    self.total_attempts = 0
                    self.successful_attempts = 0
                    self.session_start = time.time()
                    self.letter_times = {}
                    self.set_next_target()
                    self.show_message("Progress reset!", (255, 255, 0))
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Show final statistics
            self.show_final_stats()
            
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
    
    def show_final_stats(self):
        """Display final session statistics."""
        print()
        print("=" * 70)
        print("  SESSION COMPLETE!")
        print("=" * 70)
        print()
        print(f"Letters completed: {len(self.letters_completed)}/{len(self.class_names)}")
        print(f"Total attempts: {self.total_attempts}")
        
        if self.total_attempts > 0:
            accuracy = self.successful_attempts / self.total_attempts * 100
            print(f"Accuracy: {accuracy:.1f}%")
        
        elapsed = time.time() - self.session_start
        print(f"Total time: {elapsed/60:.1f} minutes")
        
        if self.letter_times:
            avg_time = sum(self.letter_times.values()) / len(self.letter_times)
            print(f"Average time per letter: {avg_time:.1f} seconds")
            
            print()
            print("Fastest letters:")
            sorted_times = sorted(self.letter_times.items(), key=lambda x: x[1])
            for letter, time_taken in sorted_times[:5]:
                print(f"  {letter}: {time_taken:.1f}s")
        
        if self.letters_completed:
            print()
            print(f"Completed letters: {', '.join(self.letters_completed)}")
        
        print()
        print("=" * 70)
        print("  Great practice! Keep learning ASL!")
        print("=" * 70)

if __name__ == "__main__":
    app = ASLInteractiveLearning()
    app.run()
