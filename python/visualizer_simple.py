"""
Simple Python Hand Tracking Visualizer with OpenCV Window
Shows camera feed with hand landmarks overlaid in real-time
"""

import cv2
import mediapipe as mp
import time

print("=" * 60)
print("  ASL HAND RECOGNITION - PYTHON VISUALIZER")
print("=" * 60)
print()
print("This will open a window showing your camera with hand landmarks")
print("Press 'Q' or 'ESC' to quit")
print()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=0
)

# Open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("✓ Camera opened")
print("✓ MediaPipe initialized")
print("\n📹 Opening window...\n")

# Stats
detection_count = 0
frame_count = 0
start_time = time.time()
fps = 0
last_fps_update = start_time

cv2.namedWindow('ASL Hand Recognition', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ASL Hand Recognition', 800, 600)

print("👋 Show your hand to the camera!")
print("The window should now be open.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame_count += 1
    
    # Flip for mirror effect (more natural)
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        detection_count += 1
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw connections
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    # Calculate FPS
    if frame_count % 10 == 0:
        current_time = time.time()
        elapsed = current_time - last_fps_update
        fps = 10 / elapsed
        last_fps_update = current_time
    
    # Draw HUD
    status_text = "HAND DETECTED" if results.multi_hand_landmarks else "No hand visible"
    status_color = (0, 255, 0) if results.multi_hand_landmarks else (0, 165, 255)
    
    # Background for text
    cv2.rectangle(frame, (10, 10), (400, 140), (50, 50, 50), -1)
    
    # Text
    cv2.putText(frame, "ASL Hand Recognition", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Detections: {detection_count}", (20, 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, status_text, (20, 125), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Instructions at bottom
    cv2.rectangle(frame, (10, frame.shape[0] - 40), (350, frame.shape[0] - 10), 
                  (50, 50, 50), -1)
    cv2.putText(frame, "Press Q or ESC to quit", (20, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display
    cv2.imshow('ASL Hand Recognition', frame)
    
    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()

# Summary
total_time = time.time() - start_time
avg_fps = frame_count / total_time
detection_rate = (detection_count / frame_count * 100) if frame_count > 0 else 0

print("\n" + "=" * 60)
print("  SESSION SUMMARY")
print("=" * 60)
print(f"  Frames:          {frame_count}")
print(f"  Detections:      {detection_count}")
print(f"  Detection Rate:  {detection_rate:.1f}%")
print(f"  Average FPS:     {avg_fps:.1f}")
print(f"  Duration:        {int(total_time)}s")
print("=" * 60)
print()

if detection_rate > 50:
    print("✅ Excellent! Hand tracking works great!")
elif detection_rate > 20:
    print("⚠️  Moderate detection. Try better lighting.")
else:
    print("❌ Low detection. Check camera and hand visibility.")

print("\n🎉 Python visualizer test complete!")
print("The hand tracking system is working!\n")
