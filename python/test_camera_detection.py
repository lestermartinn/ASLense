"""
Camera and MediaPipe Diagnostic Tool
Tests if camera opens and MediaPipe can detect hands
"""

import cv2
import mediapipe as mp
import sys

print("=" * 60)
print("  CAMERA & MEDIAPIPE DIAGNOSTIC TEST")
print("=" * 60)
print()

# Test 1: Camera Opening
print("[1/4] Testing camera access...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ FAILED: Camera won't open with CAP_DSHOW")
    print("   Trying without backend specification...")
    cap = cv2.VideoCapture(0)
    
if not cap.isOpened():
    print("❌ CRITICAL: Cannot open camera at all!")
    print("   - Check if camera is in use by another app")
    print("   - Check camera permissions")
    print("   - Try unplugging/replugging USB camera")
    sys.exit(1)

print("✓ Camera opened successfully!")

# Test 2: Camera Properties
print("\n[2/4] Checking camera properties...")
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"   Resolution: {int(width)}x{int(height)}")
print(f"   FPS: {fps}")

# Test 3: Frame Capture
print("\n[3/4] Testing frame capture...")
ret, frame = cap.read()
if not ret or frame is None:
    print("❌ FAILED: Cannot read frames from camera")
    sys.exit(1)

print(f"✓ Frame captured! Shape: {frame.shape}")
print(f"   (Height={frame.shape[0]}, Width={frame.shape[1]}, Channels={frame.shape[2]})")

# Test 4: MediaPipe Detection
print("\n[4/4] Testing MediaPipe hand detection...")
print("   Initializing MediaPipe Hands...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,  # Lower threshold for testing
    min_tracking_confidence=0.5,
    model_complexity=0
)

print("✓ MediaPipe initialized")
print("\n   Capturing 100 frames to test detection...")
print("   👋 SHOW YOUR HAND TO THE CAMERA NOW!")
print()

detection_count = 0
frame_count = 0
max_frames = 100

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame_count += 1
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        detection_count += 1
        if detection_count == 1:
            print(f"   🎉 FIRST DETECTION at frame {frame_count}!")
            # Print first landmark as proof
            landmark = results.multi_hand_landmarks[0].landmark[0]
            print(f"      Wrist position: x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}")
    
    # Progress indicator every 10 frames
    if frame_count % 10 == 0:
        percentage = (frame_count / max_frames) * 100
        bar = "█" * (frame_count // 5) + "░" * ((max_frames - frame_count) // 5)
        print(f"   [{bar}] {percentage:.0f}% - Detections so far: {detection_count}", end='\r')

print()  # New line after progress bar
print()

# Cleanup
cap.release()
hands.close()

# Results
print("=" * 60)
print("  TEST RESULTS")
print("=" * 60)
print(f"  Frames processed: {frame_count}")
print(f"  Hand detections:  {detection_count}")
print(f"  Detection rate:   {(detection_count/frame_count*100):.1f}%")
print()

if detection_count == 0:
    print("❌ NO HANDS DETECTED")
    print()
    print("Possible issues:")
    print("  1. Hand not visible to camera")
    print("  2. Poor lighting conditions")
    print("  3. Hand too close or too far from camera")
    print("  4. Background too similar to hand color")
    print()
    print("Recommendations:")
    print("  ✓ Ensure your hand is 1-3 feet from camera")
    print("  ✓ Use good lighting (bright room)")
    print("  ✓ Show your full hand (all fingers visible)")
    print("  ✓ Move your hand slowly")
    print("  ✓ Try different hand orientations")
elif detection_count < 10:
    print("⚠️  FEW DETECTIONS")
    print()
    print("MediaPipe works but detection is inconsistent.")
    print("Tips for better detection:")
    print("  ✓ Improve lighting")
    print("  ✓ Keep hand steady in camera view")
    print("  ✓ Show palm facing camera")
elif detection_count < 50:
    print("✓ MODERATE DETECTION")
    print()
    print("Hand tracking is working but could be better.")
    print("For best results:")
    print("  ✓ Keep hand in center of camera view")
    print("  ✓ Ensure good, even lighting")
else:
    print("✅ EXCELLENT DETECTION!")
    print()
    print("Hand tracking is working great!")
    print("The system should work well for ASL recognition.")

print()
print("=" * 60)
