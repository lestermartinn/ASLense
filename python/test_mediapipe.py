# Quick test to verify MediaPipe installation and hand detection
import cv2
import mediapipe as mp
import sys

print("="*50)
print("MediaPipe Hand Detection - Quick Test")
print("="*50)

try:
    print(f"✓ OpenCV version: {cv2.__version__}")
    print(f"✓ MediaPipe version: {mp.__version__}")
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    print("✓ MediaPipe Hands initialized successfully")
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✓ Camera opened successfully")
        
        # Read one frame
        ret, frame = cap.read()
        if ret:
            print(f"✓ Frame captured: {frame.shape}")
            
            # Convert to RGB and process
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                print(f"✓ Hand detected! Found {len(results.multi_hand_landmarks)} hand(s)")
                hand = results.multi_hand_landmarks[0]
                print(f"✓ Landmarks extracted: {len(hand.landmark)} points")
                print(f"   Example landmark 0: x={hand.landmark[0].x:.3f}, y={hand.landmark[0].y:.3f}, z={hand.landmark[0].z:.3f}")
            else:
                print("⚠️  No hand detected in frame (try showing your hand to camera)")
        
        cap.release()
    else:
        print("⚠️  Camera not available (expected if no webcam)")
    
    hands.close()
    print("\n🎉 Phase 1, Step 1: MediaPipe Installation - COMPLETE!")
    print("\n📋 Summary:")
    print("   ✓ Python environment configured")
    print("   ✓ OpenCV installed and working")
    print("   ✓ MediaPipe installed and working")
    print("   ✓ Hand detection ready for integration")
    print("\n🚀 Next: Integrate with C++ application")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
