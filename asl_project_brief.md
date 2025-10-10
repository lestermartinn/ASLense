# ASL Hand Gesture Recognition System - Complete Project Brief

## Project Overview
Build a real-time American Sign Language (ASL) recognition system that detects and classifies hand gestures using computer vision and machine learning. The system will provide immediate feedback to users learning ASL, making it an interactive educational tool.

**Timeline**: 6-7 weeks
**Primary Language**: C++ (with Python for ML training)
**Development Environment**: VSCode with GitHub Copilot

---

## Technical Stack

### Core Technologies
- **Language**: C++ (C++17 or later)
- **Computer Vision**: OpenCV 4.x + MediaPipe Hands
- **Machine Learning Inference**: TensorFlow Lite C++ API
- **Model Training**: Python with TensorFlow/Keras
- **GUI (MVP)**: OpenCV's `imshow()` and drawing functions
- **GUI (Optional)**: Qt or SDL for polish phase
- **Build System**: CMake 3.15+
- **Version Control**: Git

### Why These Choices?
- **MediaPipe Hands**: State-of-the-art hand detection with 21 3D landmarks out-of-the-box
- **Landmark-based ML**: More robust than raw images (lighting/background invariant), faster inference
- **OpenCV GUI**: Simplest approach for MVP, avoids Qt/SDL learning curve
- **TensorFlow Lite**: Optimized for edge device inference, well-documented C++ API

---

## Project Architecture

```
┌─────────────────────────────────────────┐
│     User Interface (OpenCV imshow)       │
│  - Camera feed display                   │
│  - Prediction overlay                    │
│  - Practice mode feedback                │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│    Application Controller (C++)          │
│  - Mode management (Free/Practice)       │
│  - Feedback logic                        │
│  - Statistics tracking                   │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│   Computer Vision Pipeline (C++)         │
│  - Frame capture (OpenCV VideoCapture)   │
│  - Hand detection (MediaPipe Hands)      │
│  - Landmark extraction & normalization   │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│    ML Inference Engine (C++)             │
│  - Load TFLite model                     │
│  - Real-time classification              │
│  - Confidence scoring                    │
└──────────────────────────────────────────┘
```

---

## Development Phases

### Phase 0: Environment Setup (Week 1, possibly Week 2)
**CRITICAL**: This is the "boss battle." Take it step-by-step.

#### Step-by-Step Setup Process
1. **Day 1-2**: Minimal C++ program
   - Install C++ compiler (GCC/Clang/MSVC)
   - Install CMake (3.15+)
   - Create basic "Hello World" C++ project with CMake
   - Successfully compile and run

2. **Day 3-4**: OpenCV integration
   - Install OpenCV (4.x recommended)
   - Add OpenCV to CMakeLists.txt
   - Write minimal webcam capture program
   - Verify video feed displays in window

3. **Day 5-7**: MediaPipe Hands integration
   - Install MediaPipe C++ library
   - Integrate into CMake project
   - Display detected hand landmarks on video feed
   - Print landmark coordinates to console

4. **Day 8-10**: TensorFlow Lite setup
   - Install TensorFlow Lite C++ library
   - Add to CMakeLists.txt
   - Load a dummy .tflite model successfully
   - Verify inference works (even with random model)

#### Success Criteria
✅ Webcam feed displays in OpenCV window
✅ 21 hand landmarks tracked and displayed in real-time
✅ Can wave hand and see dots following fingers
✅ TFLite library linked and can load models

#### Troubleshooting Notes
- Keep detailed notes of CMakeLists.txt configurations
- Document every library path and linking issue
- If C++ integration too painful: Python prototype → port to C++ later

---

### Phase 1: ML Model Training (Week 2-3)

#### Why Landmark-Based Model?
- **Faster**: Trains in minutes, not hours
- **Robust**: Invariant to lighting, background, skin tone
- **Lightweight**: Tiny model, perfect for real-time
- **Debuggable**: 63 numbers easier to inspect than pixels

#### Model Architecture (Feedforward Neural Network)
```
Input: 63 features (21 landmarks × 3 coords: x, y, z)
↓
Dense Layer (128 units, ReLU activation)
Dropout (0.3)
↓
Dense Layer (64 units, ReLU activation)
Dropout (0.3)
↓
Output Layer (24 units, Softmax) [A-Z excluding J, Z]
```

#### Dataset
- **Primary**: ASL Alphabet Dataset from Kaggle (~87,000 images)
- **Process**: Extract MediaPipe landmarks from each image
- **Format**: CSV with columns: `[letter, x1, y1, z1, x2, y2, z2, ..., x21, y21, z21]`

#### Training Pipeline (Python)
```python
# 1. Data Preprocessing
- Load ASL image dataset
- Run MediaPipe on each image to extract landmarks
- Normalize landmarks relative to hand bounding box
- Save as CSV or numpy arrays
- Split: 70% train, 15% validation, 15% test

# 2. Model Training
- Define feedforward architecture in TensorFlow/Keras
- Compile with Adam optimizer, categorical cross-entropy loss
- Train with early stopping (monitor validation accuracy)
- Target: >95% validation accuracy

# 3. Model Export
- Convert trained model to TensorFlow Lite (.tflite)
- Optimize for inference (dynamic range quantization)
- Test inference in Python before moving to C++
```

#### Deliverables
- `asl_model.tflite` (trained model file)
- `landmarks_train.csv` (preprocessed training data)
- `train_model.py` (training script)
- Training plots (accuracy/loss curves)

---

### Phase 2: Real-Time Recognition (Week 3-4)

#### Core Functionality
1. Capture webcam frame
2. Detect hand with MediaPipe
3. Extract 21 landmarks
4. Normalize landmarks
5. Run inference with TFLite model
6. Display prediction on frame

#### OpenCV GUI Implementation
```cpp
// Pseudo-code for main loop
while (true) {
    // Capture frame
    cap >> frame;
    
    // Detect hand and extract landmarks
    auto landmarks = mediapipe.detectHand(frame);
    
    // Run inference if hand detected
    if (landmarks.size() == 21) {
        auto [prediction, confidence] = model.predict(landmarks);
        
        // Draw prediction on frame
        cv::putText(frame, 
                    "Predicted: " + prediction + " (" + to_string(confidence) + "%)",
                    cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0, cv::Scalar(0, 255, 0), 2);
        
        // Draw landmarks
        for (auto& landmark : landmarks) {
            cv::circle(frame, landmark, 3, cv::Scalar(255, 0, 0), -1);
        }
    }
    
    cv::imshow("ASL Recognition", frame);
    
    // Exit on ESC key
    if (cv::waitKey(1) == 27) break;
}
```

#### Features to Implement
- Real-time prediction overlay (letter + confidence %)
- Hand landmark visualization (21 colored dots)
- FPS counter display
- Keyboard controls:
  - ESC: Exit program
  - SPACE: Toggle practice mode
  - R: Reset statistics

#### Success Criteria
- Achieves 30+ FPS on standard laptop
- Predictions update in real-time (<100ms latency)
- Clean, readable display

---

### Phase 3: Interactive Learning Mode (Week 4-5)

#### Practice Mode Flow
```
1. Display target letter (e.g., "Show me: A")
2. User performs gesture
3. System continuously predicts
4. When prediction matches target:
   - Show visual feedback (green border)
   - Optional: Play success sound
   - Update statistics
   - Move to next letter after 2 seconds
5. If incorrect for >5 seconds:
   - Show hint or reference image
```

#### Visual Feedback with OpenCV
```cpp
// Correct gesture
cv::rectangle(frame, 
              cv::Rect(0, 0, frame.cols, frame.rows),
              cv::Scalar(0, 255, 0), // Green
              20); // Border thickness
cv::putText(frame, "Correct! ✓", cv::Point(50, 100), ...);

// Incorrect gesture
cv::rectangle(frame, 
              cv::Rect(0, 0, frame.cols, frame.rows),
              cv::Scalar(0, 0, 255), // Red
              20);
cv::putText(frame, "Try again ✗", cv::Point(50, 100), ...);
```

#### Statistics Tracking (In-Memory)
```cpp
struct PracticeStats {
    int totalAttempts;
    int correctAttempts;
    double avgConfidence;
    std::map<char, int> letterAccuracy; // Per-letter tracking
    double sessionStartTime;
};
```

#### Display Statistics
- Current accuracy rate (%)
- Letters practiced count
- Average confidence score
- Session duration
- Best/worst performing letters

---

### Phase 4: Optimization & Polish (Week 6-7)

#### Performance Optimization
- **Profile the code**: Find bottlenecks (hand detection, inference, drawing)
- **Reduce redundant processing**: Cache results, skip frames if needed
- **Optimize inference**: Ensure model runs efficiently
- **Frame skipping**: Process every 2-3 frames if needed for speed

#### Robustness Improvements
- **Confidence threshold**: Only show prediction if >70% confident
- **No hand handling**: Display "No hand detected" message
- **Calibration mode**: Ask user to show hand in various positions
- **Multi-lighting test**: Verify works in bright/dim environments
- **Background invariance**: Test with cluttered backgrounds

#### Documentation
1. **README.md**:
   - Project description
   - Features list
   - Installation instructions (step-by-step)
   - Usage guide
   - Demo screenshots/GIF
   - Architecture overview
   - Future improvements

2. **Code Documentation**:
   - Clear comments explaining complex logic
   - Function/class docstrings
   - CMakeLists.txt comments

3. **Demo Assets**:
   - Screen recording showing real-time recognition
   - GIF of practice mode in action
   - Architecture diagram

---

## Project Structure

```
asl-recognition/
├── CMakeLists.txt
├── README.md
├── .gitignore
├── src/
│   ├── main.cpp                    # Entry point
│   ├── hand_detector.hpp/cpp       # MediaPipe wrapper
│   ├── model_inference.hpp/cpp     # TFLite inference
│   ├── practice_mode.hpp/cpp       # Interactive learning logic
│   ├── ui_renderer.hpp/cpp         # OpenCV drawing utilities
│   └── utils.hpp/cpp               # Helper functions
├── models/
│   └── asl_model.tflite           # Trained model
├── training/
│   ├── train_model.py             # Model training script
│   ├── preprocess_dataset.py     # Landmark extraction
│   ├── requirements.txt           # Python dependencies
│   └── data/                      # Raw dataset (not in git)
├── assets/
│   ├── demo.gif
│   └── screenshots/
└── docs/
    └── architecture.md
```

---

## Milestones & Timeline

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1-2 | Environment Setup | C++ project compiles, webcam + MediaPipe working |
| 2-3 | ML Model | Trained .tflite model with >95% accuracy |
| 3-4 | Real-Time Recognition | Live prediction with OpenCV GUI |
| 4-5 | Practice Mode | Interactive learning mode functional |
| 5-6 | Statistics & Feedback | Tracking, visual feedback polished |
| 6-7 | Optimization & Docs | Performance tuned, README complete |

---

## Success Metrics

### Technical Performance
- **FPS**: 30+ frames per second
- **Latency**: <100ms from gesture to prediction
- **Accuracy**: >90% on test set, >85% real-world
- **Model Size**: <5MB for TFLite model

### Functional Requirements
- Recognizes 24 static ASL letters (A-Z excluding J, Z)
- Provides real-time feedback with confidence scores
- Practice mode cycles through alphabet
- Displays accuracy statistics
- Handles "no hand detected" gracefully

---

## Risk Mitigation

### Risk 1: C++ Setup Takes >2 Weeks
**Mitigation**: Build Python prototype first (same architecture)
**Fallback**: Python frontend + C++ backend for inference only
**Still valid**: Demonstrates systems thinking and optimization

### Risk 2: Landmark Model Not Accurate Enough
**Mitigation**: 
- Collect additional training data
- Try data augmentation on landmarks (rotation, scaling)
**Fallback**: Switch to image-based CNN (MobileNetV2)

### Risk 3: MediaPipe Integration Issues
**Mitigation**: Use MediaPipe Python bindings if C++ fails
**Keep C++ value**: Core inference and processing stays in C++

### Risk 4: Real-Time Performance Issues
**Mitigation**:
- Profile and optimize bottlenecks
- Reduce model complexity
- Process every 2nd/3rd frame
**Fallback**: Accept 20+ FPS (still usable)

---

## Resume Bullets (Polished)

Use these when updating your resume:

1. **Architected** a real-time ASL recognition system in C++ and OpenCV, **optimizing the vision pipeline** to achieve <33ms latency (30+ FPS) and >90% classification accuracy on live webcam feed

2. **Engineered** an end-to-end ML workflow using MediaPipe for hand landmark extraction and TensorFlow Lite C++ API, **deploying a lightweight neural network** optimized for real-time inference on consumer hardware

3. **Designed** an interactive learning mode with immediate visual feedback, creating an educational tool for ASL alphabet practice with accuracy tracking and progressive difficulty

4. **Implemented** robust computer vision pipeline handling diverse lighting conditions and backgrounds through geometric feature extraction from 21 hand landmarks

---

## Technical Challenges & Learning Outcomes

### Key Challenges
1. **C++ Library Integration**: CMake, linking, cross-platform builds
2. **Real-Time Performance**: Balancing accuracy and speed
3. **Computer Vision Pipeline**: Hand detection, landmark extraction
4. **ML Deployment**: Training in Python, deploying in C++
5. **User Experience**: Intuitive feedback, clear visuals

### Skills Demonstrated
- **Systems Programming**: C++, memory management, performance optimization
- **Computer Vision**: OpenCV, MediaPipe, image processing
- **Machine Learning**: Model training, deployment, inference optimization
- **Software Engineering**: Project architecture, build systems, documentation
- **Problem Solving**: Debugging, optimization, trade-off decisions

---

## Extensions (Post-MVP)

If you have extra time or want to continue after completing MVP:

1. **Dynamic Gestures**: Add J and Z using temporal models (LSTM/GRU)
2. **Common Words**: Expand beyond alphabet to phrases like "hello", "thank you"
3. **Two-Handed Signs**: Multi-hand detection for complex signs
4. **Mobile Port**: iOS/Android app with shared C++ core
5. **Difficulty Levels**: Adaptive practice based on user performance
6. **Community Features**: Share progress, leaderboards
7. **Accessibility**: Text-to-speech for predictions, high-contrast mode

---

## Resources & References

### Documentation
- MediaPipe Hands: https://google.github.io/mediapipe/solutions/hands
- OpenCV C++ Tutorials: https://docs.opencv.org/4.x/d9/df8/tutorial_root.html
- TensorFlow Lite C++ API: https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_c

### Datasets
- ASL Alphabet (Kaggle): https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- ASL MNIST: https://www.kaggle.com/datasets/datamunge/sign-language-mnist

### Example Projects (for inspiration, not copying)
- MediaPipe Hands examples in C++
- TFLite image classification samples

---

## First Week Action Items

### Days 1-2: Basic C++ Setup
- [ ] Install C++ compiler (g++/clang/MSVC)
- [ ] Install CMake (3.15+)
- [ ] Create "Hello World" CMake project
- [ ] Verify compilation and execution

### Days 3-4: OpenCV Integration
- [ ] Install OpenCV (via package manager or build from source)
- [ ] Add OpenCV to CMakeLists.txt
- [ ] Write webcam capture program
- [ ] Display video feed in window

### Days 5-7: MediaPipe Hands
- [ ] Install MediaPipe C++ library
- [ ] Integrate into CMake
- [ ] Run hand detection on video feed
- [ ] Display 21 landmarks as colored dots

### Days 8-10: TensorFlow Lite
- [ ] Install TFLite C++ library
- [ ] Add to CMakeLists.txt
- [ ] Load dummy .tflite model
- [ ] Run test inference

**Success Check**: By end of Week 1, you should see your hand with 21 dots on screen in real-time.

---

## Notes for GitHub Copilot

When working with GitHub Copilot in VSCode:

1. **Reference this document**: "Based on the ASL project brief..."
2. **Be specific with prompts**: "Write the CMakeLists.txt for Phase 0 with OpenCV and MediaPipe dependencies"
3. **Request explanations**: "Explain how to normalize hand landmarks relative to bounding box"
4. **Iterate on code**: "Optimize this landmark extraction for real-time performance"
5. **Ask for debugging help**: "Why is MediaPipe not detecting hands in low light?"

---

## Final Notes

- **Don't rush the setup phase**: A solid foundation prevents headaches later
- **Test incrementally**: Each component should work before moving to next
- **Document as you go**: Future you will thank present you
- **Commit often**: Git is your safety net
- **Ask for help**: Use Copilot, Stack Overflow, documentation liberally
- **Focus on MVP first**: Polish comes after functionality

Good luck! This project will be a strong addition to your portfolio and demonstrates skills that most CS students don't have (C++, CV, real-time systems).

---

**Project Brief Version**: 1.0  
**Last Updated**: September 30, 2025  
**Prepared for**: Lester Martin, Emory University CS Student