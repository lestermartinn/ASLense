# ASLense 

**Real-time American Sign Language recognition system powered by computer vision and machine learning.**

ASLense recognizes ASL alphabet letters (A-Y) in real-time using hand landmark detection and a neural network. The system achieves 99.25% accuracy and includes an interactive learning mode for practicing ASL.

---

## How It Works

**Technology Stack:**
- **MediaPipe Hands**: Detects 21 hand landmarks in 3D space from webcam feed
- **TensorFlow Lite**: Runs neural network inference (72.6KB model, 48ms per prediction)
- **OpenCV**: Handles video capture and visualization
- **Python/C++ Hybrid**: Python for ML processing, optional C++ client for system integration

**Recognition Pipeline:**
1. Webcam captures video frames at 30 FPS
2. MediaPipe extracts 21 hand landmarks (x, y, z coordinates)
3. Landmarks normalized and fed to TFLite model
4. Neural network predicts letter with confidence score
5. Results displayed in real-time with visual feedback

**Supports 24 letters** (A-Y). Letters J and Z require motion tracking and are not included.

---

## Getting Started

### Prerequisites
- Python 3.12+
- Webcam
- (Optional) C++ compiler for C++ client

### Installation & Usage

1. **Clone and install dependencies:**
```bash
git clone https://github.com/lestermartinn/ASLense.git
cd ASLense
pip install -r python/requirements.txt
```

2. **Run the interactive learning mode:**
```bash
python -m venv .venv

.venv\Scripts\Activate.ps1

python python/interactive_learning.py
```
Practice ASL letters with real-time feedback. The system shows you a target letter, and you make the sign. Hold it steady for 10 frames to advance.

**Controls:** `SPACE` (skip letter) | `R` (reset) | `ESC` (quit)

3. **Or run demo recognition:**
```bash
python python/demo_recognition.py
```
See real-time letter predictions with confidence scores in an OpenCV window.

---

## Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 99.25% |
| **Inference Speed** | 48ms (25-30 FPS) |
| **Model Size** | 72.6 KB |
| **Training Samples** | 110,836 images |
| **Classes** | 24 letters (A-Y) |

---

## Project Structure

```
ASLense/
├── python/
│   ├── interactive_learning.py    # Interactive practice mode
│   ├── demo_recognition.py        # Real-time demo
│   ├── recognition_service.py     # Core ML service
│   ├── model_training.py          # Model training
│   └── requirements.txt
├── src/                           # C++ client (optional)
├── models/
│   ├── asl_model.tflite          # TensorFlow Lite model
│   └── label_encoder.pkl         # Label mappings
└── data/                          # Training dataset (not included)
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Created by Lester Martin** | [GitHub](https://github.com/lestermartinn/ASLense)


