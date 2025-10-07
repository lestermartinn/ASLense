# ASL Hand Recognition Visualizer - Quick Reference

## Starting the Visualizer

### Method 1: Using Batch File (Recommended)
```cmd
run_visualizer.bat
```

### Method 2: Direct Execution
Make sure MSYS64/mingw64/bin is in PATH:
```powershell
$env:PATH = "C:\msys64\mingw64\bin;$env:PATH"
bin\visualizer.exe
```

---

## Controls

| Key | Action |
|-----|--------|
| **Q** | Quit application |
| **ESC** | Quit application (alternative) |
| **S** | Save screenshot to project root |

---

## What You'll See

### Visual Elements

1. **Hand Landmarks** 
   - 21 green dots with black borders
   - Each dot represents a joint or tip

2. **Hand Skeleton**
   - Orange lines connecting landmarks
   - Shows natural hand structure

3. **HUD (Top Left)**
   - Title: "ASL Hand Recognition - Visualizer"
   - FPS: Real-time frames per second
   - Detections: Count of hand detections
   - Status: "Hand: DETECTED" (green) or "Hand: Not visible" (orange)

4. **Instructions (Bottom)**
   - Quit instruction (left)
   - Screenshot instruction (right)

### Background
- Dark gray canvas (easier on eyes)
- No camera feed (Python service owns camera)

---

## Hand Landmark Map

```
        4        8       12       16       20
       (tip)   (tip)   (tip)    (tip)   (tip)
        |       |       |        |       |
        3       7      11       15      19
        |       |       |        |       |
        2       6      10       14      18
        |       |       |        |       |
        1       5       9       13      17
        |       |       |        |       |
        └───────┴───────┴────────┴───────┘
                     0 (wrist)
```

### Landmark IDs
- **0**: Wrist
- **1-4**: Thumb (base to tip)
- **5-8**: Index finger (base to tip)
- **9-12**: Middle finger (base to tip)  
- **13-16**: Ring finger (base to tip)
- **17-20**: Pinky (base to tip)

---

## Troubleshooting

### Window Doesn't Open
**Problem:** Visualizer exits immediately

**Solutions:**
1. Check Qt6 is installed:
   ```bash
   pacman -S mingw-w64-x86_64-qt6-base
   ```

2. Use `run_visualizer.bat` (sets PATH automatically)

3. Verify Python service can start:
   ```powershell
   .\.venv\Scripts\python.exe python/landmark_service.py
   ```
   Should see `{"type": "ready", ...}` message

### Low FPS
**Problem:** Visualization runs slower than expected

**Solutions:**
- Close other camera applications
- Reduce MediaPipe model complexity (already at 0 = Lite)
- Check CPU usage (MediaPipe is the bottleneck)

### No Hand Detected
**Problem:** Status always shows "Hand: Not visible"

**Solutions:**
- Show your hand clearly to the camera
- Ensure good lighting
- Keep hand at moderate distance (1-3 feet)
- Make sure Python service has camera access

### Screenshots Not Saving
**Problem:** Press S but no file appears

**Solutions:**
- Check project root directory for `screenshot_N.jpg`
- Ensure write permissions in project folder
- Check console output for save confirmation

---

## Expected Performance

| Metric | Value |
|--------|-------|
| FPS (hand visible) | 28-30 |
| FPS (no hand) | 30+ |
| Startup time | 1-2 seconds |
| Latency | <40ms |
| CPU usage | Moderate (mostly MediaPipe) |
| Memory | ~200-300 MB |

---

## Understanding the Output

### FPS Counter
- **>25 FPS**: Excellent, real-time
- **15-25 FPS**: Good, slight lag
- **<15 FPS**: Slow, check CPU usage

### Detection Count
- Increments each time a hand is detected
- Use this to verify system responsiveness
- Should increment smoothly when hand is visible

### Status Indicator
- **Green "Hand: DETECTED"**: System sees your hand
- **Orange "Hand: Not visible"**: Show hand to camera

---

## Tips for Best Results

### Lighting
- Use bright, even lighting
- Avoid backlighting (window behind you)
- No harsh shadows on hand

### Hand Position
- Keep hand 1-3 feet from camera
- Show full hand (all fingers visible)
- Move slowly for smooth tracking

### Camera
- Clean camera lens
- Stable camera position
- Good camera angle (slight above eye level)

---

## Keyboard Shortcuts Summary

```
┌─────────────────────────────────┐
│  Q / ESC    →  Quit             │
│  S          →  Screenshot       │
└─────────────────────────────────┘
```

---

## File Outputs

### Screenshots
- **Location:** Project root directory
- **Naming:** `screenshot_1.jpg`, `screenshot_2.jpg`, etc.
- **Format:** JPEG
- **Resolution:** 640x480 (same as visualization)

### Console Output
```
Starting MediaPipe landmark service... ✓ Service started
Display resolution: 640x480
Opening visualization window...

Controls:
  Q or ESC - Quit
  S        - Save screenshot

📹 Visualization active! Show your hand to the camera...
```

When you press S:
```
📸 Screenshot saved: screenshot_1.jpg
```

When you quit (Q/ESC):
```
🛑 Quit requested by user

╔════════════════════════════════════════════════════════╗
║              Session Summary                           ║
╚════════════════════════════════════════════════════════╝
  Total frames: 2840
  Hand detections: 1523
  Detection rate: 53.6%
  Average FPS: 28.4
  Duration: 100 seconds
  Screenshots saved: 3

🎉 Phase 1, Step 3: Visualization - COMPLETE!
```

---

## Color Reference

| Element | RGB | Hex | Description |
|---------|-----|-----|-------------|
| Background | (30,30,30) | #1E1E1E | Dark canvas |
| Landmarks | (0,255,0) | #00FF00 | Bright green |
| Landmark border | (0,0,0) | #000000 | Black outline |
| Connections | (255,200,0) | #FFC800 | Orange lines |
| Text | (255,255,255) | #FFFFFF | White |
| Text background | (50,50,50) | #323232 | Gray overlay |
| Detected status | (0,255,0) | #00FF00 | Green indicator |
| Not detected status | (255,165,0) | #FFA500 | Orange indicator |

---

## Integration with Other Tools

### With integration_test.cpp
Both programs can't run simultaneously (camera conflict). Use visualizer for:
- Visual debugging
- Demos
- Data quality verification

Use integration_test for:
- Automated testing
- Performance benchmarks
- Console-only environments

### With Future ML Model
When Phase 3 (TFLite inference) is complete:
- Visualizer will show predicted letter
- Confidence percentage displayed
- Real-time ASL recognition in the window

---

## Advanced Usage

### Custom Resolution
Edit `visualizer.cpp`:
```cpp
int frameWidth = 1280;   // Change from 640
int frameHeight = 720;   // Change from 480
```

### Custom Colors
Edit color constants:
```cpp
const Scalar COLOR_LANDMARKS(0, 255, 0);      // Green
const Scalar COLOR_CONNECTIONS(0, 200, 255);  // Orange
```

### Show More Landmark IDs
Edit the condition:
```cpp
if (i < 21) {  // Change from i < 5 to show all IDs
    string idxText = to_string(i);
    putText(frame, idxText, ...);
}
```

---

**Enjoy visualizing your hand tracking!** 👋✨
