/*
 * ASL Hand Recognition - Real-time Visualizer
 * Phase 1, Step 3: OpenCV Visualization with MediaPipe Landmarks
 *
 * This program displays live camera feed with hand landmarks overlaid,
 * showing the hand skeleton structure in real-time.
 */

#include "landmark_service.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>

using namespace std;
using namespace cv;

// MediaPipe Hand Landmarks connection indices
// These define the hand skeleton structure (which landmarks connect to which)
const vector<pair<int, int>> HAND_CONNECTIONS = {
    // Thumb
    {0, 1},
    {1, 2},
    {2, 3},
    {3, 4},
    // Index finger
    {0, 5},
    {5, 6},
    {6, 7},
    {7, 8},
    // Middle finger
    {0, 9},
    {9, 10},
    {10, 11},
    {11, 12},
    // Ring finger
    {0, 13},
    {13, 14},
    {14, 15},
    {15, 16},
    // Pinky
    {0, 17},
    {17, 18},
    {18, 19},
    {19, 20},
    // Palm
    {5, 9},
    {9, 13},
    {13, 17}};

// Color scheme (BGR format for OpenCV)
const Scalar COLOR_LANDMARKS(0, 255, 0);     // Green dots
const Scalar COLOR_CONNECTIONS(0, 200, 255); // Orange lines
const Scalar COLOR_TEXT(255, 255, 255);      // White text
const Scalar COLOR_BG(50, 50, 50);           // Dark gray background for text

// Draw a filled circle with border for better visibility
void drawLandmark(Mat &frame, Point center, int radius = 5)
{
    circle(frame, center, radius, COLOR_LANDMARKS, -1); // Filled circle
    circle(frame, center, radius, Scalar(0, 0, 0), 1);  // Black border
}

// Draw hand skeleton connections
void drawConnections(Mat &frame, const vector<Point> &points)
{
    for (const auto &connection : HAND_CONNECTIONS)
    {
        int idx1 = connection.first;
        int idx2 = connection.second;

        if (idx1 < points.size() && idx2 < points.size())
        {
            line(frame, points[idx1], points[idx2], COLOR_CONNECTIONS, 2);
        }
    }
}

// Draw text with background for better readability
void drawTextWithBackground(Mat &frame, const string &text, Point pos,
                            double scale = 0.6, int thickness = 2)
{
    int baseline = 0;
    Size textSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);

    // Draw background rectangle
    rectangle(frame,
              Point(pos.x - 5, pos.y - textSize.height - 5),
              Point(pos.x + textSize.width + 5, pos.y + 5),
              COLOR_BG, -1);

    // Draw text
    putText(frame, text, pos, FONT_HERSHEY_SIMPLEX, scale, COLOR_TEXT, thickness);
}

// Draw HUD (Heads-Up Display) with stats
void drawHUD(Mat &frame, double fps, int detectionCount, bool handDetected)
{
    int y = 30;
    int spacing = 35;

    // Title
    drawTextWithBackground(frame, "ASL Hand Recognition - Visualizer", Point(10, y), 0.7, 2);
    y += spacing;

    // FPS
    string fpsText = "FPS: " + to_string(static_cast<int>(fps));
    drawTextWithBackground(frame, fpsText, Point(10, y));
    y += spacing;

    // Detection count
    string countText = "Detections: " + to_string(detectionCount);
    drawTextWithBackground(frame, countText, Point(10, y));
    y += spacing;

    // Status
    string status = handDetected ? "Hand: DETECTED" : "Hand: Not visible";
    Scalar statusColor = handDetected ? Scalar(0, 255, 0) : Scalar(0, 165, 255);

    // Draw status with colored indicator
    int baseline = 0;
    Size textSize = getTextSize(status, FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
    rectangle(frame,
              Point(5, y - textSize.height - 5),
              Point(textSize.width + 15, y + 5),
              statusColor, -1);
    putText(frame, status, Point(10, y), FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BG, 2);

    // Instructions at bottom
    int bottomY = frame.rows - 20;
    drawTextWithBackground(frame, "Press 'Q' or 'ESC' to quit",
                           Point(10, bottomY), 0.5, 1);

    drawTextWithBackground(frame, "Press 'S' to save screenshot",
                           Point(frame.cols - 250, bottomY), 0.5, 1);
}

// Convert normalized landmarks to pixel coordinates
vector<Point> landmarksToPixels(const asl::LandmarkFeatures &landmarks,
                                int frameWidth, int frameHeight)
{
    vector<Point> points;

    // Extract 21 landmarks (each has x, y, z; we use only x, y for 2D display)
    for (size_t i = 0; i < 21; i++)
    {
        float x = landmarks[i * 3];     // X coordinate (normalized 0-1)
        float y = landmarks[i * 3 + 1]; // Y coordinate (normalized 0-1)
        // float z = landmarks[i * 3 + 2]; // Z coordinate (depth, not used for 2D viz)

        // Convert normalized coordinates to pixel coordinates
        int px = static_cast<int>(x * frameWidth);
        int py = static_cast<int>(y * frameHeight);

        points.push_back(Point(px, py));
    }

    return points;
}

int main()
{
    cout << "\n╔════════════════════════════════════════════════════════╗\n";
    cout << "║  ASL Hand Recognition - Real-time Visualizer          ║\n";
    cout << "║  Phase 1, Step 3: OpenCV Visualization                ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";

    // Paths to Python environment and landmark service
    string python_exe = "C:/Users/leste/OneDrive/Desktop/cs-projects/asl hand recognition/.venv/Scripts/python.exe";
    string script_path = "C:/Users/leste/OneDrive/Desktop/cs-projects/asl hand recognition/python/landmark_service.py";

    cout << "Starting MediaPipe landmark service... " << flush;
    asl::LandmarkService service(python_exe, script_path);

    if (!service.start())
    {
        cerr << "✗ Failed to start landmark service!" << endl;
        return 1;
    }
    cout << "✓ Service started\n\n";

    // Note: Python service already has the camera open
    // We'll create a blank canvas to visualize landmarks
    int frameWidth = 640;
    int frameHeight = 480;

    cout << "Display resolution: " << frameWidth << "x" << frameHeight << "\n";
    cout << "Opening visualization window...\n\n";
    cout << "Controls:\n";
    cout << "  Q or ESC - Quit\n";
    cout << "  S        - Save screenshot\n\n";

    // Create window
    const string windowName = "ASL Hand Recognition";
    namedWindow(windowName, WINDOW_NORMAL);
    resizeWindow(windowName, 800, 600);

    // Statistics
    int detectionCount = 0;
    int frameCount = 0;
    int screenshotCount = 0;
    auto startTime = chrono::high_resolution_clock::now();
    auto lastFpsUpdate = startTime;
    double fps = 0.0;
    bool handDetected = false;

    cout << "📹 Visualization active! Show your hand to the camera...\n\n";

    // Main visualization loop
    while (true)
    {
        // Create a dark canvas for visualization
        Mat frame = Mat::zeros(frameHeight, frameWidth, CV_8UC3);
        frame = Scalar(30, 30, 30); // Dark gray background

        frameCount++;

        // Get landmarks from service
        auto landmarks = service.get_landmarks();

        if (landmarks)
        {
            detectionCount++;
            handDetected = true;

            // Convert normalized landmarks to pixel coordinates
            vector<Point> points = landmarksToPixels(*landmarks, frameWidth, frameHeight);

            // Draw hand skeleton
            drawConnections(frame, points);

            // Draw landmarks
            for (size_t i = 0; i < points.size(); i++)
            {
                drawLandmark(frame, points[i]);

                // Draw landmark index for first few points (optional, for debugging)
                if (i < 5)
                {
                    string idxText = to_string(i);
                    putText(frame, idxText, points[i] + Point(10, 0),
                            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 0), 1);
                }
            }
        }
        else
        {
            handDetected = false;
        }

        // Calculate FPS every 10 frames
        if (frameCount % 10 == 0)
        {
            auto now = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = now - lastFpsUpdate;
            fps = 10.0 / elapsed.count();
            lastFpsUpdate = now;
        }

        // Draw HUD
        drawHUD(frame, fps, detectionCount, handDetected);

        // Display frame
        imshow(windowName, frame);

        // Handle keyboard input
        int key = waitKey(1) & 0xFF;

        if (key == 'q' || key == 'Q' || key == 27)
        { // Q or ESC
            cout << "\n🛑 Quit requested by user\n";
            break;
        }
        else if (key == 's' || key == 'S')
        { // S for screenshot
            screenshotCount++;
            string filename = "screenshot_" + to_string(screenshotCount) + ".jpg";
            imwrite(filename, frame);
            cout << "📸 Screenshot saved: " << filename << endl;
        }

        // Small delay to prevent CPU overload
        this_thread::sleep_for(chrono::milliseconds(1));
    }

    // Cleanup
    destroyAllWindows();

    // Print summary
    auto endTime = chrono::high_resolution_clock::now();
    chrono::duration<double> totalTime = endTime - startTime;

    cout << "\n╔════════════════════════════════════════════════════════╗\n";
    cout << "║              Session Summary                           ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n";
    cout << "  Total frames: " << frameCount << "\n";
    cout << "  Hand detections: " << detectionCount << "\n";
    cout << "  Detection rate: " << (frameCount > 0 ? (100.0 * detectionCount / frameCount) : 0) << "%\n";
    cout << "  Average FPS: " << (frameCount / totalTime.count()) << "\n";
    cout << "  Duration: " << static_cast<int>(totalTime.count()) << " seconds\n";
    cout << "  Screenshots saved: " << screenshotCount << "\n";

    cout << "\n🎉 Phase 1, Step 3: Visualization - COMPLETE!\n\n";
    cout << "Next steps:\n";
    cout << "  → Phase 2: Collect ASL dataset and train ML model\n";
    cout << "  → Phase 3: Integrate TensorFlow Lite for inference\n";
    cout << "  → Phase 4: Build interactive learning mode\n\n";

    return 0;
}
