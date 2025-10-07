/*
 * ASL Hand Recognition - File-based Visualizer
 * Saves visualization frames to disk instead of showing in window
 * This works around Qt6/OpenCV GUI issues
 */

#include "landmark_service.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace cv;

// MediaPipe Hand Landmarks connection indices
const vector<pair<int, int>> HAND_CONNECTIONS = {
    {0, 1}, {1, 2}, {2, 3}, {3, 4}, // Thumb
    {0, 5},
    {5, 6},
    {6, 7},
    {7, 8}, // Index
    {0, 9},
    {9, 10},
    {10, 11},
    {11, 12}, // Middle
    {0, 13},
    {13, 14},
    {14, 15},
    {15, 16}, // Ring
    {0, 17},
    {17, 18},
    {18, 19},
    {19, 20}, // Pinky
    {5, 9},
    {9, 13},
    {13, 17} // Palm
};

const Scalar COLOR_LANDMARKS(0, 255, 0);
const Scalar COLOR_CONNECTIONS(0, 200, 255);
const Scalar COLOR_TEXT(255, 255, 255);
const Scalar COLOR_BG(50, 50, 50);

void drawLandmark(Mat &frame, Point center, int radius = 5)
{
    circle(frame, center, radius, COLOR_LANDMARKS, -1);
    circle(frame, center, radius, Scalar(0, 0, 0), 1);
}

void drawConnections(Mat &frame, const vector<Point> &points)
{
    for (const auto &connection : HAND_CONNECTIONS)
    {
        int idx1 = connection.first;
        int idx2 = connection.second;
        if (idx1 < static_cast<int>(points.size()) && idx2 < static_cast<int>(points.size()))
        {
            line(frame, points[idx1], points[idx2], COLOR_CONNECTIONS, 2);
        }
    }
}

void drawTextWithBackground(Mat &frame, const string &text, Point pos,
                            double scale = 0.6, int thickness = 2)
{
    int baseline = 0;
    Size textSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
    rectangle(frame,
              Point(pos.x - 5, pos.y - textSize.height - 5),
              Point(pos.x + textSize.width + 5, pos.y + 5),
              COLOR_BG, -1);
    putText(frame, text, pos, FONT_HERSHEY_SIMPLEX, scale, COLOR_TEXT, thickness);
}

void drawHUD(Mat &frame, double fps, int detectionCount, bool handDetected, int frameNum)
{
    int y = 30;
    drawTextWithBackground(frame, "ASL Hand Recognition - Visualizer", Point(10, y), 0.7, 2);
    y += 35;

    ostringstream fpsText;
    fpsText << "FPS: " << fixed << setprecision(1) << fps;
    drawTextWithBackground(frame, fpsText.str(), Point(10, y));
    y += 35;

    drawTextWithBackground(frame, "Detections: " + to_string(detectionCount), Point(10, y));
    y += 35;

    string status = handDetected ? "Hand: DETECTED" : "Hand: Not visible";
    Scalar statusColor = handDetected ? Scalar(0, 255, 0) : Scalar(0, 165, 255);
    int baseline = 0;
    Size textSize = getTextSize(status, FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
    rectangle(frame,
              Point(5, y - textSize.height - 5),
              Point(textSize.width + 15, y + 5),
              statusColor, -1);
    putText(frame, status, Point(10, y), FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BG, 2);

    // Frame number at bottom
    int bottomY = frame.rows - 20;
    drawTextWithBackground(frame, "Frame: " + to_string(frameNum), Point(10, bottomY), 0.5, 1);
}

vector<Point> landmarksToPixels(const asl::LandmarkFeatures &landmarks,
                                int frameWidth, int frameHeight)
{
    vector<Point> points;
    for (size_t i = 0; i < 21; i++)
    {
        float x = landmarks[i * 3];
        float y = landmarks[i * 3 + 1];
        int px = static_cast<int>(x * frameWidth);
        int py = static_cast<int>(y * frameHeight);
        points.push_back(Point(px, py));
    }
    return points;
}

int main()
{
    cout << "\n╔════════════════════════════════════════════════════════╗\n";
    cout << "║  ASL Hand Recognition - File Visualizer               ║\n";
    cout << "║  Saves frames to 'visualizer_output' folder           ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";

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

    cout << "[DEBUG] Setting dimensions..." << flush;
    int frameWidth = 640;
    int frameHeight = 480;
    cout << " OK\n";

    cout << "[DEBUG] Creating output directory..." << flush;
    // Create output directory
    system("if not exist visualizer_output mkdir visualizer_output");
    cout << " OK\n";

    cout << "Saving visualization frames to: visualizer_output\\\n";
    cout << "Processing for 15 seconds...\n";
    cout << "Check the folder to see your hand tracking!\n\n";

    cout << "[DEBUG] Initializing variables..." << flush;
    int detectionCount = 0;
    int frameCount = 0;
    int saveInterval = 30; // Save every 30th frame
    auto startTime = chrono::high_resolution_clock::now();
    auto lastFpsUpdate = startTime;
    double fps = 0.0;
    bool handDetected = false;

    auto endTime = chrono::high_resolution_clock::now() + chrono::seconds(15);
    cout << " OK\n";

    cout << "[DEBUG] Entering main loop...\n";
    while (chrono::high_resolution_clock::now() < endTime)
    {
        cout << "[DEBUG] Creating frame..." << flush;
        Mat frame = Mat::zeros(frameHeight, frameWidth, CV_8UC3);
        cout << " OK, setting color..." << flush;
        frame = Scalar(30, 30, 30);
        cout << " OK" << endl;

        frameCount++;

        cout << "[DEBUG] Getting landmarks..." << flush;
        auto landmarks = service.get_landmarks();
        cout << " OK" << endl;

        if (landmarks)
        {
            detectionCount++;
            handDetected = true;

            vector<Point> points = landmarksToPixels(*landmarks, frameWidth, frameHeight);
            drawConnections(frame, points);
            for (const auto &point : points)
            {
                drawLandmark(frame, point);
            }
        }
        else
        {
            handDetected = false;
        }

        if (frameCount % 10 == 0)
        {
            auto now = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = now - lastFpsUpdate;
            fps = 10.0 / elapsed.count();
            lastFpsUpdate = now;
        }

        drawHUD(frame, fps, detectionCount, handDetected, frameCount);

        // Save every Nth frame
        if (frameCount % saveInterval == 0 || (handDetected && frameCount % 10 == 0))
        {
            ostringstream filename;
            filename << "visualizer_output/frame_" << setfill('0') << setw(6) << frameCount << ".jpg";
            imwrite(filename.str(), frame);
            cout << "📸 Saved: " << filename.str() << " (FPS: " << fixed << setprecision(1) << fps << ")" << endl;
        }

        this_thread::sleep_for(chrono::milliseconds(10));
    }

    // Save final frame
    Mat finalFrame = Mat::zeros(frameHeight, frameWidth, CV_8UC3);
    finalFrame = Scalar(30, 30, 30);
    drawHUD(finalFrame, fps, detectionCount, false, frameCount);

    // Add summary text
    int y = frameHeight / 2 - 60;
    drawTextWithBackground(finalFrame, "SESSION COMPLETE!", Point(frameWidth / 2 - 150, y), 0.8, 2);
    y += 50;
    drawTextWithBackground(finalFrame, "Frames: " + to_string(frameCount), Point(frameWidth / 2 - 80, y));
    y += 40;
    drawTextWithBackground(finalFrame, "Detections: " + to_string(detectionCount), Point(frameWidth / 2 - 100, y));
    y += 40;
    ostringstream rateText;
    rateText << "Rate: " << fixed << setprecision(1) << (100.0 * detectionCount / frameCount) << "%";
    drawTextWithBackground(finalFrame, rateText.str(), Point(frameWidth / 2 - 60, y));

    imwrite("visualizer_output/SUMMARY.jpg", finalFrame);

    auto totalTime = chrono::high_resolution_clock::now() - startTime;
    double duration = chrono::duration<double>(totalTime).count();

    cout << "\n╔════════════════════════════════════════════════════════╗\n";
    cout << "║              Session Summary                           ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n";
    cout << "  Total frames: " << frameCount << "\n";
    cout << "  Hand detections: " << detectionCount << "\n";
    cout << "  Detection rate: " << fixed << setprecision(1) << (100.0 * detectionCount / frameCount) << "%\n";
    cout << "  Average FPS: " << (frameCount / duration) << "\n";
    cout << "  Duration: " << static_cast<int>(duration) << " seconds\n";
    cout << "\n📁 Check the 'visualizer_output' folder for saved frames!\n";
    cout << "🎉 Visualization complete!\n\n";

    return 0;
}
