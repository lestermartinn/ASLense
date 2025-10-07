/*
 * ASL Hand Recognition - Text-Only Monitor
 * Displays hand detection in terminal (no OpenCV visualization)
 */

#include "landmark_service.hpp"
#include <iostream>
#include <chrono>
#include <thread>
#include <iomanip>

using namespace std;

void clearScreen()
{
    // ANSI escape code to clear screen
    cout << "\033[2J\033[1;1H";
}

void printHandASCII(bool detected)
{
    if (detected)
    {
        cout << "       _____\n";
        cout << "      |     |\n";
        cout << "   ___|  |  |___\n";
        cout << "  |   |  |  |   |\n";
        cout << "  |   |  |  |   |\n";
        cout << "  |___|  |  |___|\n";
        cout << "      |     |\n";
        cout << "      |_____|\n";
        cout << "\n   ✋ HAND DETECTED!\n";
    }
    else
    {
        cout << "\n\n\n";
        cout << "       _____\n";
        cout << "      |     |\n";
        cout << "      |     |\n";
        cout << "      |     |\n";
        cout << "      |_____|\n";
        cout << "\n   👁️  Waiting for hand...\n";
    }
}

int main()
{
    cout << "\n╔════════════════════════════════════════════════════════╗\n";
    cout << "║  ASL Hand Recognition - Live Monitor                  ║\n";
    cout << "║  Text-based hand tracking display                     ║\n";
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

    cout << "Press Ctrl+C to quit\n";
    cout << "Showing live hand detection for 30 seconds...\n\n";
    this_thread::sleep_for(chrono::seconds(2));

    int detectionCount = 0;
    int frameCount = 0;
    auto startTime = chrono::high_resolution_clock::now();
    auto lastUpdate = startTime;
    auto endTime = startTime + chrono::seconds(30);

    while (chrono::high_resolution_clock::now() < endTime)
    {
        frameCount++;
        auto landmarks = service.get_landmarks();
        bool detected = landmarks.has_value();

        if (detected)
        {
            detectionCount++;
        }

        // Update display every 10 frames (about 3 times per second)
        if (frameCount % 10 == 0)
        {
            clearScreen();

            cout << "╔════════════════════════════════════════════════════════╗\n";
            cout << "║       ASL HAND RECOGNITION - LIVE TRACKING            ║\n";
            cout << "╚════════════════════════════════════════════════════════╝\n\n";

            printHandASCII(detected);

            auto now = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = now - startTime;
            double fps = frameCount / elapsed.count();
            double detectRate = (frameCount > 0) ? (100.0 * detectionCount / frameCount) : 0.0;
            int remaining = 30 - static_cast<int>(elapsed.count());

            cout << "\n┌────────────────────────────────────────────────────────┐\n";
            cout << "│  STATISTICS                                            │\n";
            cout << "├────────────────────────────────────────────────────────┤\n";
            cout << "│  Frames Processed:  " << setw(6) << frameCount << "                           │\n";
            cout << "│  Detections:        " << setw(6) << detectionCount << "                           │\n";
            cout << "│  Detection Rate:    " << fixed << setprecision(1) << setw(5) << detectRate << "%                         │\n";
            cout << "│  FPS:               " << fixed << setprecision(1) << setw(5) << fps << "                          │\n";
            cout << "│  Time Remaining:    " << setw(3) << max(0, remaining) << "s                           │\n";
            cout << "└────────────────────────────────────────────────────────┘\n\n";

            if (detected && landmarks)
            {
                cout << "📍 Sample Landmark Data (first 3 points):\n";
                for (int i = 0; i < 3 && i < 21; i++)
                {
                    float x = (*landmarks)[i * 3];
                    float y = (*landmarks)[i * 3 + 1];
                    float z = (*landmarks)[i * 3 + 2];
                    cout << "   Point " << i << ": ("
                         << fixed << setprecision(3) << x << ", "
                         << y << ", " << z << ")\n";
                }
            }

            cout << "\n💡 TIP: Show your hand clearly to the camera!\n";
            cout << "⏱️  Time remaining: " << max(0, remaining) << " seconds\n";
        }

        this_thread::sleep_for(chrono::milliseconds(10));
    }

    clearScreen();

    auto totalTime = chrono::high_resolution_clock::now() - startTime;
    double duration = chrono::duration<double>(totalTime).count();
    double avgFps = frameCount / duration;
    double detectRate = (frameCount > 0) ? (100.0 * detectionCount / frameCount) : 0.0;

    cout << "\n╔════════════════════════════════════════════════════════╗\n";
    cout << "║              SESSION COMPLETE! 🎉                      ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    cout << "📊 Final Statistics:\n";
    cout << "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    cout << "  Total Frames:       " << frameCount << "\n";
    cout << "  Hand Detections:    " << detectionCount << "\n";
    cout << "  Detection Rate:     " << fixed << setprecision(1) << detectRate << "%\n";
    cout << "  Average FPS:        " << fixed << setprecision(1) << avgFps << "\n";
    cout << "  Duration:           " << static_cast<int>(duration) << " seconds\n";
    cout << "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

    if (detectRate > 50)
    {
        cout << "✅ Great! Hand tracking is working well!\n\n";
    }
    else if (detectRate > 20)
    {
        cout << "⚠️  Moderate detection. Try better lighting and hand visibility.\n\n";
    }
    else
    {
        cout << "❌ Low detection rate. Check camera and show hand clearly.\n\n";
    }

    cout << "🎉 Phase 1 Complete - Hand Landmark Extraction Working!\n\n";
    cout << "Next: Phase 2 - ML Model Training\n";
    cout << "  → Collect ASL dataset\n";
    cout << "  → Train neural network\n";
    cout << "  → Integrate TensorFlow Lite\n\n";

    return 0;
}
