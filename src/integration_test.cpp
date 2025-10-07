#include <iostream>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>
#include "landmark_service.hpp"

int main()
{
    std::cout << "=================================\n";
    std::cout << "ASL Hand Recognition System\n";
    std::cout << "Phase 1: C++ <-> Python Integration Test\n";
    std::cout << "=================================\n\n";

    // Paths to Python executable and script
    std::string python_exe = "C:/Users/leste/OneDrive/Desktop/cs-projects/asl hand recognition/.venv/Scripts/python.exe";
    std::string script_path = "python/landmark_service.py"; // Real MediaPipe service

    std::cout << "Starting MediaPipe landmark service...\n";

    // Create and start the service
    asl::LandmarkService service(python_exe, script_path);

    if (!service.start())
    {
        std::cerr << "❌ Failed to start landmark service\n";
        return -1;
    }

    std::cout << "✓ Service started\n\n";
    std::cout << "📋 Real-time hand tracking active!\n";
    std::cout << "   🖐️  Show your hand to the camera\n";
    std::cout << "   📹 Camera should be opening now...\n";
    std::cout << "   ⌨️  Press Ctrl+C to exit\n";
    std::cout << "   ⏱️  Test will run for 30 seconds or 100 detections\n\n";

    int hand_detected_count = 0;
    int no_hand_count = 0;
    auto start_time = std::chrono::steady_clock::now();

    try
    {
        while (true)
        {
            // Get latest landmarks
            auto landmarks = service.get_landmarks();

            if (landmarks)
            {
                hand_detected_count++;

                // Print first few features
                if (hand_detected_count % 30 == 1)
                { // Every 30 detections
                    std::cout << "\n✓ Hand detected! Landmark features:\n";
                    std::cout << "   [";
                    for (int i = 0; i < 9; ++i)
                    {
                        std::cout << landmarks->at(i);
                        if (i < 8)
                            std::cout << ", ";
                    }
                    std::cout << ", ...]\n";
                    std::cout << "   Total detections: " << hand_detected_count << "\n";
                    std::cout << "   FPS: " << service.get_fps() << "\n";
                }
            }
            else
            {
                no_hand_count++;

                if (no_hand_count % 100 == 1)
                {
                    std::cout << "⚠️  No hand detected (show hand to camera)\n";
                }
            }

            // Small delay to avoid busy waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            // Check for exit condition (run for 30 seconds or until 100 detections)
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

            if (elapsed > 30 || hand_detected_count >= 100)
            {
                std::cout << "\n✓ Test complete!\n";
                break;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "❌ Error: " << e.what() << "\n";
    }

    // Summary
    std::cout << "\n=================================\n";
    std::cout << "Test Summary:\n";
    std::cout << "=================================\n";
    std::cout << "Hand detections: " << hand_detected_count << "\n";
    std::cout << "No hand frames: " << no_hand_count << "\n";
    std::cout << "Service FPS: " << service.get_fps() << "\n";
    std::cout << "Frames processed: " << service.get_frames_processed() << "\n";

    if (hand_detected_count > 0)
    {
        std::cout << "\n🎉 Phase 1, Step 2: Integration - SUCCESS!\n";
        std::cout << "\n📋 Next Steps:\n";
        std::cout << "   1. Create OpenCV visualization with landmarks\n";
        std::cout << "   2. Implement ML model training pipeline\n";
        std::cout << "   3. Add TensorFlow Lite inference\n";
    }
    else
    {
        std::cout << "\n⚠️  No hands detected during test\n";
        std::cout << "   Make sure your camera is working and visible to the app\n";
    }

    std::cout << "\nStopping service...\n";
    service.stop();
    std::cout << "✓ Service stopped\n";

    return 0;
}
