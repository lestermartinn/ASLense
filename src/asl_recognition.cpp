#include "asl_recognition_service.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>

int main()
{
    std::cout << "======================================================================" << std::endl;
    std::cout << "  ASL RECOGNITION - C++ CLIENT" << std::endl;
    std::cout << "======================================================================" << std::endl;
    std::cout << std::endl;

    // Create service
    ASLRecognitionService service;

    // Start service
    std::cout << "Starting recognition service..." << std::endl;
    if (!service.start())
    {
        std::cerr << "Failed to start service" << std::endl;
        return 1;
    }

    std::cout << std::endl;
    std::cout << "Recognition active! Make ASL letters in front of camera." << std::endl;
    std::cout << "Press Ctrl+C to stop." << std::endl;
    std::cout << std::endl;
    std::cout << "======================================================================" << std::endl;
    std::cout << std::endl;

    // Statistics
    int frame_count = 0;
    int detection_count = 0;
    int high_confidence_count = 0;
    std::string last_letter = "";
    int same_letter_count = 0;

    auto start_time = std::chrono::steady_clock::now();

    // Main loop
    while (true)
    {
        ASLRecognitionService::Prediction pred;

        if (service.get_prediction(pred))
        {
            frame_count++;

            if (pred.hand_detected)
            {
                detection_count++;

                // Display prediction
                std::cout << "Frame " << std::setw(5) << pred.frame_number << " | ";

                if (pred.confidence > 0.9)
                {
                    std::cout << "\033[32m"; // Green
                    high_confidence_count++;
                }
                else if (pred.confidence > 0.7)
                {
                    std::cout << "\033[33m"; // Yellow
                }
                else
                {
                    std::cout << "\033[31m"; // Red
                }

                std::cout << "Letter: " << pred.letter
                          << " | Confidence: " << std::fixed << std::setprecision(1)
                          << (pred.confidence * 100) << "%\033[0m";

                // Track repeated predictions
                if (pred.letter == last_letter)
                {
                    same_letter_count++;
                    if (same_letter_count >= 10)
                    {
                        std::cout << " ← STABLE!";
                    }
                }
                else
                {
                    last_letter = pred.letter;
                    same_letter_count = 1;
                }

                std::cout << std::endl;
            }
            else
            {
                // No hand detected - only show occasionally
                if (frame_count % 30 == 0)
                {
                    std::cout << "Frame " << std::setw(5) << pred.frame_number
                              << " | No hand detected" << std::endl;
                }
            }

            // Show statistics every 100 frames
            if (frame_count % 100 == 0)
            {
                auto now = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

                std::cout << std::endl;
                std::cout << "--- Statistics ---" << std::endl;
                std::cout << "  Frames processed: " << frame_count << std::endl;
                std::cout << "  Hands detected: " << detection_count
                          << " (" << (detection_count * 100 / frame_count) << "%)" << std::endl;
                std::cout << "  High confidence: " << high_confidence_count
                          << " (" << (high_confidence_count * 100 / std::max(1, detection_count)) << "%)" << std::endl;
                std::cout << "  Runtime: " << duration << " seconds" << std::endl;
                if (duration > 0)
                {
                    std::cout << "  FPS: " << (frame_count / duration) << std::endl;
                }
                std::cout << std::endl;
            }
        }
        else
        {
            std::cerr << "Failed to get prediction" << std::endl;
            break;
        }
    }

    // Final statistics
    std::cout << std::endl;
    std::cout << "======================================================================" << std::endl;
    std::cout << "  SESSION COMPLETE" << std::endl;
    std::cout << "======================================================================" << std::endl;
    std::cout << "  Total frames: " << frame_count << std::endl;
    std::cout << "  Detections: " << detection_count << std::endl;
    std::cout << "  High confidence predictions: " << high_confidence_count << std::endl;
    std::cout << "======================================================================" << std::endl;

    return 0;
}
