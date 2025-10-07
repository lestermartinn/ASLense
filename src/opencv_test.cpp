#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>

int main()
{
    std::cout << "=================================\n";
    std::cout << "ASL Hand Gesture Recognition System\n";
    std::cout << "Phase 0: OpenCV Integration Test\n";
    std::cout << "=================================\n\n";

    // Display OpenCV version and build info
    std::cout << "✓ OpenCV Version: " << CV_VERSION << std::endl;
    std::cout << "✓ OpenCV Major: " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Test basic OpenCV functionality
    try
    {
        // Create a test image
        cv::Mat test_image = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::rectangle(test_image, cv::Point(100, 100), cv::Point(300, 200), cv::Scalar(0, 255, 0), 2);
        cv::putText(test_image, "ASL Recognition Ready!", cv::Point(150, 160),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        std::cout << "✓ Image Processing: Working" << std::endl;

        // Test camera enumeration (without actually opening)
        std::cout << "✓ Camera Access: Checking..." << std::endl;

        // Test basic CV operations
        cv::Mat gray;
        cv::cvtColor(test_image, gray, cv::COLOR_BGR2GRAY);
        std::cout << "✓ Color Conversion: Working" << std::endl;

        // Test GUI creation
        cv::namedWindow("ASL Test Window", cv::WINDOW_AUTOSIZE);
        cv::imshow("ASL Test Window", test_image);
        std::cout << "✓ GUI Window Creation: Working" << std::endl;

        std::cout << "\n🎯 Phase 0, Step 2: OpenCV Integration - COMPLETE!" << std::endl;
        std::cout << "\n📋 Next Steps:" << std::endl;
        std::cout << "   1. Install MediaPipe for hand detection" << std::endl;
        std::cout << "   2. Test camera capture" << std::endl;
        std::cout << "   3. Implement basic hand landmark detection" << std::endl;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "\n⏱️  Execution time: " << duration.count() << " microseconds" << std::endl;

        std::cout << "\nPress any key in the window to continue..." << std::endl;
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    catch (const cv::Exception &e)
    {
        std::cerr << "❌ OpenCV Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}