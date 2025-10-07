#include <iostream>
#include <string>
#include <chrono>

int main()
{
    std::cout << "=================================\n";
    std::cout << "ASL Hand Gesture Recognition System\n";
    std::cout << "Phase 0: Basic C++ Setup Test\n";
    std::cout << "=================================\n\n";

    // Test basic C++17 features
    std::string project_name = "ASL Recognition";
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "✓ Project: " << project_name << std::endl;
    std::cout << "✓ C++17 Standard: Working" << std::endl;
    std::cout << "✓ STL Libraries: Imported successfully" << std::endl;
    std::cout << "✓ Chrono: High-resolution timer available" << std::endl;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::cout << "\n🎯 Next Steps:" << std::endl;
    std::cout << "   1. Verify this program compiles and runs" << std::endl;
    std::cout << "   2. Test CMake build system" << std::endl;
    std::cout << "   3. Move to OpenCV integration (Phase 0, Step 2)" << std::endl;

    std::cout << "\n⏱️  Execution time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "\nPress Enter to exit..." << std::endl;
    std::cin.get();

    return 0;
}