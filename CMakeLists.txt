# ASL Hand Gesture Recognition System
# CMakeLists.txt - Phase 0: OpenCV Integration

cmake_minimum_required(VERSION 3.15)
project(ASLRecognition VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")

# Create executables
add_executable(asl_recognition
    src/main.cpp
)

add_executable(opencv_test
    src/opencv_test.cpp
)

add_executable(integration_test
    src/landmark_service.cpp
    src/integration_test.cpp
)

# Link OpenCV libraries
target_link_libraries(opencv_test ${OpenCV_LIBS})
target_link_libraries(integration_test ${OpenCV_LIBS})

# Set output directory
set_target_properties(asl_recognition opencv_test integration_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Compiler-specific options
if(MSVC)
    target_compile_options(asl_recognition PRIVATE /W4)
    target_compile_options(opencv_test PRIVATE /W4)
else()
    target_compile_options(asl_recognition PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(opencv_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Print build information
message(STATUS "Building ASL Recognition System")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")