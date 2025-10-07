@echo off
REM Build script for ASL Visualizer with full OpenCV support

echo ====================================
echo Building ASL Visualizer
echo ====================================
echo.

REM Compile with all required OpenCV libraries
g++ -std=c++17 -Wall -Wextra -O2 ^
    -IC:\msys64\mingw64\include\opencv4 ^
    src\landmark_service.cpp src\visualizer.cpp ^
    -LC:\msys64\mingw64\lib ^
    -lopencv_core -lopencv_imgproc -lopencv_highgui ^
    -lopencv_imgcodecs -lopencv_videoio ^
    -o bin\visualizer.exe

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ====================================
    echo Build SUCCESS!
    echo ====================================
    echo Executable: bin\visualizer.exe
    echo.
) else (
    echo.
    echo ====================================
    echo Build FAILED
    echo ====================================
    echo.
    exit /b 1
)
