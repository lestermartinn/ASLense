@echo off
echo ========================================
echo ASL Recognition - Integration Test Build
echo ========================================

REM Add MinGW to PATH
set PATH=%PATH%;C:\msys64\mingw64\bin

echo Compiling C++ integration test...
g++ -std=c++17 -Wall -Wextra -O2 ^
    -IC:\msys64\mingw64\include\opencv4 ^
    -IC:\msys64\mingw64\include\opencv4\opencv2 ^
    src\landmark_service.cpp ^
    src\integration_test.cpp ^
    -LC:\msys64\mingw64\lib ^
    -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs ^
    -o bin\integration_test.exe

if exist bin\integration_test.exe (
    echo ✓ Compilation successful!
    echo.
    echo Running integration test...
    echo.
    bin\integration_test.exe
) else (
    echo ❌ Compilation failed!
    pause
    exit /b 1
)

pause
