@echo off
echo ========================================
echo ASL Recognition - OpenCV Test Build
echo ========================================

REM Add MinGW to PATH for this session
set PATH=%PATH%;C:\msys64\mingw64\bin

echo Checking OpenCV installation...
if not exist "C:\msys64\mingw64\include\opencv4" (
    echo ❌ OpenCV headers not found. Installing...
    C:\msys64\usr\bin\bash.exe -lc "pacman -S --noconfirm mingw-w64-x86_64-opencv"
)

echo ✓ OpenCV installation found

REM Create directories
if not exist bin mkdir bin
if not exist obj mkdir obj

echo.
echo Compiling OpenCV test program...
g++ -std=c++17 -Wall -Wextra -O2 ^
    -IC:\msys64\mingw64\include\opencv4 ^
    -IC:\msys64\mingw64\include\opencv4\opencv2 ^
    src\opencv_test.cpp ^
    -LC:\msys64\mingw64\lib ^
    -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs ^
    -o bin\opencv_test.exe

if exist bin\opencv_test.exe (
    echo ✓ OpenCV compilation successful!
    echo Running OpenCV test...
    echo.
    bin\opencv_test.exe
) else (
    echo ❌ OpenCV compilation failed!
    echo.
    echo Trying with pkg-config...
    echo.
    pkg-config --cflags --libs opencv4
    echo.
    pause
    exit /b 1
)

echo.
echo ✓ Phase 0, Step 2: OpenCV Integration - COMPLETE!
echo Next: MediaPipe integration for hand tracking
pause