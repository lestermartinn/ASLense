@echo off
echo ========================================
echo ASL Hand Recognition - Build Script
echo ========================================

REM Add MinGW to PATH for this session
set PATH=%PATH%;C:\msys64\mingw64\bin

echo Checking build environment...
g++ --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ MinGW g++ not found. Please install MSYS2 and MinGW-w64
    echo Run: winget install msys2.msys2
    echo Then: pacman -S mingw-w64-x86_64-gcc
    pause
    exit /b 1
)

echo ✓ MinGW g++ compiler found

REM Create directories
if not exist bin mkdir bin
if not exist obj mkdir obj

echo Compiling ASL Recognition System...
g++ -std=c++17 -Wall -Wextra -O2 src\main.cpp -o bin\asl_recognition.exe

if exist bin\asl_recognition.exe (
    echo ✓ Compilation successful!
    echo Running program...
    echo.
    bin\asl_recognition.exe
) else (
    echo ❌ Compilation failed!
    pause
    exit /b 1
)

echo.
echo ✓ Phase 0, Step 1: Basic C++ Setup - COMPLETE!
echo Next: Install OpenCV for computer vision
pause