@echo off
echo Building ASL Recognition Application...

g++ -std=c++17 -Wall -Wextra -O2 ^
    src/asl_recognition_service.cpp ^
    src/asl_recognition.cpp ^
    -o bin/asl_recognition.exe

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ Build successful!
    echo.
    echo Run: bin\asl_recognition.exe
) else (
    echo.
    echo ✗ Build failed!
)
