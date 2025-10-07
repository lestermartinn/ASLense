@echo off
REM Launcher for ASL Visualizer - ensures DLLs are found

echo ====================================
echo   ASL Hand Recognition Visualizer
echo ====================================
echo.

REM Add MSYS64 bin to PATH for OpenCV DLLs
set PATH=C:\msys64\mingw64\bin;%PATH%

echo Starting visualizer...
echo A window will open showing hand landmarks.
echo Press Q or ESC to quit, S to save screenshots.
echo.

REM Set Qt plugin path for platform integration
set QT_QPA_PLATFORM_PLUGIN_PATH=C:\msys64\mingw64\share\qt6\plugins

REM Run the visualizer
bin\visualizer.exe

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Visualizer exited with error code: %ERRORLEVEL%
    pause
)
