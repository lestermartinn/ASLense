/*
 * Minimal test to isolate the visualizer issue
 */

#include "landmark_service.hpp"
#include <iostream>

using namespace std;

int main()
{
    cout << "Test 1: Basic output" << endl;

    cout << "Test 2: Creating service object..." << flush;
    string python_exe = "C:/Users/leste/OneDrive/Desktop/cs-projects/asl hand recognition/.venv/Scripts/python.exe";
    string script_path = "C:/Users/leste/OneDrive/Desktop/cs-projects/asl hand recognition/python/landmark_service.py";
    asl::LandmarkService service(python_exe, script_path);
    cout << " OK" << endl;

    cout << "Test 3: Starting service..." << flush;
    bool started = service.start();
    cout << (started ? " OK" : " FAILED") << endl;

    if (started)
    {
        cout << "Test 4: Reading one message..." << flush;
        auto msg = service.read_message();
        cout << (msg.has_value() ? " OK" : " FAILED") << endl;

        if (msg.has_value())
        {
            cout << "Message type: " << static_cast<int>(msg->type) << endl;
        }
    }

    cout << "\nAll tests complete!" << endl;
    return 0;
}
