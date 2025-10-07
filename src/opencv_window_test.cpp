/*
 * Test OpenCV window creation
 */

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    cout << "Test 1: Creating window..." << endl;

    try
    {
        namedWindow("Test Window", WINDOW_NORMAL);
        cout << " OK" << endl;

        cout << "Test 2: Creating image..." << endl;
        Mat img = Mat::zeros(480, 640, CV_8UC3);
        img = Scalar(50, 50, 50);
        cout << " OK" << endl;

        cout << "Test 3: Displaying image..." << endl;
        imshow("Test Window", img);
        cout << " OK" << endl;

        cout << "Test 4: Waiting for key (press any key to continue)..." << endl;
        waitKey(2000); // Wait 2 seconds
        cout << " OK" << endl;

        destroyAllWindows();
        cout << "\nAll OpenCV tests passed!" << endl;
    }
    catch (const exception &e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
