#ifndef ASL_RECOGNITION_SERVICE_HPP
#define ASL_RECOGNITION_SERVICE_HPP

#include <string>
#include <vector>
#include <memory>
#include <cstdio>

/**
 * ASL Recognition Service Client
 * Communicates with Python recognition service via pipes
 * Receives hand landmarks + ASL letter predictions
 */
class ASLRecognitionService
{
public:
    struct Prediction
    {
        std::vector<float> landmarks; // 63 landmark values
        std::string letter;           // Predicted letter (A-Y)
        float confidence;             // Confidence score (0-1)
        bool hand_detected;           // Whether hand was detected
        int frame_number;             // Frame number
        double timestamp;             // Timestamp
    };

    ASLRecognitionService();
    ~ASLRecognitionService();

    // Start the Python service
    bool start();

    // Stop the service
    void stop();

    // Get next prediction (blocking)
    bool get_prediction(Prediction &pred);

    // Check if service is running
    bool is_running() const { return pipe_ != nullptr; }

private:
    FILE *pipe_;
    std::string python_path_;
    std::string script_path_;

    // Read one line from pipe
    bool read_line(std::string &line);

    // Parse JSON message
    bool parse_message(const std::string &json, Prediction &pred);
};

#endif // ASL_RECOGNITION_SERVICE_HPP
