#ifndef LANDMARK_SERVICE_HPP
#define LANDMARK_SERVICE_HPP

#include <string>
#include <vector>
#include <array>
#include <memory>
#include <optional>

// JSON parsing (we'll use a simple approach for now)
#include <sstream>
#include <cstdio>

namespace asl
{

    /**
     * Represents the 63-element feature vector from hand landmarks
     * Format: [x1,y1,z1, x2,y2,z2, ..., x21,y21,z21]
     */
    using LandmarkFeatures = std::array<float, 63>;

    /**
     * Message types from the landmark service
     */
    enum class MessageType
    {
        READY,
        LANDMARKS,
        NO_HAND,
        STATS,
        ERROR,
        SHUTDOWN,
        UNKNOWN
    };

    /**
     * Parsed message from the Python landmark service
     */
    struct ServiceMessage
    {
        MessageType type;
        double timestamp;
        int frame;

        // Data fields (depend on message type)
        std::optional<LandmarkFeatures> landmarks;
        std::string error_message;
        double fps;
        int frames_processed;
    };

    /**
     * Interface to the Python MediaPipe landmark extraction service.
     * Manages the Python subprocess and communication via JSON.
     */
    class LandmarkService
    {
    public:
        LandmarkService(const std::string &python_exe, const std::string &script_path);
        ~LandmarkService();

        // Disable copy
        LandmarkService(const LandmarkService &) = delete;
        LandmarkService &operator=(const LandmarkService &) = delete;

        /**
         * Start the Python service subprocess
         */
        bool start();

        /**
         * Stop the Python service
         */
        void stop();

        /**
         * Check if the service is running
         */
        bool is_running() const { return running_; }

        /**
         * Read the next message from the service (non-blocking)
         * Returns nullopt if no message available
         */
        std::optional<ServiceMessage> read_message();

        /**
         * Get the latest landmark features
         * Returns nullopt if no hand detected or service not ready
         */
        std::optional<LandmarkFeatures> get_landmarks();

        /**
         * Get current FPS from the service
         */
        double get_fps() const { return current_fps_; }

        /**
         * Get total frames processed
         */
        int get_frames_processed() const { return frames_processed_; }

    private:
        std::string python_exe_;
        std::string script_path_;
        FILE *process_pipe_;
        bool running_;

        // Latest state
        std::optional<LandmarkFeatures> latest_landmarks_;
        double current_fps_;
        int frames_processed_;

        /**
         * Parse a JSON message line
         */
        std::optional<ServiceMessage> parse_message(const std::string &json_line);

        /**
         * Simple JSON value extraction (lightweight, no external deps)
         */
        std::string extract_string_value(const std::string &json, const std::string &key);
        double extract_double_value(const std::string &json, const std::string &key);
        int extract_int_value(const std::string &json, const std::string &key);
        std::vector<float> extract_array_value(const std::string &json, const std::string &key);
    };

} // namespace asl

#endif // LANDMARK_SERVICE_HPP
