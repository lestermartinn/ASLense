#include "landmark_service.hpp"
#include <iostream>
#include <cstring>
#include <algorithm>

#ifdef _WIN32
#define POPEN _popen
#define PCLOSE _pclose
#else
#define POPEN popen
#define PCLOSE pclose
#endif

namespace asl
{

    LandmarkService::LandmarkService(const std::string &python_exe, const std::string &script_path)
        : python_exe_(python_exe), script_path_(script_path), process_pipe_(nullptr), running_(false), current_fps_(0.0), frames_processed_(0)
    {
    }

    LandmarkService::~LandmarkService()
    {
        stop();
    }

    bool LandmarkService::start()
    {
        if (running_)
        {
            return true;
        }

        // Build command with proper quoting for Windows paths with spaces
        // Redirect stderr to NUL to suppress MediaPipe/TensorFlow warnings
        std::string command = "\"\"" + python_exe_ + "\" \"" + script_path_ + "\"\" 2>NUL";

        std::cout << "Starting landmark service..." << std::endl;

        // Open pipe to Python process
        process_pipe_ = POPEN(command.c_str(), "r");

        if (!process_pipe_)
        {
            std::cerr << "Failed to start Python service" << std::endl;
            return false;
        }

        running_ = true;
        std::cout << "✓ Landmark service started" << std::endl;

        return true;
    }

    void LandmarkService::stop()
    {
        if (process_pipe_)
        {
            PCLOSE(process_pipe_);
            process_pipe_ = nullptr;
        }
        running_ = false;
    }

    std::optional<ServiceMessage> LandmarkService::read_message()
    {
        if (!running_ || !process_pipe_)
        {
            return std::nullopt;
        }

        char buffer[4096];
        if (fgets(buffer, sizeof(buffer), process_pipe_) != nullptr)
        {
            std::string line(buffer);

            // Remove trailing newline
            if (!line.empty() && line.back() == '\n')
            {
                line.pop_back();
            }

            return parse_message(line);
        }

        return std::nullopt;
    }

    std::optional<LandmarkFeatures> LandmarkService::get_landmarks()
    {
        // Read all available messages and update state
        while (auto msg = read_message())
        {
            if (msg->type == MessageType::LANDMARKS && msg->landmarks)
            {
                latest_landmarks_ = msg->landmarks;
            }
            else if (msg->type == MessageType::NO_HAND)
            {
                latest_landmarks_ = std::nullopt;
            }
            else if (msg->type == MessageType::STATS)
            {
                current_fps_ = msg->fps;
                frames_processed_ = msg->frames_processed;
            }
        }

        return latest_landmarks_;
    }

    std::optional<ServiceMessage> LandmarkService::parse_message(const std::string &json_line)
    {
        ServiceMessage msg;

        // Extract basic fields
        std::string type_str = extract_string_value(json_line, "type");
        msg.timestamp = extract_double_value(json_line, "timestamp");
        msg.frame = extract_int_value(json_line, "frame");

        // Determine message type
        if (type_str == "ready")
        {
            msg.type = MessageType::READY;
        }
        else if (type_str == "landmarks")
        {
            msg.type = MessageType::LANDMARKS;

            // Extract the features array from nested "data" object
            // JSON structure: {"type": "landmarks", "data": {"features": [...]}}
            size_t data_start = json_line.find("\"data\": {");
            if (data_start != std::string::npos)
            {
                std::string data_section = json_line.substr(data_start);
                auto features = extract_array_value(data_section, "features");
                if (features.size() == 63)
                {
                    LandmarkFeatures lm_features;
                    std::copy(features.begin(), features.end(), lm_features.begin());
                    msg.landmarks = lm_features;
                }
            }
        }
        else if (type_str == "no_hand")
        {
            msg.type = MessageType::NO_HAND;
        }
        else if (type_str == "stats")
        {
            msg.type = MessageType::STATS;
            // Extract from nested "data" object
            size_t data_start = json_line.find("\"data\": {");
            if (data_start != std::string::npos)
            {
                std::string data_section = json_line.substr(data_start);
                msg.fps = extract_double_value(data_section, "fps");
                msg.frames_processed = extract_int_value(data_section, "frames_processed");
            }
        }
        else if (type_str == "error")
        {
            msg.type = MessageType::ERROR;
            msg.error_message = extract_string_value(json_line, "message");
        }
        else if (type_str == "shutdown")
        {
            msg.type = MessageType::SHUTDOWN;
        }
        else
        {
            msg.type = MessageType::UNKNOWN;
        }

        return msg;
    }

    // Simple JSON parsers (lightweight, no external library needed)

    std::string LandmarkService::extract_string_value(const std::string &json, const std::string &key)
    {
        std::string search = "\"" + key + "\": \"";
        size_t start = json.find(search);
        if (start == std::string::npos)
        {
            return "";
        }
        start += search.length();

        size_t end = json.find("\"", start);
        if (end == std::string::npos)
        {
            return "";
        }

        return json.substr(start, end - start);
    }

    double LandmarkService::extract_double_value(const std::string &json, const std::string &key)
    {
        std::string search = "\"" + key + "\": ";
        size_t start = json.find(search);
        if (start == std::string::npos)
        {
            return 0.0;
        }
        start += search.length();

        // Find the end (comma, brace, or bracket)
        size_t end = json.find_first_of(",}]", start);
        if (end == std::string::npos)
        {
            return 0.0;
        }

        std::string value_str = json.substr(start, end - start);
        try
        {
            return std::stod(value_str);
        }
        catch (...)
        {
            return 0.0;
        }
    }

    int LandmarkService::extract_int_value(const std::string &json, const std::string &key)
    {
        return static_cast<int>(extract_double_value(json, key));
    }

    std::vector<float> LandmarkService::extract_array_value(const std::string &json, const std::string &key)
    {
        std::vector<float> result;

        std::string search = "\"" + key + "\": [";
        size_t start = json.find(search);
        if (start == std::string::npos)
        {
            return result;
        }
        start += search.length();

        size_t end = json.find("]", start);
        if (end == std::string::npos)
        {
            return result;
        }

        std::string array_content = json.substr(start, end - start);

        // Parse comma-separated values
        std::istringstream iss(array_content);
        std::string token;

        while (std::getline(iss, token, ','))
        {
            try
            {
                result.push_back(std::stof(token));
            }
            catch (...)
            {
                // Skip invalid values
            }
        }

        return result;
    }

} // namespace asl
