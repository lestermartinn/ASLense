#include "asl_recognition_service.hpp"
#include <iostream>
#include <sstream>
#include <cstring>

// Simple JSON parsing (finds values after keys)
static std::string extract_string_value(const std::string &json, const std::string &key)
{
    std::string search = "\"" + key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos)
        return "";

    pos += search.length();
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '"'))
        pos++;

    size_t end = pos;
    while (end < json.length() && json[end] != '"' && json[end] != ',' && json[end] != '}')
        end++;

    return json.substr(pos, end - pos);
}

static double extract_number_value(const std::string &json, const std::string &key)
{
    std::string search = "\"" + key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos)
        return 0.0;

    pos += search.length();
    while (pos < json.length() && json[pos] == ' ')
        pos++;

    return std::stod(json.substr(pos));
}

static std::vector<float> extract_array_value(const std::string &json, const std::string &key)
{
    std::vector<float> result;

    std::string search = "\"" + key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos)
        return result;

    pos += search.length();
    size_t start = json.find('[', pos);
    size_t end = json.find(']', start);

    if (start == std::string::npos || end == std::string::npos)
        return result;

    std::string array_content = json.substr(start + 1, end - start - 1);
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

ASLRecognitionService::ASLRecognitionService() : pipe_(nullptr)
{
    // Get paths
    python_path_ = R"(C:/Users/leste/OneDrive/Desktop/cs-projects/asl hand recognition/.venv/Scripts/python.exe)";
    script_path_ = R"(C:/Users/leste/OneDrive/Desktop/cs-projects/asl hand recognition/python/recognition_service.py)";
}

ASLRecognitionService::~ASLRecognitionService()
{
    stop();
}

bool ASLRecognitionService::start()
{
    if (pipe_)
    {
        std::cerr << "Service already running" << std::endl;
        return false;
    }

    // Build command
    std::string command = "\"\"" + python_path_ + "\" \"" + script_path_ + "\"\" 2>NUL";

    // Open pipe
    pipe_ = _popen(command.c_str(), "r");

    if (!pipe_)
    {
        std::cerr << "Failed to start recognition service" << std::endl;
        return false;
    }

    // Wait for ready message OR any valid message (service might start very quickly)
    // NOTE: Python's JSON has spaces after colons, so search for "type": not "type":
    std::string line;
    for (int i = 0; i < 10; i++)
    {
        if (read_line(line))
        {
            // Check if it's a ready message
            if (line.find("\"type\": \"ready\"") != std::string::npos)
            {
                std::cout << "✓ Recognition service started (ready message received)" << std::endl;
                return true;
            }
            // Or check if it's any other valid message (no_hand, prediction)
            // This means service is already running
            if (line.find("\"type\": \"no_hand\"") != std::string::npos ||
                line.find("\"type\": \"prediction\"") != std::string::npos)
            {
                std::cout << "✓ Recognition service started (already processing frames)" << std::endl;
                return true;
            }
        }
        else
        {
            break; // No more data available
        }
    }

    std::cerr << "Service failed to send ready message" << std::endl;
    std::cerr << "Last line read: " << line << std::endl;
    stop();
    return false;
}

void ASLRecognitionService::stop()
{
    if (pipe_)
    {
        _pclose(pipe_);
        pipe_ = nullptr;
    }
}

bool ASLRecognitionService::read_line(std::string &line)
{
    if (!pipe_)
        return false;

    char buffer[8192];
    if (fgets(buffer, sizeof(buffer), pipe_))
    {
        line = buffer;
        // Remove trailing newline
        if (!line.empty() && line.back() == '\n')
        {
            line.pop_back();
        }
        return true;
    }

    return false;
}

bool ASLRecognitionService::parse_message(const std::string &json, Prediction &pred)
{
    // Extract type
    std::string msg_type = extract_string_value(json, "type");

    if (msg_type == "no_hand")
    {
        pred.hand_detected = false;
        pred.letter = "";
        pred.confidence = 0.0;
        pred.landmarks.clear();
        pred.frame_number = static_cast<int>(extract_number_value(json, "frame"));
        pred.timestamp = extract_number_value(json, "timestamp");
        return true;
    }

    if (msg_type == "prediction")
    {
        pred.hand_detected = true;
        pred.frame_number = static_cast<int>(extract_number_value(json, "frame"));
        pred.timestamp = extract_number_value(json, "timestamp");

        // Find the "data" object
        size_t data_pos = json.find("\"data\":");
        if (data_pos != std::string::npos)
        {
            std::string data_section = json.substr(data_pos);

            pred.letter = extract_string_value(data_section, "letter");
            pred.confidence = static_cast<float>(extract_number_value(data_section, "confidence"));
            pred.landmarks = extract_array_value(data_section, "landmarks");

            return pred.landmarks.size() == 63; // Verify we got all landmarks
        }
    }

    return false;
}

bool ASLRecognitionService::get_prediction(Prediction &pred)
{
    if (!pipe_)
        return false;

    std::string line;
    if (read_line(line))
    {
        return parse_message(line, pred);
    }

    return false;
}
