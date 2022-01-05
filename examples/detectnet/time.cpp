#include "function.hpp"

std::string return_current_time_and_date(){
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream dd;
    dd << std::put_time(std::localtime(&in_time_t), "%m-%d_%X");
    // ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%X");
    return dd.str();
}
std::string to_hour(){
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream zz;
    zz << std::put_time(std::localtime(&in_time_t), "%H");
    // ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%X");
    return zz.str();
}
std::string to_time(){
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream zz;
    zz << std::put_time(std::localtime(&in_time_t), "%H%M%S");
    // ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%X");
    return zz.str();
}
std::string to_date(){
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ee;
    ee << std::put_time(std::localtime(&in_time_t), "%y%m%d");
    // ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%X");
    return ee.str();
}
