//
// Created by Feus on 2022/01/12.
//
#include <chrono>
#include <memory>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
class Detection {
private:
    std::shared_ptr<Detection> detect;
    std::shared_ptr<Detection> off;
    std::shared_ptr<Detection> empty;
    std::chrono::system_clock::time_point start_clock;
    std::chrono::duration<double> end_clock;
    bool det_signal;
    bool logger;
    bool on_detect;
    bool off_detect;
    bool empty_detect;

public:
    /* Member Initializer & Constructor */
    Detection()
        : det_signal(false),logger(false), on_detect(false), off_detect(false), empty_detect(false) {}

    /* Getter */
    std::chrono::system_clock::time_point getStartClock() { return this->start_clock;}
    std::chrono::duration<double> getEndClock() { return this->end_clock;}
    bool getDetSignal() {return this-> det_signal;}
    bool getOnDetect() { return this->on_detect; }
    bool getOffDetect() { return this->off_detect; }
    bool getEmptyDetect() { return this->empty_detect; }

    /* Setter */
    void setStartClock(std::chrono::system_clock::time_point _start_clock) { this->start_clock = _start_clock; }
    void setEndClock(std::chrono::duration<double> _end_clock) { this->end_clock = _end_clock; }
    void setDetSignal(bool _det_signal) { this->det_signal = _det_signal;}

    int OnDetection();
    int OffDetection();
    int EmptyDetection();

    int OnSeconds();
    int OffSeconds();
    int EmptySeconds();

    void ResetOnDetection();
    void ResetOffDetection();
    void ResetEmptyDetection();
    void AllReset();

    void UseCount();
    void Logging(int &in_object, std::ofstream &log_file);
};

