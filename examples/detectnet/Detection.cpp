//
// Created by Feus on 2022/01/12.
//

#include "Detection.h"

void Detection::UseCount() {
    std::shared_ptr <Detection> ptr = this->detect;
    std::cout << "use count : " << ptr.use_count() << std::endl;

}


/* On Function */
int Detection::OnDetection() {

    if(!this->on_detect){
        std::shared_ptr <Detection> ptr(new Detection());
        ptr.get()->setStartClock(std::chrono::system_clock::now());
        this->detect = ptr;
        this->on_detect = true;
    }

    return 1;
}

int Detection::OnSeconds() {
    std::shared_ptr <Detection> ptr = this->detect;
    ptr.get()->setEndClock(std::chrono::system_clock::now() - ptr.get()->getStartClock());
    return static_cast<int>(ptr.get()->getEndClock().count());
}


void Detection::ResetOnDetection() {
    if(this->on_detect) {
        std::shared_ptr <Detection> ptr = this->detect;
        ptr.reset();
        this->on_detect = false;

        if(this->logger) this->logger = false;
    }
}


/* Off Function */
int Detection::OffDetection() {

    if(!this->off_detect){
        std::shared_ptr <Detection> ptr(new Detection());
        ptr.get()->setStartClock(std::chrono::system_clock::now());
        this->off = ptr;
        this->off_detect = true;
    }

    return 1;
}


int Detection::OffSeconds() {

    std::shared_ptr <Detection> ptr = this->off;
    ptr.get()->setEndClock(std::chrono::system_clock::now() - ptr.get()->getStartClock());
    return static_cast<int>(ptr.get()->getEndClock().count());

}

void Detection::ResetOffDetection() {

    if(this->off_detect) {
        std::shared_ptr <Detection> ptr = this->off;
        ptr.reset();
        this->off_detect = false;
    }
}


/* Empty Function */

int Detection::EmptyDetection() {

    if(!this->empty_detect){
        std::shared_ptr <Detection> ptr(new Detection());
        ptr.get()->setStartClock(std::chrono::system_clock::now());
        this->empty = ptr;
        this->empty_detect = true;
    }

    return 1;
}


int Detection::EmptySeconds() {

    std::shared_ptr <Detection> ptr = this->empty;
    ptr.get()->setEndClock(std::chrono::system_clock::now() - ptr.get()->getStartClock());
    return static_cast<int>(ptr.get()->getEndClock().count());

}

void Detection::ResetEmptyDetection() {
    if(this->empty_detect) {
        std::shared_ptr <Detection> ptr = this->empty;
        ptr.reset();
        this->empty_detect = false;
    }
}



/* */
void Detection::AllReset() {

    if (this->detect != nullptr)  this->detect.reset();
    if (this->off != nullptr) this->off.reset();
    if (this->empty != nullptr) this->empty.reset();
    if (this->on_detect) this->on_detect = false;
    if (this->off_detect) this->off_detect = false;
    if (this->empty_detect) this->empty_detect = false;
    if (this->logger) this->logger = false;
}


void Detection::Logging(int &car_in, std::ofstream &log_file){

    if(!this->logger) {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream date_time;
        date_time << std::put_time(std::localtime(&in_time_t), "%m-%d_%X");
        car_in++;
        std::cout << "Detect : " << car_in << "\t" << date_time.str() << std::endl;
        log_file << "Detect : " << car_in << "\t" << date_time.str() << std::endl;
        this->logger = true;
    }

}