#include "gstCamera.h"
#include "glDisplay.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "detectNet.h"
#include "commandLine.h"
#include "filesystem.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <array>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <signal.h>
#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>
// UART

#include <fcntl.h> // Contains file controls like O_RDWR
#include <errno.h> // Error integer and strerror() function
#include <termios.h> // Contains POSIX terminal control definitions
#include <unistd.h> // write(), read(), close()

const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);
const cv::Scalar SCALAR_BLUE = cv::Scalar(255.0, 0.0, 0.0);


const int org_width = 320;  // original video width
const int org_height = 240; // original video height

const int det_width = 320;  // detectnet program video width
const int det_height = 265; // detectnet program video height

const int save_video_fps = 30;
const int one_second = 9;
using namespace cv;
using namespace std;



class CSharedMemory
{

private :

    int m_shmid;
    key_t m_key;
    char *m_shared_memory[8];


public :

    void setShmId( int key );
    int getShmId();
    void setKey( key_t key );

    void setupSharedMemory( int size );
    void attachSharedMemory();
    void copyToSharedMemory(bool det_code);
//   void close();
};





///Time


std::string current_datetime();
std::string to_hour();
std::string to_time();
std::string to_date();




///// Draw 

extern bool DoesROIOverlap( cv::Rect rect,std::vector<cv::Point> ctr , std::string &res );
void draw_polygon(Mat src,std::vector<cv::Point> vertices, Scalar color);
bool DoesROIOverlap( cv::Rect boundingbox,std::vector<cv::Point> contour, std::string &res);
void onMouse(int event, int x, int y, int flags, void* userdata);
void SendStatusValueInToPixel(Mat &image,std::vector<cv::Point> vertices, unsigned char detected, unsigned char LMB, unsigned char CAN, bool OnSignal, bool OffSignal, unsigned char StdDev);
std::string* StringSplit(string strOrigin, string strTok);
void draw_region(const std::string &region);



/// utility

void stddev_modify(const std::string &std, int &stddev);
int get_stddev(const std::string &std_file);
void save_nextvideo(cv::VideoWriter &video, const std::string &record_video_file, const std::string &record_dir_by_date);
void c_region(const std::string &region, const std::string &std_file);
bool check_bad_weather(int img_stddev, int get_cfg_stddev, std::ofstream &log_file);