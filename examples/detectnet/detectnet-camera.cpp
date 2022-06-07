/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *////
#include "function.hpp"
#include "MJPEGWriter.h"
#include "Detection.h"

using namespace cv;
using namespace std;

#define SIZE 1024


/**
 * @attention these are global variables
 * @var img: 초기에 처리하는 영상데이터. draw.cpp와 sharing
 * @var RoiVtx: 초기에 처리하는 검지영역, draw.cpp와 sharing
 * @var signal_received: 프로그램 종료 신호를 체크하는 변수
 * @class MJPEGWriter: MJPEG 스트리밍을 하기위한 클래스, 기본 포트는 7777, MJPEGWriter.cpp와 sharing
 * @class Detection: 좌회전 감응 시스템 클래스, Detection.h와 sharing
 */
extern cv::Mat img;
extern std::vector <cv::Point> RoiVtx;
bool signal_received = false;
MJPEGWriter test(7777);
Detection detection;

extern string *StringSplit(string strOrigin, string strTok);
int usage() {
    printf("usage: detectnet-camera [-h] [--network NETWORK] [--camera CAMERA]\n");
    printf("                        [--width WIDTH] [--height HEIGHT]\n\n");
    printf("Locate objects in a live camera stream using an object detection DNN.\n\n");
    printf("optional arguments:\n");
    printf("  --help           show this help message and exit\n");
    printf("  --camera CAMERA  index of the MIPI CSI camera to use (NULL for CSI camera 0),\n");
    printf("                   or for VL42 cameras the /dev/video node to use (/dev/video0).\n");
    printf("                   by default, MIPI CSI camera 0 will be used.\n");
    printf("  --width WIDTH    desired width of camera stream (default is 1280 pixels)\n");
    printf("  --height HEIGHT  desired height of camera stream (default is 720 pixels)\n\n");
    printf("%s\n", detectNet::Usage());

    return 0;
}

void sig_handler(int signo) {
    if (signo == SIGINT) {
        printf("received SIGINT\n");
        signal_received = true;
        test.stop();
        exit(1);
    }
}

int main(int argc, char **argv) {

    /**
     * @var dzWaitSeconds: 최소점유대기시간, 해당시간 이상을 점유하면 CAN컨트롤러에 감응 신호를 보낸다. 기본 값은 3(초)
     * @var dzLeaveSeconds: 최소미점유대기시간, 검지영역에서 벗어난 시간으로부터
     *                      해당시간 이상지났으면 검지신호(uart)를 미감응상태(uart=false)로 변경한다. 기본 값은 1(초)
     */
    const int dzWaitSeconds = 3;
    const int dzLeaveSeconds = 1;


    /**
     * @var search_exec_proc: 영상처리 프로그램이 실행중인지 확인하는 디렉토리의 경로
     * @example 프로그램 실행시 이미 폴더가 생성이 되었으면 삭제
     */

    std::string search_exec_proc = "/home/user/jetson-inference/dbict/control/run";
    if (fileExists(search_exec_proc)) system("rmdir /home/user/jetson-inference/dbict/control/run");

    commandLine cmdLine(argc, argv);

    /**
     * @var fps_clock, @var duration
     * @example 영상처리 프로그램 프레임 확인시 필요한 변수
     */

    double fps_clock = 0;
    double duration = 0;

    /**
     * @var uart:
     * @example 실제 감응 신호를 보내는 변수, 미감응 0, 감응 1
     * @var total_frame: 프레임 카운트
     */

    unsigned char uart = 0;
    unsigned char total_frame = 0;


    /**
     * @var connect_LMB: LMB가 연결됐는지 확인하는 변수
     * @var connect_CAN: CAN이 연결됐는지 확인하는 변수
     * @example LMB는 /home/user/jetson-inference/dbict/control/lmberr 가 있으면 lmb 연결 실패
     * @example CAN은 /home/user/jetson-inference/dbict/control/canerr 가 있으면 can 연결 실패
     */

    unsigned char connect_LMB = 0;
    unsigned char connect_CAN = 0;

    /**
     * @var car_in
     * @example 1시간 마다 좌회전 신호의 수를 로깅할 때 쓰는 변수
     */
    int car_in = 0;


    /**
     * @var record_path: 영상처리 화면을 저장할 때 사용하는 부모 폴더                                /home/user/record
     * @var log_path: 검지 로그를 저장할 때 사용하는 폴더                                          /home/user/detlog
     * @var record_dir_by_date: 영상파일을 녹화할때 사용하는 폴더, 날짜로 분류                     ex) /home/user/record/300101
     * @var record_file_by_date: 해당날짜에 저장하게 될 현재 시간 값을 가진 영상파일                ex) /home/user/record/300101/00.mp4
     * @var log_dir_by_date: 검지 로그를 저장할 때 사용하는 폴더, 날짜로 분류                      ex) /home/user/detlog/300101
     * @var pic_dir_by_date: 검지한 사진을 분류하는 폴더, 날짜로 분류                             ex) /home/user/detlog/300101/pic
     * @var b_pic_dir_by_date: 화질이 좋지 않은(표준편차가 낮은) 검지사진을 분류하는 폴더, 날짜로 분류   ex) /home/user/detlog/300101/badpic
     */

    std::string record_path = "/home/user/record";
    std::string log_path = "/home/user/detlog";
    std::string record_dir_by_date = record_path + "/" + to_date();
    std::string record_file_by_date = record_dir_by_date + '/' + to_hour() + ".mp4";
    std::string log_dir_by_date = log_path + "/" + to_date();
    std::string pic_dir_by_date = log_dir_by_date + "/pic";
    std::string b_pic_dir_by_date = log_dir_by_date + "/badpic";



    /// @example 프로그램 실행 시 위 디렉토리들 중 생성되지 않은 폴더들을 생성.

    if (!fileExists(record_path)) mkdir(record_path.c_str(), 0777);
    if (!fileExists(log_path)) mkdir(log_path.c_str(), 0777);
    if (!fileExists(record_dir_by_date)) mkdir(record_dir_by_date.c_str(), 0777);
    if (!fileExists(log_dir_by_date)) mkdir(log_dir_by_date.c_str(), 0777);
    if (!fileExists(pic_dir_by_date)) mkdir(pic_dir_by_date.c_str(), 0777);
    if (!fileExists(b_pic_dir_by_date)) mkdir(b_pic_dir_by_date.c_str(), 0777);


    /**
     * @var ctl_dir: 좌회전 감응 프로그램을 제어 또는 강제로 작동시킬 때 사용하는 폴더
     * @var uarton: 검지여부와 관계 없이 강제로 검지 신호를 줄 때 사용하는 폴더, 생성이 되면 강제 감응, 없을 시 정상운용
     * @var uartoff: 검지여부와 관계 없이 강제로 검지 신호를 주지 않을 때 사용 하는 폴더.
     *          차량이 검지영역에 들어와 검지를 해도 미검지로 처리, 생성이 되면 강제 미검지, 없을 시 정상운용
     * @var mod: 센터에서 kafka로 검지영역을 수정해서 전달할 때 영역 수정이 되었음을 알려줄 때 사용하는 폴더, 생성이 되면 검지영역 변경
     * @var std: 센터에서 kafka로 표준편차의 값을 수정해서 전달할 때 편차 값이 수정되었음을 알려줄 때 사용하는 폴더, 생성이 되면 표준편차 값 변경
     * @var canerr: CAN 컨트롤러가 연결돼 있는지 확인할 때 사용하는 폴더, 생성이 되면 연결 실패, jetson-inference/dbict/python/uart.py와 연결돼 있다.
     * @var lmberr: LMB가 연결돼 있는지 확인할 때 사용하는 폴더, 생성이 되면 연결 실패, jetson-inference/dbict/python/uart.py와 연결돼 있다.
     */


    std::string ctl_dir = "/home/user/jetson-inference/dbict/control/";
    std::string uarton = ctl_dir + "uarton";
    std::string uartoff = ctl_dir + "uartoff";
    std::string mod = ctl_dir + "mod";
    std::string std = ctl_dir + "std";
    std::string canerr = ctl_dir + "canerr";
    std::string lmberr = ctl_dir + "lmberr";


    /// @var video: opencv에서의 영상 데이터의 변수
    VideoWriter video;

    /**
     * @var stdDevValues: 영상의 표준편차를 확인할 때 사용하는 변수.
     * @deprecated meanValues는 사용하지 않는다.
     */
    cv::Scalar meanValues, stdDevValues;

    /**
     * @class CSharedMemory
     * @example 1000번지에 공유메모리를 256바이트만큼 초기화한다.
     * @example 검지 여부를 write.
     * @see shared-memory.cpp 참조
     */
    CSharedMemory m;
    m.setKey(0x1000);
    m.setupSharedMemory(256);
    m.attachSharedMemory();


    /**
     * @var region: 검지 영역의 좌표를 저장한 텍스트파일, 영역의 x값, y값을 한칸 씩 띄어서 저장,
     * @example region.txt = 99 68 125 178 282 163 239 47 으로 저장되어 있음
     *
     * @fn draw_region: 검지영역을 그려주는 메소드
     *  @param region
     * @return void
     * @example region.txt이 존재하면 그 좌표로 검지영역을 그리고 없으면 새로 검지영역을 그린 후 region.txt에 좌표를 저장
     * @see draw.cpp의 draw_region() 참조
     */
    std::string region("/home/user/jetson-inference/dbict/control/region.txt");
    draw_region(region);


    /**
     * @var stdnum: 표준편차 값이 들어있는 텍스트 파일의 경로,
     *
     * @fn get_stddev: 텍스트파일에 저장된 표준편차을 가져오는 메소드
     *  @param stdnum
     * @return int
     *
     * @example default는 10, 해당 값보다 영상의 표준편차가 낮으면 강제 감응
     * @see utility.cpp의 get_stddev() 참조
     */

    std::string stdnum("/home/user/jetson-inference/dbict/control/stddev.txt");
    int stddev = get_stddev(stdnum);


    if (cmdLine.GetFlag("help"))
        return usage();

    /*
     * attach signal handler
     */
    if (signal(SIGINT, sig_handler) == SIG_ERR) {
        printf("\ncan't catch SIGINT\n");
        test.stop();
    }

    /*
     * create the camera device
     */
    gstCamera *camera = gstCamera::Create(cmdLine.GetInt("width", gstCamera::DefaultWidth),
                                          cmdLine.GetInt("height", gstCamera::DefaultHeight),
                                          cmdLine.GetString("camera"));

    if (!camera) {
        printf("\ndetectnet-camera:  failed to initialize camera device\n");
        return 0;
    }

    printf("\ndetectnet-camera:  successfully initialized camera device\n");
    printf("    width:  %u\n", camera->GetWidth());
    printf("   height:  %u\n", camera->GetHeight());
    // printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());

    /*
     * create detection network
     */
    detectNet *net = detectNet::Create(argc, argv);

    if (!net) {
        printf("detectnet-camera:   failed to load detectNet model\n");
        return 0;
    }

    // create openGL windowRect

    // glDisplay* display = glDisplay::Create();

    // if( !display )
    // 	printf("detectnet-camera:  failed to create openGL display\n");

    // start streaming/

    if (!camera->Open()) {
        printf("detectnet-camera:  failed to open camera for streaming\n");
        return 0;
    }

    printf("detectnet-camera:  camera open for streaming\n");

    /*
     * processing loop
     */


    /**
     * @fn save_nextvideo: 프로그램 실행 시 영상을 생성해 초기화하는 메소드
     *  @param video: 영상 데이터
     *  @param record_file_by_date: 시간 별로 분류한 영상 파일의 이름
     *  @param record_dir_by_date: 시간 별로 분류한 영상 파일을 저장할 폴더
     *
     * @example am 10시에 예상치못한 에러로 프로그램이 재시작하여 해당하는 시간에 존재하는 영상파일이 있다면
     * @example 영상은 opencv상에서 붙일 수 없기 때문에
     * @example 영상파일 뒤에 '_1'를 추가해 10_1.mp4을 생성한 후 영상을 저장한다.
     */
    save_nextvideo(video, record_file_by_date, record_dir_by_date);


    /// @fn MJPEGWriter.start(): MJPEG 스트리밍을 시작하는 메서드
    test.start();

    while (!signal_received) {


        /// 프레임 카운트
        total_frame++;

        /**
         * @var ForcedSignal: 강제 감응 신호가 작동하는지 확인하는 변수
         * @var ForcedNotSignal: 강제 미감응 신호가 작동하는지 확인하는 변수
         */

        bool ForcedSignal = 0;
        bool ForcedNotSignal = 0;

        /// @var duration: 영상의 fps를 계산할 때 사용하는 변수

        duration = static_cast<double>(cv::getTickCount());

        /**
         * @var white_bg: 영상처리 화면 상단에 시간과 기타 정보를 렌더링 할 때 사용하는 bg, 사이즈는 320 x 25
         * @image jetson-inference/dbict/picture/white.jpg
         */

        cv::Mat white_bg = imread("/home/user/jetson-inference/dbict/picture/white.jpg", IMREAD_COLOR);
        if (white_bg.empty()) {
            cout << "could not open or find the image" << endl;
            return -1;
        }


        /**
         * @attention 정시지나 날짜가 바뀌거나 한 시간 단위로 시간이 바뀌게 되면
         * @attention 영상처리 프로그램이 돌아가는 중에도 폴더 또는 파일을 새로 생성해야 하기 때문에
         * @attention 저장하는 폴더와 파일의 경로를 체크해야 한다.
         *
         * @var record_path: 영상처리 화면을 저장할 때 사용하는 부모 폴더                                /home/user/record
         * @var log_path: 검지 로그를 저장할 때 사용하는 폴더                                          /home/user/detlog
         * @var record_dir_by_date: 영상파일을 녹화할때 사용하는 폴더, 날짜로 분류                     ex) /home/user/record/300101
         * @var record_file_by_date: 해당날짜에 저장하게 될 현재 시간 값을 가진 영상파일                ex) /home/user/record/300101/00.mp4
         * @var log_dir_by_date: 검지 로그를 저장할 때 사용하는 폴더, 날짜로 분류                      ex) /home/user/detlog/300101
         * @var pic_dir_by_date: 검지한 사진을 분류하는 폴더, 날짜로 분류                             ex) /home/user/detlog/300101/pic
         * @var b_pic_dir_by_date: 화질이 좋지 않은(표준편차가 낮은) 검지사진을 분류하는 폴더, 날짜로 분류   ex) /home/user/detlog/300101/badpic
         * @var log_by_date: 검지 로그를 저장하는 텍스트 파일
         */

        std::string record_path = "/home/user/record";
        std::string log_path = "/home/user/detlog";
        std::string record_dir_by_date  = record_path + "/" + to_date();
        std::string record_file_by_date = record_dir_by_date + '/' + to_hour() + ".mp4";
        std::string log_dir_by_date = log_path + "/" + to_date();
        std::string pic_dir_by_date = log_dir_by_date + "/pic";
        std::string b_pic_dir_by_date = log_dir_by_date + "/badpic";
        std::string log_by_date = log_dir_by_date + "/" + to_hour() + "_log.txt";

        /**
         * @var log_file: log_by_date 경로의 파일을 검지 로그를 저장하기 위해 open 한다.
         */

        std::ofstream log_file;
        log_file.open(log_by_date, std::ios_base::app);


        /**
         * @example 영상 저장 부모 폴더가 없으면 폴더 새로 생성
         * @example 로그 저장 부모 폴더가 없으면 폴더 새로 생성
         */

        if (!fileExists(record_path)) mkdir(record_path.c_str(), 0777);
        if (!fileExists(log_path)) mkdir(log_path.c_str(), 0777);


        /// @example 프로그램 운용 중, record_dir_by_date가 사라지면 날짜가 바뀐 것이기 때문에 로깅 폴더들을 재생성한다.
        if (!fileExists(record_dir_by_date)) {
            std::cout << "the current time is 00:00" << std::endl;
            std::cout << "change directory" << std::endl;
            mkdir(record_dir_by_date.c_str(), 0777);
            mkdir(log_dir_by_date.c_str(), 0777);
            mkdir(pic_dir_by_date.c_str(), 0777);
            mkdir(b_pic_dir_by_date.c_str(), 0777);
        }

        /// @example 프로그램 운용 중, record_file_by_date가 사라지면 시간이 바뀐 것이기 때문에 로깅 폴더들을 재생성한다.
        /// @example 또한 영상 데이터의 영상 저장 파일 경로도 변경 후 초기화 한다.

        if (!fileExists(record_file_by_date)) {
            car_in = 0;
            std::cout << "--------------------------------------------------------------------------------" << std::endl;
            cout << "The current time is " << to_time() << endl;
            cout << "Change video" << endl;

            video.open(record_file_by_date, VideoWriter::fourcc('a', 'v', 'c', '1'), 30,
                       cv::Size(det_width, det_height), true);

            if (!video.isOpened()) {
                cout << to_hour() << ".mp4 out failed" << endl;
                return -1;
            }
        }

        /**
         * @fn c_region: change region. 센터에서 변경한 검지영역로 적용
         * @fn stddev_modify: 표준편차를 센터에서 변경한 표준편차로 적용
         */

        c_region(region, mod); // changed region drew
        stddev_modify(std, stddev);



        /// @var CarInWaitZone: 검지영역에 오브젝트가 검지되면 true
        bool CarInWaitZone = false;

        // capture RGBA image
        float *imgRGBA = NULL;


        if (!camera->CaptureRGBA(&imgRGBA, 1000, 1))
            printf("detectnet-camera:  failed to capture RGBA image from camera\n");


        // detect objects in the frame
        detectNet::Detection *detections = NULL;
        const int numDetections = net->Detect(imgRGBA, camera->GetWidth(), camera->GetHeight(), &detections);



        /**
         * 1 @var cv_img: 실제로 처리될 영상 데이터
         * 2 @var last_img: MJPEG 스트리밍과 출력을 하게될 최종 영상 데이터
         * 3 @example 32FC4 타입을 8UC3 타입으로 변환, 32fc4는 명암(alpha)이 포함된 4채널의 RBGA이고 8UC3은 BGR 타입이다.
         * 4 @example 실제 영상데이터의 컬러도 RGBA to BGR로 변환
         * 5 @fn meanStdDev: 영상 데이터의 표준편차를 @var stdDevValues에 저장
         * 6 @var det_img: 원본 영상 데이터을 복사, 실제 카메라 영상이며 이미지 데이터를 수집할 때 사용
         * 7 @fn resize: cv_img를 320 x 240으로 resize한 것을 det_img에 적용
         */

        cv::Mat cv_img = cv::Mat(camera->GetHeight(), camera->GetWidth(), CV_32FC4, imgRGBA);
        cv::Mat last_img;
        cv_img.convertTo(cv_img, CV_8UC3);
        cv::cvtColor(cv_img, cv_img, COLOR_RGBA2BGR);
        meanStdDev(cv_img, meanValues, stdDevValues);
        cv::Mat det_img = cv_img.clone();
        cv::resize(cv_img, det_img, cv::Size(320, 240), 1);


        /// @example 10진수 27 = ASCII 'ESC'
        if (waitKey(10) == 27) {
            video.release();
            signal_received = true;
        }

        /**
         * @fn draw_polygon: 검지영역을 화면에 그리는 메서드
         *  @param cv_img: 원본 데이터
         *  @param RoiVtx: 검지영역의 좌표
         *  @param SCALAR_WHITE: BGR값으로 된 WHITE색의 스칼라
         */
        draw_polygon(cv_img, RoiVtx, SCALAR_WHITE);


        /// @var img_stddev: double형으로 되어있는 stdDevValues의 BGR값 중 G값을 int형으로 변환
        int img_stddev = int(stdDevValues[1]);

        /**
         * @var str_stddev: int형인 img_stddev를 string형으로 변환, 이미지 파일 저장할 때 사용
         * @var normal_pic: 카메라 표준편차가 높은(품질이 좋은) 이미지 데이터를 저장할 때 사용하는 파일 경로
         * @var bad_pic: 카메라 표준편차가 낮은(품질이 낮은) 이미지 데이터를 저장할 때 사용하는 파일 경로
         */

        std::string str_stddev = to_string(int(stdDevValues[1]));
        std::string normal_pic = pic_dir_by_date + "/" + current_datetime() + "_" + str_stddev + ".jpg";
        std::string bad_pic = b_pic_dir_by_date + "/" + current_datetime() + "_" + str_stddev + ".jpg";


        /// @example 영상처리 중인 것을 확인하기 위해 run폴더를 생성
        if (!fileExists(search_exec_proc)) system("mkdir /home/user/jetson-inference/dbict/control/run");



        if (numDetections > 0) {

            /**
             * @var numDetections: 영상 내에 존재하는 object의 개수
             * @var rect: 영상 내에 존재하는 object의 bounding box
             */

            for (int n = 0; n < numDetections; n++) {
                Rect rect(int(detections[n].Left), int(detections[n].Top),
                          int(detections[n].Right - detections[n].Left), int(detections[n].Bottom - detections[n].Top));
                std::string res;

                /**
                 * @fn DoesROIOverlap: 검지영역에 객체가 있는지 확인하는 메서드
                 *  @param rect: 오브젝트의 bounding box
                 *  @param RoiVtx: 검지영역 좌표
                 *  @param res: 검지 영역과 오브젝트의 상관관계를 확인하기 위한 임계 값 출력 용도의 string타입 파라미터, 연구/개발에 사용
                 *
                 * @example DoesROIOverlap이 true이면 검지, 그렇지 않으면 미검지
                 * @see draw.cpp의 DoesROIOverlap() 참고
                 */

                if (DoesROIOverlap(rect, RoiVtx, res)) {
                    cv::rectangle(cv_img, rect, SCALAR_BLUE, 2);
                    CarInWaitZone = true;
                } else {
                    cv::rectangle(cv_img, rect, SCALAR_RED, 2);
                }
            }

            /// @example 오브젝트를 검지한 상태
            if (CarInWaitZone) {

                /**
                 * @fn OnDetection: 검지 상태가 되었음을 선언하는 메서드
                 * @fn ResetOffDetection: 미검지 상태를 초기화
                 */
                detection.OnDetection();
                detection.ResetOffDetection();

                /**
                 * @fn OnSeconds: 점유시간을 카운트하는 메서드, OnDetection이 선언된 직후부터 카운트가 되기 시작한다.
                 * @var dzWaitSeconds: 최소점유대기시간, 해당시간 이상을 점유하면 CAN컨트롤러에 감응 신호를 보낸다. 기본 값은 3(초)
                 * @var uart: CAN 컨트롤러에 보낼 감응신호, 감응상태는 1, 미감응상태는 0
                 *
                 * @example 점유시간(OnSeconds)이 최소점유대기시간(dzWaitSeconds)를 넘어가면 검지신호(uart)를 감응상태(uart=true)로 변경한다.
                 */
                if (detection.OnSeconds() >= dzWaitSeconds) {
                    uart = 1;

                    draw_polygon(cv_img, RoiVtx, SCALAR_GREEN);

                    /// @fn Logging: 점유 상태를 파일에 로깅한다.
                    detection.Logging(car_in, log_file);

                    /// @example 표준편차가 20 이상일 때 normal_pic으로 분류, 이하일 때 bad_pic으로 분류
                    if (img_stddev > 20) cv::imwrite(normal_pic, det_img);
                    else cv::imwrite(bad_pic, det_img);
                }
            } else {

                draw_polygon(cv_img, RoiVtx, SCALAR_WHITE);

                /// @fn OffDetection: 미점유 상태가 되었음을 선언하는 메서드
                detection.OffDetection();
            }

            /// @fn getOffDetect: 미점유상태의 여부를 가져오는 메서드, true이면 미점유상태
            if (detection.getOffDetect()) {

                /**
                 * @fn OffSeconds: 미점유시간을 카운트하는 메서드, OffDetection이 선언된 직후부터 카운트가 되기 시작한다.
                 * @var dzLeaveSeconds: 최소미점유대기시간, 검지영역에서 벗어난 시간으로부터
                 *                      해당시간 이상지났으면 검지신호(uart)를 미감응상태(uart=false)로 변경한다. 기본 값은 1(초)
                 * @fn AllReset: 모든 검지상태를 초기화한다.
                 */
                if (uart && detection.OffSeconds() > dzLeaveSeconds) {
                    detection.AllReset();
                    uart = 0;
                }
            }

            /**
             * @example 교통량이 매우 많거나, 과검지 및 오검지로 인해 제대로 하지 못하는 경우, 검지상태나 미검지상태가 제대로 초기화가 되지 않을 수도 있기 때문에
             * @example 검지상태와 미검지상태가 중첩이 되었을 때 모든 검지상태를 초기화한다.
             */

            if (detection.getOnDetect() && detection.getOffDetect()) {
                if (detection.OnSeconds() < 3 && detection.OffSeconds() > 3) {
                    detection.AllReset();
                    uart = 0;
                }
            }
        }

        /**
         * @fn EmptyDetection: 화면에 오브젝트가 없으면 비어있음을 선언
         * @fn EmptySeconds: 오브젝트가 없음을 표시, EmptyDetection이 선언된 직후부터 카운트가 되기 시작한다.
         *
         * @example 오브젝트가 없을 때 1초마다 모든 상태 초기화
         */
        if (numDetections == 0) {
            detection.EmptyDetection();
            if (detection.EmptySeconds() > 1) {
                detection.AllReset();
                uart = 0;

            }
        }

        /**
         * @fn check_bad_weather: 화질의 품질이 떨어졌는 지 표준편차를 이용해 계산하는 메서드
         *  @param img_stddev: 영상데이터의 표준편차
         *  @param stddev: 텍스트파일에 지정된 표준편차, 해당 값보다 img_stddev의 값이 내려가면 check_bad_weather = true
         *  @param log_file: 강제감응신호 발생을 로깅하는 파일
         */
        if (check_bad_weather(img_stddev, stddev, log_file)) {
            uart = 1;
            draw_polygon(cv_img, RoiVtx, SCALAR_YELLOW);
        }

        /// @example 강제감응 신호를 요청하는 폴더가 존재하는 경우 강제감응 발생
        if (fileExists(uarton)) {
            uart = 1;
            draw_polygon(cv_img, RoiVtx, SCALAR_GREEN);
            ForcedSignal = 1;
        }

        /// @example 강제미감응 신호를 요청하는 폴더가 존재하는 경우 강제미감응 발생
        if (fileExists(uartoff)) {
            uart = 1;
            draw_polygon(cv_img, RoiVtx, SCALAR_RED);
            ForcedNotSignal = 1;
        }


        /**
         * @example 영상 상단에 출력되는 정보.
         * @example 시간, 표준편차, CAN연결여부, LMB연결여부를 출력
         */
        std::ostringstream out;
        out.str("");
        out << fixed;
        out.precision(2);
        out << current_datetime();
        out << " Std:" << stdDevValues[1];
        if (fileExists(canerr)) {
            out << "  CAN: X";
            connect_CAN = 0;
        } else {
            out << "  CAN: O";
            connect_CAN = 1;
        }
        if (fileExists(lmberr)) {
            out << "  LMB: X";
            connect_LMB = 0;
        } else {
            out << "  LMB: O";
            connect_LMB = 1;
        }
        cv::putText(white_bg, out.str(), cv::Point2f(5, 16), cv::FONT_HERSHEY_COMPLEX, 0.42, cv::Scalar(0, 0, 0));


        /// @example 프레임 초기화
        if(total_frame == 255) total_frame = 0;

        /// @example 감응신호 여부를 SharedMemory에 write
        m.copyToSharedMemory(total_frame, uart);

        cv::resize(cv_img, cv_img, cv::Size(320, 240), 1);

        /// @fn vconcat: 시간, 표준편차등의 정보를 출력하는 상단(white_bg)과 영상 데이터(cv_img)를 세로로 잇는 메서드
        cv::vconcat(white_bg, cv_img, last_img);

        /**
         * @attention 영상처리하고있는 정보의 메타데이터를 화면의 픽셀데이터에 출력하기 위한 로직
         * @fn line: y=0의 가로로된 선을 전부 black으로 초기화한다.
         * @fn SendStatusValueInToPixel: 특정 값들을 픽셀데이터화 시키는 메서드
         *  @param last_img: 픽셀데이터를 입력시킬 최종 영상데이터
         *  @param ...MetaData: 아래와 같은 매개변수들을 최종 영상데이터에 입력한다.
         *
         * @see draw.cpp의 SendStatusValueInToPixel() 참조
         */
        line(last_img, Point(0, 0), Point(320, 0), cv::Scalar(0, 0, 0), 1);
        SendStatusValueInToPixel(last_img, RoiVtx, uart, connect_LMB, connect_CAN, ForcedSignal, ForcedNotSignal, stddev);

        /// @fn imshow: 영상데이터를 화면에 출력한다.
        cv::imshow("destination image", last_img);

        /**
         * @example video.write: 영상데이터를 파일에 저장한다.
         * @example test.write: 영상데이터를 MJPEG 형태로 스트리밍한다.
         */
        video.write(last_img);
        test.write(last_img);


        /// @example 영상처리 프로그램의 fps를 계산하는 로직
        duration = static_cast<double>(cv::getTickCount()) - duration;
        duration = duration / cv::getTickFrequency();
        fps_clock = 1 / duration;
//        std::cout << "FPS: " << fps_clock << endl;
#if 0
#endif
    }
    return 0;
}
