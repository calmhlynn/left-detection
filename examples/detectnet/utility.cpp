#include "function.hpp"

extern std::vector<cv::Point> RoiVtx;

extern int number_frame;
extern int foggy_or_storm_frame;
extern int normal_frame;

void stddev_modify(const std::string &std, int &stddev) {
    if (fileExists(std)) {
        std::ifstream std("/home/user/jetson-inference/dbict/control/stddev.txt");
        std::string get_std;
        getline(std, get_std);
        stddev = stoi(get_std);
        system("rmdir /home/user/jetson-inference/dbict/control/std");
    }
}


int get_stddev(const std::string &std_file){

    std::ifstream std_num(std_file);
    std::string get_std;
    getline(std_num, get_std);
    return stoi(get_std);
}


void save_nextvideo(cv::VideoWriter &video, const std::string &record_video_file, const std::string &record_dir_by_date){

    if (fileExists(record_video_file)) {
        char fName[100];
        char gName[100];
        char sys_rm[100];

        // check video file size
        FILE *f;
        FILE *g;
        strcpy(fName, record_video_file.c_str());
        f = fopen(fName, "r");
        fseek(f, 0, SEEK_END);
        int fileLength = ftell(f);
        fclose(f);

        if (fileLength < 100) {
            std::string remove_cli = "rm " + record_video_file;
            strcpy(sys_rm, remove_cli.c_str());
            system(sys_rm);

            video.open(record_video_file, VideoWriter::fourcc('a','v','c','1'), save_video_fps, cv::Size(det_width,det_height), true);
        }else{
            for (int i = 1; i < 10; ++i) {
                std::string next_video_file = record_dir_by_date + "/" + to_hour() + "_" + to_string(i) + ".mp4";

                if (fileExists(next_video_file)) {
                    strcpy(gName, next_video_file.c_str());
                    g = fopen(gName, "r");
                    fseek(g, 0, SEEK_END);
                    int fileLengths = ftell(g);
                    fclose(g);

                    if (fileLength < 100) {
                        std::string remove_cli = "rm " + next_video_file;
                        strcpy(sys_rm, remove_cli.c_str());
                        system(sys_rm);

                        video.open(record_video_file, VideoWriter::fourcc('a','v','c','1'), save_video_fps, cv::Size(det_width,det_height), true);
                        break;
                    }
                }else{
                    video.open(record_video_file, VideoWriter::fourcc('a','v','c','1'), save_video_fps, cv::Size(det_width,det_height), true);
                }
            }
        }
    }
}


void c_region(const std::string &region, const std::string &std_file){

    if(fileExists(std_file)){
        char poly[100];
        std::ifstream c_region(region);
        if(c_region.is_open()){
            RoiVtx.clear();
            while(!c_region.eof()){
                std::string position;
                getline(c_region,position);
                string* points = new string[256];
                points = StringSplit(position," ");
                for(int i=0; i <= 30 ; i++){
                    poly[i*2] = atoi(points[i*2].c_str());
                    poly[i*2+1] = atoi(points[i*2+1].c_str());
                    if(poly[i*2] != 0 && poly[i*2+1] != 0){
                        RoiVtx.push_back(Point(poly[i*2],poly[i*2+1]));
                    }
                }
            }
        }
        std::string rmdir_cli = "rmdir " + std_file;
        system(rmdir_cli.c_str());
    }
}


bool check_bad_weather(int img_stddev, int get_cfg_stddev, std::ofstream &log_file) {

    if (img_stddev < get_cfg_stddev) {
        foggy_or_storm_frame++;
        normal_frame = 0;
        if (foggy_or_storm_frame >= one_second * 2) {
            normal_frame = 0;
            if (foggy_or_storm_frame == one_second * 2) {
                std::cout << "start bad weather:\t" << current_datetime() << "\tStdDev:\t"
                          << img_stddev << endl;
                log_file << "start bad weather:\t" << current_datetime() << "\tStdDev:\t"
                         << img_stddev << endl;

            }
            return 1;
        }
    } else {
        normal_frame++;
        return 0;
    }
    if (foggy_or_storm_frame > one_second * 2 && normal_frame > one_second * 3) {
        foggy_or_storm_frame = 0;
        std::cout << "stop bad weather:\t" << current_datetime() << "\tStdDev:\t" << img_stddev
                  << endl;
        log_file << "stop bad weather:\t" << current_datetime() << "\tStdDev:\t" << img_stddev
                 << endl;
    }
    if (foggy_or_storm_frame < one_second * 2 && normal_frame > one_second * 3) foggy_or_storm_frame = 0;

    return 0;
}