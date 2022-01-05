#include "function.hpp"

cv::Mat img;
std::vector<cv::Point> RoiVtx;



void SendStatusValueInToPixel(Mat &image,std::vector<cv::Point> vertices, unsigned char detected, unsigned char LMB, unsigned char CAN, bool OnSignal, bool OffSignal, unsigned char StdDev){

    image.at<Vec3b>(0,0)[0] = detected; // Detected Left Turn Signal                    // Blue in BGR  -> [0,0] point pixel
    image.at<Vec3b>(0,0)[1] = LMB;       // On or Off LMB                                // Green in BGR -> [0,0] point pixel
    image.at<Vec3b>(0,0)[2] = CAN;      // On or Off CAN                                // Red in BGR   -> [0,0] point pixel

    /*
     *  Forced On or Off Detect Signal                                                  // Blue in BGR  -> [0,1] point pixel
     */

    if (!OnSignal && !OffSignal){
        image.at<Vec3b>(0,1)[0] = 0;
    }else if(OnSignal && !OffSignal){
        image.at<Vec3b>(0,1)[0] = 1;
    }else{
        image.at<Vec3b>(0,1)[0] = 2;
    }


    image.at<Vec3b>(0,1)[1] = StdDev;                                                   // Green in BGR -> [0,1] point pixel
    image.at<Vec3b>(0,1)[2] = 0;                                                        // Blue in BGR  -> [0,1] point pixel
    
    image.at<Vec3b>(0,2)[0] = 0; 
    image.at<Vec3b>(0,2)[1] = 0; 
    image.at<Vec3b>(0,2)[2] = 0; 


    image.at<Vec3b>(0,3)[0] = 0; 
    image.at<Vec3b>(0,3)[1] = 0; 
    image.at<Vec3b>(0,3)[2] = 0;

    for(int i=0; i <= vertices.size(); i++){
        
        if(vertices[i].x > 100){
            image.at<Vec3b>(0,i*2+10)[0] = vertices[i].x / 100;
            image.at<Vec3b>(0,i*2+10)[1] = (vertices[i].x % 100) / 10;
            image.at<Vec3b>(0,i*2+10)[2] = (vertices[i].x % 100) % 10;
            
        }else if(100 > vertices[i].x && vertices[i].x >= 10){
            image.at<Vec3b>(0,i*2+10)[0] = 0;
            image.at<Vec3b>(0,i*2+10)[1] = vertices[i].x / 10;
            image.at<Vec3b>(0,i*2+10)[2] = vertices[i].x % 10;

        }else{
            image.at<Vec3b>(0,i*2+10)[0] = 0;
            image.at<Vec3b>(0,i*2+10)[1] = 0;
            image.at<Vec3b>(0,i*2+10)[2] = vertices[i].x;
        }

        if(vertices[i].y > 100){
            image.at<Vec3b>(0,i*2+11)[0] = vertices[i].y / 100;
            image.at<Vec3b>(0,i*2+11)[1] = (vertices[i].y % 100) / 10;
            image.at<Vec3b>(0,i*2+11)[2] = (vertices[i].y % 100) % 10;
            
        }else if(100 > vertices[i].y && vertices[i].y >= 10){
            image.at<Vec3b>(0,i*2+11)[0] = 0;
            image.at<Vec3b>(0,i*2+11)[1] = vertices[i].y / 10;
            image.at<Vec3b>(0,i*2+11)[2] = vertices[i].y % 10;

        }else{
            image.at<Vec3b>(0,i*2+11)[0] = 0;
            image.at<Vec3b>(0,i*2+11)[1] = 0;
            image.at<Vec3b>(0,i*2+11)[2] = vertices[i].y;
        }
        
    }


}

void draw_ploygon(Mat src,std::vector<cv::Point> vertices, Scalar color){

    if (vertices.size()>0) {

                for( int j = 0; j < vertices.size(); j++ ){
                    line( src, vertices[j],  vertices[(j+1)%vertices.size()], color, 2);
                }
    }
}
bool DoesROIOverlap( cv::Rect boundingbox,std::vector<cv::Point> contour, std::string &res) {

	//Get the corner points.
	
    const cv::Point *pts = (const cv::Point*) Mat(contour).data;
	int npts = Mat(contour).rows;
    double C_area =contourArea(contour);
	

    int xCenter = boundingbox.x+(boundingbox.width/2);
    int yCenter = boundingbox.y+(boundingbox.height/2);
    float AR =(float)boundingbox.width/(float)boundingbox.height;
	//std::cout<<"x:"<<boundingbox.x<<" y:"<<boundingbox.y << " w:"<<boundingbox.width<<" h:"<<boundingbox.height<<endl;

	

   // if ((pointPolygonTest(Mat(contour), Point2f(xCenter,yCenter), true) < 0))
    // if((boundingbox.x<0)||(boundingbox.y<0)||(boundingbox.x+boundingbox.width>640)||(boundingbox.y+boundingbox.height>480))
    // {
    //    res ="invalid XY";
    //   // 	std::cout<<"Feus1 :"<<res<<endl;
    //    return false;
    // }
    // if (boundingbox.area()< 1000)
    // {
    //     res="TooSmall";
    //     return false;
    // }
    // if((AR >2.0)||(AR<0.5))
    //  {
    //    res ="AR :"+to_string(AR);
    //    return false;
    // }

    int IntersectionArea = 0;

	int xmin = boundingbox.x;
    int xmax = boundingbox.x+(boundingbox.width);
    int ymin = boundingbox.y;
    int ymax = boundingbox.y+(boundingbox.height);

    for(float x=xmin; x<xmax;x++){
        for(float y=ymin; y<ymax;y++){
           if (pointPolygonTest(Mat(contour), Point2f(x,y), true) > 0)
             IntersectionArea++;
        }
    }
    int ratio = 100*IntersectionArea/boundingbox.area();

    int ratio_poly = 100*IntersectionArea/(int)C_area;

    // res =" contour size:"+to_string(C_area)+" ratio:"+to_string(ratio_poly) +"% size:"+std::to_string(boundingbox.area())+" intersect:"+std::to_string(IntersectionArea)+" ratio: "+std::to_string(ratio)+"%";
	
	if(boundingbox.area() > C_area*0.8){
		if(ratio > 20){
			res =  "bi...BIG!!!" ;
			return true;
		}
	}
    if(ratio > 30){
        res =" ration>"+to_string(C_area)+" ratio:"+to_string(ratio_poly) +"% size:"+std::to_string(boundingbox.area())+" intersect:"+std::to_string(IntersectionArea)+" ratio: "+std::to_string(ratio)+"%";

		 return true;
    }else{
		return false;
	}

    // if(IntersectionArea>20000){
    //     res =" Area>"+to_string(C_area)+" ratio:"+to_string(ratio_poly) +"% size:"+std::to_string(boundingbox.area())+" intersect:"+std::to_string(IntersectionArea)+" ratio: "+std::to_string(ratio)+"%";

	//    return true;

    // }else
    // {
    //     // res =" false:"+to_string(C_area)+" ratio:"+to_string(ratio_poly) +"% size:"+std::to_string(boundingbox.area())+" intersect:"+std::to_string(IntersectionArea)+" ratio: "+std::to_string(ratio)+"%";

    //     return false;
    // }


}

void onMouse(int event, int x, int y, int flags, void* userdata){


	if(event == EVENT_LBUTTONDOWN){
		RoiVtx.push_back(cv::Point(x,y));
	}
}
std::string* StringSplit(string strOrigin, string strTok){
	int cutAt;
	int index = 0;
	string* strResult = new string[256];

	while((cutAt = strOrigin.find_first_of(strTok)) != strOrigin.npos){
		if(cutAt > 0){
			strResult[index++] = strOrigin.substr(0, cutAt);
		}
		strOrigin = strOrigin.substr(cutAt+1);
	}
	if(strOrigin.length()>0){
		strResult[index++] = strOrigin.substr(0, cutAt);
	}
	return strResult;
}



void draw_region(const std::string &region){
    int poly[256];

    if(!fileExists(region)){
        std::string winname = "window";
        std::ifstream url("/home/user/jetson-inference/dbict/control/url.txt");
        std::string get_url;
        getline(url, get_url);
        VideoCapture cap(get_url);

        if(!cap.isOpened()){
            "VIDEO load failed";
            exit(0);
        }

        cv::namedWindow(winname);

        while(cap.isOpened()){
            int key = waitKey(100);
            ofstream edit;
            edit.open("/home/user/jetson-inference/dbict/control/region.txt", std::ios_base::out | std::ios_base::app);
            cv::moveWindow(winname, 10, 10);
            cap >> img;


            cv::waitKey(1);
            cv::setMouseCallback(winname, onMouse, NULL);
            draw_ploygon(img, RoiVtx,SCALAR_WHITE);

            if(key=='o'){
                std::cout << "Good" << endl;
                for(int i=0; i < (RoiVtx.size()) ; i++){
                    edit << RoiVtx[i].x << " " << RoiVtx[i].y << " ";
                }
                cv::destroyWindow(winname);
                break;
            }else if(key== 'x'){
                std::cout << "Region clear" << endl ;
                RoiVtx.clear();
            }
            cv::imshow(winname, img);
        }
    }else{
        std::ifstream region_txt("/home/user/jetson-inference/dbict/control/region.txt");
        while(!region_txt.eof()){

            std::string position;
            getline(region_txt,position);
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

        region_txt.close();
    }
}
