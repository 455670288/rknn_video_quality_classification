#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp> 
#include <opencv2/highgui.hpp>  
#include <opencv2/imgproc.hpp>

class NormalDet
{
public:
    NormalDet();
    ~NormalDet();

    bool FreezeDet(const cv::Mat& currentFrame,double threshold=0.0, double minDifferenceRatio =0.99);

    bool BlackScreenDet(const cv::Mat& Img);

private:

    int32_t blurWidth;
    int32_t blurHeight;

    
    /*
    画面冻结第一帧不检测
    */
    bool isfirst = true;

    /*
    画面冻结次数记录
    */
    int Freeze_count = 10; 

    /*
    前一帧缓存
    */
   cv::Mat lastFrame;
   /*
   第一次检测
    */
//    bool first = true;
};
