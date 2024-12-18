#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>  // For VideoCapture and VideoWriter
#include <opencv2/highgui.hpp>  // For highgui functions (like imshow)
#include <opencv2/imgproc.hpp> 
#include <chrono>
#include <iomanip>
#include <normal_det.h>





void start_detect() {
    // cv::Mat image = cv::imread("../../model/dog_224x224.jpg");
  
    cv::VideoCapture capture("rtsp://172.16.40.84:553/live", cv::CAP_FFMPEG);
    if (!capture.isOpened()) {
        std::cerr << "Error: Unable to open video file." << std::endl;
        return;
    }

    // 设置起始帧
    double fps = capture.get(cv::CAP_PROP_FPS);
    std::cout << "FPS: " << fps << std::endl;

    int w = int(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = int(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

    NormalDet freezedet;   //实例化
    cv::Mat frame;

    
    int frame_count =0;
    int frame_interval = 2;
    while (true) {
        capture >> frame; // 逐帧读取视频
        if (frame.empty()) {
            break; // 如果没有读取到帧，退出循环
        }

        if(frame_count % frame_interval ==0){
          
          /*
          黑屏检测（信号丢失检测）
          */
          if(freezedet.BlackScreenDet(frame)){
            std::cout << "黑屏" << std::endl;
          }

          /* 冻结检测 */
          if (freezedet.FreezeDet(frame)){
            std::cout << "画面冻结" << std::endl;
          }
   
    
          cv::imshow("Frame", frame);
        }
        
        frame_count++;
        if (cv::waitKey(1) == 'q')
            break;
        
    }
    capture.release();
}

int main() {
    start_detect();
    return 0;
}