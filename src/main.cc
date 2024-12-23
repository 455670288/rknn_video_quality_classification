// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

#include <fstream>
#include <iostream>

#include "AbnormalAnalysis.h"
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <chrono>

// constexpr const char* Imagenet_classes_file_path = "../video_quality_label.txt";


struct Text_Result {
    bool is_DarkDet= false;
    bool is_Freeze = false;
    bool is_ColorBar = false;
    bool is_Cover =false;
    bool is_Blur = false;
    bool is_Frige = false;
    bool is_Snow = false;
};




/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{

    const char* model_path = "../../model/resnet50_video_quality_multi_labels_20241220.rknn";

    cv::VideoCapture cap("rtsp://172.16.40.84:553/live",cv::CAP_FFMPEG);

    int ret;
    Abnormal::AbnormalAnalysis resnet;
    ret = resnet.init_model(model_path);
    if (ret != 0) {
        printf("init_mobilenet_model fail! ret=%d model_path=%s\n", ret, model_path);
        return -1;
    }
    
    ///////////////////////////////////
    // cv::VideoCapture cap("concatenated_video.mp4");
    // 获取视频的帧率、分辨率等信息
    double fps = cap.get(cv::CAP_PROP_FPS);
    int frame_width = (int) cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = (int) cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    // 创建输出视频文件
    // cv::VideoWriter writer("output_video.mp4", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, cv::Size(frame_width, frame_height));
    //////////////////////////////////



    int frame_count = 0;
    int frame_interval = 2;

    float threshold = 0;
    bool is_abnormal = false;
    float conf = 0;
//  cv::namedWindow("AbnormalDetect", cv::WINDOW_NORMAL);


    cv::Mat orig_img; 
    Text_Result text_result;
    while(true){
        cap >> orig_img;

        if(orig_img.empty()){
            std::cout << "image is empty !" << std::endl;
            break;
        }

        if(frame_count % frame_interval == 0){

        
        cv::resize(orig_img, orig_img, cv::Size(640, 360), 0, 0, cv::INTER_AREA);
        cv::Mat rgb_img;
        cv::cvtColor(orig_img, rgb_img, cv::COLOR_BGR2RGB);

        auto start = std::chrono::high_resolution_clock::now();
        ret = resnet.infer(rgb_img);
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << "ms" << std::endl;

        resnet.detect(Abnormal::AbnormalType::darkDet, threshold, is_abnormal, conf);
        text_result.is_DarkDet = is_abnormal;
 
        resnet.detect(Abnormal::AbnormalType::freezeDet, threshold, is_abnormal, conf);
        text_result.is_Freeze = is_abnormal;

        ret = resnet.detect(Abnormal::AbnormalType::blurDet, 0.5, is_abnormal, conf);
        text_result.is_Blur = is_abnormal;

        ret = resnet.detect(Abnormal::AbnormalType::colorBarDet, 0.5, is_abnormal, conf);
        text_result.is_ColorBar = is_abnormal;


        ret = resnet.detect(Abnormal::AbnormalType::coverDet, 0.5, is_abnormal, conf);
        text_result.is_Cover = is_abnormal;


        ret = resnet.detect(Abnormal::AbnormalType::fringeNoiseDet, 0.5, is_abnormal, conf);
        text_result.is_Frige = is_abnormal;

        ret = resnet.detect(Abnormal::AbnormalType::snowNoiseDet, 0.5, is_abnormal, conf);
        text_result.is_Snow = is_abnormal;

        int y_offset = frame_height - 10;
        int line_height = 10; 

        // 检测和显示各种异常的文本
        std::vector<std::pair<std::string, bool>> abnormal_texts = {
            {"Signal Loss", text_result.is_DarkDet},
            {"Video Freezing", text_result.is_Freeze},
            {"Blur", text_result.is_Blur},
            {"ColorBar", text_result.is_ColorBar},
            {"Cover", text_result.is_Cover},
            {"FringeNoise", text_result.is_Frige},
            {"SnowNoise", text_result.is_Snow}
        };

        for (const auto& text_pair : abnormal_texts) {
            if (text_pair.second) {  // 如果当前检测到异常
            std::string text = text_pair.first;
        
            // 获取当前文本的大小
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1, 2, &baseline);
        
            // 计算文本的y坐标，并更新y_offset
            cv::Point text_org(10, y_offset);
            y_offset -= text_size.height + line_height;  // 每次绘制后，y_offset往上移动
        
            // 绘制文本
            cv::putText(orig_img, text, text_org, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            }
        }


        // writer.write(orig_img);
        // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << "ms" << std::endl;
        // std::cout << "#######################" << std::endl;

        // Show result
        cv::imshow("AbnormalDetect", orig_img);
        }
        
        frame_count++;

        if (ret != 0) {
          printf("init_mobilenet_model fail! ret=%d\n", ret);
        }

        // // Show result
        // cv::imshow("detect", orig_img);
        if (cv::waitKey(1) == 'q')
            break;
    }
    cap.release();
    // writer.release();
    cv::destroyAllWindows();
    
    printf("end test.....\n");

    return 0;
}
