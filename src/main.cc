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


/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
//    if (argc != 2) {
//         printf("%s <model_path>\n", argv[0]);
//         return -1;
//     }

    const char* model_path = "../../model/resnet50_video_quality_20241216.rknn";

    cv::VideoCapture cap("rtsp://172.16.40.84:553/live",cv::CAP_FFMPEG);

    int ret;
    Abnormal::AbnormalAnalysis resnet;
    ret = resnet.init_model(model_path);
    if (ret != 0) {
        printf("init_mobilenet_model fail! ret=%d model_path=%s\n", ret, model_path);
        return -1;
    }

    int frame_count = 0;
    int frame_interval = 2;

    float threshold = 0;
    bool is_abnormal = false;
    float conf = 0;

    cv::Mat orig_img;
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
        // ret = resnet.detect(Abnormal::AbnormalType::freezeDet, threshold, is_abnormal, conf);
        ret = resnet.detect(Abnormal::AbnormalType::colorBarDet, threshold, is_abnormal, conf);


        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << "ms" << std::endl;

        std::cout << "################################################" << std::endl;
        // Show result
        cv::imshow("detect", orig_img);
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
    cv::destroyAllWindows();
    
    printf("end test.....\n");

    return 0;
}
