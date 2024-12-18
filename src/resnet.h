#ifndef _RKNN_DEMO_RESNET_H_
#define _RKNN_DEMO_RESNET_H_

#pragma once
#include "rknn_api.h"
#include "common.h"
#include "normal_det.h"
#include "AbnormalAnalysis.h"
#include "rk_cls_utils.h"


#include <opencv2/opencv.hpp>



class Abnormal::AbnormalAnalysis::Impl{

public:
  Impl();
  ~Impl();



  int init_resnet_model(const char* model_path,const int &NPUcore = -1);

  int inference_resnet_model(const cv::Mat &image, resnet_result* result_cls, int topk =5);

private:
   

   // 存储RK推理变量，指针在构造函数分配内存并初始化
   rknn_app_context_t* app_ctx;

   // 释放RK推理变量
   int release_resnet_model();
    
    //存储自定义类别
    // int line_count;
    // char** lines;


};

#endif //_RKNN_DEMO_RESNET_H_