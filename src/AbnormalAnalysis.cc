#include <iostream>
#include "AbnormalAnalysis.h"
#include "resnet.h"

using namespace Abnormal;

NormalDet normaldet;
resnet_result result_cls[5];
bool is_dark = false;
bool is_freeze = false;


AbnormalAnalysis::AbnormalAnalysis(){
    pImpl = new Impl();
}

AbnormalAnalysis::~AbnormalAnalysis(){
    delete pImpl;
}

int AbnormalAnalysis::init_model(const char* model_path){
    return pImpl->init_resnet_model(model_path);
}


int Abnormal::AbnormalAnalysis::infer(const cv::Mat& image){
    int ret = pImpl->inference_resnet_model(image,result_cls);
    if(ret < 0){
        std::cout << "rknn model inference fail !" << std::endl;
    }
    is_dark = normaldet.BlackScreenDet(image);
    is_freeze = normaldet.FreezeDet(image);

    return 0;
}


int AbnormalAnalysis::detect(AbnormalType abt, float threshold, bool &is_abnormal, float &conf){
    
    if(abt == darkDet){
        if(is_dark){
            std::cout << "黑屏" << std::endl;
            is_abnormal = true;
            conf = 1;
        }else{
            is_abnormal = false;
            conf = 0;
        }
    }
    else if (abt == freezeDet)
    {
        if(is_freeze){
            std::cout << "冻结" << std::endl;
            is_abnormal = true;
            conf = 1;            
        }else{
            is_abnormal = false;
            conf = 0;
        }
    }
    else if (abt == blurDet){
        if(classes_labels[result_cls[0].cls] == "blur" && result_cls[0].score >= threshold){
            is_abnormal = true;
            conf = result_cls[0].score;
        }else{
            is_abnormal = false;
            conf = 0;
        }
    }
    else if (abt == colorBarDet){
        if(classes_labels[result_cls[0].cls] == "color-bar" && result_cls[0].score >= threshold){
            is_abnormal = true;
            conf = result_cls[0].score;
            // std::cout << "彩条" << std::endl;
        }else{
            is_abnormal = false;
            conf = 0;
        }
    }
    else if (abt == coverDet){
        if(classes_labels[result_cls[0].cls] == "cover" && result_cls[0].score >= threshold){
            is_abnormal = true;
            conf = result_cls[0].score;
        }else{
            is_abnormal = false;
            conf = 0;
        }
    }
    else if (abt == fringeNoiseDet){
        if(classes_labels[result_cls[0].cls] == "fringe-noise" && result_cls[0].score >= threshold){
            is_abnormal = true;
            conf = result_cls[0].score;
        }else{
            is_abnormal = false;
            conf = 0;
        }
    }
    else if (abt == snowNoiseDet){
        if(classes_labels[result_cls[0].cls] == "snow-noise" && result_cls[0].score >= threshold){
            is_abnormal = true;
            conf = result_cls[0].score;
        }else{
            is_abnormal = false;
            conf = 0;
        }
    }

    return 0;

}