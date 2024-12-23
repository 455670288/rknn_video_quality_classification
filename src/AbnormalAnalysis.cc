#include <iostream>
#include "AbnormalAnalysis.h"
#include "resnet.h"

using namespace Abnormal;

NormalDet normaldet;          //实例化黑屏、冻结检测
resnet_result result_cls[6];  //存储分类模型输出
bool is_dark = false; 
bool is_freeze = false;


static int count = 0;



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
            count ++;
            std::cout << "无视频" << " " << count << std::endl;
            std::cout << std::endl;
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
            count ++;
            std::cout << "画面冻结" << " "  << count<< std::endl;
            std::cout << std::endl;
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
            count ++;
            std::cout << "清晰度异常" << " " << count << std::endl;
            std::cout << std::endl;
        }else{
            is_abnormal = false;
            conf = 0;
        }
    }
    else if (abt == colorBarDet){
        if(classes_labels[result_cls[1].cls] == "color-bar" && result_cls[1].score >= threshold){
            is_abnormal = true;
            conf = result_cls[1].score;
            count ++;
            std::cout << "彩条" << " " << count << std::endl;
            std::cout << std::endl;
        }else{
            is_abnormal = false;
            conf = 0;
        }
    }
    else if (abt == coverDet){
        if(classes_labels[result_cls[2].cls] == "cover" && result_cls[2].score >= threshold){
            is_abnormal = true;
            conf = result_cls[2].score;
            count ++;
            std::cout << "遮挡异常" << " " << count << std::endl;
            std::cout << std::endl;
        }else{
            is_abnormal = false;
            conf = 0;
        }
    }
    else if (abt == fringeNoiseDet){
        if(classes_labels[result_cls[3].cls] == "fringe-noise" && result_cls[3].score >= threshold){
            is_abnormal = true;
            conf = result_cls[3].score;
            count ++;
            std::cout << "条纹噪声" << " "  << count << std::endl;
            std::cout << std::endl;
        }else{
            is_abnormal = false;
            conf = 0;
        }
    }
    else if (abt == snowNoiseDet){
        if(classes_labels[result_cls[5].cls] == "snow-noise" && result_cls[5].score >= threshold){
            is_abnormal = true;
            conf = result_cls[5].score;
            count ++;
            std::cout << "雪花噪声" << " " << count << std::endl;
            std::cout << std::endl;
        }else{
            is_abnormal = false;
            conf = 0;
        }
    }


    return 0;

}