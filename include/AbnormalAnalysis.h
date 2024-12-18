#pragma once
#include <opencv2/opencv.hpp>


namespace Abnormal
{
    enum AbnormalType
    {
        darkDet = 0, //黑屏检测
        freezeDet = 1, //冻结检测
        blurDet = 2, 
        colorBarDet = 3,
        coverDet = 4,
        fringeNoiseDet = 5,
        snowNoiseDet = 6,
    };

    class AbnormalAnalysis
    {
    public:
        AbnormalAnalysis();
        ~AbnormalAnalysis();
        

        int init_model(const char* model_path);
        int infer(const cv::Mat &image);
        int detect(AbnormalType abt, float threshold, bool &is_abnormal, float &conf);

    private:
        class Impl;
        Impl* pImpl;

        const std::vector<std::string> classes_labels={"blur","color-bar","cover","fringe-noise","normal","snow-noise"};
    };
};
