#include "normal_det.h"
#include "iostream"

NormalDet::NormalDet(){

}

NormalDet::~NormalDet(){

}


bool NormalDet::FreezeDet(const cv::Mat& currentFrame,double threshold, double minDifferenceRatio) {
       static int Freezetime = 0;
       if(isfirst){
         isfirst = false;
         lastFrame = currentFrame.clone();  // 缓存
         return false;
       }
    
       //计算两张图像绝对差值
       cv::Mat diff;
       cv::absdiff(currentFrame, lastFrame, diff);           
       cv::Mat grayDiff;
       cv::cvtColor(diff, grayDiff, cv::COLOR_BGR2GRAY);
       
       // 计算小于阈值的像素个数,统计前后帧相同像素的个数
       int nonZeroCount = cv::countNonZero(grayDiff <= 255 * threshold);
       int totalPixels = currentFrame.total();

      //  std::cout << "nonZeroCount: " << nonZeroCount << std::endl;
      //  std::cout << "totalPixels: " << totalPixels << std::endl;
      //  std::cout << "前后帧相似度：" <<nonZeroCount / double(totalPixels) << std::endl;
       
       //前后帧相同的像素比例。
        if(nonZeroCount / double(totalPixels) >= minDifferenceRatio){
            Freezetime += 1;
            //冻结记录次数连续达到阈值，输出冻结
            if(Freezetime == Freeze_count) { 
                Freezetime *= 0;
                lastFrame = currentFrame.clone();
                return true;
            }else{
                 lastFrame = currentFrame.clone();
                 return false;
            }  
        }else{
          //若冻结记录不是连续的，则认定为误判，不予以计数
          Freezetime *= 0;
          lastFrame = currentFrame.clone();
          return false;
        }
    }

bool NormalDet::BlackScreenDet(const cv::Mat& Img){
    if(Img.empty()) return 0;
    
    cv::Mat resizedImg, grayImg;
    cv::resize(Img, resizedImg,cv::Size(Img.cols * 0.25, Img.rows * 0.25));
    cv::cvtColor(resizedImg, grayImg,cv::COLOR_BGR2GRAY);

     
    int32_t blurWidth = static_cast<int32_t>(0.2 * grayImg.cols);
    if (blurWidth % 2 == 0) {
      blurWidth += 1;
    }

    int32_t blurHeight = static_cast<int32_t>(0.3 * grayImg.rows);
    if (blurHeight % 2 == 0) {
      blurHeight += 1;
    }

    // 应用高斯模糊和Canny边缘检测
    cv::GaussianBlur(grayImg, grayImg, cv::Size(blurWidth, blurHeight), 0);
    cv::Canny(grayImg, grayImg, 0, 0);


    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(grayImg, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
 
  
    double sum = static_cast<double>(contours.size());
    double imgArea = static_cast<double>(grayImg.cols * grayImg.rows);

    //计算轮廓密集程度(放大轮廓影响)
    double res = (sum * sum) / imgArea;
    res *= res;
    
    //计算黑屏连通区域与整幅图像面积比例
    double result = 1.0 - (res / 100.0 > 1.0 ? 1.0 : res / 100.0);
    // std::cout << "黑屏区域比例: " << result << std::endl;

    bool black_screen = result >= 0.99 ? true : false;
  

    return black_screen;


}