#ifndef _DrawingAnimation_HPP_
#define _DrawingAnimation_HPP_

#include <opencv2/opencv.hpp>

cv::Mat Draw2DClassificationData(string strWindowName ,vector<vector<double>> XYData, LKYDeepNN* _nn, string strPutText="LKY",
    double Xmin = -10, double Xmax = 10, double Ymin = -10, double Ymax = 10)
{
    cv::Size canvasSize(400, 400); //畫布大小(X,Y)
    cv::Mat canvas(canvasSize, CV_8UC3, cv::Scalar(0));//產生畫布

    //計算修正參數
    double XscaleRate = canvasSize.width/(Xmax-Xmin);
    double YscaleRate = canvasSize.height/(Ymax-Ymin);

    cv::Vec3b violet(255, 0, 204);
    cv::Vec3b yellow(51, 153, 255);
    cv::Vec3b white(255, 255, 255);
    
    //寫入對樣本空間的估計
    for (int pixel_X = 0 ; pixel_X < canvasSize.width ; pixel_X++)
    {
        double resvX = pixel_X/XscaleRate + Xmin; //正規化

        for (int pixel_Y = 0 ; pixel_Y < canvasSize.height ; pixel_Y++)
        {
            double resvY = pixel_Y/YscaleRate + Ymin; //正規化
            vector<double> result = _nn->ForwardPropagation(vector<double>{resvX,resvY});

            if(result[0] > result[1])
            {
                canvas.at<cv::Vec3b>(pixel_Y, pixel_X) = violet + (white-violet)*(1-(result[0]-result[1]));
                //canvas.at<cv::Vec3b>(pixel_Y, pixel_X) = cv::Vec3b(255*2*result[1], 255, 255*2*result[1]);
                //canvas.at<cv::Vec3b>(pixel_Y, pixel_X) = cv::Vec3b(255*2*(int)(result[0]+0.5), 255, 255*2*(int)(result[0]+0.5));

            }
            if(result[1] > result[0])
            {
                canvas.at<cv::Vec3b>(pixel_Y, pixel_X) = yellow + (white-yellow)*(1-(result[1]-result[0]));
                //canvas.at<cv::Vec3b>(pixel_Y, pixel_X) = cv::Vec3b(255*2*result[0], 255*2*result[0], 255);
                //canvas.at<cv::Vec3b>(pixel_Y, pixel_X) = cv::Vec3b(255*2*(int)(result[1]+0.5), 255*2*(int)(result[1]+0.5), 255);
            }
        }
    }

    //寫入每個資料點，以圓圈代表
    for (size_t i = 0 ; i < XYData.size() ; i++)
    {
        int newY = YscaleRate*(XYData[i][1]-Ymin);
        int newX = XscaleRate*(XYData[i][0]-Xmin);

        cv::Scalar circleColor;
        if(*(XYData[i].end()-2) == 1){circleColor = violet*0.7;}//顏色稍微調暗
        if(*(XYData[i].end()-1) == 1){circleColor = yellow*0.7;}//顏色稍微調暗

        const int radius = 5, thickness = 2;
        cv::circle(canvas, cv::Point(newX, newY), radius, circleColor, thickness);
    }

    //cv::resize(canvas,canvas,cv::Size(800,800));
    cv::putText(canvas,strPutText.c_str(), cv::Point(20,40), cv::FONT_HERSHEY_COMPLEX,0.5, cv::Scalar(0x00));
    cv::putText(canvas,"LKYDeepNN, mosdeo@gmail.com", cv::Point(20,canvas.rows-20), cv::FONT_HERSHEY_COMPLEX,0.5, cv::Scalar(0x00));

    //顯示
    cv::imshow(strWindowName.c_str(), canvas);
    cv::waitKey(1);

    return canvas;
};


cv::Mat Draw2DRegressionData(string strWindowName ,vector<vector<double>> XYData, LKYDeepNN* _nn,string strPutText="LKY", double Xmin = 0, double Xmax = 6.4, double Ymin = -3, double Ymax = 3)
{
    cv::Size canvasSize(400, 400); //畫布大小
    cv::Mat canvas(canvasSize, CV_8U, cv::Scalar(0));//產生畫布

    //計算修正參數
    double XscaleRate = canvasSize.width/(Xmax-Xmin);
    double YscaleRate = canvasSize.height/(Ymax-Ymin);
    
    //畫上原始訓練資料
    for (size_t i = 0 ; i < XYData.size() ; i++)
    {
        int newY = YscaleRate*(XYData[i][1]-Ymin);
        int newX = XscaleRate*(XYData[i][0]-Xmin);
        canvas.at<unsigned char>(newY, newX)=127;//pixel write
    }

    //畫上模型預結果
    size_t numItems = 120;
    vector<vector<double>> perdictData(numItems,vector<double>(2));
   
    for(size_t i=0;i<numItems;i++)
    {//產生所有取樣點
        perdictData[i][0] = i*(2*M_PI)/(double)numItems;
        perdictData[i][1] =_nn->ForwardPropagation(vector<double>{perdictData[i][0]}).front();
    }
    for (size_t i = 0 ; i < perdictData.size() ; i++)
    {
        int newY = YscaleRate*(perdictData[i][1]-Ymin);
        int newX = XscaleRate*(perdictData[i][0]-Xmin);
        
        newY = min(newY, canvasSize.height); //防止數值爆掉造成記憶體錯誤
        newY = max(newY, 0);

        canvas.at<unsigned char>(newY, newX)=255;//pixel write
    }

    cv::dilate(canvas, canvas, cv::Mat()); //使畫素膨脹

    cv::putText(canvas,strPutText.c_str(),cv::Point(20,40), cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(255));
    cv::putText(canvas,"LKYDeepNN, mosdeo@gmail.com",cv::Point(20,canvas.rows-20), cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(127));

    //顯示
    cv::imshow(strWindowName.c_str(), canvas);
    cv::waitKey(1);

    return canvas;
};

#endif