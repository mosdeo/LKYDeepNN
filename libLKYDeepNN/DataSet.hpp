#ifndef _DATASET_HPP_
#define _DATASET_HPP_
#include <opencv2/opencv.hpp>
#include <tuple>
#include <random>
using namespace std;

vector<vector<double>> Make2DBinaryTrainingData(int numTariningData = 40)
{
    //make 2*numItems 2D vector
    vector<vector<double>> trainData(numTariningData, vector<double>(4));
    std::mt19937 rng(0);    // random-number engine used (Mersenne-Twister in this case)
    std::uniform_real_distribution<double> uni_noise(-0.5, 0.5); // guaranteed unbiased

    //產生兩個類別的資料點
    double A_centerX = 2,   A_centerY = 2;
    double B_centerX = -2, B_centerY = -2;
    double noiseRate = 7;

    for (size_t i = 0; i < trainData.size(); ++i)
    {
        if(i>trainData.size()/2)
        {
            trainData[i][0] = A_centerX + noiseRate*uni_noise(rng);
            trainData[i][1] = A_centerY + noiseRate*uni_noise(rng);
            trainData[i][2] = 1;
            trainData[i][3] = 0;
        }
        else
        {
            trainData[i][0] = B_centerX + noiseRate*uni_noise(rng);
            trainData[i][1] = B_centerY + noiseRate*uni_noise(rng);
            trainData[i][2] = 0;
            trainData[i][3] = 1;
        }
    }

    return trainData;
}

vector<vector<double>> classifyCircleData(int numSamples=80, double noise=0.1)
{
    vector<vector<double>> points(0, vector<double>(4));
    double radius = 5;

    auto getCircleLabel = [](std::tuple<double, double> p, std::tuple<double, double> center, double radius)
    {
        double dist_p_to_center = pow((get<0>(p) - get<0>(center)),2) + pow((get<1>(p) - get<1>(center)),2);
        dist_p_to_center = pow(dist_p_to_center, 0.5);
        return (dist_p_to_center < (radius * 0.5)) ? 1 : 0;
    };

    //std::random_device rd;     // only used once to initialise (seed) engine
    std::mt19937 rng(0);    // random-number engine used (Mersenne-Twister in this case)
    //std::minstd_rand0 rng();
    std::uniform_real_distribution<double> uni_r; // guaranteed unbiased
    std::uniform_real_distribution<double> uni_angle(0, 2 * M_PI); // guaranteed unbiased
    std::uniform_real_distribution<double> uni_noise(-radius, radius); // guaranteed unbiased
    //auto random_integer = uni(rng);

    // Generate positive points inside the circle.
    uni_r = std::uniform_real_distribution<double>(0, radius * 0.5);
    for (int i = 0; i < numSamples / 2; i++)
    {
        double r = uni_r(rng);
        double angle = uni_angle(rng);
        double x = r * sin(angle);
        double y = r * cos(angle);

        double noiseX = uni_noise(rng) * noise;
        double noiseY = uni_noise(rng) * noise;
        std::tuple<double, double> noise(x+noiseX, y+noiseY);

        //int label = getCircleLabel(noise, std::tuple<double, double>(0,0),radius);
        //points.push_back({x, y, (double)label});
        points.push_back({x, y, 1.0F, 0.0F});
    }

    // // Generate negative points outside the circle.
    uni_r = std::uniform_real_distribution<double>(radius * 0.7, radius);
    for (int i = 0; i < numSamples / 2; i++)
    {
        double r = uni_r(rng);
        double angle = uni_angle(rng);
        double x = r * sin(angle);
        double y = r * cos(angle);
        
        double noiseX = uni_noise(rng) * noise;
        double noiseY = uni_noise(rng) * noise;
        std::tuple<double, double> noise(x+noiseX, y+noiseY);

        //int label = getCircleLabel(noise, std::tuple<double, double>(0,0),radius);
        //points.push_back({x, y, (double)label});
        points.push_back({x, y, 0.0F, 1.0F});
    }

    return points;
}

cv::Mat Draw2DClassificationData(string strWindowName ,vector<vector<double>> XYData, LKYDeepNN& _nn, string strPutText="LKY",
    double Xmin = -10, double Xmax = 10, double Ymin = -10, double Ymax = 10)
{
    cv::Size canvasSize(400, 400); //畫布大小
    cv::Mat canvas(canvasSize, CV_8UC3, cv::Scalar(0));//產生畫布

    //計算修正參數
    double XscaleRate = canvasSize.width/(Xmax-Xmin);
    double YscaleRate = canvasSize.height/(Ymax-Ymin);
    
    //寫入機率密度分佈
    for (int pixel_X = 0 ; pixel_X < canvasSize.width ; pixel_X++)
    {
        for (int pixel_Y = 0 ; pixel_Y < canvasSize.height ; pixel_Y++)
        {
            double resvY = pixel_Y/YscaleRate + Ymin; 
            double resvX = pixel_X/XscaleRate + Xmin;
            //vector<double> result = _nn.ForwardPropagation(vector<double>{resvX,resvY,0,0});
            vector<double> result(2,0.5);

            if(result[0] < result[1])
            {
                //canvas.at<cv::Vec3b>(pixel_Y, pixel_X) = cv::Vec3b(255*2*result[0], 255, 255*2*result[0]);
                canvas.at<cv::Vec3b>(pixel_Y, pixel_X) = cv::Vec3b(255*2*(int)(result[0]+0.5), 255, 255*2*(int)(result[0]+0.5));

            }
            if(result[1] < result[0])
            {
                //canvas.at<cv::Vec3b>(pixel_Y, pixel_X) = cv::Vec3b(255*2*result[1], 255*2*result[1], 255);
                canvas.at<cv::Vec3b>(pixel_Y, pixel_X) = cv::Vec3b(255*2*(int)(result[1]+0.5), 255*2*(int)(result[1]+0.5), 255);
            }
        }
    }

    //寫入每個資料點畫素
    for (size_t i = 0 ; i < XYData.size() ; i++)
    {
        int newY = YscaleRate*(XYData[i][1]-Ymin);
        int newX = XscaleRate*(XYData[i][0]-Xmin);

        cv::Scalar circleColor;
        if(XYData[i][2]== 1){circleColor = cv::Scalar(0, 0, 205);}//鮮紅色
        if(XYData[i][3]== 1){circleColor = cv::Scalar(0, 205, 0);}//深綠色onst

        const int radius = 5;
        const int thickness = 2;
        cv::circle(canvas, cv::Point(newY, newX), radius, circleColor, thickness);
    }

    //cv::resize(canvas,canvas,cv::Size(800,800));
    cv::putText(canvas,strPutText.c_str(), cv::Point(20,40), cv::FONT_HERSHEY_COMPLEX,0.5, cv::Scalar(0x00));
    cv::putText(canvas,"Lin Kao-Yuan, mosdeo@gmail.com", cv::Point(20,canvas.rows-20), cv::FONT_HERSHEY_COMPLEX,0.5, cv::Scalar(0x00));

    //顯示
    cv::imshow(strWindowName.c_str(), canvas);
    cv::waitKey(1);

    return canvas;
};


cv::Mat Draw2DRegressionData(string strWindowName ,vector<vector<double>> XYData, string strPutText="LKY", double Xmin = 0, double Xmax = 6.4, double Ymin = -3, double Ymax = 3)
{
    cv::Size canvasSize(640, 480); //畫布大小
    cv::Mat canvas(canvasSize, CV_8U, cv::Scalar(0));//產生畫布

    //計算修正參數
    double XscaleRate = canvasSize.width/(Xmax-Xmin);
    double YscaleRate = canvasSize.height/(Ymax-Ymin);
    
    //寫入每個資料點畫素
    for (size_t i = 0 ; i < XYData.size() ; i++)
    {
        int newY = YscaleRate*(XYData[i][1]-Ymin);
        int newX = XscaleRate*(XYData[i][0]-Xmin);
        canvas.at<unsigned char>(newY, newX)=255;//pixel write
    }

    cv::dilate(canvas, canvas, cv::Mat()); //使畫素膨脹

    cv::putText(canvas,strPutText.c_str(),cv::Point(20,40), cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(255));
    cv::putText(canvas,"Lin Kao-Yuan, mosdeo@gmail.com",cv::Point(20,canvas.rows-20), cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(127));

    //顯示
    cv::imshow(strWindowName.c_str(), canvas);
    cv::waitKey(1);

    return canvas;
};

#endif