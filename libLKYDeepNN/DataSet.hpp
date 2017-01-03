#ifndef _DATASET_HPP_
#define _DATASET_HPP_
#include <opencv2/opencv.hpp>
#include <tuple>
#include <random>
using namespace std;

vector<vector<double>> Make2DBinaryTrainingData(int numTariningData = 120)
{
    //make 2*numItems 2D vector
    vector<vector<double>> trainData(numTariningData, vector<double>(4));
    std::mt19937_64 rng(0);    // random-number engine used (Mersenne-Twister in this case)
    std::uniform_real_distribution<double> uni_noise(-0.5, 0.5); // guaranteed unbiased

    //產生兩個類別的資料點
    double A_centerX = 2,   A_centerY = 2;
    double B_centerX = -2, B_centerY = -2;
    double noiseRate = 6;

    for (size_t i = 0; i < trainData.size(); ++i)
    {
        if(i>trainData.size()/2)
        {
            trainData[i][0] = A_centerX + noiseRate*uni_noise(rng);
            trainData[i][1] = A_centerY + noiseRate*uni_noise(rng);
            trainData[i][2] = 0;
            trainData[i][3] = 1;
        }
        else
        {
            trainData[i][0] = B_centerX + noiseRate*uni_noise(rng);
            trainData[i][1] = B_centerY + noiseRate*uni_noise(rng);
            trainData[i][2] = 1;
            trainData[i][3] = 0;
        }
    }

    return trainData;
}

vector<vector<double>> classifySpiralData(double xBias, double yBias, int numSamples=120, double noise=0.1)
{
    vector<vector<double>> points;
    int n = numSamples / 2;

    auto genSpiral = [](vector<vector<double>>& points, int numSamples, double xBias, double yBias, double noise, int deltaT, int label)
    {
        std::mt19937_64 rng(0);
        
        std::uniform_real_distribution<double> uni_noise(-1, 1); // guaranteed unbiased

        for(int i=0; i < numSamples ; i++)
        {
            double r = 8*(double)i/numSamples;
            double t = 1.75 *i/numSamples*2 * M_PI + deltaT;
            double x = xBias + r * sin(t) + uni_noise(rng) * noise;
            double y = yBias + r * cos(t) + uni_noise(rng) * noise;

            // if(1==label)  points.push_back({x, y, x*y, 5*sin(x), 5*sin(y), 1, 0});
            // if(-1==label) points.push_back({x, y, x*y, 5*sin(x), 5*sin(y), 0, 1});
            if(1==label)  points.push_back({x, y, 1, 0});
            if(-1==label) points.push_back({x, y, 0, 1});

            for(double d:points.back())
            {
                printf(" %lf,",d);
            }printf("\n");
        }
    };

    genSpiral(points,n, xBias, yBias, noise,0, 1); // Positive examples.
    genSpiral(points,n, xBias, yBias, noise,M_PI, -1); // Negative examples.
    return points;
}

vector<vector<double>> classifyCircleData(double xBias, double yBias, int numSamples=120, double noise=0.9)
{
    vector<vector<double>> points(0, vector<double>(4));

    auto GenSampleCircle = [](vector<vector<double>>& points,
        int numSamples,double xBias, double yBias ,double radius, double maxRadius, double minRadius, double noise ,bool label)
    {
        //std::random_device rd;     // only used once to initialise (seed) engine
        std::mt19937_64 rng(0);    // random-number engine used (Mersenne-Twister in this case)
        std::uniform_real_distribution<double> uni_r; // guaranteed unbiased
        std::uniform_real_distribution<double> uni_angle(0, 2 * M_PI); // guaranteed unbiased
        std::uniform_real_distribution<double> uni_noise(-radius, radius); // guaranteed unbiased

        // Generate points inside the circle.
        uni_r = std::uniform_real_distribution<double>(minRadius, maxRadius);
        for (int i = 0; i < numSamples; i++)
        {
            double r = uni_r(rng);
            double angle = uni_angle(rng);
            double x = xBias + r * sin(angle);
            double y = yBias + r * cos(angle);

            double noiseX = uni_noise(rng) * noise;
            double noiseY = uni_noise(rng) * noise;
            std::tuple<double, double> noise(x+noiseX, y+noiseY);

            points.push_back({x, y, label ? 1.0 : 0.0 , label ? 0.0 : 1.0});
        }
    };

    double radius = 5;
    GenSampleCircle(points, numSamples/2, xBias ,yBias , 5,        0, radius*0.5, noise, true);
    GenSampleCircle(points, numSamples/2, xBias ,yBias , 5, radius * 0.7, radius, noise, false);

    return points;
}

cv::Mat Draw2DClassificationData(string strWindowName ,vector<vector<double>> XYData, LKYDeepNN* _nn, string strPutText="LKY",
    double Xmin = -10, double Xmax = 10, double Ymin = -10, double Ymax = 10)
{
    cv::Size canvasSize(400, 400); //畫布大小
    cv::Mat canvas(canvasSize, CV_8UC3, cv::Scalar(0));//產生畫布

    //計算修正參數
    double XscaleRate = canvasSize.width/(Xmax-Xmin);
    double YscaleRate = canvasSize.height/(Ymax-Ymin);

    cv::Vec3b violet(255, 0, 204);
    cv::Vec3b yellow(51, 153, 255);
    cv::Vec3b white(255, 255, 255);
    
    //寫入機率密度分佈
    for (int pixel_X = 0 ; pixel_X < canvasSize.width ; pixel_X++)
    {
        for (int pixel_Y = 0 ; pixel_Y < canvasSize.height ; pixel_Y++)
        {
            double resvY = pixel_Y/YscaleRate + Ymin; 
            double resvX = pixel_X/XscaleRate + Xmin;
            vector<double> result = _nn->ForwardPropagation(vector<double>{resvX,resvY});

            if(result[0] > result[1])
            {
                canvas.at<cv::Vec3b>(pixel_Y, pixel_X) = violet + (white-violet)*(1-(result[0]-result[1]));
                //canvas.at<cv::Vec3b>(pixel_Y, pixel_X) = cv::Vec3b(255*2*result[0], 255, 255*2*result[0]);
                //canvas.at<cv::Vec3b>(pixel_Y, pixel_X) = cv::Vec3b(255*2*(int)(result[0]+0.5), 255, 255*2*(int)(result[0]+0.5));

            }
            if(result[1] > result[0])
            {
                canvas.at<cv::Vec3b>(pixel_Y, pixel_X) = yellow + (white-yellow)*(1-(result[1]-result[0]));
                //canvas.at<cv::Vec3b>(pixel_Y, pixel_X) = cv::Vec3b(255*2*result[1], 255*2*result[1], 255);
                //canvas.at<cv::Vec3b>(pixel_Y, pixel_X) = cv::Vec3b(255*2*(int)(result[1]+0.5), 255*2*(int)(result[1]+0.5), 255);
            }
        }
    }

    //寫入每個資料點畫素
    for (size_t i = 0 ; i < XYData.size() ; i++)
    {
        int newY = YscaleRate*(XYData[i][1]-Ymin);
        int newX = XscaleRate*(XYData[i][0]-Xmin);

        cv::Scalar circleColor;
        if(*(XYData[i].end()-2) == 1){circleColor = violet*0.7;}//顏色稍微調暗
        if(*(XYData[i].end()-1) == 1){circleColor = yellow*0.7;}//顏色稍微調暗

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
    cv::putText(canvas,"Lin Kao-Yuan, mosdeo@gmail.com",cv::Point(20,canvas.rows-20), cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(127));

    //顯示
    cv::imshow(strWindowName.c_str(), canvas);
    cv::waitKey(1);

    return canvas;
};

vector<vector<double>> WaveData(int numTrainingData=80)
{
    //make 2*numItems 2D vector
    vector<vector<double>> trainData(numTrainingData, vector<double>(2));

    //std::random_device rd;     // only used once to initialise (seed) engine
    std::mt19937 rng(0);    // random-number engine used (Mersenne-Twister in this case)
    std::uniform_real_distribution<double> uni_noise(0, 1); // guaranteed unbiased

    //產生一個周期內的80個sin取樣點
    for (int i = 0; i < numTrainingData; ++i)
    {
        double x = 2*M_PI*uni_noise(rng); // [0 to 2PI]
        double sx = cos(2*x);
        trainData[i][0] = x;
        //trainData[i][1] = sin(x);
        trainData[i].back() = sx;
        //printf("x=%lf, sx=%lf\n", x, sx);
    }

    return trainData;
}

#endif