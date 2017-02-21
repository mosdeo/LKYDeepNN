#ifndef _DATASET_HPP_
#define _DATASET_HPP_

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

vector<vector<double>> classifySpiralData(double xBias, double yBias, int numSamples=230, double noise=0.1)
{
    vector<vector<double>> points;
    int n = numSamples / 2;

    auto genSpiral = [](vector<vector<double>>& points, int numSamples, double xBias, double yBias, double noise, double deltaT, int label)
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

            //for(double d:points.back()){ printf(" %lf,",d);　}printf("\n");
        }
    };

    genSpiral(points,n, xBias, yBias, noise, 0, 1); // Positive examples.
    genSpiral(points,n, xBias, yBias, noise, M_PI, -1); // Negative examples.
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
        double sx = (cos(2*x)+sin(3*x));
        trainData[i][0] = x;
        //trainData[i][1] = sin(x);
        trainData[i].back() = sx;
        //printf("x=%lf, sx=%lf\n", x, sx);
    }

    return trainData;
}

#endif