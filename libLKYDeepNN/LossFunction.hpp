#ifndef _LossFunction_hpp_
#define _LossFunction_hpp_

#include <cmath>
#include <vector>
using namespace std;

class LossFunction
{
    public: virtual double Error(const double, const double)=0;
    public: virtual double Derivative(const double, const double)=0;
};

class Square: public LossFunction
{
    public: double Error(const double output, const double target)
    {
        return 0.5*pow(output - target, 2);
    }

    public: double Derivative(const double output, const double target)
    {
        return output - target;
    }
};

class CrossEntropy: public LossFunction
{
    public: double Error(const double output, const double target)
    {
        // bool isBinaryClassification = true;
        // if(isBinaryClassification)
        // {
            double a = target*log(output);
            double b = (1-target)*log(1-output);
            return a+b;
        // }
        //return target*log(output);
    }

    public: double Derivative(const double output, const double target)
    {
        return (output - target)*output*(1-output);
    }
};

#endif