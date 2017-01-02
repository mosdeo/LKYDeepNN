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

class Square: LossFunction
{
    public: double Error(const double target, const double output)
    {
        return 0.5*pow(output - target, 2);
    }

    public: double Derivative(const double, const double)
    {
        return output - target;
    }
};

class CrossEntropy: LossFunction
{
    public: double Error(const double target, const double output)
    {
        bool isBinaryClassification = true;
        if(isBinaryClassification)
        {
            doube a = target*log(output);
            doube b = (1-target)*log(1-output);
            return a+b;
        }
        //return target*log(output);
    }

    public: double Derivative(const double, const double)
    {
        return output - target;
    }
};

#endif