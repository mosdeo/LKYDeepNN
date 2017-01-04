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
    public: Square(){ cout << "LossFunction is Square." << endl;}
    public: ~Square(){cout << "~Square()" << endl;}

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
    public: CrossEntropy(){ cout << "LossFunction is CrossEntropy." << endl;}
    public: ~CrossEntropy(){cout << "~CrossEntropy()" << endl;}

    public: double Error(const double output, const double target)
    {
        return target*log(output+1) + (1-target)*log(1-output+1);
    }

    public: double Derivative(const double output, const double target)
    {
        return (output - target)*output*(1-output);
    }
};

#endif