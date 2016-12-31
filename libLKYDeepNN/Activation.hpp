#ifndef _Activation_hpp_
#define _Activation_hpp_

#include <cmath>
#include <vector>
using namespace std;

class Activation
{
    public: virtual vector<double> Forward(const vector<double>&)=0;
    public: virtual double Derivative(const double)=0;
};

class WithoutActivation: public Activation
{
    public: WithoutActivation()
    {
        cout << "Activation is WithoutActivation." << endl;
    }

    public: ~WithoutActivation()
    {
        cout << "~WithoutActivation()" << endl;
    }

    public: vector<double> Forward(const vector<double>& nodeSum)
    {
        vector<double> result(nodeSum.size());
        
        for (size_t i = 0; i < nodeSum.size(); ++i)
        {
            result[i] = nodeSum[i];
        }

        return result; // now scaled so that xi sum to 1.0
    }

    public: double Derivative(const double x)
    {
        return 1;
    }
};


class Softmax: public Activation
{
    public: Softmax()
    {
        cout << "Activation is Softmax." << endl;
    }

    public: ~Softmax()
    {
        cout << "~Softmax()" << endl;
    }

    public: vector<double> Forward(const vector<double>& nodeSum)
    {
        // does all output nodes at once so scale
        // doesn't have to be re-computed each time

        // if (oSums.Length < 2) throw . . .
        vector<double> result(nodeSum.size());

        double sum = 0.0;
        for (size_t i = 0; i < nodeSum.size(); ++i)
            sum += exp(nodeSum[i]);

        for (size_t i = 0; i < nodeSum.size(); ++i)
            result[i] = exp(nodeSum[i]) / sum;

        return result; // now scaled so that xi sum to 1.0
    }

    public: double Derivative(const double x)
    {
        return x*(1-x);
    }
};

class Tanh: public Activation
{
    public: Tanh()
    {
        cout << "Activation is Tanh." << endl;
    }

    public: ~Tanh()
    {
        cout << "~Tanh()" << endl;
    }


    public: vector<double> Forward(const vector<double>& nodeSum)
    {
        vector<double> result(nodeSum.size());
        
        for (size_t i = 0; i < nodeSum.size(); ++i){
            result[i] = tanh(nodeSum[i]);}

        return result;
    }

    public: double Derivative(const double x)
    {
        return 1 - pow(tanh(x), 2);
    }
};

class ReLU: public Activation
{
    public: ReLU()
    {
        cout << "Activation is ReLU." << endl;
    }

    public: vector<double> Forward(const vector<double>& nodeSum)
    {
        vector<double> result(nodeSum.size());
        
        for (size_t i = 0; i < nodeSum.size(); ++i)
        {
            result[i] = max(nodeSum[i],0.0);
        }

        return result;
    }

    public: double Derivative(const double x)
    {
        if(x > 0){
            return 1;}
        else{
            return 0.01;}
    }
};

#endif