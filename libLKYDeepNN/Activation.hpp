#ifndef _Activation_hpp_
#define _Activation_hpp_

#include <cmath>
#include <vector>
#include <tuple>
using namespace std;

class Activation
{
    public: virtual vector<tuple<double,double>> Forward(vector<tuple<double,double>>&)=0;
    public: virtual double Derivative(const double)=0;
};

class Sigmoid: public Activation
{
    public: Sigmoid(){ cout << "Activation is Sigmoid." << endl;}
    public: ~Sigmoid(){cout << "~Sigmoid()" << endl;}

    public: vector<tuple<double,double>> Forward(vector<tuple<double,double>>& nodeVector)
    {
        for(tuple<double,double>& node : nodeVector)
        {
            get<1>(node) = 1 / (1 + exp(-get<0>(node)));
        }
        return nodeVector;
    }

    public: double Derivative(const double x)
    {
        double output = 1 / (1 + exp(-x));
        return output * (1 - output);
    }
};

class Linear: public Activation
{
    public: Linear(){ cout << "Activation is Linear." << endl;}
    public: ~Linear(){cout << "~Linear()" << endl;}

    public: vector<tuple<double,double>> Forward(vector<tuple<double,double>>& nodeVector)
    {
        for(tuple<double,double>& node : nodeVector)
        {
            get<1>(node) = get<0>(node);
        }
        return nodeVector;
    }

    public: double Derivative(const double x)
    {
        return 1;
    }
};


class Softmax: public Activation
{
    public: Softmax(){ cout << "Activation is Softmax." << endl;}
    public: ~Softmax(){cout << "~Softmax()" << endl;}

    public: vector<tuple<double,double>> Forward(vector<tuple<double,double>>& nodeSum)
    {
        // does all output nodes at once so scale
        // doesn't have to be re-computed each time

        // if (oSums.Length < 2) throw . . .
        double sumExp= 0.0;
        for (size_t i = 0; i < nodeSum.size(); ++i)
            sumExp += exp(get<0>(nodeSum[i]));

        for (size_t i = 0; i < nodeSum.size(); ++i)
            get<1>(nodeSum[i]) = exp(get<0>(nodeSum[i])) / sumExp;

        return nodeSum; // now scaled so that xi sum to 1.0
    }

    public: double Derivative(const double x)
    {
        return x*(1-x);
    }
};

class Tanh: public Activation
{
    public: Tanh(){ cout << "Activation is Tanh." << endl;}
    public: ~Tanh(){cout << "~Tanh()" << endl;}

    public: vector<tuple<double,double>> Forward(vector<tuple<double,double>>& nodeVector)
    {
        for(tuple<double,double>& node : nodeVector)
        {
            get<1>(node) = tanh(get<0>(node));
        }
        return nodeVector;
    }

    public: double Derivative(const double x)
    {
        return 1 - pow(tanh(x), 2);
    }
};

class ReLU: public Activation
{
    public: ReLU() {cout << "Activation is ReLU." << endl;}
    public: ~ReLU(){cout << "~ReLU()" << endl;}

    public: vector<tuple<double,double>> Forward(vector<tuple<double,double>>& nodeVector)
    {
        for(tuple<double,double>& node : nodeVector)
        {
            get<1>(node) = max(0.0, get<0>(node));
        }
        return nodeVector;
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