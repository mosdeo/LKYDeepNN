#include <cmath>
#include <vector>
using namespace std;

class Activation
{
    public: virtual vector<double> Forward(const vector<double>&)=0;
    public: virtual double Derivative(const double)=0;
};

class Tanh: public Activation
{
    public: Tanh()
    {
        cout << "Activation is Tanh." << endl;
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
        if(x >= 0){
            return 1;}
        else{
            return 0.01;}
    }
};