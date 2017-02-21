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
    private: double sumExp;

    public: vector<tuple<double,double>> Forward(vector<tuple<double,double>>& nodeVector)
    {
        if(nodeVector.size()<2)
        {
            cout << "小於兩個輸出點，使用 Softmax 活化得不到有意義的輸出結果。" << endl;
            exit(EXIT_FAILURE);
        }

        // //找出最大，作為後續計算偏移量 for stable issue
        // double maxNode = std::numeric_limits<double>::min();
        // for(auto& node :nodeVector)
        //     if(maxNode < get<0>(node))
        //         maxNode = get<0>(node);

        //計算分母
        this->sumExp = 0.0;
        for(auto& node :nodeVector)
            this->sumExp += exp((get<0>(node)));

        //計算 分子/分母
        for(auto& node :nodeVector)
            get<1>(node) = exp(get<0>(node))/this->sumExp;

        return nodeVector;
    }

    public: double Derivative(const double x)
    {
        double softmax = exp(x)/this->sumExp;
        return softmax*(1-softmax);
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
            return 0;}
    }
};

class LReLU: public Activation
{
    //Leaky Rectified Linear Unit.
    //A Leaky ReLU can help fix the “dying ReLU” problem.
    //ReLU’s can “die” if a large enough gradient changes the weights such
    //that the neuron never activates on new data.
    public: LReLU() {cout << "Activation is LReLU." << endl;}
    public: LReLU(double alpha)
    {
        this->alpha = alpha;
        cout << "Activation is LReLU, alpha = " << this->alpha << "." << endl;
    }
    public: ~LReLU(){cout << "~LReLU()" << endl;}
    private: double alpha = 0.01;//default

    public: vector<tuple<double,double>> Forward(vector<tuple<double,double>>& nodeVector)
    {
        for(tuple<double,double>& node : nodeVector)
        {
            double x = get<0>(node);
            get<1>(node) = (x > 0) ? x : x*alpha;
        }
        return nodeVector;
    }

    public: double Derivative(const double x)
    {
        if(x > 0){
            return 1;}
        else{
            return alpha;}
    }
};

#endif