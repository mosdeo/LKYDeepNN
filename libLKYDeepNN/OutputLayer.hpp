#ifndef _OutputLayer_hpp_
#define _OutputLayer_hpp_

#include "BackPropagationLayer.hpp"
#include "LossFunction.hpp"
#include <memory>

class HiddenLayer;

class OutputLayer: public BackPropagationLayer
{
    private: shared_ptr<LossFunction> lossFunction;

    public: ~OutputLayer()
    {
        cout << "~OutputLayer()" << endl;
    }

    public: void ForwardPropagation();
    public: void BackPropagation(double , vector<double>);
    public: string ToString(){ return "class OutputLayer";}
    public: void SetPrevLayer(HiddenLayer*);
    public: void SetLossFunction(LossFunction* lossFunction);
};

#endif