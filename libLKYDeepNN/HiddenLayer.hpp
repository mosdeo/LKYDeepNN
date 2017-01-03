#ifndef _HiddenLayer_hpp_
#define _HiddenLayer_hpp_

#include "Layer.hpp"
#include "InputLayer.hpp"
#include "OutputLayer.hpp"

#include <memory>

class HiddenLayer: public BackPropagationLayer
{
    friend class OutputLayer;
    private: static int count;
    private: int serialNum;
    public: int GetSerialNum(){return this->serialNum;}
    public: int GetCount(){return count;}

    //下一層
    public: shared_ptr<BackPropagationLayer> nextLayer;

    public: HiddenLayer()
    {
        HiddenLayer::count++;
        this->serialNum = HiddenLayer::count;
    }

    public: ~HiddenLayer()
    {
        cout << "~HiddenLayer(): " << HiddenLayer::count << endl;
        HiddenLayer::count--;
    }

    public: void ForwardPropagation();

    public: void BackPropagation(double);

    public: vector<double> GetOutput();

    public: string ToString(){ return "class HiddenLayer";} 

    public: void SetNextLayer(BackPropagationLayer*);
    public: void SetPrevLayer(Layer*);
};

#endif