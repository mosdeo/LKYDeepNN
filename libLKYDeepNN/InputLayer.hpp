#ifndef _InputLayer_hpp_
#define _InputLayer_hpp_

#include "Layer.hpp"
#include "HiddenLayer.hpp"

using namespace std;

class InputLayer: public Layer
{
    //後層
    protected: HiddenLayer* nextLayer;

    public: InputLayer();
    public: InputLayer(int, HiddenLayer*);

    public: void Input(const vector<double>);

    public: string ToString(){ return "class InputLayer";}

    public: void SetNextLayer(HiddenLayer*);
};

#endif