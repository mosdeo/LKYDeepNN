#include "InputLayer.hpp"
#include "HiddenLayer.hpp"
#include "OutputLayer.hpp"

void OutputLayer::ForwardPropagation()
{
    if(NULL == this->activation)
    {
        cout << "WARNING: Output Layer 沒有配置活化函數，要做 Regression 嗎？ 那應該配置 Linear" << endl;
        exit(EXIT_FAILURE);
    }

    //節點的乘積與和 //[this][perv]
    for (size_t j = 0; j < this->nodes.size(); ++j) // compute i-h sum of weights * inputNodes
    {
        //將自己的節點歸零，因為要存放上一級傳來的運算結果，不能累積。
        this->nodes[j] = make_tuple<double,double>(0,0);

        for (size_t i = 0; i < this->previousLayer->nodes.size(); ++i)
        {
            get<0>(this->nodes[j]) += get<1>(this->previousLayer->nodes[i]) * this->intoWeights[j][i]; // note +=
        }

        get<0>(this->nodes[j]) += this->intoBiases[j];  //加上截距
        //get<1>(this->nodes[j]) = this->activation->Forward(get<0>(this->nodes[j])); //活化函數
    }

    //將自身節點全部跑一次活化函數
    this->nodes = this->activation->Forward(this->nodes);
}

void OutputLayer::BackPropagation(double learningRate, vector<double> desiredOutValues)
{
    // cout << "OutputLayer::BackPropagation" << endl;
    // cout << "  prev Layer: " <<this->previousLayer->ToString() << endl;

    if(desiredOutValues.size() != this->nodes.size())
    {
        cout << "ERROR: desiredOutValues.size() != this->nodes.size()" << endl;
        exit(EXIT_FAILURE);
    }

    //[this][perv]
    for(size_t j=0 ; j < this->wDelta.size() ; j++)
    {
        //求損失函數微分
        double cost = this->lossFunction->Derivative(get<1>(this->nodes[j]),desiredOutValues[j]); 
            
        //此節點微分值 (get<0>:節點之前, get<1>:節點之後)
        //Softmax 微分比較麻煩，先暫時拿節點後的數值來運算
        double derivativeActivation = this->activation->Derivative(get<1>(this->nodes[j]));
        double delta = cost * derivativeActivation;

        for(size_t i=0 ; i < this->wDelta[j].size() ; i++)
        {
            this->wDelta[j][i] = delta;
            double pervInput = get<1>(this->previousLayer->nodes[i]);

            //更新權重
            this->intoWeights[j][i] -= learningRate*(this->wDelta[j][i]*pervInput);
        }

        //更新基底權重
        this->bDelta[j] = delta;
        this->intoBiases[j] -= learningRate*this->bDelta[j];
    }
    //cout << "end\n" << endl;
}

void OutputLayer::SetPrevLayer(HiddenLayer* pervLayer)
{
    this->previousLayer.reset(pervLayer);
}

void OutputLayer::SetLossFunction(LossFunction* lossFunction)
{
    this->lossFunction.reset(lossFunction);
}