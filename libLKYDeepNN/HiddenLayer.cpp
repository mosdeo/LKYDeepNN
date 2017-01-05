#include "Layer.hpp"
#include "InputLayer.hpp"
#include "HiddenLayer.hpp"
#include "OutputLayer.hpp"
using namespace std;

int HiddenLayer::count = 0;

void HiddenLayer::ForwardPropagation()
{
    if(NULL == this->activation)
    {
        cout << "ERROR: Hidden Layer 沒有配置活化函數。" << endl;
        exit(EXIT_FAILURE);
    }

    //節點的乘積與和
    for (size_t j = 0; j < this->nodes.size(); ++j) // compute i-h sum of weights * inputNodes
    {
        //將自己的節點歸零，因為要存放上一級傳來的運算結果，不能累積。
        this->nodes[j] = make_tuple<double,double>(0,0);

        for (size_t i = 0; i < this->previousLayer->NodesSize(); ++i)
        {
            get<0>(this->nodes[j]) += get<1>(this->previousLayer->nodes[i]) * this->intoWeights[j][i]; // note +=
        }
        
        get<0>(this->nodes[j]) += this->intoBiases[j];  //加上截距
        //get<1>(this->nodes[j]) = this->activation->Forward(get<0>(this->nodes[j])); //活化函數
    }

    //將自身節點全部跑一次活化函數
    this->nodes = this->activation->Forward(this->nodes);
}

void HiddenLayer::BackPropagation(double learningRate)
{
    if(NULL == this->previousLayer)
    {
        cout << "NULL == this->previousLayer" << endl;
        exit(EXIT_FAILURE);
    }

    //[this][perv]
    for(size_t j=0 ; j <  this->intoWeights.size() ; j++)
    {
        //求出此節點所有順向影響的 sum(Delta*Weight)
        double sigmaDeltaWeight = 0;
        for(size_t k=0;k<this->nextLayer->nodes.size();k++)
        {
            sigmaDeltaWeight +=
                this->nextLayer->wDelta[k][j]*
                dynamic_pointer_cast<BackPropagationLayer>(this->nextLayer)->intoWeights[k][j];
        }

        //此節點微分值 (get<0>:節點之前, get<1>:節點之後)
        double derivativeActivation = this->activation->Derivative(get<0>(this->nodes[j]));

        for(size_t i=0 ; i < this->intoWeights[j].size() ; i++)
        {
            this->wDelta[j][i] = sigmaDeltaWeight*derivativeActivation;
            double pervNode = get<1>(this->previousLayer->nodes[i]);

            //更新權重
            this->intoWeights[j][i] -= learningRate*this->wDelta[j][i]*pervNode;
        }

        //更新基底權重
        this->bDelta[j] = derivativeActivation;
        this->intoBiases[j] -= learningRate*(this->bDelta[j]*sigmaDeltaWeight);
    }
    //cout << "end\n" << endl;
}

void HiddenLayer::SetNextLayer(BackPropagationLayer* nextLayer)
{
    this->nextLayer.reset(nextLayer);
}

void HiddenLayer::SetPrevLayer(Layer* pervLayer)
{
    this->previousLayer.reset(pervLayer);
}