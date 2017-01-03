#include "Layer.hpp"
#include "InputLayer.hpp"
#include "HiddenLayer.hpp"
#include "OutputLayer.hpp"
using namespace std;

int HiddenLayer::count = 0;

void HiddenLayer::ForwardPropagation()
{
    // cout << "HiddenLayer::ForwardPropagation" << endl;
    // cout << typeid(*(this->nextLayer)).name() << endl;
    // cout << typeid(this->previousLayer).name() << endl;
    // cout << "  prev Layer: " <<this->previousLayer->ToString() << endl;
    // cout << "  next Layer: " << this->nextLayer->ToString() << endl;

    //將自己的節點歸零，因為要存放上一級傳來的運算結果，不能累積。
    this->nodes = vector<double>(this->nodes.size() ,0.0);

    //節點的乘積與和
    //cout << "this->nodes.size() = " << this->nodes.size() << endl;
    for (size_t j = 0; j < this->nodes.size(); ++j) // compute i-h sum of weights * inputNodes
    {
        //cout << "this->previousLayer->nodes.size() = " << this->previousLayer->nodes.size() << endl;
        for (size_t i = 0; i < this->previousLayer->NodesSize(); ++i)
        {
            this->nodes[j] += this->previousLayer->nodes[i] * this->intoWeights[j][i]; // note +=
        }

        this->nodes[j] += this->intoBiases[j];
    }

    //活化函數
    if(NULL != this->activation)
    {//將自身節點全部跑一次活化函數
        this->nodes = this->activation->Forward(this->nodes);
    }
    else
    {
        cout << "ERROR: HiddenLayer 沒有配置活化函數" << endl;
        exit(EXIT_FAILURE);
    }
}

void HiddenLayer::BackPropagation(double learningRate)
{
    // cout << "HiddenLayer::BackPropagation" << endl;
    // cout << "  prev Layer: " <<this->previousLayer->ToString() << endl;
    // cout << "  next Layer: " << this->nextLayer->ToString() << endl;
    //printf("HiddenLayer: this->wDelta.size() = %ld, this->wDelta[0].size() = %ld\n", this->wDelta.size(), this->wDelta[0].size());
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

        //此節點微分值
        double derivativeActivation = this->activation->Derivative(this->nodes[j]);

        for(size_t i=0 ; i < this->intoWeights[j].size() ; i++)
        {
            this->wDelta[j][i] = sigmaDeltaWeight*derivativeActivation;
            double pervNode = this->previousLayer->nodes[i];

            //更新權重
            this->intoWeights[j][i] -= learningRate*this->wDelta[j][i]*pervNode;
        }

        //更新基底權重
        this->bDelta[j] = derivativeActivation;
        this->intoBiases[j] -= learningRate*(this->bDelta[j]*sigmaDeltaWeight);
    }
    //cout << "end\n" << endl;
}

vector<double> HiddenLayer::GetOutput()
{
    return this->nodes;
}

void HiddenLayer::SetNextLayer(BackPropagationLayer* nextLayer)
{
    this->nextLayer.reset(nextLayer);
}

void HiddenLayer::SetPrevLayer(Layer* pervLayer)
{
    this->previousLayer.reset(pervLayer);
}