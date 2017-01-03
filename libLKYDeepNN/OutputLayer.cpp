#include "Layer.hpp"
#include "InputLayer.hpp"
#include "HiddenLayer.hpp"
#include "OutputLayer.hpp"

void OutputLayer::ForwardPropagation()
{
    //將自己的節點歸零，因為要存放上一級傳來的運算結果，不能累積。
    this->nodes = vector<double>(this->nodes.size() ,0.0);

    //節點的乘積與和 //[this][perv]
    for (size_t j = 0; j < this->nodes.size(); ++j) // compute i-h sum of weights * inputNodes
    {
        for (size_t i = 0; i < this->previousLayer->nodes.size(); ++i)
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
        cout << "WARNING: Output Layer 沒有配置活化函數，要做 Regression 嗎？ 那應該配置 Linear" << endl;
        exit(EXIT_FAILURE);
    }
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

    //printf("OutputLayer: this->wDelta.size() = %ld, this->wDelta[0].size() = %ld\n", this->wDelta.size(), this->wDelta[0].size());

    //[this][perv]
    for(size_t j=0 ; j < this->wDelta.size() ; j++)
    {
        //double err = this->nodes[j] - desiredOutValues[j];//Output-target(Square Loss Function的微分)
        double err = this->lossFunction->Derivative(this->nodes[j], desiredOutValues[j]); 
        double derivativeActivation = this->activation->Derivative(this->nodes[j]);

        for(size_t i=0 ; i < this->wDelta[j].size() ; i++)
        {
            double pervInput = this->previousLayer->nodes[i];
            this->wDelta[j][i] = err*derivativeActivation;

            //更新權重
            this->intoWeights[j][i] -= learningRate*(this->wDelta[j][i]*pervInput);
        }

        //更新基底權重
        this->bDelta[j] = err*derivativeActivation;
        this->intoBiases[j] -= learningRate*this->bDelta[j];
    }

    //cout << "end\n" << endl;
}

vector<double> OutputLayer::GetOutput()
{
    return this->nodes;
}

void OutputLayer::SetPrevLayer(HiddenLayer* pervLayer)
{
    this->previousLayer.reset(pervLayer);
}

void OutputLayer::SetLossFunction(LossFunction* lossFunction)
{
    this->lossFunction.reset(lossFunction);
}