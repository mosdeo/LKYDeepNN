#include "Layer.hpp"
#include "InputLayer.hpp"
#include "HiddenLayer.hpp"
#include "OutputLayer.hpp"

void OutputLayer::SetActivation(Activation* activation)
{
    this->activation = activation;
}

void OutputLayer::InitializeWeights()
{
    this->intoWeights = MakeMatrix(this->previousLayer->nodes.size(), this->nodes.size(), 1.0);
    this->outBiases = vector<double>(this->nodes.size(), 0); //numNodes double with value 0
    this->wGrads = MakeMatrix(this->previousLayer->nodes.size(), this->nodes.size(), 0.0);
    this->oGrads = vector<double>(this->outBiases.size());

    const double hi = 1/(sqrt(this->nodes.size()));
    const double lo = -hi;

    //std::random_device rd;     // only used once to initialise (seed) engine
    std::mt19937 rng(0);    // random-number engine used (Mersenne-Twister in this case)
    std::uniform_real_distribution<double> uni_noise(lo, hi); // guaranteed unbiased

    for (size_t j = 0; j < this->intoWeights.size(); ++j)
    {
        for (size_t i = 0; i < this->intoWeights[j].size(); ++i)
        {
            this->intoWeights[j][i] = uni_noise(rng);
        }
    }

    for(double& bias : this->outBiases)
    {
        bias = uni_noise(rng);
    }

    cout << "Completed output Layer InitializeWeights()" << endl;
}

void OutputLayer::ForwardPropagation()
{
    //將自己的節點歸零，因為要存放上一級傳來的運算結果，不能累積。
    this->nodes = vector<double>(this->nodes.size() ,0.0);

    //節點的乘積與和
    for (size_t j = 0; j < this->nodes.size(); ++j) // compute i-h sum of weights * inputNodes
    {
        for (size_t i = 0; i < this->previousLayer->nodes.size(); ++i)
        {
            this->nodes[j] += this->previousLayer->nodes[i] * this->intoWeights[i][j]; // note +=
        }

        this->nodes[j] += this->outBiases[j];
    }

    //活化函數
    if(NULL == this->activation)
    {
        //cout << "WARNING: Output Layer 沒有配置活化函數，要做 Regression 嗎？" << endl;
        //exit(EXIT_FAILURE);
    }
    else
    {//將自身節點全部跑一次活化函數
        this->nodes = this->activation->Forward(this->nodes);
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

    for(size_t j=0 ; j < this->wGrads.size() ; j++)
    {
        for(size_t i=0 ; i < this->wGrads[j].size() ; i++)
        {
            double err = this->nodes[i] - desiredOutValues[i];//Output-target
            double derivativeActivation = (NULL == this->activation)?
                1:
                this->activation->Derivative(this->nodes[i]);
            double pervInput = this->previousLayer->nodes[j];
            this->wGrads[j][i] = err*derivativeActivation*pervInput;

            //更新權重
            this->intoWeights[j][i] -= learningRate*this->wGrads[j][i];
        }
    }
    //cout << "end\n" << endl;
}

vector<double> OutputLayer::GetOutput()
{
    return this->nodes;
}

void OutputLayer::SetPrevLayer(HiddenLayer* pervLayer)
{
    this->previousLayer = pervLayer;
}