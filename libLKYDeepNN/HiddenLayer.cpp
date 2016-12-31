#include "Layer.hpp"
#include "InputLayer.hpp"
#include "HiddenLayer.hpp"
#include "OutputLayer.hpp"
using namespace std;

void HiddenLayer::SetActivation(Activation* activation)
{
    this->activation = activation;
}

void HiddenLayer::InitializeWeights()
{   
    this->intoWeights = MakeMatrix(this->previousLayer->nodes.size(), this->nodes.size(), 1.0);
    this->hiddenBiases = vector<double>(this->nodes.size() ,0); //numNodes double with value 0
    this->wGrads = MakeMatrix(this->previousLayer->nodes.size(), this->nodes.size(), 0.0);
    this->oGrads = vector<double>(this->hiddenBiases.size());

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

    for(double& bias : this->hiddenBiases)
    {
        bias = uni_noise(rng);
    }

    // cout << "  prev Layer: " << this->previousLayer->ToString() << endl;
    // cout << "  next Layer: " << this->nextLayer->ToString() << endl;
    // cout << "Completed hidden Layer InitializeWeights()" << endl;
}

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
        for (size_t i = 0; i < this->previousLayer->nodes.size(); ++i)
        {
            this->nodes[j] += this->previousLayer->nodes[i] * this->intoWeights[i][j]; // note +=
        }

        //cout << "mark" << endl;
        //cout << "this->hiddenBiases[j] = " << this->hiddenBiases[j] << endl;
        this->nodes[j] += this->hiddenBiases[j];
        //cout << "mark" << endl;
    }

    //活化函數
    if(NULL == this->activation)
    {
        cout << "ERROR: 忘記配置活化函數" << endl;
        exit(EXIT_FAILURE);
    }
    else
    {//將自身節點全部跑一次活化函數
        this->nodes = this->activation->Forward(this->nodes);
    }
}

void HiddenLayer::BackPropagation(double learningRate)
{
    // cout << "HiddenLayer::BackPropagation" << endl;
    // cout << "  prev Layer: " <<this->previousLayer->ToString() << endl;
    // cout << "  next Layer: " << this->nextLayer->ToString() << endl;

    for(size_t j=0 ; j < this->wGrads.size() ; j++)
    {
        for(size_t i=0 ; i < this->wGrads[j].size() ; i++)
        {
            // if(typeid(*(this->nextLayer)) == typeid(InputLayer))
            // {
            //     cout << "ERROR: HiddenLayer 的下一層不能是 InputLayer." << endl;
            //     exit(EXIT_FAILURE);
            // }

            // if(typeid(*(this->nextLayer)) == typeid(OutputLayer))
            // {
            //     cout << "ERROR: HiddenLayer 的下一層不能是 OutputLayer." << endl;
            //     exit(EXIT_FAILURE);
            // }          

            if(NULL == this->previousLayer)
            {
                cout << "NULL == this->previousLayer" << endl;
                exit(EXIT_FAILURE);
            }

            double pervGrad = this->nextLayer->wGrads[j][i]; //取得下一層算過的梯度
            double derivativeActivation = this->activation->Derivative(this->nodes[i]);//取得進出這個節點的梯度
            double pervInput = this->previousLayer->nodes[j];//取得上一個節點的值
            this->wGrads[j][i] = pervGrad*derivativeActivation*pervInput;

            //更新權重
            this->intoWeights[j][i] -= learningRate*this->wGrads[j][i];
        }
    }
    //cout << "end\n" << endl;
}

vector<double> HiddenLayer::GetOutput()
{
    return this->nodes;
}

void HiddenLayer::SetNextLayer(Layer* nextLayer)
{
    this->nextLayer = nextLayer;
}

void HiddenLayer::SetPrevLayer(Layer* pervLayer)
{
    this->previousLayer = pervLayer;
}