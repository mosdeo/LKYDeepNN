#include "Layer.hpp"

#ifndef _BackPropagationLayer_hpp_
#define _BackPropagationLayer_hpp_

class BackPropagationLayer : public Layer
{
    friend class HiddenLayer;
    friend class OutputLayer;

    //前層
    protected: shared_ptr<Layer> previousLayer;

    //順向進入的權重與基底
    protected: vector<vector<double>> intoWeights;
    protected: vector<double> intoBiases;
    public: vector<vector<double>> GetWeights(){ return intoWeights;}
    public: vector<double> GetBiases(){ return intoBiases;}

    //倒傳遞的梯度
    protected: vector<vector<double>> wDelta;
    protected: vector<double> bDelta;

    //活化函數
    protected: shared_ptr<Activation> activation;
    public: void SetActivation(Activation* activation)
    {
        this->activation.reset(activation);//activation;
    }

    public: void InitializeWeights()
    {
        this->intoWeights = MakeMatrix(this->nodes.size(), this->previousLayer->NodesSize(), 1.0);
        this->intoBiases = vector<double>(this->nodes.size(), 0); //numNodes double with value 0
        this->wDelta = MakeMatrix(this->nodes.size(), this->previousLayer->NodesSize(), 0.0);
        this->bDelta = vector<double>(this->intoBiases.size());

        const double hi = 1/(sqrt(this->nodes.size()));
        const double lo = -hi;

        std::random_device rd;     // only used once to initialise (seed) engine
        std::mt19937_64 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
        std::uniform_real_distribution<double> uni_noise(lo, hi); // guaranteed unbiased

        for (size_t j = 0; j < this->intoWeights.size(); ++j)
        {
            for (size_t i = 0; i < this->intoWeights[j].size(); ++i)
            {
                this->intoWeights[j][i] = uni_noise(rng);
            }
        }

        for(double& bias : this->intoBiases)
        {
            bias = uni_noise(rng);
        }

        cout << "Completed BackPropagation Layer InitializeWeights()" << endl;
    }

    public: vector<double> GetOutput()
    {
        vector<double> outputNodeArray(this->nodes.size());
        
        for(size_t i=0 ; i < this->Layer::nodes.size() ; ++i)
        {
            outputNodeArray[i] = get<1>(this->Layer::nodes[i]);
        }
        
        return outputNodeArray;
    }
};

#endif