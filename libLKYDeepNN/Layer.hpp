#ifndef _Layer_hpp_
#define _Layer_hpp_

#include <iostream>
#include <cstdlib>
#include <vector>
#include <random>
#include <typeinfo>
#include <memory>
#include "Activation.hpp"
using namespace std;

class Layer
{
    friend class HiddenLayer;

    //層節點
    protected: vector<double> nodes;
    public: size_t NodesSize()
    {
        return this->nodes.size();
    }

    protected: vector<vector<double>> MakeMatrix(int rows, int cols, double v) // helper for ctor, Train
    {
        vector<double> row;
        row.assign(cols, v); //配置一個row的大小
        vector<vector<double>> array_2D;
        array_2D.assign(rows, row); //配置2維

        return array_2D;
    }

    public: virtual string ToString(){ return "class Layer";} 

    public: void SetNode(int numNodes)
    {
        this->nodes  = vector<double>(numNodes);
    }
};

class BackPropagationLayer : public Layer
{
    friend class HiddenLayer;

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
};

#endif