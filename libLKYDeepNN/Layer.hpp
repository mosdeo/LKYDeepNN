#ifndef _Layer_hpp_
#define _Layer_hpp_

#include <iostream>
#include <cstdlib>
#include <vector>
#include <random>
#include <typeinfo>
#include <memory>
#include <tuple>
#include "Activation.hpp"
using namespace std;

class Layer
{
    friend class HiddenLayer;
    friend class OutputLayer;

    //層節點 (get<0>:節點之前, get<1>:節點之後)
    protected: vector<tuple<double,double>> nodes;
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
        this->nodes  = vector<tuple<double,double>>(numNodes);
    }
};

#endif