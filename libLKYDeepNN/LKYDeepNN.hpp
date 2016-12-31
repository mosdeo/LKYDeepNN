#include "Layer.hpp"
#include "InputLayer.hpp"
#include "HiddenLayer.hpp"
#include "OutputLayer.hpp"

class LKYDeepNN
{
    //各層指標
    private: InputLayer* inputLayer;
    private: vector<HiddenLayer*> hiddenLayerArray;
    private: OutputLayer* outputLayer;
    private: Activation* activation = NULL;

    public: ~LKYDeepNN()
    {
        delete inputLayer;
        for (auto hiddenLayer : this->hiddenLayerArray)
        {
            delete hiddenLayer;
        }
        delete outputLayer;
        cout << "~LKYDeepNN() completed." << endl;
    }
    
    public: LKYDeepNN(int numInputNodes, vector<int> numHiddenNodes, int numOutputNodes)
    {
        //===================== step 1: 各層實體配置 ===================== 
        this->inputLayer = new InputLayer();
        this->hiddenLayerArray = vector<HiddenLayer*>(numHiddenNodes.size()); //這行要先做, 不然沒東西傳入InputLayer建構子
        for(auto& aHiddenLayer : this->hiddenLayerArray)
        {
            aHiddenLayer = new HiddenLayer();
        }
        //printf("最後一個隱藏層位址=%p\n",hiddenLayerArray.back());
        this->outputLayer = new OutputLayer();


        //===================== step 2: 各層連結配置 & 節點初始化 =====================
        //noteic: 這一層不能再做實體配置，不然會改變各層的位址，先前建立好的link會壞掉
        // 輸入層
        this->inputLayer->SetNextLayer(hiddenLayerArray.front());
        (this->inputLayer)->SetNode(numInputNodes);

        //隱藏層
        if(1 ==  this->hiddenLayerArray.size())
        {//如果hidden layer只有一層就這樣處理
            int numNode = numHiddenNodes.front();
            this->hiddenLayerArray.front()->SetPrevLayer((Layer*)inputLayer);
            this->hiddenLayerArray.front()->SetNextLayer((Layer*)outputLayer);
            this->hiddenLayerArray.front()->SetNode(numNode);
            //this->hiddenLayerArray.front()->SetActivation(new Tanh());
            this->hiddenLayerArray.front()->SetActivation(this->activation);
        }
        else
        {//如果hidden layer是多層
            for(vector<HiddenLayer*>::iterator it=hiddenLayerArray.begin(); it!=hiddenLayerArray.end(); it++)
            {
                //取得此層節點數
                int numNode = numHiddenNodes[it-hiddenLayerArray.begin()];
                
                if(it==hiddenLayerArray.begin())
                {//第一個隱藏層連結配置
                    this->hiddenLayerArray.front()->SetPrevLayer((Layer*)inputLayer);
                    this->hiddenLayerArray.front()->SetNextLayer((Layer*)*(it+1));
                }
                else if(it==hiddenLayerArray.end()-1)
                {//最後一個隱藏層連結配置
                    this->hiddenLayerArray.back()->SetPrevLayer((Layer*)*(it-1));
                    this->hiddenLayerArray.back()->SetNextLayer((Layer*)outputLayer);
                    //printf("最後一個隱藏層位址=%p\n",*it);
                }
                else
                {//中間隱藏層連結配置
                    (*it)->SetPrevLayer((Layer*)*(it-1));
                    (*it)->SetNextLayer((Layer*)*(it+1));
                    //cout << "中間隱藏層連結配置" << endl;
                }

                //節點 & 活化函數配置
                (*it)->SetNode(numNode);
                //(*it)->SetActivation(new Tanh());
                (*it)->SetActivation(this->activation);
            }
        }

        //輸出層連結 & 活化函數配置
        this->outputLayer->SetPrevLayer(hiddenLayerArray.back());
        this->outputLayer->SetNode(numOutputNodes);
        //this->outputLayer->SetActivation(new Tanh());
        
        //printf("最後一個隱藏層位址=%p\n",hiddenLayerArray.back());
        
        //===================== step 3: 統一權重初始化 =====================
        this->InitializeWeights();
    }

    public: void InitializeWeights()
    {
        //權重初始化
        //cout << "權重初始化" << endl;
        for (auto hiddenLayer : this->hiddenLayerArray)
        {
            hiddenLayer->InitializeWeights();
        }
        this->outputLayer->InitializeWeights();
    }

    public: void SetActivation(Activation* activation)
    {
        this->activation = activation;

        for (auto hiddenLayer : this->hiddenLayerArray)
        {
            hiddenLayer->SetActivation(activation);
        }
        //outputLayer->SetActivation(activation);
    }

    public: vector<double> ForwardPropagation(vector<double> inputArray)
    {
        this->ActivationExistCheck();

        //輸入資料到輸入層節點
        //cout << "輸入資料到輸入層節點" << endl;
        this->inputLayer->Input(inputArray);

        //隱藏層順傳遞
        //cout << "隱藏層順傳遞" << endl;
        for (auto hiddenLayer : hiddenLayerArray)
        {
            hiddenLayer->ForwardPropagation();
        }

        //最後一個隱藏層到輸出層的順傳遞
        //cout << "最後一個隱藏層到輸出層的順傳遞" << endl;
        this->outputLayer->ForwardPropagation();

        //回傳輸出層輸出
        return this->outputLayer->GetOutput();
    }

    public: void Training(double learningRate, vector<double> desiredOutValues)
    {
        this->ActivationExistCheck();

        this->outputLayer->BackPropagation(learningRate, desiredOutValues);

        for(vector<HiddenLayer*>::reverse_iterator r_it=hiddenLayerArray.rbegin(); r_it!=hiddenLayerArray.rend(); r_it++)
        {
            (*r_it)->BackPropagation(learningRate);
        }
    }

    private: void ActivationExistCheck()
    {//活化函數檢查
        if(NULL == this->activation)
        {
            cout << "ERROR: 沒有配置活化函數." << endl;
            exit(EXIT_FAILURE);
        }
    }
};