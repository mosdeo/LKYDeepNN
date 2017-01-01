#include "Layer.hpp"
#include "InputLayer.hpp"
#include "HiddenLayer.hpp"
#include "OutputLayer.hpp"
#include <thread>
#include <algorithm>
using namespace std;

class LKYDeepNN
{
    //各層指標
    private: InputLayer* inputLayer;
    private: vector<HiddenLayer*> hiddenLayerArray;
    private: OutputLayer* outputLayer;
    private: shared_ptr<Activation> hiddenActivation;
    private: shared_ptr<Activation> outputActivation;

    private: vector<double> trainError;
    public: vector<double> GetTrainError(){return this->trainError;}

    public: ~LKYDeepNN()
    {
        cout << "~LKYDeepNN() completed." << endl;
    }

    public: string ToString()
    {
        string strMsg;
        strMsg  = "====== LKYDeepNN ======\n";
        strMsg += "Layer: \n";
        strMsg += "   Input: "+to_string(inputLayer->NodesSize())+"\n";
        strMsg += "  Hidden: ";for(HiddenLayer* numNode : hiddenLayerArray){strMsg += to_string(numNode->NodesSize()) + "-";} strMsg += "\n";
        strMsg += "  Output: "+to_string(outputLayer->NodesSize())+"\n";

        strMsg += "Activation Function: \n";
        strMsg += "  Hidden: "+string(typeid(*hiddenActivation).name()).substr(1)+"\n";
        strMsg += "  Output: "+string(typeid(*outputActivation).name()).substr(1)+"\n";
        strMsg += "=======================\n";

        return strMsg;
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

                //節點配置
                (*it)->SetNode(numNode);
            }
        }

        //輸出層連結
        this->outputLayer->SetPrevLayer(hiddenLayerArray.back());
        this->outputLayer->SetNode(numOutputNodes);        
        //printf("最後一個隱藏層位址=%p\n",hiddenLayerArray.back());
        
        //===================== step 3: 統一權重初始化 =====================
        this->InitializeWeights();

        //===================== step 4: 預設活化函數配置 =====================
        this->SetActivation(new Tanh(), new Softmax());
        //this->SetActivation(make_shared<ReLU>(), make_shared<Softmax>());
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

    public: void SetActivation(Activation* hiddenLayerActivation, Activation* outputLayerActivation)
    {
        this->hiddenActivation.reset(hiddenLayerActivation);
        this->outputActivation.reset(outputLayerActivation);

        for (auto hiddenLayer : this->hiddenLayerArray)
        {
            hiddenLayer->SetActivation(hiddenLayerActivation);
        }
        outputLayer->SetActivation(outputLayerActivation);
    }

    // public: void SetActivation(shared_ptr<Activation> hiddenLayerActivation, shared_ptr<Activation> outputLayerActivation)
    // {
    //     this->hiddenActivation = hiddenLayerActivation;
    //     this->outputActivation = outputLayerActivation;

    //     for (auto hiddenLayer : this->hiddenLayerArray)
    //     {
    //         hiddenLayer->SetActivation(hiddenLayerActivation.get());
    //     }
    //     outputLayer->SetActivation(outputLayerActivation.get());
    // }

    public: vector<double> ForwardPropagation(vector<double> aFeaturesAndLabels)
    {
        this->ActivationExistCheck();

        //分離出 feature, 去除label
        vector<double> features(inputLayer->NodesSize());  // features
        std::copy(aFeaturesAndLabels.begin(), aFeaturesAndLabels.begin() + inputLayer->NodesSize(),features.begin());

        //輸入資料到輸入層節點
        //cout << "輸入資料到輸入層節點" << endl;
        this->inputLayer->Input(features);

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

    public: void Training(double learningRate, int epochs, vector<vector<double>> trainData)
    {
        this->ActivationExistCheck();
        
        vector<double> inputValues(inputLayer->NodesSize(),0);  // features
        vector<double> targetValues(outputLayer->NodesSize(),0); // labels
        
        vector<int> sequence(trainData.size());
        for (size_t i = 0; i < trainData.size(); ++i)
        {
            sequence[i] = i;
        } 
        
        //開始訓練
        for(int currentEpochs=0 ; currentEpochs < epochs ; ++currentEpochs)
        {
            Shuffle(sequence);//訓練順序亂數洗牌
            for (size_t i = 0; i < trainData.size(); ++i)
            {
                size_t rand_i = sequence[i];

                //分離出 feature & label
                std::copy(trainData[rand_i].begin(), trainData[rand_i].begin() + inputLayer->NodesSize(), inputValues.begin());
                std::copy(trainData[rand_i].begin()+inputLayer->NodesSize(),
                        trainData[rand_i].begin()+inputLayer->NodesSize()+outputLayer->NodesSize(),
                        targetValues.begin());

                //檢驗資料是否正確被切開
                // cout << "==================" << endl;
                // for (double const output : trainData[i]){ printf("  %lf, ",output);}cout << endl;
                // for (double const output : inputValues){ printf("x=%lf, ",output);}
                // for (double const output : targetValues){ printf("t=%lf, ",output);}cout << endl;
                // cout << "==================" << endl;

                //順傳遞
                this->ForwardPropagation(inputValues);                

                //倒傳遞
                this->outputLayer->BackPropagation(learningRate, targetValues);
                for(vector<HiddenLayer*>::reverse_iterator r_it=hiddenLayerArray.rbegin(); r_it!=hiddenLayerArray.rend(); r_it++)
                {
                    (*r_it)->BackPropagation(learningRate);
                }
            }

            trainError.push_back(this->MeanSquaredError(trainData));
            if(0 == currentEpochs % 200)
                cout << "MeanSquaredError = " << this->GetTrainError().back() << endl;

            if(NULL != this->eventInTraining) //繪製訓練過程testData
            {//呼叫事件
                thread th(this->eventInTraining, this, epochs, currentEpochs, trainData);
                th.join();
            }
        }

        cout << "訓練完成" << endl;
    }

    private: void ActivationExistCheck()
    {//活化函數檢查
        if(NULL == this->hiddenActivation)
        {
            cout << "ERROR: 沒有配置活化函數." << endl;
            exit(EXIT_FAILURE);
        }
    }

    private: void Shuffle(vector<int> sequence) // an instance method
    {
        //std::default_random_engine{}; //relatively casual, inexpert, and/or lightweight use.
        std::srand(time(NULL));
        std::random_shuffle(sequence.begin(), sequence.end());
    } // Shuffle

    private: double MeanSquaredError(vector<vector<double>> data)
    {
        // MSE == average squared error per training item
        double sumSquaredError = 0.0;
        vector<double> features(inputLayer->NodesSize());  // first numInputNodes values in trainData
        vector<double> targets(outputLayer->NodesSize()); // last numOutputNodes values

        // walk thru each training case
        for (size_t i = 0; i < data.size(); ++i)
        {
            std::copy(data[i].begin(), data[i].begin() + inputLayer->NodesSize(), features.begin());
            std::copy(data[i].begin() + inputLayer->NodesSize(),
                    data[i].begin() + inputLayer->NodesSize() + outputLayer->NodesSize(),
                    targets.begin());

            // //檢驗資料是否正確被切開
            // cout << "==================" << endl;
            // for (double const output : data[i]){ printf("  %lf, ",output);}cout << endl;
            // for (double const output : features){ printf("x=%lf, ",output);}
            // for (double const output : targets){ printf("t=%lf, ",output);}cout << endl;
            // cout << "==================" << endl;

            vector<double> yValues = this->ForwardPropagation(features);

            for (size_t j = 0; j < yValues.size(); ++j)
            {
                double err = targets[j] - yValues[j];
                sumSquaredError += err * err;
            }
        }
        return sumSquaredError / data.size();
    } // Error

    public: void (*eventInTraining)(LKYDeepNN* ,int ,int ,const vector<vector<double>>& displayData) = NULL;

    // LKYDeepNN& operator=(const LKYDeepNN& Obj)
    // {
    //     cout << "LKYDeepNN& operator=(const LKYDeepNN& Obj)" << endl;
    //     exit(EXIT_SUCCESS);
    // }

    // LKYDeepNN(const LKYDeepNN& Obj)
    // {
    //     cout << "LKYDeepNN(const LKYDeepNN& Obj)" << endl;

    //     // private: InputLayer* inputLayer;
    //     //this->inputLayer = Obj.inputLayer

    //     // private: vector<HiddenLayer*> hiddenLayerArray;
    //     // private: OutputLayer* outputLayer;
    //     // private: Activation* activation = NULL;

    //     // private: vector<double> trainError;
    //     this->trainError = Obj.trainError;
    //     exit(EXIT_SUCCESS);
    // }
};

