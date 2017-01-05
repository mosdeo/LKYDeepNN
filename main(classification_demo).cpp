#include "libLKYDeepNN/LKYDeepNN.hpp"
#include "libLKYDeepNN/DataSet.hpp"

void DrawTraining(LKYDeepNN* _nn, int maxEpochs, int currentEpochs, const vector<vector<double>>& displayData)
{ 
    string strPngName = "classification_Spiral_demo_PNGs/訓練途中" + to_string(currentEpochs) + ".png";
    string strPutText = "Epoch:"+to_string(currentEpochs)+"/"+to_string(maxEpochs)+"  Err:" + to_string(_nn->GetTrainLoss().back());
    cv::Mat shot = Draw2DClassificationData("訓練途中", displayData, _nn, strPutText);

    //PNG maker
    if(0 == currentEpochs % 10)
    {    
        //cv::imwrite(strPngName.c_str(), shot);
    }
    //fgetc(stdin);
}

int main()
{
    vector<tuple<double,double>> nodes{tuple<double, double>(-1000,0),
                                       tuple<double, double>(2000,0),
                                       tuple<double, double>(3000,0)};



    Softmax softmax;
    nodes = softmax.Forward(nodes);

    for(auto node : nodes)
    {
        printf("%lf ",get<0>(node));
    }cout << endl;

    for(auto node : nodes)
    {
        printf("%lf ",get<1>(node));
    }cout << endl;

    // exit(EXIT_SUCCESS);

    // vector<vector<double>> trainData = Make2DBinaryTrainingData();//
    double bias = 0;
    //vector<vector<double>> trainData = classifyCircleData(bias ,bias);//
    vector<vector<double>> trainData = classifySpiralData(bias ,bias);
    //int numHiddenNodesInEachLayer = 8;
    //int numHiddenLayers = 3;
    //LKYDeepNN nn(2, vector<int>(numHiddenLayers, numHiddenNodesInEachLayer), 2);
    LKYDeepNN nn(trainData.front().size()-2, vector<int>{8,8,8}, 2);
    nn.SetActivation(new ReLU(), new Softmax());
    //nn.SetLossFunction(new CrossEntropy());
    nn.SetLossFunction(new Square());
    cout << nn.ToString() << endl;
    nn.eventInTraining = DrawTraining;//將包有視覺化的事件傳入

    cout << "訓練開始" <<endl;
    double learningRate = 0.007;
    int epochs = 3500;
    printf("learningRate=%lf\n",learningRate);
    nn.Training(learningRate, epochs, trainData);
    cout << nn.WeightsToString()<<endl;
    cout << "訓練完成" <<endl;
    cv::waitKey(0);
}