#include "libLKYDeepNN/LKYDeepNN.hpp"
#include "libLKYDeepNN/DataSet.hpp"

void DrawTraining(LKYDeepNN* _nn, int maxEpochs, int currentEpochs, const vector<vector<double>>& displayData)
{   //size_t numItems = 80;
    string strPngName = "png/訓練途中" + to_string(currentEpochs) + ".png";
    string strPutText = "Epoch:"+to_string(currentEpochs)+"/"+to_string(maxEpochs)+"  Err:" + to_string(_nn->GetTrainError().back());

    //cv::imwrite(strPngName.c_str(),DrawData("訓練途中", displayData, strPutText));
    Draw2DClassificationData("訓練途中", displayData, _nn, strPutText);
    //Draw2DRegressionData("訓練途中", displayData, _nn, strPutText);
    //fgetc(stdin);
}

int main()
{
    // vector<vector<double>> trainData = Make2DBinaryTrainingData();//
    //vector<vector<double>> trainData = classifyCircleData();//
    //vector<vector<double>> trainData = WaveData();//
    vector<vector<double>> trainData = classifySpiralData();
    //int numHiddenNodesInEachLayer = 4;
    //int numHiddenLayers = 3;
    LKYDeepNN nn(5, vector<int>{16,16}, 2);
    //LKYDeepNN nn(2, vector<int>(numHiddenLayers, numHiddenNodesInEachLayer), 2);
    //nn.SetActivation(new Tanh(), new Softmax());
    cout << nn.ToString() << endl;
    nn.eventInTraining = DrawTraining;//將包有視覺化的事件傳入

    cout << "訓練開始" <<endl;
    double learningRate = 0.0001;
    int epochs = 9999999;
    printf("learningRate=%lf\n",learningRate);
    nn.Training(learningRate, epochs, trainData);
    //nn.Training(0.001, epochs, trainData);
    cout << "訓練完成" <<endl;
}