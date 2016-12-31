#include "libLKYDeepNN/LKYDeepNN.hpp"
#include "libLKYDeepNN/DataSet.hpp"

void DrawTraining(LKYDeepNN _nn, int maxEpochs, int currentEpochs, const vector<vector<double>>& displayData)
{   //size_t numItems = 80;
    string strPngName = "png/訓練途中" + to_string(currentEpochs) + ".png";
    string strPutText = "Epoch:"+to_string(currentEpochs)+"/"+to_string(maxEpochs)+"  Err:" + to_string(_nn.GetTrainError().back());

    //cv::imwrite(strPngName.c_str(),DrawData("訓練途中", displayData, strPutText));
    Draw2DClassificationData("訓練途中", displayData, _nn, strPutText);
    //fgetc(stdin);
}

int main()
{
    vector<vector<double>> trainData = Make2DBinaryTrainingData(1);
    int numHiddenNodesInEachLayer = 5;
    int numHiddenLayers = 10;
    LKYDeepNN nn(2, vector<int>(numHiddenLayers, numHiddenNodesInEachLayer), 2);
    nn.SetActivation(new Tanh());
    //nn.eventInTraining = DrawTraining;//將包有視覺化的事件傳入

    cout << "訓練開始" <<endl;
    int epochs = 1;
    nn.Training(0.0001, epochs, trainData);
    cout << "訓練完成" <<endl;

    // vector<double> outputArray = nn.ForwardPropagation(vector<double>{2,2});
    // cout << "outputArray: ";
    // for (double const output : outputArray)
    // {//print
    //     printf("%lf, ",output);
    // }
    // cout << "順傳遞測試完成" <<endl;
}